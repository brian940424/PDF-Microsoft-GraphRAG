import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import BatchQuestion, GoldEvidence, QuestionSetService
from automotive_graphrag.retrieval_evaluation import RetrievalEvaluationService, compute_source_metrics


def evidence(rank: int, chunk_id: str, document_id: str = "WW.pdf", page: int = 25) -> Evidence:
    return Evidence(
        evidence_id=f"E{rank}",
        rank=rank,
        context_id=str(rank),
        text_unit_id=f"tu-{chunk_id}",
        chunk_id=chunk_id,
        section_id="WW",
        section_name="Wiper & Washer",
        document_id=document_id,
        page=page,
        block_id="b01",
        text=f"來源 {chunk_id}",
    )


class EvaluationQueryRunner:
    def ask(self, project_id: str, question: str, method: str = "local", source: str | None = None) -> QueryResult:
        evidence_by_question = {
            "問題一": (
                evidence(1, "irrelevant", page=10),
                evidence(2, "gold-q1", page=25),
            ),
            "問題二": (evidence(1, "gold-q2", document_id="PG.pdf", page=42),),
            "問題三": (evidence(1, "irrelevant-q3", page=11),),
        }
        now = datetime.now(timezone.utc).isoformat()
        return QueryResult(
            query_id=f"query-{question}",
            project_id=project_id,
            question=question,
            method=method,
            status="COMPLETED",
            answer="回答",
            error=None,
            started_at=now,
            completed_at=now,
            duration_seconds=0.2,
            evidence=evidence_by_question[question],
        )


class RetrievalEvaluationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = ProjectStore(self.root / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        self.store.update_status("L33-SM3E", "INDEXED")
        source = self.root / "questions.json"
        source.write_text(
            json.dumps(
                {
                    "questions": [
                        {"number": 1, "question": "問題一", "expected_answer": "答案一", "question_sources": [], "answer_sources": []},
                        {"number": 2, "question": "問題二", "expected_answer": "答案二", "question_sources": [], "answer_sources": []},
                        {"number": 3, "question": "問題三", "expected_answer": "答案三", "question_sources": [], "answer_sources": []},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.question_sets = QuestionSetService(self.store, EvaluationQueryRunner())
        self.question_set = self.question_sets.import_file("L33-SM3E", source)
        gold_by_question = {
            "Q0001": (GoldEvidence("WW.pdf", (25,), ("gold-q1",)),),
            "Q0002": (GoldEvidence("PG.pdf", (42,), ("gold-q2",)),),
            "Q0003": (GoldEvidence("WW.pdf", (99,), ("missing-q3",)),),
        }
        for question_id, gold in gold_by_question.items():
            self.question_sets.update_gold_evidence(
                "L33-SM3E", self.question_set.question_set_id, question_id, gold
            )
        self.question_sets.run("L33-SM3E", self.question_set.question_set_id)
        self.service = RetrievalEvaluationService(self.store, self.question_sets)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_evaluate_computes_unranked_evidence_metrics_and_latency(self) -> None:
        result = self.service.evaluate("L33-SM3E", self.question_set.question_set_id, top_k=3)

        self.assertAlmostEqual(result.evidence_hit_rate, 2 / 3)
        self.assertAlmostEqual(result.evidence_precision, 0.5)
        self.assertAlmostEqual(result.document_hit_rate, 1.0)
        self.assertEqual(result.no_evidence_question_count, 0)
        self.assertAlmostEqual(result.average_latency_seconds or 0, 0.2)
        self.assertEqual([item.evidence_hit for item in result.items], [True, True, False])
        self.assertEqual([item.relevant_evidence_count for item in result.items], [1, 1, 0])
        self.assertEqual(
            self.service.last_result("L33-SM3E", self.question_set.question_set_id),
            result,
        )

    def test_metrics_do_not_depend_on_context_order(self) -> None:
        question = self.question_sets.get("L33-SM3E", self.question_set.question_set_id).questions[0]
        retrieved = tuple(evidence(rank, f"irrelevant-{rank}", page=10) for rank in range(1, 10)) + (
            evidence(10, "gold-q1"),
        )

        first = self.service._evaluate_question(replace(question, retrieved_evidence=retrieved))
        second = self.service._evaluate_question(replace(question, retrieved_evidence=tuple(reversed(retrieved))))

        self.assertEqual(first.evidence_hit, second.evidence_hit)
        self.assertEqual(first.evidence_precision, second.evidence_precision)
        self.assertEqual(first.document_hit, second.document_hit)

    def test_precision_is_macro_averaged_page_precision_and_empty_retrieval_is_zero(self) -> None:
        gold = (GoldEvidence("WW.pdf", (25,), ("gold-chunk",)),)
        samples = (
            # Different chunk IDs on the same document/page are one relevant page.
            ((evidence(1, "other-chunk", page=25), evidence(2, "another-chunk", page=25)), gold),
            ((evidence(1, "gold-chunk", page=25), evidence(2, "noise", page=10)), gold),
            ((), gold),
        )

        hit_rate, precision, document_hit_rate, no_evidence = compute_source_metrics(samples)

        self.assertAlmostEqual(hit_rate, 2 / 3)
        self.assertAlmostEqual(precision, 0.5)  # mean(1.0, 0.5, 0.0), not pooled 2/3
        self.assertAlmostEqual(document_hit_rate, 2 / 3)
        self.assertEqual(no_evidence, 1)
        empty_result = RetrievalEvaluationService._evaluate_question(
            BatchQuestion("Q", "問題", "答案", gold_evidence=gold, retrieved_evidence=())
        )
        self.assertEqual(empty_result.evidence_precision, 0.0)
        self.assertEqual(empty_result.retrieved_evidence_count, 0)

    def test_gold_without_pages_is_excluded_from_precision_average(self) -> None:
        with_pages = (GoldEvidence("WW.pdf", (25,), ()),)
        without_pages = (GoldEvidence("WW.pdf", (), ("old-chunk-only",)),)

        _, precision, _, _ = compute_source_metrics((
            ((evidence(1, "matching-page", page=25),), with_pages),
            ((evidence(1, "unmatchable", page=25),), without_pages),
        ))

        self.assertEqual(precision, 1.0)
        unscorable = RetrievalEvaluationService._evaluate_question(
            BatchQuestion("Q", "問題", "答案", gold_evidence=without_pages,
                          retrieved_evidence=(evidence(1, "unmatchable", page=25),))
        )
        self.assertIsNone(unscorable.evidence_precision)

    def test_evaluate_can_rerun_only_questions_with_gold_evidence(self) -> None:
        result = self.service.evaluate(
            "L33-SM3E",
            self.question_set.question_set_id,
            rerun_queries=True,
        )

        self.assertEqual(result.question_count, 3)
        self.assertEqual(result.items[0].retrieved_evidence[1].chunk_id, "gold-q1")

    def test_export_writes_summary_json_and_item_csv(self) -> None:
        self.service.evaluate("L33-SM3E", self.question_set.question_set_id)

        json_path, csv_path = self.service.export("L33-SM3E", self.question_set.question_set_id)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["question_count"], 3)
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["question_id"], "Q0001")
        self.assertEqual(rows[0]["evidence_hit"], "True")

    def test_drift_metrics_are_calculated_after_rerunning_and_mapping_its_sources(self) -> None:
        result = self.service.evaluate(
            "L33-SM3E", self.question_set.question_set_id, method="drift", rerun_queries=True
        )

        self.assertAlmostEqual(result.evidence_hit_rate or 0, 2 / 3)
        self.assertAlmostEqual(result.evidence_precision or 0, 0.5)
        self.assertEqual(result.document_hit_rate, 1.0)
        self.assertIn("不使用檢索排名", result.retrieval_metrics_status)

    def test_auto_answers_keep_routed_strategy_and_metrics_skip_global_questions(self) -> None:
        current = self.question_sets.get("L33-SM3E", self.question_set.question_set_id)
        routed = {"Q0001": "local", "Q0002": "global", "Q0003": "basic"}
        results = {
            item.question_id: replace(
                EvaluationQueryRunner().ask("L33-SM3E", item.question, routed[item.question_id]),
                requested_method="auto", routing_reason=f"選 {routed[item.question_id]}",
            )
            for item in current.questions
        }
        updated = self.question_sets.update_answers(
            "L33-SM3E", self.question_set.question_set_id, results, "auto"
        )

        self.assertEqual(updated.method, "auto")
        self.assertEqual([item.routed_method for item in updated.questions], ["local", "global", "basic"])
        self.assertEqual(updated.questions[0].routing_reason, "選 local")
        reloaded = self.question_sets.get("L33-SM3E", self.question_set.question_set_id)
        self.assertEqual(reloaded.questions[2].routed_method, "basic")

        result = self.service.evaluate("L33-SM3E", self.question_set.question_set_id, method="auto")

        self.assertAlmostEqual(result.evidence_hit_rate, 0.5)
        self.assertEqual([item.evidence_hit for item in result.items], [True, None, False])
        self.assertIn("2/3", result.retrieval_metrics_status)

    def test_global_metrics_remain_unavailable_for_unmapped_community_reports(self) -> None:
        result = self.service.evaluate(
            "L33-SM3E", self.question_set.question_set_id, method="global", rerun_queries=True
        )

        self.assertIsNone(result.evidence_hit_rate)
        self.assertIsNone(result.evidence_precision)
        self.assertIsNone(result.document_hit_rate)
        self.assertIn("GLOBAL", result.retrieval_metrics_status)

    def test_evaluate_requires_gold_evidence(self) -> None:
        source = self.root / "no-gold.json"
        source.write_text(
            json.dumps({"questions": [{"number": 1, "question": "問題", "expected_answer": "答案", "question_sources": [], "answer_sources": []}]}),
            encoding="utf-8",
        )
        question_set = self.question_sets.import_file("L33-SM3E", source)

        with self.assertRaisesRegex(ProjectError, "沒有可評估"):
            self.service.evaluate("L33-SM3E", question_set.question_set_id)


if __name__ == "__main__":
    unittest.main()
