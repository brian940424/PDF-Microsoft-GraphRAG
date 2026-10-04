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
from automotive_graphrag.question_sets import GoldEvidence, QuestionSetService
from automotive_graphrag.retrieval_evaluation import RetrievalEvaluationService


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
    def ask(self, project_id: str, question: str, method: str = "local") -> QueryResult:
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
                    "name": "Retrieval 評估集",
                    "questions": [
                        {"question_id": "Q001", "question": "問題一"},
                        {"question_id": "Q002", "question": "問題二"},
                        {"question_id": "Q003", "question": "問題三"},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.question_sets = QuestionSetService(self.store, EvaluationQueryRunner())
        self.question_set = self.question_sets.import_file("L33-SM3E", source)
        gold_by_question = {
            "Q001": (GoldEvidence("WW.pdf", (25,), ("gold-q1",)),),
            "Q002": (GoldEvidence("PG.pdf", (42,), ("gold-q2",)),),
            "Q003": (GoldEvidence("WW.pdf", (99,), ("missing-q3",)),),
        }
        for question_id, gold in gold_by_question.items():
            self.question_sets.update_gold_evidence(
                "L33-SM3E", self.question_set.question_set_id, question_id, gold
            )
        self.question_sets.run("L33-SM3E", self.question_set.question_set_id)
        self.service = RetrievalEvaluationService(self.store, self.question_sets)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_evaluate_computes_recall_mrr_rank_accuracy_and_latency(self) -> None:
        result = self.service.evaluate("L33-SM3E", self.question_set.question_set_id, top_k=3)

        self.assertAlmostEqual(result.recall_at_5, 2 / 3)
        self.assertAlmostEqual(result.recall_at_10, 2 / 3)
        self.assertAlmostEqual(result.mrr, 0.5)
        self.assertEqual(result.average_first_relevant_rank, 1.5)
        self.assertEqual(result.evidence_source_accuracy, 0.5)
        self.assertAlmostEqual(result.average_latency_seconds or 0, 0.2)
        self.assertEqual([item.first_relevant_rank for item in result.items], [2, 1, None])
        self.assertEqual([item.passed_at_k for item in result.items], [True, True, False])
        self.assertEqual(
            self.service.last_result("L33-SM3E", self.question_set.question_set_id),
            result,
        )

    def test_recall_at_10_includes_a_relevant_result_at_rank_ten(self) -> None:
        question = self.question_sets.get("L33-SM3E", self.question_set.question_set_id).questions[0]
        retrieved = tuple(evidence(rank, f"irrelevant-{rank}", page=10) for rank in range(1, 10)) + (
            evidence(10, "gold-q1"),
        )

        item = self.service._evaluate_question(replace(question, retrieved_evidence=retrieved), top_k=10)

        self.assertFalse(item.recall_at_5)
        self.assertTrue(item.recall_at_10)
        self.assertEqual(item.first_relevant_rank, 10)

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
        self.assertEqual(rows[0]["question_id"], "Q001")
        self.assertEqual(rows[0]["first_relevant_rank"], "2")

    def test_evaluate_requires_gold_evidence(self) -> None:
        source = self.root / "no-gold.json"
        source.write_text(
            json.dumps({"name": "No Gold", "questions": [{"question_id": "Q1", "question": "問題"}]}),
            encoding="utf-8",
        )
        question_set = self.question_sets.import_file("L33-SM3E", source)

        with self.assertRaisesRegex(ProjectError, "沒有可評估"):
            self.service.evaluate("L33-SM3E", question_set.question_set_id)


if __name__ == "__main__":
    unittest.main()
