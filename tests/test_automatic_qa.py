import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.automatic_qa import AutomaticQATestService
from automotive_graphrag.question_generation import GeneratedQuestion
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import BatchQuestion, GoldEvidence, QuestionSetService
from automotive_graphrag.source_sampling import SourceSample


class ConcurrentQuery:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.models = []
        self.lock = threading.Lock()

    def ask(self, project_id, question, method="local", chat_model=None):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.models.append(chat_model)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        now = datetime.now(timezone.utc).isoformat()
        return QueryResult(
            query_id=question, project_id=project_id, question=question, method=method,
            status="COMPLETED", answer="回答", error=None, started_at=now,
            completed_at=now, duration_seconds=0.02,
        )


class NoJudge:
    def evaluate(self, *args, **kwargs):
        raise ProjectError("test judge unavailable")

    def clear_result(self, *args, **kwargs):
        return None


class NoRetrieval:
    def evaluate(self, *args, **kwargs):
        raise ProjectError("test gold unavailable")


class FakeSampling:
    def __init__(self, documents):
        self.samples = [
            SourceSample(
                sample_id=name + "-chunk", chunk_id=name + "-chunk", project_id="TEST",
                section_id=name, section_name=name, document_id=name, page=1, block_id="b1",
                content_type="general", character_count=100, text="原文內容" * 30,
            )
            for name in documents
        ]

    def scan(self, project_id, minimum_characters=1):
        return self.samples


class ConcurrentGeneration:
    def __init__(self):
        self.active = 0
        self.maximum_active = 0
        self.active_by_document = {}
        self.maximum_by_document = {}
        self.lock = threading.Lock()

    def generate_for_document(self, project_id, document_id, samples, count, model, **kwargs):
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            self.active_by_document[document_id] = self.active_by_document.get(document_id, 0) + 1
            self.maximum_by_document[document_id] = max(
                self.maximum_by_document.get(document_id, 0), self.active_by_document[document_id]
            )
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
            self.active_by_document[document_id] -= 1
        sample = next(item for item in samples if item.document_id == document_id)
        return tuple(
            GeneratedQuestion(
                question_id=f"generated-{document_id}-{index}",
                question=f"{document_id} 的測試問題 {index}？",
                reference_answer="依照手冊內容處理。",
                gold_evidence=(GoldEvidence(document_id, (1,), (sample.chunk_id,)),),
                source_sample_ids=(sample.sample_id,), difficulty="simple",
                generation_status="pending_review",
            )
            for index in range(count)
        )


class AutomaticQATests(unittest.TestCase):
    def test_answer_existing_persists_answers_without_judging(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProjectStore(Path(temporary) / "projects")
            store.create(
                project_id="TEST", display_name="Test", vehicle_name="Vehicle", manual_version="Version"
            )
            store.update_status("TEST", "INDEXED")
            sets = QuestionSetService(store)
            question_set = sets.create_with_questions(
                "TEST", "automatic", [BatchQuestion("Q0001", "問題", "正解")]
            )
            query = ConcurrentQuery()
            service = AutomaticQATestService(
                store, object(), object(), query, sets, NoJudge(), NoRetrieval()
            )

            with self.assertRaisesRegex(ProjectError, "完成所有題目"):
                service.evaluate_existing(
                    "TEST", question_set.question_set_id,
                    "gpt-4.1-mini", "gpt-4o-mini", "local",
                )

            report = service.answer_existing(
                "TEST", question_set.question_set_id, "gpt-4.1-mini", "local", 1
            )

            self.assertIsNone(report.judge)
            self.assertEqual(report.question_set.questions[0].answer, "回答")
            self.assertEqual(report.question_set.questions[0].status, "COMPLETED")
            evaluated = service.evaluate_existing(
                "TEST", question_set.question_set_id, "gpt-4.1-mini", "gpt-4o-mini", "local"
            )
            self.assertEqual(len(query.models), 1)
            self.assertIn("test judge unavailable", evaluated.judge_error)

    def test_answer_concurrency_and_model_are_applied_to_all_questions(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = ProjectStore(Path(temporary) / "projects")
            store.create(
                project_id="TEST", display_name="Test", vehicle_name="Vehicle", manual_version="Version"
            )
            store.update_status("TEST", "INDEXED")
            query = ConcurrentQuery()
            sets = QuestionSetService(store)
            question_set = sets.create_with_questions(
                "TEST", "automatic", [
                    BatchQuestion("Q0001", "問題一", "答案一"),
                    BatchQuestion("Q0002", "問題二", "答案二"),
                    BatchQuestion("Q0003", "問題三", "答案三"),
                ],
            )
            service = AutomaticQATestService(
                store, object(), object(), query, sets, NoJudge(), NoRetrieval()
            )

            report = service.run_existing(
                "TEST", question_set.question_set_id, "gpt-4.1-mini",
                "gpt-4o-mini", "local", answer_concurrency=2,
            )

            self.assertEqual(query.maximum_active, 2)
            self.assertEqual(query.models, ["gpt-4.1-mini"] * 3)
            self.assertTrue(all(item.status == "COMPLETED" for item in report.question_set.questions))
            self.assertIn("test judge unavailable", report.judge_error)

    def test_parallel_generation_never_overlaps_for_the_same_pdf(self):
        for parallel, expected_maximum in ((False, 1), (True, 3)):
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as temporary:
                store = ProjectStore(Path(temporary) / "projects")
                project = store.create(
                    project_id="TEST", display_name="Test", vehicle_name="Vehicle", manual_version="Version"
                )
                store.update_status("TEST", "INDEXED")
                source = store.path_for(project.project_id) / "source"
                for document in ("one.pdf", "two.pdf", "three.pdf", "four.pdf"):
                    (source / document).write_bytes(b"")
                generation = ConcurrentGeneration()
                service = AutomaticQATestService(
                    store, FakeSampling(["one.pdf", "two.pdf", "three.pdf", "four.pdf"]), generation, ConcurrentQuery(),
                    QuestionSetService(store), NoJudge(), NoRetrieval(),
                )

                report = service.generate_and_run(
                    "TEST", 1, parallel, "gpt-4o-mini", "gpt-4o-mini",
                    "gpt-4o-mini", "local", 1,
                )

                self.assertEqual(len(report.question_set.questions), 4)
                self.assertEqual(generation.maximum_active, expected_maximum)
                self.assertEqual(set(generation.maximum_by_document.values()), {1})


if __name__ == "__main__":
    unittest.main()
