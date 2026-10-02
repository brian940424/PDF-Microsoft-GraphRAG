import json
import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.question_generation import QuestionGenerationService
from automotive_graphrag.source_sampling import SourceSamplingService


class FakeGenerationClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, str, str, str]] = []

    def __call__(self, base_url: str, api_key: str, model: str, prompt: str) -> object:
        self.calls.append((base_url, api_key, model, prompt))
        return self.response


class QuestionGenerationServiceTests(unittest.TestCase):
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
        records = [
            self.record("WW", "WW.pdf", 25, "雨刷馬達不作動時，先檢查保險絲與電源供應。" * 20),
            self.record("WW", "WW.pdf", 26, "確認電源正常後，依程序檢查雨刷馬達與線束。" * 20),
            self.record("PG", "PG.pdf", 42, "電源供應系統的保險絲位置與檢查方式。" * 20),
        ]
        processed = self.store.path_for("L33-SM3E") / "processed" / "input.jsonl"
        processed.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
            encoding="utf-8",
        )
        self.sampling = SourceSamplingService(self.store)
        self.sample_batch = self.sampling.sample(
            "L33-SM3E",
            count=3,
            minimum_characters=20,
            seed=1,
        )
        self.connections = ConnectionSettings(self.store.root)
        self.connections.save("https://api.openai.com/v1", "test-key")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def record(section: str, document: str, page: int, text: str) -> dict[str, object]:
        chunk_id = f"L33-SM3E-{section}-p{page:04d}-b01"
        return {
            "id": chunk_id.removesuffix("-b01"),
            "chunk_id": chunk_id,
            "project_id": "L33-SM3E",
            "section_id": section,
            "section_name": section,
            "document_id": document,
            "page": page,
            "block_id": "b01",
            "text": text,
        }

    def test_generate_uses_one_low_cost_call_and_builds_grounded_questions(self) -> None:
        sample_ids = [item.sample_id for item in self.sample_batch.samples]
        client = FakeGenerationClient(
            {
                "questions": [
                    {
                        "question": "雨刷不作動時應先檢查什麼？",
                        "reference_answer": "先檢查保險絲與電源供應。",
                        "source_sample_ids": [sample_ids[0]],
                    },
                    {
                        "question": "電源系統應如何檢查？",
                        "reference_answer": "依手冊檢查保險絲位置。",
                        "source_sample_ids": [sample_ids[1]],
                    },
                ]
            }
        )
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)

        batch = service.generate("L33-SM3E", self.sample_batch.sample_batch_id, 2, "simple", 300)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][2], "gpt-4o-mini")
        self.assertEqual(len(batch.questions), 2)
        self.assertTrue(all(item.generation_status == "pending_review" for item in batch.questions))
        self.assertEqual(batch.questions[0].gold_evidence[0].chunk_ids, (sample_ids[0],))
        self.assertEqual(service.get("L33-SM3E", batch.generation_batch_id), batch)
        self.assertLess(len(client.calls[0][3]), 3000)

    def test_generate_rejects_unknown_source_id_from_model(self) -> None:
        client = FakeGenerationClient(
            {
                "questions": [
                    {
                        "question": "問題",
                        "reference_answer": "答案",
                        "source_sample_ids": ["invented-source"],
                    }
                ]
            }
        )
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)

        with self.assertRaisesRegex(ProjectError, "不存在的 sample_id"):
            service.generate("L33-SM3E", self.sample_batch.sample_batch_id, 1)

    def test_cross_section_requires_sources_from_two_sections(self) -> None:
        ww_batch = self.sampling.sample(
            "L33-SM3E",
            count=2,
            section_ids=["WW"],
            minimum_characters=20,
        )
        client = FakeGenerationClient({"questions": []})
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)

        with self.assertRaisesRegex(ProjectError, "兩個不同章節"):
            service.generate("L33-SM3E", ww_batch.sample_batch_id, 1, "cross_section")

        self.assertEqual(client.calls, [])

    def test_human_edit_approval_rejection_and_question_set_export(self) -> None:
        sample_id = self.sample_batch.samples[0].sample_id
        client = FakeGenerationClient(
            {
                "questions": [
                    {
                        "question": "原問題",
                        "reference_answer": "原答案",
                        "source_sample_ids": [sample_id],
                    },
                    {
                        "question": "淘汰問題",
                        "reference_answer": "淘汰答案",
                        "source_sample_ids": [self.sample_batch.samples[1].sample_id],
                    },
                ]
            }
        )
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)
        batch = service.generate("L33-SM3E", self.sample_batch.sample_batch_id, 2)

        updated = service.update_question(
            "L33-SM3E",
            batch.generation_batch_id,
            batch.questions[0].question_id,
            "編輯後問題",
            "編輯後答案",
            "approved",
        )
        service.update_question(
            "L33-SM3E",
            batch.generation_batch_id,
            batch.questions[1].question_id,
            batch.questions[1].question,
            batch.questions[1].reference_answer,
            "rejected",
        )
        self.assertEqual(updated.questions[0].question, "編輯後問題")

        export_path = service.export_question_set("L33-SM3E", batch.generation_batch_id)
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["questions"]), 1)
        self.assertEqual(payload["questions"][0]["question"], "編輯後問題")
        self.assertEqual(payload["questions"][0]["generation_status"], "approved")

    def test_app_builds_with_question_generation_review_controls(self) -> None:
        app = create_app(self.root / "ui-projects")

        labels = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }
        self.assertIn("生成題數", labels)
        self.assertIn("選擇要審核的題目", labels)
        self.assertIn("Question Set JSON", labels)
        page_values = {
            component.get_config().get("label"): component.get_config().get("value")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
            and component.get_config().get("label") in {"起始頁碼", "結束頁碼"}
        }
        self.assertEqual(page_values, {"起始頁碼": 1, "結束頁碼": 1})


if __name__ == "__main__":
    unittest.main()
