import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.question_generation import QuestionGenerationService, normalize_question
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
        self.assertIn("必須使用繁體中文", client.calls[0][3])
        self.assertIn("參考答案要完整且具體", client.calls[0][3])

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

    def test_generate_rejects_english_only_question_or_answer(self) -> None:
        sample_id = self.sample_batch.samples[0].sample_id
        client = FakeGenerationClient(
            {
                "questions": [
                    {
                        "question": "What should be checked first?",
                        "reference_answer": "Check the fuse and power supply.",
                        "source_sample_ids": [sample_id],
                    }
                ]
            }
        )
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)

        with self.assertRaisesRegex(ProjectError, "未使用繁體中文"):
            service.generate("L33-SM3E", self.sample_batch.sample_batch_id, 1)

    def test_generate_for_document_uses_selected_model_and_only_its_samples(self) -> None:
        document_sample = next(item for item in self.sample_batch.samples if item.document_id == "WW.pdf")
        client = FakeGenerationClient({
            "questions": [{
                "question": "雨刷馬達不作動先檢查什麼？",
                "reference_answer": "先檢查保險絲與電源供應。",
                "source_sample_ids": [document_sample.sample_id],
            }]
        })
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)

        questions = service.generate_for_document(
            "L33-SM3E", "WW.pdf", self.sample_batch.samples, 1, "gpt-4.1-mini"
        )

        self.assertEqual(client.calls[0][2], "gpt-4.1-mini")
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].gold_evidence[0].document_id, "WW.pdf")

    def test_document_generation_randomly_selects_consecutive_five_page_windows(self) -> None:
        sample = self.sample_batch.samples[0]
        samples = [
            replace(sample, sample_id=f"sample-{index}", chunk_id=f"chunk-{index}", page=index + 1)
            for index in range(60)
        ]

        windows = QuestionGenerationService._sample_page_windows(samples, question_count=10)

        self.assertEqual(len(windows), 10)
        for start, end, selected in windows:
            self.assertEqual(end - start + 1, 5)
            self.assertTrue(all(start <= item.page <= end for item in selected))
            self.assertEqual({item.page for item in selected}, set(range(start, end + 1)))

    def test_generated_question_tracks_question_and_answer_pages_separately(self) -> None:
        samples = [item for item in self.sampling.scan("L33-SM3E", minimum_characters=1) if item.document_id == "WW.pdf"]
        client = FakeGenerationClient({
            "questions": [{
                "question": "雨刷馬達的檢查程序為何？",
                "reference_answer": "檢查馬達與線束。",
                "question_source_sample_ids": [samples[0].sample_id],
                "answer_source_sample_ids": [samples[1].sample_id],
            }]
        })
        service = QuestionGenerationService(self.store, self.sampling, self.connections, client)

        question = service.generate_for_document(
            "L33-SM3E", "WW.pdf", samples, 1, "gpt-4o-mini"
        )[0]

        self.assertEqual(question.question_source_evidence[0].pages, (25,))
        self.assertEqual(question.answer_source_evidence[0].pages, (26,))
        self.assertEqual(question.gold_evidence[0].pages, (26,))

    def test_duplicate_question_normalization_ignores_punctuation_and_width(self) -> None:
        self.assertEqual(normalize_question("ＡＢＣ？ 雨刷！"), normalize_question("abc 雨刷"))

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
