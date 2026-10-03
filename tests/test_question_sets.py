import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import QuestionSetService


class FakeQuestionRunner:
    def __init__(self, failing_questions: set[str] | None = None) -> None:
        self.failing_questions = failing_questions or set()
        self.questions: list[str] = []

    def ask(self, project_id: str, question: str, method: str = "local") -> QueryResult:
        self.questions.append(question)
        if question in self.failing_questions:
            raise RuntimeError("temporary query error")
        now = datetime.now(timezone.utc).isoformat()
        return QueryResult(
            query_id=f"query-{len(self.questions)}",
            project_id=project_id,
            question=question,
            method=method,
            status="COMPLETED",
            answer=f"回答：{question}",
            error=None,
            started_at=now,
            completed_at=now,
            duration_seconds=0.25,
        )


class QuestionSetServiceTests(unittest.TestCase):
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
        self.runner = FakeQuestionRunner()
        self.service = QuestionSetService(self.store, self.runner)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_question_set(self, value: object) -> Path:
        path = self.root / "question_set.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def valid_value(self) -> dict[str, object]:
        return {
            "name": "雨刷測試集",
            "description": "批次測試",
            "questions": [
                {"question_id": "Q001", "question": "問題一", "reference_answer": "參考答案一"},
                {"question_id": "Q002", "question": "問題二"},
                {"question_id": "Q003", "question": "問題三"},
            ],
        }

    def test_import_validates_and_persists_question_set(self) -> None:
        question_set = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))

        self.assertEqual(question_set.name, "雨刷測試集")
        self.assertEqual([item.status for item in question_set.questions], ["PENDING"] * 3)
        self.assertEqual(question_set.questions[0].reference_answer, "參考答案一")
        self.assertEqual(self.service.get("L33-SM3E", question_set.question_set_id), question_set)

    def test_create_set_and_append_question_with_answer(self) -> None:
        question_set = self.service.create("L33-SM3E", "單次問答收藏", "滿意的回答")

        updated = self.service.append_answered_question(
            "L33-SM3E",
            question_set.question_set_id,
            "如何檢查雨刷馬達？",
            "先確認保險絲，再檢查馬達供電。",
        )

        self.assertEqual(len(updated.questions), 1)
        self.assertEqual(updated.questions[0].question, "如何檢查雨刷馬達？")
        self.assertEqual(updated.questions[0].reference_answer, "先確認保險絲，再檢查馬達供電。")
        self.assertEqual(updated.questions[0].answer, updated.questions[0].reference_answer)
        self.assertEqual(
            self.service.get("L33-SM3E", question_set.question_set_id),
            updated,
        )

    def test_append_rejects_missing_answer(self) -> None:
        question_set = self.service.create("L33-SM3E", "空題目集")

        with self.assertRaisesRegex(ProjectError, "系統回答不可空白"):
            self.service.append_answered_question(
                "L33-SM3E",
                question_set.question_set_id,
                "沒有答案的問題",
                "",
            )

    def test_import_reports_duplicate_id_with_array_position(self) -> None:
        value = self.valid_value()
        value["questions"][1]["question_id"] = "Q001"  # type: ignore[index]

        with self.assertRaisesRegex(ProjectError, r"questions\[1\].*重複題號 Q001"):
            self.service.import_file("L33-SM3E", self.write_question_set(value))

    def test_batch_continues_after_one_question_fails(self) -> None:
        runner = FakeQuestionRunner({"問題二"})
        service = QuestionSetService(self.store, runner)
        imported = service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))

        result = service.run("L33-SM3E", imported.question_set_id)

        self.assertEqual(runner.questions, ["問題一", "問題二", "問題三"])
        self.assertEqual([item.status for item in result.questions], ["COMPLETED", "FAILED", "COMPLETED"])
        self.assertEqual(result.questions[1].error, "temporary query error")
        self.assertEqual((service.summary(result).completed, service.summary(result).failed), (2, 1))

    def test_selected_run_and_resume_only_process_unfinished_questions(self) -> None:
        imported = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))

        selected = self.service.run(
            "L33-SM3E",
            imported.question_set_id,
            selected_question_ids=["Q002"],
        )
        self.assertEqual([item.status for item in selected.questions], ["PENDING", "COMPLETED", "PENDING"])

        resumed = self.service.run("L33-SM3E", imported.question_set_id, resume=True)
        self.assertEqual([item.status for item in resumed.questions], ["COMPLETED"] * 3)
        self.assertEqual(self.runner.questions, ["問題二", "問題一", "問題三"])

    def test_export_writes_json_and_csv(self) -> None:
        imported = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))
        self.service.run("L33-SM3E", imported.question_set_id, selected_question_ids=["Q001"])

        json_path, csv_path = self.service.export("L33-SM3E", imported.question_set_id)

        exported = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(exported["name"], "雨刷測試集")
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["question_id"], "Q001")
        self.assertEqual(rows[0]["status"], "COMPLETED")

    def test_batch_requires_indexed_project(self) -> None:
        imported = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))
        self.store.update_status("L33-SM3E", "STALE")

        with self.assertRaisesRegex(ProjectError, "尚未完成建圖"):
            self.service.run("L33-SM3E", imported.question_set_id)

    def test_batch_rerun_preserves_imported_gold_evidence(self) -> None:
        value = self.valid_value()
        value["questions"][0]["gold_evidence"] = [  # type: ignore[index]
            {
                "document_id": "WW.pdf",
                "pages": [25],
                "chunk_ids": ["L33-SM3E-WW-p0025-b01"],
            }
        ]
        imported = self.service.import_file("L33-SM3E", self.write_question_set(value))

        result = self.service.run(
            "L33-SM3E",
            imported.question_set_id,
            selected_question_ids=["Q001"],
        )

        self.assertEqual(result.questions[0].gold_evidence[0].pages, (25,))
        self.assertEqual(result.questions[0].reference_answer, "參考答案一")
        self.assertEqual(
            result.questions[0].gold_evidence[0].chunk_ids,
            ("L33-SM3E-WW-p0025-b01",),
        )


if __name__ == "__main__":
    unittest.main()
