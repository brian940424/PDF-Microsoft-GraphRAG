import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import (
    BatchQuestion, GoldEvidence, QuestionSetService, infer_answer_scope, question_design_label,
)


class FakeQuestionRunner:
    def __init__(self, failing_questions: set[str] | None = None) -> None:
        self.failing_questions = failing_questions or set()
        self.questions: list[str] = []

    def ask(self, project_id: str, question: str, method: str = "local", source: str | None = None) -> QueryResult:
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
            "questions": [
                {"number": 1, "question": "問題一", "expected_answer": "參考答案一", "question_sources": [], "answer_sources": []},
                {"number": 2, "question": "問題二", "expected_answer": "參考答案二", "question_sources": [], "answer_sources": []},
                {"number": 3, "question": "問題三", "expected_answer": "參考答案三", "question_sources": [], "answer_sources": []},
            ],
        }

    def test_import_validates_and_persists_question_set(self) -> None:
        value = self.valid_value()
        question_set = self.service.import_file("L33-SM3E", self.write_question_set(value))

        self.assertEqual(question_set.name, "question_set")
        self.assertEqual([item.status for item in question_set.questions], ["PENDING"] * 3)
        self.assertEqual(question_set.questions[0].reference_answer, "參考答案一")
        self.assertEqual(question_set.questions[0].answer, "")
        self.assertEqual(question_set.questions[0].status, "PENDING")
        self.assertEqual(self.service.get("L33-SM3E", question_set.question_set_id), question_set)

    def test_answer_aliases_import_export_and_default_for_old_question_sets(self) -> None:
        value = self.valid_value()
        value["questions"][0]["answer_aliases"] = ["參考答案別名"]  # type: ignore[index]
        question_set = self.service.import_file("L33-SM3E", self.write_question_set(value))
        self.assertEqual(question_set.questions[0].answer_aliases, ("參考答案別名",))

        exported = json.loads(self.service.export("L33-SM3E", question_set.question_set_id).read_text(encoding="utf-8"))
        self.assertEqual(exported["questions"][0]["answer_aliases"], ["參考答案別名"])
        reloaded = self.service.get("L33-SM3E", question_set.question_set_id)
        self.assertEqual(reloaded.questions[0].answer_aliases, ("參考答案別名",))
        self.assertEqual(reloaded.questions[1].answer_aliases, ())

    def test_answer_aliases_must_be_string_array(self) -> None:
        value = self.valid_value()
        value["questions"][0]["answer_aliases"] = "不是陣列"  # type: ignore[index]
        with self.assertRaisesRegex(ProjectError, "answer_aliases.*字串陣列"):
            self.service.import_file("L33-SM3E", self.write_question_set(value))

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
        value["questions"][1]["number"] = 1  # type: ignore[index]

        with self.assertRaisesRegex(ProjectError, r"questions\[1\].*重複題號 1"):
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
            selected_question_ids=["Q0002"],
        )
        self.assertEqual([item.status for item in selected.questions], ["PENDING", "COMPLETED", "PENDING"])

        resumed = self.service.run("L33-SM3E", imported.question_set_id, resume=True)
        self.assertEqual([item.status for item in resumed.questions], ["COMPLETED"] * 3)
        self.assertEqual(self.runner.questions, ["問題二", "問題一", "問題三"])

    def test_export_writes_current_question_json(self) -> None:
        imported = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))
        self.service.run("L33-SM3E", imported.question_set_id, selected_question_ids=["Q0001"])

        json_path = self.service.export("L33-SM3E", imported.question_set_id)

        exported = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertNotIn("schema_version", exported)
        self.assertEqual(exported["questions"][0]["number"], 1)
        self.assertEqual(exported["questions"][0]["expected_answer"], "參考答案一")
        self.assertIn("question_sources", exported["questions"][0])
        self.assertIn("answer_sources", exported["questions"][0])
        self.assertNotIn("source_pages", exported["questions"][0])

    def test_question_set_export_import_preserves_source_page_ranges(self) -> None:
        question_set = self.service.create_with_questions(
            "L33-SM3E",
            "來源頁碼測試",
            [
                BatchQuestion(
                    "Q0001",
                    "問題",
                    reference_answer="正確答案",
                    gold_evidence=(GoldEvidence("doc-003", (27,), ("chunk-a",), "零件手冊.pdf"),),
                    question_source_evidence=(
                        GoldEvidence("doc-001", (12, 13), (), "維修手冊.pdf"),
                        GoldEvidence("doc-002", (4,), (), "保養指南.pdf"),
                    ),
                    answer_source_evidence=(
                        GoldEvidence("doc-001", (13,), (), "維修手冊.pdf"),
                        GoldEvidence("doc-003", (27,), (), "零件手冊.pdf"),
                    ),
                )
            ],
        )

        json_path = self.service.export("L33-SM3E", question_set.question_set_id)
        exported = json.loads(json_path.read_text(encoding="utf-8"))
        row = exported["questions"][0]
        self.assertEqual(row["number"], 1)
        self.assertEqual(row["question"], "問題")
        self.assertEqual(row["expected_answer"], "正確答案")
        self.assertEqual(row["question_sources"], [
            {"document_name": "維修手冊.pdf", "pages": [12, 13]},
            {"document_name": "保養指南.pdf", "pages": [4]},
        ])
        self.assertEqual(row["answer_sources"], [
            {"document_name": "維修手冊.pdf", "pages": [13]},
            {"document_name": "零件手冊.pdf", "pages": [27]},
        ])

        imported = self.service.import_file("L33-SM3E", json_path)
        self.assertEqual(
            [(source.document_id, source.document_name, source.pages) for source in imported.questions[0].question_source_evidence],
            [("維修手冊.pdf", "維修手冊.pdf", (12, 13)), ("保養指南.pdf", "保養指南.pdf", (4,))],
        )
        self.assertEqual(
            [(source.document_id, source.document_name, source.pages) for source in imported.questions[0].answer_source_evidence],
            [("維修手冊.pdf", "維修手冊.pdf", (13,)), ("零件手冊.pdf", "零件手冊.pdf", (27,))],
        )
        reloaded = self.service.get("L33-SM3E", imported.question_set_id)
        self.assertEqual(reloaded.questions[0].question_source_evidence, imported.questions[0].question_source_evidence)

    def test_imports_requested_unversioned_format_with_cross_document_sources(self) -> None:
        value = {
            "questions": [{
                "number": 1,
                "question": "如何使用 WPS-PBC？",
                "expected_answer": "在控制面板選擇 WPS-PBC，並按下路由器 WPS 按鈕。",
                "question_sources": [{"document_name": "使用手冊.pdf", "pages": [42, 43]}],
                "answer_sources": [
                    {"document_name": "使用手冊.pdf", "pages": [43]},
                    {"document_name": "網路手冊.pdf", "pages": [8, 9]},
                ],
            }],
        }

        imported = self.service.import_file("L33-SM3E", self.write_question_set(value))
        question = imported.questions[0]

        self.assertEqual(question.question_id, "Q0001")
        self.assertEqual(question.reference_answer, "在控制面板選擇 WPS-PBC，並按下路由器 WPS 按鈕。")
        self.assertEqual(question.question_source_evidence[0].document_id, "使用手冊.pdf")
        self.assertEqual(question.question_source_evidence[0].pages, (42, 43))
        self.assertEqual(
            [(item.document_name, item.pages) for item in question.answer_source_evidence],
            [("使用手冊.pdf", (43,)), ("網路手冊.pdf", (8, 9))],
        )

    def test_imports_neo4j_demo_export_ignoring_document_id(self) -> None:
        value = {
            "schema_version": 1,
            "questions": [{
                "number": 1,
                "question": "問題？",
                "expected_answer": "答案",
                "question_sources": [{"document_id": "neo4j-doc-1", "document_name": "手冊.pdf", "pages": [3]}],
                "answer_sources": [{"document_id": "", "document_name": "手冊.pdf", "pages": [3, 4]}],
            }],
        }

        imported = self.service.import_file("L33-SM3E", self.write_question_set(value))
        question = imported.questions[0]

        self.assertEqual(
            [(item.document_id, item.document_name, item.pages) for item in question.question_source_evidence],
            [("手冊.pdf", "手冊.pdf", (3,))],
        )
        self.assertEqual(
            [(item.document_id, item.pages) for item in question.answer_source_evidence],
            [("手冊.pdf", (3, 4))],
        )

    def test_rejects_old_question_set_formats(self) -> None:
        value = {
            "schema_version": 1,
            "questions": [{
                "number": 7,
                "question": "跨頁問題？",
                "expected_answer": "完整答案",
                "source_pages": [60, 61],
                "document": "手冊.pdf",
            }],
        }
        with self.assertRaisesRegex(ProjectError, "不支援的欄位 document, source_pages"):
            self.service.import_file("L33-SM3E", self.write_question_set(value))
        with self.assertRaisesRegex(ProjectError, "不支援的題目集欄位"):
            self.service.import_file("L33-SM3E", self.write_question_set({**value, "name": "x"}))

        legacy_value = {"questions": [{
            "question_id": "Q0001", "question": "問題？", "reference_answer": "答案",
        }]}
        with self.assertRaisesRegex(ProjectError, "不支援的欄位"):
            self.service.import_file("L33-SM3E", self.write_question_set(legacy_value))

    def test_batch_requires_indexed_project(self) -> None:
        imported = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))
        self.store.update_status("L33-SM3E", "STALE")

        with self.assertRaisesRegex(ProjectError, "尚未完成建圖"):
            self.service.run("L33-SM3E", imported.question_set_id)

    def test_batch_rerun_preserves_imported_gold_evidence(self) -> None:
        value = self.valid_value()
        value["questions"][0]["answer_sources"] = [  # type: ignore[index]
            {"document_name": "WW.pdf", "pages": [25]},
        ]
        imported = self.service.import_file("L33-SM3E", self.write_question_set(value))

        result = self.service.run(
            "L33-SM3E",
            imported.question_set_id,
            selected_question_ids=["Q0001"],
        )

        self.assertEqual(result.questions[0].gold_evidence[0].pages, (25,))
        self.assertEqual(result.questions[0].reference_answer, "參考答案一")
        self.assertEqual(result.questions[0].gold_evidence[0].chunk_ids, ())


    def test_infer_answer_scope_from_answer_sources(self) -> None:
        def source(name: str, *pages: int) -> GoldEvidence:
            return GoldEvidence(name, tuple(pages), (), name)

        self.assertEqual(infer_answer_scope(()), "")
        self.assertEqual(infer_answer_scope((source("BR.pdf", 49),)), "single_page")
        # The same page listed twice is still one page.
        self.assertEqual(infer_answer_scope((source("BR.pdf", 49), source("BR.pdf", 49))), "single_page")
        self.assertEqual(infer_answer_scope((source("BR.pdf", 14, 15),)), "cross_page")
        self.assertEqual(infer_answer_scope((source("BR.pdf", 13), source("BR.pdf", 48))), "cross_page")
        self.assertEqual(infer_answer_scope((source("BR.pdf", 18), source("ACC.pdf", 3))), "cross_document")
        self.assertEqual(question_design_label("local", "cross_document"), "Local／跨檔案")
        self.assertEqual(question_design_label("drift", ""), "DRIFT／未分類")
        self.assertEqual(question_design_label("", ""), "")

    def test_design_fields_are_optional_validated_and_inferred(self) -> None:
        value = self.valid_value()
        questions = value["questions"]  # type: ignore[index]
        questions[0].update(design_strategy="local", answer_scope="cross_document",
                            answer_sources=[{"document_name": "BR.pdf", "pages": [18]}])
        questions[1].update(design_strategy="basic", answer_sources=[{"document_name": "BR.pdf", "pages": [14, 15]}])

        imported = self.service.import_file("L33-SM3E", self.write_question_set(value))
        first, second, third = imported.questions

        # A hand-set scope wins over the one the sources imply.
        self.assertEqual((first.design_strategy, first.answer_scope, first.answer_scope_inferred),
                         ("local", "cross_document", False))
        self.assertEqual((second.design_strategy, second.answer_scope, second.answer_scope_inferred),
                         ("basic", "cross_page", True))
        self.assertEqual((third.design_strategy, third.answer_scope), ("", ""))
        reloaded = self.service.get("L33-SM3E", imported.question_set_id)
        self.assertEqual(
            [(item.design_strategy, item.answer_scope, item.answer_scope_inferred) for item in reloaded.questions],
            [("local", "cross_document", False), ("basic", "cross_page", True), ("", "", False)],
        )

        exported = QuestionSetService.to_question_set_json(imported.questions)["questions"]
        self.assertEqual((exported[0]["design_strategy"], exported[0]["answer_scope"]), ("local", "cross_document"))
        # Inferred scopes are not written back, so a re-import infers them from the sources again.
        self.assertEqual(exported[1]["design_strategy"], "basic")
        self.assertNotIn("answer_scope", exported[1])
        self.assertNotIn("design_strategy", exported[2])

        rerun = self.service.run("L33-SM3E", imported.question_set_id)
        self.assertEqual([item.design_strategy for item in rerun.questions], ["local", "basic", ""])
        self.assertEqual(rerun.questions[1].answer_scope, "cross_page")

    def test_invalid_design_values_name_the_field_and_allowed_values(self) -> None:
        value = self.valid_value()
        value["questions"][0]["design_strategy"] = "Basic"  # type: ignore[index]
        value["questions"][1]["answer_scope"] = "跨頁"  # type: ignore[index]

        with self.assertRaises(ProjectError) as caught:
            self.service.import_file("L33-SM3E", self.write_question_set(value))

        message = str(caught.exception)
        self.assertIn("questions[0].design_strategy：必須是 basic、local、global、drift 之一", message)
        self.assertIn("questions[1].answer_scope：必須是 single_page、cross_page、cross_document 之一", message)

    def test_old_saved_question_sets_without_design_fields_still_load(self) -> None:
        imported = self.service.import_file("L33-SM3E", self.write_question_set(self.valid_value()))
        path = self.store.path_for("L33-SM3E") / "question_sets" / f"{imported.question_set_id}.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        for item in saved["questions"]:
            for key in ("design_strategy", "answer_scope", "answer_scope_inferred"):
                item.pop(key)
        path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

        loaded = self.service.get("L33-SM3E", imported.question_set_id)

        self.assertEqual({(item.design_strategy, item.answer_scope) for item in loaded.questions}, {("", "")})


if __name__ == "__main__":
    unittest.main()
