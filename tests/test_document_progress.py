import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pymupdf

from automotive_graphrag.app import DOCUMENT_COLUMNS, create_app, document_row
from automotive_graphrag.documents import DocumentInfo
from automotive_graphrag.projects import ProjectStore


class DocumentProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "projects"
        store = ProjectStore(self.root)
        store.create(project_id="L33-SM3E", display_name="L33", vehicle_name="L33", manual_version="SM3E")
        source = store.path_for("L33-SM3E") / "source"
        for name in ("A.pdf", "B.pdf"):
            document = pymupdf.open()
            for page_number in range(3):
                document.new_page().insert_text((72, 72), f"{name} page {page_number}")
            document.save(source / name)
            document.close()
        app = create_app(self.root)
        functions = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        self.preprocess_documents = functions["preprocess_documents"]
        self.document_view = functions["document_view"]
        self.remove_documents = functions["remove_documents"]
        self.event_scripts = [fn.js for fn in app.fns.values() if fn.js]
        self.source = source

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_preprocess_selected_file_streams_progress_and_updates_status_columns(self) -> None:
        outputs = list(self.preprocess_documents("L33-SM3E", None, ["A.pdf"]))

        self.assertTrue(all("<progress" in output[1] for output in outputs))
        message, progress, table, _, targets, notice = outputs[-1]
        rows = table["value"]
        self.assertIn("✅ 前處理完成（1 份 PDF）", message)
        self.assertIn("100%", progress)
        status = {row[0]: dict(zip(DOCUMENT_COLUMNS, row)) for row in rows}
        self.assertEqual(status["A.pdf"]["前處理"], "✅ 已完成")
        self.assertEqual(status["A.pdf"]["建圖"], "⏳ 未建圖")
        self.assertEqual(status["B.pdf"]["前處理"], "⏳ 未前處理")
        # Unfinished PDFs stay selected so they can be processed or indexed again.
        self.assertEqual(targets.value, ["A.pdf", "B.pdf"])
        self.assertTrue(table["visible"])
        self.assertFalse(notice["visible"])

    def test_vision_preprocess_requires_api_key(self) -> None:
        with mock.patch.dict(os.environ, {"GRAPHRAG_API_KEY": ""}):
            outputs = list(self.preprocess_documents("L33-SM3E", None, ["A.pdf"], "gpt-6-luna", "figures"))

        self.assertEqual(len(outputs), 1)
        self.assertIn("使用 Vision 前請先", outputs[0][0])
        self.assertFalse((self.root / "L33-SM3E" / "processed" / "input.jsonl").exists())

    def test_remove_selected_documents_after_confirmation(self) -> None:
        message, table, _, targets, notice, confirmation = self.remove_documents("L33-SM3E", ["A.pdf", "B.pdf"], True)
        rows = table["value"]

        self.assertIn("✅ 已刪除 A.pdf、B.pdf", message)
        self.assertEqual(rows, [])
        self.assertEqual(targets.choices, [("+ 上傳 PDF...", "__upload_pdf__")])
        # An empty project hides the table and shows the empty-list notice instead.
        self.assertFalse(table["visible"])
        self.assertTrue(notice["visible"])
        self.assertFalse(confirmation)

    def test_cancelled_confirmation_keeps_documents(self) -> None:
        outputs = self.remove_documents("L33-SM3E", ["A.pdf"], False)

        self.assertTrue((self.source / "A.pdf").exists())
        self.assertFalse(outputs[-1])

    def test_remove_without_selection_reports_error(self) -> None:
        message, *_ = self.remove_documents("L33-SM3E", [], True)

        self.assertIn("請先在「選擇要前處理／建圖的 PDF」選擇要刪除的 PDF", message)
        self.assertTrue((self.source / "A.pdf").exists())

    def test_event_scripts_have_no_raw_newlines(self) -> None:
        # A raw newline inside a JS string literal is a syntax error that stops the whole page from loading.
        delete_script = next(script for script in self.event_scripts if "files.join" in script)
        self.assertNotIn("\n", delete_script)
        self.assertIn("files.join('\\n')", delete_script)

    def test_document_row_marks_failed_and_indexed_documents(self) -> None:
        failed = document_row(DocumentInfo("bad.pdf", None, 1, "ERROR", error="broken"))
        indexed = document_row(
            DocumentInfo("ok.pdf", 3, 1, "PROCESSED", processed_at="2026-01-01", indexed=True)
        )

        self.assertEqual(failed[3:5], ["❌ 失敗", "—"])
        self.assertEqual(indexed[3:5], ["✅ 已完成", "✅ 已建圖"])
        self.assertEqual(len(indexed), len(DOCUMENT_COLUMNS))


if __name__ == "__main__":
    unittest.main()
