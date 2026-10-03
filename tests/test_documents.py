import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from automotive_graphrag.documents import DocumentService
from automotive_graphrag.projects import ProjectError, ProjectStore


class FakePage:
    def __init__(self, text: str | None = None, error: str | None = None) -> None:
        self.text = text
        self.error = error

    def get_text(self, option: str = "text", *, clip=None) -> str | None:
        if self.error:
            raise RuntimeError(self.error)
        return self.text


class FakeDocument:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __iter__(self):
        return iter(self.pages)

    def __len__(self) -> int:
        return len(self.pages)

    def close(self) -> None:
        pass


class FakeRect:
    x0 = 0
    y0 = 0
    x1 = 100
    y1 = 100
    height = 100


class PositionedFakePage:
    rect = FakeRect()

    def __init__(self, fragments: list[tuple[str, float]]) -> None:
        self.fragments = fragments

    def get_text(self, option: str = "text", *, clip=None) -> str:
        if clip is None:
            return "".join(text for text, _ in self.fragments)
        return "".join(text for text, y_position in self.fragments if clip[1] <= y_position <= clip[3])


class DocumentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = ProjectStore(self.root / "projects")
        for project_id in ("L33-SM3E", "T30-SM5E"):
            self.store.create(
                project_id=project_id,
                display_name=project_id,
                vehicle_name=project_id.split("-")[0],
                manual_version=project_id.split("-")[1],
            )
        self.pages_by_filename = {
            "WW.pdf": FakeDocument([FakePage("雨刷檢修內容"), FakePage("  "), FakePage(error="damaged page")]),
            "PG.pdf": FakeDocument([FakePage("電源供應內容")]),
        }
        self.service = DocumentService(self.store, lambda path: self.pages_by_filename[Path(path).name])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def upload(self, name: str, content: bytes = b"fake pdf") -> Path:
        path = self.root / "uploads" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        return path

    def test_import_multiple_pdfs_keeps_projects_isolated(self) -> None:
        files = [self.upload("WW.pdf"), self.upload("PG.pdf")]

        documents = self.service.import_pdfs("L33-SM3E", files)

        self.assertEqual([item.filename for item in documents], ["PG.pdf", "WW.pdf"])
        self.assertEqual(self.store.get("L33-SM3E").status, "UPLOADED")
        self.assertEqual(self.store.document_count("L33-SM3E"), 2)
        self.assertEqual(self.store.document_count("T30-SM5E"), 0)

    def test_import_replaces_same_filename_and_marks_index_stale(self) -> None:
        uploaded = self.upload("WW.pdf", b"first")
        self.service.import_pdfs("L33-SM3E", [uploaded])
        self.store.update_status("L33-SM3E", "INDEXED")
        uploaded.write_bytes(b"replacement")

        self.service.import_pdfs("L33-SM3E", [uploaded])

        saved = self.store.path_for("L33-SM3E") / "source" / "WW.pdf"
        self.assertEqual(saved.read_bytes(), b"replacement")
        self.assertEqual(self.store.get("L33-SM3E").status, "STALE")

    def test_import_rejects_non_pdf_before_copying_any_file(self) -> None:
        pdf = self.upload("WW.pdf")
        text = self.upload("notes.txt")

        with self.assertRaisesRegex(ProjectError, "只允許匯入 PDF"):
            self.service.import_pdfs("L33-SM3E", [pdf, text])

        self.assertEqual(self.store.document_count("L33-SM3E"), 0)

    def test_preprocess_writes_page_metadata_and_error_report(self) -> None:
        self.service.import_pdfs("L33-SM3E", [self.upload("WW.pdf"), self.upload("PG.pdf")])

        report = self.service.preprocess("L33-SM3E")

        self.assertEqual((report.pages, report.successful_pages, report.empty_pages, report.error_pages), (4, 2, 1, 1))
        processed = self.store.path_for("L33-SM3E") / "processed"
        records = [json.loads(line) for line in (processed / "input.jsonl").read_text().splitlines()]
        self.assertEqual(
            records[1],
            {
                "id": "L33-SM3E-WW-p0001",
                "chunk_id": "L33-SM3E-WW-p0001-b01",
                "project_id": "L33-SM3E",
                "section_id": "WW",
                "section_name": "WW",
                "title": "WW.pdf - Page 1",
                "text": "雨刷檢修內容",
                "document_id": "WW.pdf",
                "page": 1,
                "block_id": "b01",
            },
        )
        details = json.loads((processed / "report.json").read_text())
        ww = next(item for item in details["documents"] if item["filename"] == "WW.pdf")
        self.assertEqual(ww["page_errors"], [{"page": 3, "error": "damaged page"}])
        self.assertEqual(ww["status"], "PROCESSED_WITH_ERRORS")
        self.assertEqual(self.store.get("L33-SM3E").status, "READY")

    def test_preprocess_requires_at_least_one_pdf(self) -> None:
        with self.assertRaisesRegex(ProjectError, "沒有可處理"):
            self.service.preprocess("L33-SM3E")
        self.assertEqual(self.store.get("L33-SM3E").status, "EMPTY")

    def test_preprocess_ignores_configured_header_and_footer_percentages(self) -> None:
        self.pages_by_filename["WW.pdf"] = FakeDocument(
            [
                PositionedFakePage(
                    [
                        ("重複頁首\n", 95),
                        ("維修內文\n", 50),
                        ("第 1 頁", 5),
                    ]
                )
            ]
        )
        self.service.import_pdfs("L33-SM3E", [self.upload("WW.pdf")])

        report = self.service.preprocess(
            "L33-SM3E",
            {"WW.pdf": {"header_ignore_percent": 10, "footer_ignore_percent": 10}},
        )

        processed = self.store.path_for("L33-SM3E") / "processed"
        record = json.loads((processed / "input.jsonl").read_text().strip())
        self.assertEqual(record["text"], "維修內文")
        ww_report = next(item for item in json.loads((processed / "report.json").read_text())["documents"] if item["filename"] == "WW.pdf")
        self.assertEqual((ww_report["header_ignore_percent"], ww_report["footer_ignore_percent"]), (10, 10))

    def test_pymupdf_extracts_traditional_chinese_and_clips_margins(self) -> None:
        pdf = self.root / "uploads" / "ZH.pdf"
        pdf.parent.mkdir(exist_ok=True)
        document = pymupdf.open()
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 40), "重複頁首", fontname="china-t")
        page.insert_text((72, 420), "繁體中文維修內容", fontname="china-t")
        page.insert_text((72, 810), "第 1 頁", fontname="china-t")
        document.save(pdf)
        document.close()
        service = DocumentService(self.store)
        service.import_pdfs("L33-SM3E", [pdf])

        service.preprocess(
            "L33-SM3E",
            {"ZH.pdf": {"header_ignore_percent": 10, "footer_ignore_percent": 10}},
        )

        processed = self.store.path_for("L33-SM3E") / "processed" / "input.jsonl"
        record = json.loads(processed.read_text(encoding="utf-8").strip())
        self.assertEqual(record["text"], "繁體中文維修內容")

    def test_preprocess_rejects_margins_that_remove_the_whole_page(self) -> None:
        self.service.import_pdfs("L33-SM3E", [self.upload("WW.pdf")])

        with self.assertRaisesRegex(ProjectError, "合計必須小於 100"):
            self.service.preprocess(
                "L33-SM3E",
                {"WW.pdf": {"header_ignore_percent": 50, "footer_ignore_percent": 50}},
            )

        self.assertEqual(self.store.get("L33-SM3E").status, "UPLOADED")

    def test_remove_pdf_deletes_selected_file_and_invalidates_processed_input(self) -> None:
        self.service.import_pdfs("L33-SM3E", [self.upload("WW.pdf"), self.upload("PG.pdf")])
        self.service.preprocess("L33-SM3E")
        project_path = self.store.path_for("L33-SM3E")
        graph_input = project_path / "graphrag" / "input" / "input.jsonl"
        graph_input.parent.mkdir()
        graph_input.write_text("stale")

        self.service.remove_pdf("L33-SM3E", "WW.pdf")

        self.assertEqual([item.filename for item in self.service.list_documents("L33-SM3E")], ["PG.pdf"])
        self.assertEqual(self.store.get("L33-SM3E").status, "UPLOADED")
        self.assertFalse((project_path / "processed" / "input.jsonl").exists())
        self.assertFalse((project_path / "processed" / "report.json").exists())
        self.assertFalse(graph_input.exists())

    def test_remove_last_pdf_returns_project_to_empty(self) -> None:
        self.service.import_pdfs("L33-SM3E", [self.upload("WW.pdf")])

        self.service.remove_pdf("L33-SM3E", "WW.pdf")

        self.assertEqual(self.store.get("L33-SM3E").status, "EMPTY")

    def test_remove_pdf_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ProjectError, "檔名格式"):
            self.service.remove_pdf("L33-SM3E", "../WW.pdf")

    def test_ui_safe_list_ignores_a_stale_deleted_project_selection(self) -> None:
        self.store.delete("L33-SM3E")

        self.assertEqual(self.service.list_documents_if_available("L33-SM3E"), [])


if __name__ == "__main__":
    unittest.main()
