import json
import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.documents import DocumentService
from automotive_graphrag.projects import ProjectError, ProjectStore


class FakePage:
    def __init__(self, text: str | None = None, error: str | None = None) -> None:
        self.text = text
        self.error = error

    def extract_text(self) -> str | None:
        if self.error:
            raise RuntimeError(self.error)
        return self.text


class FakeDocument:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages


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
                "title": "WW.pdf - Page 1",
                "text": "雨刷檢修內容",
                "document_id": "WW.pdf",
                "page": 1,
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


if __name__ == "__main__":
    unittest.main()
