import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from automotive_graphrag.app import create_app
from automotive_graphrag.documents import DocumentService
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.vision import SCOPE_ALL, VisionSettings


def write_pdf(path: Path, texts: list[str]) -> None:
    document = pymupdf.open()
    for text in texts:
        page = document.new_page(width=595, height=842)
        if text:
            page.insert_text((72, 420), text, fontname="china-t")
    document.save(path)
    document.close()


class PageReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = ProjectStore(self.root / "projects")
        self.store.create(project_id="L33-SM3E", display_name="L33", vehicle_name="L33", manual_version="SM3E")
        self.pdf = self.root / "manual.pdf"
        write_pdf(self.pdf, ["第一頁原文", "", "第三頁原文"])
        self.service = DocumentService(self.store, page_analyzer=lambda image, page, language, settings: f"Vision {page}")
        self.service.import_pdfs("L33-SM3E", [self.pdf])
        self.service.preprocess("L33-SM3E")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def records(self) -> dict[int, str]:
        path = self.store.path_for("L33-SM3E") / "processed" / "input.jsonl"
        return {item["page"]: item["text"] for item in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    def reviews(self):
        return {item.page: item for item in self.service.page_reviews("L33-SM3E", "manual.pdf")}

    def test_page_reviews_list_every_selected_page_with_its_source(self) -> None:
        reviews = self.reviews()

        self.assertEqual([reviews[page].source for page in (1, 2, 3)], ["text", "empty", "text"])
        self.assertEqual(reviews[1].text, "第一頁原文")
        self.assertFalse(reviews[1].reviewed or reviews[1].edited)

    def test_save_marks_reviewed_and_edit_reaches_graph_input(self) -> None:
        self.service.save_page_text("L33-SM3E", "manual.pdf", 1, "第一頁原文")
        review = self.service.save_page_text("L33-SM3E", "manual.pdf", 2, "  手動補上的空白頁內容  ")

        reviews = self.reviews()
        self.assertTrue(reviews[1].reviewed)
        self.assertFalse(reviews[1].edited)
        self.assertTrue(review.edited and review.reviewed)
        self.assertEqual(self.records()[2], "手動補上的空白頁內容")

    def test_reprocessing_keeps_edits_and_flags_changed_extraction(self) -> None:
        self.service.save_page_text("L33-SM3E", "manual.pdf", 1, "修正後的第一頁")
        self.service.save_page_text("L33-SM3E", "manual.pdf", 3, "第三頁原文")

        report = self.service.preprocess(
            "L33-SM3E", vision=VisionSettings(model="gpt-6-luna", scope=SCOPE_ALL, max_workers=1)
        )

        reviews = self.reviews()
        self.assertEqual(report.edited_pages, 1)
        self.assertEqual(self.records()[1], "修正後的第一頁")
        self.assertEqual(reviews[1].source, "vision")
        self.assertTrue(reviews[1].edited and reviews[1].source_changed)
        # A reviewed page without edits loses its mark when the extraction changes.
        self.assertEqual(self.records()[3], "Vision 3")
        self.assertFalse(reviews[3].reviewed)

    def test_reprocessing_with_same_extraction_keeps_review_mark(self) -> None:
        self.service.save_page_text("L33-SM3E", "manual.pdf", 3, "第三頁原文")

        self.service.preprocess("L33-SM3E")

        self.assertTrue(self.reviews()[3].reviewed)

    def test_revert_restores_extracted_text(self) -> None:
        self.service.save_page_text("L33-SM3E", "manual.pdf", 1, "修正後的第一頁")

        review = self.service.revert_page_text("L33-SM3E", "manual.pdf", 1)

        self.assertFalse(review.edited or review.reviewed)
        self.assertEqual(self.records()[1], "第一頁原文")
        self.assertFalse((self.store.path_for("L33-SM3E") / "processed" / "page_edits.json").exists())

    def test_edit_marks_indexed_document_as_changed(self) -> None:
        processed_at = self.service.list_documents("L33-SM3E")[0].processed_at
        graph = self.store.path_for("L33-SM3E") / "graphrag"
        graph.mkdir(exist_ok=True)
        (graph / "indexed_documents.json").write_text(json.dumps({"manual.pdf": processed_at}), encoding="utf-8")
        self.store.update_status("L33-SM3E", "INDEXED")
        self.assertTrue(self.service.list_documents("L33-SM3E")[0].indexed)

        self.service.save_page_text("L33-SM3E", "manual.pdf", 1, "修正後的第一頁")

        self.assertFalse(self.service.list_documents("L33-SM3E")[0].indexed)
        self.assertEqual(self.store.get("L33-SM3E").status, "STALE")

    def test_reimporting_pdf_drops_its_edits(self) -> None:
        self.service.save_page_text("L33-SM3E", "manual.pdf", 1, "修正後的第一頁")

        self.service.import_pdfs("L33-SM3E", [self.pdf])
        self.service.preprocess("L33-SM3E")

        self.assertEqual(self.records()[1], "第一頁原文")
        self.assertFalse(self.reviews()[1].edited)

    def test_rejects_pages_outside_the_processed_range(self) -> None:
        with self.assertRaises(ProjectError):
            self.service.save_page_text("L33-SM3E", "manual.pdf", 4, "x")
        with self.assertRaises(ProjectError):
            self.service.page_reviews("L33-SM3E", "missing.pdf")

    def test_extract_page_again_and_render_image(self) -> None:
        vision = VisionSettings(model="gpt-6-luna")

        self.assertEqual(self.service.extract_page_text("L33-SM3E", "manual.pdf", 3), "第三頁原文")
        self.assertEqual(self.service.extract_page_text("L33-SM3E", "manual.pdf", 3, vision), "Vision 3")
        self.assertTrue(self.service.render_page_image("L33-SM3E", "manual.pdf", 1).startswith(b"\x89PNG"))
        # Re-extracting does not save anything.
        self.assertEqual(self.records()[3], "第三頁原文")


class PageReviewUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "projects"
        store = ProjectStore(self.root)
        store.create(project_id="L33-SM3E", display_name="L33", vehicle_name="L33", manual_version="SM3E")
        write_pdf(store.path_for("L33-SM3E") / "source" / "A.pdf", ["甲頁", "", "丙頁"])
        DocumentService(store).preprocess("L33-SM3E")
        app = create_app(self.root)
        self.functions = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_filter_save_and_step_through_pages(self) -> None:
        refresh = self.functions["review_refresh"]
        dropdown = self.functions["review_document_choices"]("L33-SM3E", None)
        self.assertEqual(dropdown.value, "A.pdf")

        rows, page, image, text, info = refresh("L33-SM3E", "A.pdf", "全部", None)
        self.assertEqual([row[:2] for row in rows], [[1, "文字層"], [2, "空白"], [3, "文字層"]])
        self.assertEqual((page, text), (1, "甲頁"))
        self.assertEqual(image.size[0] > 0, True)
        self.assertIn("⬜ 未檢查", info)

        message = self.functions["review_save"]("L33-SM3E", "A.pdf", 1, "甲頁（修正）")
        self.assertIn("已標示為手動修改", message)
        rows, page, *_ = refresh("L33-SM3E", "A.pdf", "已修改", 1)
        self.assertEqual((len(rows), page, rows[0][3]), (1, 1, "✏️"))

        rows, page, *_ = refresh("L33-SM3E", "A.pdf", "未檢查", 1)
        self.assertEqual([row[0] for row in rows], [2, 3])
        self.assertEqual(page, 2)
        # Stepping stays within the filtered pages and stops at either end.
        self.assertEqual(self.functions["review_next_page"]("L33-SM3E", "A.pdf", "未檢查", 2)[0], 3)
        self.assertEqual(self.functions["review_next_page"]("L33-SM3E", "A.pdf", "未檢查", 3)[0], 3)
        self.assertEqual(self.functions["review_previous_page"]("L33-SM3E", "A.pdf", "全部", 3)[0], 2)


if __name__ == "__main__":
    unittest.main()
