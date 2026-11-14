import json
import tempfile
import unittest
from pathlib import Path

import gradio as gr
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

    def test_retry_vision_fallbacks_stops_before_sending_pages(self) -> None:
        import threading

        vision = VisionSettings(model="gpt-6-luna", scope=SCOPE_ALL, max_workers=1)
        self.service.page_analyzer = lambda image, page, language, settings: (_ for _ in ()).throw(RuntimeError("x"))
        self.service.preprocess("L33-SM3E", vision=vision)
        sent: list[int] = []
        self.service.page_analyzer = lambda image, page, language, settings: sent.append(page) or f"Vision {page}"
        stop = threading.Event()
        stop.set()

        recovered, failing = self.service.retry_vision_fallbacks("L33-SM3E", "manual.pdf", vision, stop_event=stop)

        self.assertEqual((sent, recovered, failing), ([], [], [1, 2, 3]))

    def test_extract_page_again_and_render_image(self) -> None:
        vision = VisionSettings(model="gpt-6-luna")

        self.assertEqual(self.service.extract_page_text("L33-SM3E", "manual.pdf", 3), "第三頁原文")
        self.assertEqual(self.service.extract_page_text("L33-SM3E", "manual.pdf", 3, vision), "Vision 3")
        self.assertTrue(self.service.render_page_image("L33-SM3E", "manual.pdf", 1).startswith(b"\x89PNG"))
        # Re-extracting does not save anything.
        self.assertEqual(self.records()[3], "第三頁原文")

    def test_retry_vision_fallbacks_saves_recovered_pages(self) -> None:
        failing = {1, 3}
        attempts: list[int] = []

        def analyzer(image, page, language, settings):
            attempts.append(page)
            if page in failing:
                raise RuntimeError(f"timeout {page}")
            return f"Vision {page}"

        self.service.page_analyzer = analyzer
        vision = VisionSettings(model="gpt-6-luna", scope=SCOPE_ALL, max_workers=1)
        self.service.preprocess("L33-SM3E", vision=vision)
        self.service.save_page_text("L33-SM3E", "manual.pdf", 1, "修正後的第一頁")
        self.assertEqual([self.reviews()[page].source for page in (1, 3)], ["fallback", "fallback"])

        failing = {1}
        attempts.clear()
        progress: list[tuple[int, int, str]] = []
        recovered, still_failing = self.service.retry_vision_fallbacks(
            "L33-SM3E", "manual.pdf", vision, progress_callback=lambda *update: progress.append(update),
        )
        self.assertEqual(progress, [(0, 2, "manual.pdf"), (1, 2, "manual.pdf"), (2, 2, "manual.pdf")])

        # Only the failed pages are sent again.
        self.assertEqual(sorted(attempts), [1, 3])
        self.assertEqual((recovered, still_failing), ([3], [1]))
        reviews = self.reviews()
        self.assertEqual((reviews[1].source, reviews[1].error), ("fallback", "timeout 1"))
        self.assertEqual((reviews[3].source, self.records()[3]), ("vision", "Vision 3"))
        # The manual edit on a still-failing page is untouched.
        self.assertEqual(self.records()[1], "修正後的第一頁")
        self.assertEqual(self.service.list_documents("L33-SM3E")[0].status, "PROCESSED_WITH_WARNINGS")

        failing = set()
        self.assertEqual(self.service.retry_vision_fallbacks("L33-SM3E", "manual.pdf", vision), ([1], []))
        reviews = self.reviews()
        self.assertEqual(reviews[1].source, "vision")
        # A recovered page with a manual edit keeps the edit and is flagged.
        self.assertTrue(reviews[1].edited and reviews[1].source_changed)
        self.assertEqual(self.records()[1], "修正後的第一頁")
        # With no Vision failures left the warning status clears.
        self.assertEqual(self.service.list_documents("L33-SM3E")[0].status, "PROCESSED")
        with self.assertRaises(ProjectError):
            self.service.retry_vision_fallbacks("L33-SM3E", "manual.pdf", vision)


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
        page, image, text, info = refresh("L33-SM3E", "A.pdf", "全部", None)
        self.assertEqual((page, text), (1, "甲頁"))
        self.assertEqual(image.size[0] > 0, True)
        self.assertIn("**A.pdf　第 1 頁**", info)
        self.assertIn("⬜ 未檢查", info)

        save_next = self.functions["review_save_next"]
        message, page, _, text, _ = save_next("L33-SM3E", "A.pdf", "全部", 1, "甲頁（修正）")
        self.assertIn("已標示為手動修改", message)
        # Saving moves on to the next page matching the filter.
        self.assertEqual((page, text), (2, ""))
        page, *_ = refresh("L33-SM3E", "A.pdf", "已修改", 3)
        self.assertEqual(page, 1)
        self.assertEqual(refresh("L33-SM3E", "A.pdf", "未檢查", 1)[0], 2)

        # With the "unchecked" filter the saved page drops out and the next unchecked page follows.
        message, page, *_ = save_next("L33-SM3E", "A.pdf", "未檢查", 2, "")
        self.assertEqual(page, 3)
        message, page, *_ = save_next("L33-SM3E", "A.pdf", "全部", 3, "丙頁")
        self.assertEqual(page, 3)
        self.assertIn("已是最後一頁", message)
        self.assertIn("沒有符合「未檢查」的頁面", refresh("L33-SM3E", "A.pdf", "未檢查", 1)[3])
        self.assertEqual(self.functions["review_previous_page"]("L33-SM3E", "A.pdf", "全部", 3)[0], 2)
        # A failed save stays on the page.
        message, *rest = save_next("L33-SM3E", None, "全部", 1, "x")
        self.assertTrue(message.startswith("❌"))

    def test_retry_vision_button_requires_a_vision_model(self) -> None:
        retry = self.functions["review_retry_vision"]
        # Outputs: status, progress, review screen, page, image, text, info, filter, then the job buttons.
        (message, *rest), = list(retry("L33-SM3E", "A.pdf", "全部", 1, None, "__vision_disabled__", None))
        self.assertTrue(message.startswith("❌"))
        self.assertEqual(len(rest), 9)
        (message, *_), = list(retry("L33-SM3E", None, "全部", 1, None, None, None))
        self.assertTrue(message.startswith("❌"))

    def test_options_table_rows_choose_the_reviewed_pdf(self) -> None:
        write_pdf(ProjectStore(self.root).path_for("L33-SM3E") / "source" / "B.pdf", ["乙頁", "乙二"])
        rows = [["A.pdf", 3, 0, 0, 0, 0, 0, 2], ["B.pdf", 2, 0, 0, 0, 0, 0, 1]]
        sync = self.functions["review_sync"]

        filename, page, _, text, info, filter_update = sync("L33-SM3E", rows, None, "全部", None)
        # Filter choices carry page counts; A.pdf has a blank page 2 and no Vision failures,
        # so a newly reviewed PDF opens on the blank/error filter.
        self.assertEqual((filename, page, text), ("A.pdf", 2, ""))
        self.assertEqual(filter_update["value"], "空白／錯誤")
        self.assertIn(("全部（3）", "全部"), filter_update["choices"])
        self.assertIn(("空白／錯誤（1）", "空白／錯誤"), filter_update["choices"])
        self.assertIn(("Vision 失敗（0）", "Vision 失敗"), filter_update["choices"])
        page, _, text, _ = self.functions["review_refresh"]("L33-SM3E", "A.pdf", "全部", None, rows)
        self.assertEqual((page, text), (1, "甲頁"))
        # Editing the table while A.pdf stays listed keeps the PDF, page and editor text.
        result = sync("L33-SM3E", rows, "A.pdf", "全部", 3)
        self.assertFalse(isinstance(result[0], str))
        self.assertFalse(isinstance(result[3], str))
        # A PDF no longer in the table falls back to the first row.
        self.assertEqual(sync("L33-SM3E", rows[1:], "A.pdf", "全部", 3)[:2], ("B.pdf", 1))
        # After preprocessing (force) the filter resets to the default, so the blank page comes up.
        self.assertEqual(sync("L33-SM3E", rows, "A.pdf", "全部", 3, force=True)[:2], ("A.pdf", 2))

        pick = self.functions["review_pick"]
        clicked = gr.SelectData(None, {"index": [1, 5], "value": 0, "row_value": rows[1]})
        filename, page, _, text, info, _ = pick("L33-SM3E", rows, "A.pdf", "全部", clicked)
        self.assertEqual((filename, page, text), ("B.pdf", 1, "乙頁"))
        self.assertIn("尚未前處理", info)
        # Clicking a cell of the PDF already under review (e.g. to edit a margin) changes nothing.
        same = gr.SelectData(None, {"index": [1, 5], "value": 0, "row_value": rows[1]})
        self.assertTrue(all(not isinstance(value, str) for value in pick("L33-SM3E", rows, "B.pdf", "全部", same)))

    def test_review_shades_margins_from_unsaved_table_and_keeps_editor_text(self) -> None:
        rows = [["A.pdf", 3, 0, 0, 0, 10, 0, 2]]
        page, image, text, info = self.functions["review_refresh"]("L33-SM3E", "A.pdf", "全部", None, rows)
        self.assertEqual((page, text), (1, "甲頁"))
        self.assertGreater(image.size[0], 0)
        self.assertIn("右 59.5 pt", info)
        self.assertIn("與上次前處理不同", info)
        # Without table values the saved (zero) margins apply and nothing is shaded.
        self.assertNotIn("忽略範圍", self.functions["review_page_view"]("L33-SM3E", "A.pdf", 1)[2])

        # Editing the table redraws the image but leaves the editor alone for preprocessed PDFs.
        *_, image, text, info, _ = self.functions["review_sync"]("L33-SM3E", rows, "A.pdf", "全部", 1)
        self.assertGreater(image.size[0], 0)
        self.assertFalse(isinstance(text, str))
        self.assertIn("右 59.5 pt", info)
        rows[0][4], rows[0][5] = 60, 50
        self.assertIn("❌ 忽略比例", self.functions["review_sync"]("L33-SM3E", rows, "A.pdf", "全部", 1)[4])

    def test_pdf_not_yet_preprocessed_previews_clipped_text_layer(self) -> None:
        store = ProjectStore(self.root)
        source = store.path_for("L33-SM3E") / "source" / "B.pdf"
        document = pymupdf.open()
        for _ in range(3):
            page = document.new_page(width=612, height=792)
            page.insert_text((52, 300), "Body text", fontname="helv")
            page.insert_text((575, 100), "TAB", fontname="helv")
        document.save(source)
        document.close()

        rows = [["B.pdf", 3, 0, 0, 0, 8, 1, 2]]
        page, _, text, info = self.functions["review_refresh"]("L33-SM3E", "B.pdf", "全部", None, rows)
        # Only pages inside the table's range (indexes 1–2) are listed.
        self.assertEqual((page, text), (2, "Body text"))
        self.assertEqual(self.functions["review_previous_page"]("L33-SM3E", "B.pdf", "全部", 2, rows)[0], 2)
        self.assertIn("｜ 尚未前處理", info)
        self.assertIn("**裁掉文字**：`TAB`", info)
        *_, text, _, _ = self.functions["review_sync"]("L33-SM3E", [["B.pdf", 3, 0, 0, 0, 0, 1, 2]], "B.pdf", "全部", 2)
        self.assertEqual(text, "Body text\nTAB")
        message, page, *_ = self.functions["review_save_next"]("L33-SM3E", "B.pdf", "全部", 2, "x", rows)
        self.assertIn("尚未前處理", message)
        self.assertFalse(isinstance(page, int))

    def test_settings_and_review_section_is_open_whenever_pdfs_are_selected(self) -> None:
        update = self.functions["processing_section_update"]
        self.assertEqual((update("L33-SM3E", ["A.pdf"]).visible, update("L33-SM3E", ["A.pdf"]).open), (True, True))
        self.assertEqual(update("L33-SM3E", []).visible, False)
        write_pdf(ProjectStore(self.root).path_for("L33-SM3E") / "source" / "B.pdf", ["乙頁"])
        self.assertEqual(update("L33-SM3E", ["A.pdf", "B.pdf"]).open, True)
        _, section = self.functions["processing_options_view"]("L33-SM3E", ["B.pdf"], None)
        self.assertEqual((section.visible, section.open), (True, True))

if __name__ == "__main__":
    unittest.main()
