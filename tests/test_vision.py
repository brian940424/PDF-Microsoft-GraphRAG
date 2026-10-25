import base64
import io
import json
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import pymupdf

from automotive_graphrag.documents import DocumentService
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.vision import (
    SCOPE_ALL,
    SCOPE_FIGURES,
    VisionAbort,
    VisionSettings,
    analyze_page,
    detect_document_language,
    page_needs_vision,
    serialize_result,
)


def write_pdf(path: Path) -> None:
    """Page 1: text only; page 2: text and an illustration; page 3: no text layer (like a scan)."""
    document = pymupdf.open()
    document.new_page(width=595, height=842).insert_text((72, 420), "雨刷馬達拆卸步驟說明", fontname="china-t")
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 100), "雨刷連桿位置圖", fontname="china-t")
    page.draw_circle((300, 420), 80)
    document.new_page(width=595, height=842)
    document.save(path)
    document.close()


class RecordingAnalyzer:
    def __init__(self, fail_pages: dict[int, Exception] | None = None) -> None:
        self.fail_pages = fail_pages or {}
        self.calls: list[tuple[int, str | None]] = []
        self.images: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, image: str, page_number: int, language: str | None, settings: VisionSettings) -> str:
        with self.lock:
            self.calls.append((page_number, language))
            self.images.append(image)
        if page_number in self.fail_pages:
            raise self.fail_pages[page_number]
        return f"Vision 第 {page_number} 頁"


class VisionPreprocessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = ProjectStore(self.root / "projects")
        self.store.create(project_id="L33-SM3E", display_name="L33", vehicle_name="L33", manual_version="SM3E")
        self.pdf = self.root / "manual.pdf"
        write_pdf(self.pdf)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def preprocess(self, analyzer: RecordingAnalyzer, scope: str = SCOPE_FIGURES, max_workers: int = 4):
        service = DocumentService(self.store, page_analyzer=analyzer)
        service.import_pdfs("L33-SM3E", [self.pdf])
        report = service.preprocess(
            "L33-SM3E", vision=VisionSettings(model="gpt-6-luna", scope=scope, max_workers=max_workers)
        )
        processed = self.store.path_for("L33-SM3E") / "processed"
        records = {
            record["page"]: record["text"]
            for record in map(json.loads, (processed / "input.jsonl").read_text(encoding="utf-8").splitlines())
        }
        document_report = json.loads((processed / "report.json").read_text(encoding="utf-8"))["documents"][0]
        return report, records, document_report

    def test_figures_scope_sends_only_illustrated_and_textless_pages(self) -> None:
        analyzer = RecordingAnalyzer()

        report, records, document_report = self.preprocess(analyzer)

        self.assertEqual(sorted(page for page, _ in analyzer.calls), [2, 3])
        self.assertEqual(records, {1: "雨刷馬達拆卸步驟說明", 2: "Vision 第 2 頁", 3: "Vision 第 3 頁"})
        self.assertEqual((report.vision_model, report.vision_pages, report.vision_fallback_pages), ("gpt-6-luna", 2, 0))
        self.assertEqual(report.empty_pages, 0)
        self.assertEqual(document_report["vision_pages"], [2, 3])
        self.assertEqual(document_report["status"], "PROCESSED")
        self.assertTrue(base64.b64decode(analyzer.images[0]).startswith(b"\x89PNG"))

    def test_all_scope_sends_every_page_with_detected_language(self) -> None:
        analyzer = RecordingAnalyzer()

        _, records, _ = self.preprocess(analyzer, scope=SCOPE_ALL)

        self.assertEqual(sorted(analyzer.calls), [(1, None), (2, None), (3, None)])
        self.assertEqual(records[1], "Vision 第 1 頁")

    def test_failed_page_falls_back_to_text_layer(self) -> None:
        analyzer = RecordingAnalyzer({2: ValueError("bad json")})

        report, records, document_report = self.preprocess(analyzer)

        self.assertEqual(records[2], "雨刷連桿位置圖")
        self.assertEqual(report.vision_fallback_pages, 1)
        self.assertEqual(document_report["vision_fallback_pages"], [{"page": 2, "error": "bad json"}])
        self.assertEqual(document_report["status"], "PROCESSED_WITH_WARNINGS")

    def test_abort_error_stops_further_requests(self) -> None:
        analyzer = RecordingAnalyzer({2: VisionAbort("HTTP 401"), 3: VisionAbort("HTTP 401")})

        report, records, document_report = self.preprocess(analyzer, max_workers=1)

        self.assertEqual(analyzer.calls, [(2, None)])
        self.assertEqual(report.vision_abort_reason, "HTTP 401")
        self.assertEqual(report.vision_fallback_pages, 2)
        # The textless page has no fallback text, so it counts as empty.
        self.assertEqual(sorted(records), [1, 2])
        self.assertEqual(report.empty_pages, 1)

    def test_settings_reject_unknown_model_and_scope(self) -> None:
        with self.assertRaises(ProjectError):
            VisionSettings(model="gpt-unknown")
        with self.assertRaises(ProjectError):
            VisionSettings(model="gpt-6-luna", scope="everything")


class PageNeedsVisionTests(unittest.TestCase):
    def test_textless_page_and_all_scope_always_need_vision(self) -> None:
        document = pymupdf.open()
        page = document.new_page()
        self.assertTrue(page_needs_vision(page, SCOPE_FIGURES, ""))
        self.assertTrue(page_needs_vision(page, SCOPE_ALL, "text"))
        self.assertFalse(page_needs_vision(page, SCOPE_FIGURES, "text"))

    def test_illustration_outside_body_clip_is_ignored(self) -> None:
        document = pymupdf.open()
        page = document.new_page(width=595, height=842)
        page.draw_circle((300, 60), 50)
        self.assertTrue(page_needs_vision(page, SCOPE_FIGURES, "text"))
        self.assertFalse(page_needs_vision(page, SCOPE_FIGURES, "text", pymupdf.Rect(0, 200, 595, 842)))


class AnalyzePageTests(unittest.TestCase):
    settings = VisionSettings(model="gpt-6-luna", api_base_url="https://api.example/v1/", api_key="sk-test")

    def test_sends_image_and_serializes_json_content(self) -> None:
        content = {
            "page_number": 3,
            "content": [
                {"type": "text", "data": "雨刷規格"},
                {"type": "table", "data": "| 項目 | 值 |\n|---|---|\n| 電壓 | 12V |"},
                {"type": "diagram", "data": "連桿位於馬達右側"},
            ],
        }
        response = {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]}
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            text = analyze_page("aW1n", 3, "Traditional Chinese", self.settings)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "https://api.example/v1/chat/completions")
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertIn("written in Traditional Chinese", payload["messages"][0]["content"])
        self.assertIn("Do not transcribe page numbers", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["content"][1]["image_url"]["url"], "data:image/png;base64,aW1n")
        self.assertEqual(text, "雨刷規格\n\n| 項目 | 值 |\n|---|---|\n| 電壓 | 12V |\n\n連桿位於馬達右側")

    def test_authentication_error_aborts(self) -> None:
        error = urllib.error.HTTPError("url", 401, "Unauthorized", {}, io.BytesIO(b"invalid key"))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(VisionAbort):
                analyze_page("aW1n", 1, None, self.settings)

    def test_rejects_malformed_content(self) -> None:
        response = {"choices": [{"message": {"content": '{"content": [{"type": "chart", "data": "x"}]}'}}]}
        with mock.patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            with self.assertRaises(ValueError):
                analyze_page("aW1n", 1, None, self.settings)


class HelperTests(unittest.TestCase):
    def test_detect_document_language(self) -> None:
        self.assertEqual(detect_document_language(["這個車門開關的說明" * 3]), "Traditional Chinese")
        self.assertEqual(detect_document_language(["这个车门开关的说明" * 3]), "Simplified Chinese")
        self.assertEqual(detect_document_language(["Remove the wiper motor " * 10]), "English")
        self.assertIsNone(detect_document_language(["短"]))

    def test_serialize_skips_blank_items(self) -> None:
        result = {"content": [{"type": "text", "data": "a"}, {"type": "text", "data": "  "}, {"type": "diagram", "data": "b"}]}
        self.assertEqual(serialize_result(result), "a\nb")


if __name__ == "__main__":
    unittest.main()
