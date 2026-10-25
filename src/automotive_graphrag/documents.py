"""PDF import and page-level preprocessing for GraphRAG input."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol

import pymupdf

from .projects import ProjectError, ProjectStore
from .vision import PageAnalyzer, VisionAbort, VisionSettings, analyze_page, detect_document_language, page_needs_vision, render_page


class PdfPage(Protocol):
    rect: Any

    def get_text(self, option: str = "text", *, clip: Any = None) -> str: ...


class PdfDocument(Protocol):
    def __iter__(self) -> Iterator[PdfPage]: ...

    def __len__(self) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    filename: str
    pages: int | None
    size_bytes: int
    status: str
    empty_pages: int = 0
    error_pages: int = 0
    error: str | None = None
    processed_at: str | None = None
    indexed: bool = False

    @property
    def preprocessed(self) -> bool:
        return self.processed_at is not None and self.status.startswith("PROCESSED")


# Called with (completed pages, total selected pages, current filename).
ProgressCallback = Callable[[int, int, str], None]

INDEXED_DOCUMENTS_FILE = "indexed_documents.json"
# Manual page edits and review marks, kept across preprocessing runs.
PAGE_EDITS_FILE = "page_edits.json"
PAGE_RENDER_SCALE = 1.5


@dataclass(frozen=True, slots=True)
class PageReview:
    page: int
    # "vision", "text", "fallback" (vision failed, text layer kept), "empty" or "error".
    source: str
    text: str
    reviewed: bool = False
    edited: bool = False
    # The extraction changed after the page was edited, so the edit may be outdated.
    source_changed: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProcessingReport:
    project_id: str
    documents: int
    pages: int
    successful_pages: int
    empty_pages: int
    error_pages: int
    documents_with_errors: int
    output: str
    completed_at: str
    vision_model: str | None = None
    vision_pages: int = 0
    vision_fallback_pages: int = 0
    vision_abort_reason: str | None = None
    edited_pages: int = 0


class DocumentService:
    def __init__(
        self,
        projects: ProjectStore,
        reader_factory: Callable[[str | Path], PdfDocument] = pymupdf.open,
        page_analyzer: PageAnalyzer = analyze_page,
    ) -> None:
        self.projects = projects
        self.reader_factory = reader_factory
        self.page_analyzer = page_analyzer

    def import_pdfs(self, project_id: str, files: Iterable[str | Path]) -> list[DocumentInfo]:
        project = self.projects.get(project_id)
        sources = [Path(file) for file in files]
        if not sources:
            raise ProjectError("請至少選擇一份 PDF")
        for source in sources:
            if source.suffix.lower() != ".pdf":
                raise ProjectError(f"只允許匯入 PDF：{source.name}")
            if not source.is_file():
                raise ProjectError(f"找不到上傳檔案：{source.name}")

        source_directory = self.projects.path_for(project_id) / "source"
        for source in sources:
            self._atomic_copy(source, source_directory / source.name)
        # A replaced PDF must be preprocessed again; other PDFs keep their output.
        self._drop_processed_output(project_id, {source.name for source in sources})

        next_status = "STALE" if project.status in {"INDEXED", "STALE"} else "UPLOADED"
        self.projects.update_status(project_id, next_status)
        return self.list_documents(project_id)

    def list_documents(self, project_id: str) -> list[DocumentInfo]:
        source_directory = self.projects.path_for(project_id) / "source"
        report_by_name = self._document_reports(project_id)
        indexed_by_name = self.indexed_documents(project_id)
        documents: list[DocumentInfo] = []
        for path in sorted(source_directory.glob("*"), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            previous = report_by_name.get(path.name, {})
            pages: int | None = previous.get("pages")
            status = previous.get("status", "UPLOADED")
            error = previous.get("error")
            processed_at = previous.get("processed_at") if status.startswith("PROCESSED") else None
            if pages is None:
                document: PdfDocument | None = None
                try:
                    document = self.reader_factory(path)
                    pages = len(document)
                except Exception as exc:
                    status = "ERROR"
                    error = str(exc)
                finally:
                    if document is not None:
                        self._close_document(document)
            documents.append(
                DocumentInfo(
                    filename=path.name,
                    pages=pages,
                    size_bytes=path.stat().st_size,
                    status=status,
                    empty_pages=previous.get("empty_pages", 0),
                    error_pages=previous.get("error_pages", 0),
                    error=error,
                    processed_at=processed_at,
                    indexed=processed_at is not None and indexed_by_name.get(path.name) == processed_at,
                )
            )
        return documents

    def indexed_documents(self, project_id: str) -> dict[str, str]:
        """Return filename -> processed_at of the documents in the last successful graph."""
        path = self.projects.path_for(project_id) / "graphrag" / INDEXED_DOCUMENTS_FILE
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # Graphs built before this file existed: an INDEXED project covers its processed PDFs.
            if self.projects.get(project_id).status != "INDEXED":
                return {}
            return {
                name: item["processed_at"]
                for name, item in self._document_reports(project_id).items()
                if item.get("processed_at") and str(item.get("status", "")).startswith("PROCESSED")
            }
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def list_documents_if_available(self, project_id: str | None) -> list[DocumentInfo]:
        """Return an empty list when a UI selection refers to a deleted project."""
        if not project_id:
            return []
        try:
            return self.list_documents(project_id)
        except ProjectError:
            return []

    def processing_options(self, project_id: str) -> dict[str, dict[str, object]]:
        report_by_name = self._document_reports(project_id)
        option_names = (
            "header_ignore_percent",
            "footer_ignore_percent",
            "start_page",
            "end_page",
        )
        return {
            filename: {name: value[name] for name in option_names if name in value}
            for filename, value in report_by_name.items()
        }

    def remove_pdf(self, project_id: str, filename: str) -> None:
        project_path = self.projects.path_for(project_id)
        if not filename or Path(filename).name != filename or Path(filename).suffix.lower() != ".pdf":
            raise ProjectError("PDF 檔名格式不正確")
        if (project_path / "graphrag" / ".indexing.lock").exists():
            raise ProjectError("建圖工作執行中，無法移除 PDF")
        source = project_path / "source" / filename
        if not source.is_file():
            raise ProjectError(f"找不到 PDF：{filename}")

        source.unlink()
        records = self._drop_processed_output(project_id, {filename})
        (project_path / "graphrag" / "input" / "input.jsonl").unlink(missing_ok=True)
        if not self.projects.document_count(project_id):
            next_status = "EMPTY"
        elif (project_path / "graphrag" / "output").is_dir():
            next_status = "STALE"
        elif records:
            next_status = "READY"
        else:
            next_status = "UPLOADED"
        self.projects.update_status(project_id, next_status)

    def preprocess(
        self,
        project_id: str,
        options_by_filename: Mapping[str, Mapping[str, object]] | None = None,
        filenames: Iterable[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        vision: VisionSettings | None = None,
    ) -> ProcessingReport:
        """Preprocess all PDFs, or only `filenames` while keeping other PDFs' previous output.

        With `vision`, pages selected by its scope are transcribed by the vision
        model; a page whose transcription fails keeps its PDF text layer.
        """
        project = self.projects.get(project_id)
        project_path = self.projects.path_for(project_id)
        all_sources = sorted((project_path / "source").glob("*.pdf"), key=lambda item: item.name.casefold())
        if not all_sources:
            raise ProjectError("專案中沒有可處理的 PDF")
        source_names = {source.name for source in all_sources}
        unknown_options = set(options_by_filename or {}) - source_names
        if unknown_options:
            raise ProjectError(f"前處理設定含有不存在的 PDF：{', '.join(sorted(unknown_options))}")
        selected_names = set(filenames) if filenames is not None else source_names
        unknown_selection = selected_names - source_names
        if unknown_selection:
            raise ProjectError(f"找不到要前處理的 PDF：{', '.join(sorted(unknown_selection))}")
        if not selected_names:
            raise ProjectError("請至少選擇一份要前處理的 PDF")
        sources = [source for source in all_sources if source.name in selected_names]

        self.projects.update_status(project_id, "PROCESSING")
        processed_directory = project_path / "processed"
        previous_reports = self._document_reports(project_id)
        kept_records = [
            record
            for record in self._read_jsonl(processed_directory / "input.jsonl")
            if record.get("document_id") in source_names - selected_names
        ]
        records: list[dict[str, Any]] = []
        document_reports: list[dict[str, Any]] = []
        totals = {
            "pages": 0, "successful_pages": 0, "empty_pages": 0, "error_pages": 0,
            "vision_pages": 0, "vision_fallback_pages": 0, "edited_pages": 0,
        }
        all_edits = self._read_edits(project_id)
        progress = _PageProgress(
            progress_callback,
            {source.name: self._page_estimate(source, previous_reports, options_by_filename) for source in sources},
        )

        try:
            for source in sources:
                document_report = self._process_document(
                    project_id,
                    source,
                    records,
                    totals,
                    (options_by_filename or {}).get(source.name, {}),
                    progress,
                    vision,
                    all_edits.setdefault(source.name, {}),
                )
                document_reports.append(document_report)
            completed_at = datetime.now(timezone.utc).isoformat()
            for document_report in document_reports:
                document_report["processed_at"] = completed_at
            reports_by_name = {
                name: item for name, item in previous_reports.items() if name in source_names - selected_names
            }
            reports_by_name.update((item["filename"], item) for item in document_reports)
            report = ProcessingReport(
                project_id=project_id,
                documents=len(sources),
                documents_with_errors=sum(1 for item in document_reports if item["error"] or item["error_pages"]),
                output="input.jsonl",
                completed_at=completed_at,
                vision_model=vision.model if vision else None,
                vision_abort_reason=next(
                    (item["vision_abort_reason"] for item in document_reports if item.get("vision_abort_reason")), None
                ),
                **totals,
            )
            self._write_jsonl(processed_directory / "input.jsonl", self._sorted_records(kept_records + records))
            self._write_edits(project_id, all_edits)
            self._write_json(
                processed_directory / "report.json",
                {
                    "summary": asdict(report),
                    "documents": [reports_by_name[source.name] for source in all_sources if source.name in reports_by_name],
                },
            )
            self.projects.update_status(project_id, "STALE" if project.status in {"INDEXED", "STALE"} else "READY")
            progress.finish()
            return report
        except Exception as exc:
            self.projects.update_status(
                project_id,
                project.status if isinstance(exc, ProjectError) else "FAILED",
            )
            raise

    def _process_document(
        self,
        project_id: str,
        source: Path,
        records: list[dict[str, Any]],
        totals: dict[str, int],
        options: Mapping[str, object],
        progress: _PageProgress | None = None,
        vision: VisionSettings | None = None,
        edits: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Extract the selected pages; pages with a manual edit in `edits` keep the edited text."""
        edits = {} if edits is None else edits
        report: dict[str, Any] = {
            "filename": source.name,
            "pages": 0,
            "status": "PROCESSED",
            "empty_pages": 0,
            "error_pages": 0,
            "error": None,
            "page_errors": [],
        }
        document: PdfDocument | None = None
        try:
            document = self.reader_factory(source)
            pages = list(document)
        except Exception as exc:
            if document is not None:
                self._close_document(document)
            report.update(status="ERROR", error=str(exc))
            if progress is not None:
                progress.skip_document(source.name)
            return report

        page_count = len(pages)
        report["pages"] = page_count
        if page_count == 0:
            report.update(status="ERROR", error="PDF 沒有頁面")
            self._close_document(document)
            if progress is not None:
                progress.skip_document(source.name)
            return report
        try:
            header_ignore_percent, footer_ignore_percent = self._validate_ignored_margins(
                options.get("header_ignore_percent", 0),
                options.get("footer_ignore_percent", 0),
            )
            start_page = self._page_index(options.get("start_page", 0), "起始頁")
            end_value = options.get("end_page", page_count - 1)
            end_page = page_count - 1 if end_value in (None, "") else self._page_index(end_value, "結束頁")
        except Exception:
            self._close_document(document)
            raise
        if start_page > end_page:
            self._close_document(document)
            raise ProjectError(f"{source.name} 的起始頁不可大於結束頁")
        if end_page >= page_count:
            self._close_document(document)
            raise ProjectError(f"{source.name} 的結束頁超出範圍（最後一頁索引為 {page_count - 1}）")
        report.update(
            start_page=start_page,
            end_page=end_page,
            header_ignore_percent=header_ignore_percent,
            footer_ignore_percent=footer_ignore_percent,
            selected_pages=end_page - start_page + 1,
        )
        totals["pages"] += end_page - start_page + 1
        if progress is not None:
            progress.start_document(source.name, end_page - start_page + 1)
        selected = [(index + 1, page) for index, page in enumerate(pages) if start_page <= index <= end_page]
        texts: dict[int, str] = {}
        vision_texts: dict[int, str] = {}
        try:
            if vision is not None:
                report.update(vision_model=vision.model, vision_scope=vision.scope)
                vision_texts = self._transcribe_pages(
                    selected, texts, report, totals, header_ignore_percent, footer_ignore_percent, vision, progress,
                )
            else:
                for page_number, page in selected:
                    if progress is not None:
                        progress.advance()
                    self._read_page_text(page, page_number, texts, report, totals, header_ignore_percent, footer_ignore_percent)
        finally:
            self._close_document(document)
        report["edited_pages"] = []
        for page_number, _ in selected:
            extracted = (vision_texts.get(page_number) or texts[page_number]) if page_number in texts else ""
            entry = edits.get(str(page_number))
            if entry is not None and entry.get("extracted_text", "") != extracted:
                # A changed extraction outdates the review mark; an edit is kept but flagged.
                if "text" in entry:
                    entry["source_changed"] = True
                else:
                    entry["reviewed"] = False
                entry["extracted_text"] = extracted
            if entry is not None and "text" not in entry and not entry.get("reviewed"):
                del edits[str(page_number)]
                entry = None
            if entry is not None and "text" in entry:
                text = entry["text"]
                report["edited_pages"].append(page_number)
                totals["edited_pages"] += 1
            elif page_number not in texts:
                continue
            else:
                text = extracted
            if not text:
                report["empty_pages"] += 1
                totals["empty_pages"] += 1
                continue
            records.append(self._page_record(project_id, source.name, page_number, text))
            totals["successful_pages"] += 1
        if report["error_pages"]:
            report["status"] = "PROCESSED_WITH_ERRORS"
        elif report["empty_pages"] or report.get("vision_fallback_pages"):
            report["status"] = "PROCESSED_WITH_WARNINGS"
        return report

    @classmethod
    def _page_record(cls, project_id: str, filename: str, page_number: int, text: str) -> dict[str, Any]:
        stem = Path(filename).stem
        document_stem = cls._safe_identifier(stem)
        return {
            "id": f"{project_id}-{document_stem}-p{page_number:04d}",
            "chunk_id": f"{project_id}-{document_stem}-p{page_number:04d}-b01",
            "project_id": project_id,
            "section_id": document_stem,
            "section_name": stem,
            "title": f"{filename} - Page {page_number}",
            "text": text,
            "document_id": filename,
            "page": page_number,
            "block_id": "b01",
        }

    def page_reviews(self, project_id: str, filename: str) -> list[PageReview]:
        """Every selected page of a preprocessed PDF with its final text and review state."""
        document_report = self._processed_report(project_id, filename)
        texts = {
            record["page"]: record["text"]
            for record in self._read_jsonl(self.projects.path_for(project_id) / "processed" / "input.jsonl")
            if record.get("document_id") == filename
        }
        edits = self._read_edits(project_id).get(filename, {})
        errors = {item["page"]: item["error"] for item in document_report.get("page_errors", [])}
        fallbacks = {item["page"]: item["error"] for item in document_report.get("vision_fallback_pages", [])}
        vision_pages = set(document_report.get("vision_pages", []))
        reviews: list[PageReview] = []
        for page_number in self._report_page_numbers(document_report):
            entry = edits.get(str(page_number), {})
            text = texts.get(page_number, "")
            if page_number in errors:
                source = "error"
            elif page_number in fallbacks:
                source = "fallback"
            elif page_number in vision_pages:
                source = "vision"
            elif page_number not in texts and "text" not in entry:
                source = "empty"
            else:
                source = "text"
            reviews.append(
                PageReview(
                    page=page_number,
                    source=source,
                    text=text,
                    reviewed=bool(entry.get("reviewed")),
                    edited="text" in entry,
                    source_changed=bool(entry.get("source_changed")),
                    error=errors.get(page_number) or fallbacks.get(page_number),
                )
            )
        return reviews

    def save_page_text(self, project_id: str, filename: str, page_number: int, text: str) -> PageReview:
        """Save a page's reviewed text; text that differs from the extraction is kept as a manual edit."""
        text = text.strip()
        return self._update_page(project_id, filename, page_number, lambda extracted: text)

    def revert_page_text(self, project_id: str, filename: str, page_number: int) -> PageReview:
        """Drop a page's manual edit and use the extracted text again."""
        return self._update_page(project_id, filename, page_number, lambda extracted: extracted, reviewed=False)

    def _update_page(
        self,
        project_id: str,
        filename: str,
        page_number: int,
        new_text: Callable[[str], str],
        reviewed: bool = True,
    ) -> PageReview:
        project = self.projects.get(project_id)
        project_path = self.projects.path_for(project_id)
        if (project_path / "graphrag" / ".indexing.lock").exists():
            raise ProjectError("建圖工作執行中，無法修改頁面內容")
        document_report = self._processed_report(project_id, filename)
        if page_number not in self._report_page_numbers(document_report):
            raise ProjectError(f"第 {page_number} 頁不在 {filename} 的前處理範圍內")
        processed_directory = project_path / "processed"
        records = self._read_jsonl(processed_directory / "input.jsonl")
        edits = self._read_edits(project_id)
        document_edits = edits.setdefault(filename, {})
        entry = document_edits.get(str(page_number), {})
        record = next(
            (item for item in records if item.get("document_id") == filename and item.get("page") == page_number), None
        )
        extracted = entry["extracted_text"] if "extracted_text" in entry else (record or {}).get("text", "")
        text = new_text(extracted)
        entry = {"extracted_text": extracted, "reviewed": reviewed}
        if text != extracted:
            entry["text"] = text
        if entry["reviewed"] or "text" in entry:
            document_edits[str(page_number)] = entry
        else:
            document_edits.pop(str(page_number), None)
        records = [item for item in records if item is not record]
        if text:
            records.append(self._page_record(project_id, filename, page_number, text))
        # A new processed_at marks the PDF as changed since the last graph.
        report = self._read_report(project_id)
        for item in report.get("documents", []):
            if item.get("filename") == filename:
                item["processed_at"] = datetime.now(timezone.utc).isoformat()
                item["edited_pages"] = sorted(int(page) for page, value in document_edits.items() if "text" in value)
        self._write_jsonl(processed_directory / "input.jsonl", self._sorted_records(records))
        self._write_edits(project_id, edits)
        self._write_json(processed_directory / "report.json", report)
        if project.status == "INDEXED":
            self.projects.update_status(project_id, "STALE")
        return next(item for item in self.page_reviews(project_id, filename) if item.page == page_number)

    def extract_page_text(
        self, project_id: str, filename: str, page_number: int, vision: VisionSettings | None = None,
    ) -> str:
        """Extract one page again, from its text layer or with `vision`, without saving it."""
        document_report = self._processed_report(project_id, filename)
        header = float(document_report.get("header_ignore_percent", 0))
        footer = float(document_report.get("footer_ignore_percent", 0))
        document = pymupdf.open(self._source_path(project_id, filename))
        try:
            page = self._load_page(document, page_number)
            if vision is None:
                return self._extract_page_text(page, header, footer).strip()
            language = detect_document_language(
                self._extract_page_text(document[index], header, footer) for index in self._report_indices(document_report)
            )
            image = render_page(page, self._body_clip(page, header, footer))
        finally:
            document.close()
        return self.page_analyzer(image, page_number, language, vision).strip()

    def render_page_image(self, project_id: str, filename: str, page_number: int) -> bytes:
        """The whole 1-based page as PNG, for comparing with its extracted text."""
        document = pymupdf.open(self._source_path(project_id, filename))
        try:
            page = self._load_page(document, page_number)
            return page.get_pixmap(matrix=pymupdf.Matrix(PAGE_RENDER_SCALE, PAGE_RENDER_SCALE), alpha=False).tobytes("png")
        finally:
            document.close()

    def _source_path(self, project_id: str, filename: str) -> Path:
        if not filename or Path(filename).name != filename:
            raise ProjectError("PDF 檔名格式不正確")
        path = self.projects.path_for(project_id) / "source" / filename
        if not path.is_file():
            raise ProjectError(f"找不到 PDF：{filename}")
        return path

    @staticmethod
    def _load_page(document: Any, page_number: int) -> Any:
        if not 1 <= page_number <= len(document):
            raise ProjectError(f"頁碼超出範圍；PDF 共 {len(document)} 頁")
        return document[page_number - 1]

    def _processed_report(self, project_id: str, filename: str) -> dict[str, Any]:
        document_report = self._document_reports(project_id).get(filename)
        if not document_report or not str(document_report.get("status", "")).startswith("PROCESSED"):
            raise ProjectError(f"{filename} 尚未完成前處理")
        return document_report

    @staticmethod
    def _report_indices(document_report: Mapping[str, Any]) -> range:
        """0-based indices of the preprocessed pages; reports before page ranges cover the whole PDF."""
        start = int(document_report.get("start_page", 0))
        end = int(document_report.get("end_page", int(document_report.get("pages") or 0) - 1))
        return range(start, end + 1)

    @classmethod
    def _report_page_numbers(cls, document_report: Mapping[str, Any]) -> list[int]:
        return [index + 1 for index in cls._report_indices(document_report)]

    def _read_edits(self, project_id: str) -> dict[str, dict[str, dict[str, Any]]]:
        path = self.projects.path_for(project_id) / "processed" / PAGE_EDITS_FILE
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_edits(self, project_id: str, edits: Mapping[str, Mapping[str, Any]]) -> None:
        path = self.projects.path_for(project_id) / "processed" / PAGE_EDITS_FILE
        kept = {filename: pages for filename, pages in edits.items() if pages}
        if kept:
            self._write_json(path, kept)
        else:
            path.unlink(missing_ok=True)

    def _read_page_text(
        self,
        page: PdfPage,
        page_number: int,
        texts: dict[int, str],
        report: dict[str, Any],
        totals: dict[str, int],
        header_percent: float,
        footer_percent: float,
    ) -> None:
        """Store the page's text layer in `texts`, or record a page error."""
        try:
            texts[page_number] = self._extract_page_text(page, header_percent, footer_percent).strip()
        except Exception as exc:
            report["error_pages"] += 1
            totals["error_pages"] += 1
            report["page_errors"].append({"page": page_number, "error": str(exc)})

    def _transcribe_pages(
        self,
        selected: list[tuple[int, PdfPage]],
        texts: dict[int, str],
        report: dict[str, Any],
        totals: dict[str, int],
        header_percent: float,
        footer_percent: float,
        vision: VisionSettings,
        progress: _PageProgress | None,
    ) -> dict[int, str]:
        """Read every page's text layer and transcribe the pages in scope with the vision model.

        Pages are rendered in this thread (PyMuPDF documents are not thread-safe)
        and only the API calls run concurrently. A failed page keeps its text
        layer and is listed in `vision_fallback_pages`; an error that would
        repeat on every page (bad key, unknown model) stops further requests.
        """
        for page_number, page in selected:
            self._read_page_text(page, page_number, texts, report, totals, header_percent, footer_percent)
        language = detect_document_language(texts.values())
        report.update(language=language, vision_pages=[], vision_fallback_pages=[])
        vision_texts: dict[int, str] = {}
        stop = threading.Event()
        abort_reasons: list[str] = []

        def advance() -> None:
            if progress is not None:
                progress.advance()

        def fall_back(page_number: int, reason: str) -> None:
            report["vision_fallback_pages"].append({"page": page_number, "error": reason})
            totals["vision_fallback_pages"] += 1
            advance()

        def analyze(page_number: int, image: str) -> str:
            if stop.is_set():
                raise VisionAbort("Vision 解析已中止")
            try:
                return self.page_analyzer(image, page_number, language, vision)
            except VisionAbort as exc:
                # Set here, not in collect(), so queued pages skip the API at once.
                stop.set()
                abort_reasons.append(str(exc))
                raise

        def collect(pending: dict[Future[str], int]) -> None:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                page_number = pending.pop(future)
                try:
                    vision_texts[page_number] = future.result().strip()
                except VisionAbort:
                    fall_back(page_number, "Vision 解析已中止")
                except Exception as exc:
                    fall_back(page_number, str(exc) or type(exc).__name__)
                else:
                    report["vision_pages"].append(page_number)
                    totals["vision_pages"] += 1
                    advance()

        pending: dict[Future[str], int] = {}
        with ThreadPoolExecutor(max_workers=vision.max_workers) as pool:
            for page_number, page in selected:
                if page_number not in texts:
                    advance()
                    continue
                body = self._body_clip(page, header_percent, footer_percent)
                if not page_needs_vision(page, vision.scope, texts[page_number], body):
                    advance()
                    continue
                if stop.is_set():
                    fall_back(page_number, "Vision 解析已中止")
                    continue
                try:
                    image = render_page(page, body)
                except Exception as exc:
                    fall_back(page_number, f"頁面渲染失敗：{exc}")
                    continue
                pending[pool.submit(analyze, page_number, image)] = page_number
                # Bound the number of rendered images held in memory.
                if len(pending) >= vision.max_workers * 2:
                    collect(pending)
            while pending:
                collect(pending)
        report["vision_abort_reason"] = abort_reasons[0] if abort_reasons else None
        report["vision_pages"].sort()
        report["vision_fallback_pages"].sort(key=lambda item: item["page"])
        return vision_texts

    @staticmethod
    def _body_clip(page: PdfPage, header_percent: float, footer_percent: float) -> pymupdf.Rect | None:
        """The page area left after ignoring the header and footer, or None for the whole page."""
        if header_percent == 0 and footer_percent == 0:
            return None
        rect = page.rect
        return pymupdf.Rect(
            rect.x0,
            rect.y0 + rect.height * header_percent / 100,
            rect.x1,
            rect.y1 - rect.height * footer_percent / 100,
        )

    @staticmethod
    def _validate_ignored_margins(header_percent: float, footer_percent: float) -> tuple[float, float]:
        try:
            header = float(header_percent)
            footer = float(footer_percent)
        except (TypeError, ValueError) as exc:
            raise ProjectError("頁首與頁尾忽略比例必須是數字") from exc
        if not 0 <= header < 100 or not 0 <= footer < 100:
            raise ProjectError("頁首與頁尾忽略比例必須介於 0%（含）與 100%（不含）")
        if header + footer >= 100:
            raise ProjectError("頁首與頁尾忽略比例合計必須小於 100%")
        return header, footer

    @staticmethod
    def _page_index(value: object, label: str) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"{label}必須是整數頁面索引") from exc
        if not number.is_integer() or number < 0:
            raise ProjectError(f"{label}必須是 0 或更大的整數頁面索引")
        return int(number)

    @staticmethod
    def _extract_page_text(page: PdfPage, header_percent: float, footer_percent: float) -> str:
        if header_percent == 0 and footer_percent == 0:
            return page.get_text("text") or ""

        try:
            page_left = float(page.rect.x0)
            page_top = float(page.rect.y0)
            page_right = float(page.rect.x1)
            page_bottom = float(page.rect.y1)
            page_height = float(page.rect.height)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("無法取得 PDF 頁面高度以套用頁首頁尾忽略比例") from exc
        clip_top = page_top + page_height * header_percent / 100
        clip_bottom = page_bottom - page_height * footer_percent / 100
        return page.get_text("text", clip=(page_left, clip_top, page_right, clip_bottom)) or ""

    @staticmethod
    def _close_document(document: PdfDocument) -> None:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _page_estimate(
        source: Path,
        previous_reports: Mapping[str, Mapping[str, Any]],
        options_by_filename: Mapping[str, Mapping[str, object]] | None,
    ) -> int:
        """Estimate selected pages before opening the PDF; corrected once the PDF is read."""
        options = (options_by_filename or {}).get(source.name, {})
        pages = previous_reports.get(source.name, {}).get("pages") or 0
        try:
            start = int(float(options.get("start_page", 0) or 0))
            end_value = options.get("end_page")
            end = pages - 1 if end_value in (None, "") else int(float(end_value))
        except (TypeError, ValueError):
            return pages
        return max(end - start + 1, 0) if pages else 0

    @staticmethod
    def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(records, key=lambda record: (str(record.get("document_id", "")).casefold(), record.get("page", 0)))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def _drop_processed_output(self, project_id: str, filenames: set[str]) -> list[dict[str, Any]]:
        """Remove the given PDFs from processed output and return the remaining records."""
        processed_directory = self.projects.path_for(project_id) / "processed"
        edits = self._read_edits(project_id)
        if filenames & set(edits):
            self._write_edits(project_id, {name: pages for name, pages in edits.items() if name not in filenames})
        report = self._read_report(project_id)
        records = self._read_jsonl(processed_directory / "input.jsonl")
        document_reports = report.get("documents", [])
        if not any(item.get("filename") in filenames for item in document_reports):
            return records
        records = [record for record in records if record.get("document_id") not in filenames]
        document_reports = [item for item in document_reports if item.get("filename") not in filenames]
        if document_reports:
            self._write_jsonl(processed_directory / "input.jsonl", records)
            self._write_json(
                processed_directory / "report.json",
                {"summary": report.get("summary", {}), "documents": document_reports},
            )
        else:
            (processed_directory / "input.jsonl").unlink(missing_ok=True)
            (processed_directory / "report.json").unlink(missing_ok=True)
        return records

    def _read_report(self, project_id: str) -> dict[str, Any]:
        report_path = self.projects.path_for(project_id) / "processed" / "report.json"
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _document_reports(self, project_id: str) -> dict[str, dict[str, Any]]:
        report = self._read_report(project_id)
        # Reports written before per-document timestamps share the run's completion time.
        fallback = report.get("summary", {}).get("completed_at")
        reports: dict[str, dict[str, Any]] = {}
        for item in report.get("documents", []):
            if fallback and "processed_at" not in item:
                item = {**item, "processed_at": fallback}
            reports[item["filename"]] = item
        return reports

    @staticmethod
    def _safe_identifier(value: str) -> str:
        safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
        return safe.strip("-") or "document"

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix=".upload-", suffix=".pdf")
        os.close(handle)
        try:
            shutil.copy2(source, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        DocumentService._atomic_write(path, content)

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        DocumentService._atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


class _PageProgress:
    """Aggregate per-page progress across documents for a ProgressCallback."""

    def __init__(self, callback: ProgressCallback | None, estimates: Mapping[str, int]) -> None:
        self.callback = callback
        self.estimates = dict(estimates)
        self.total = sum(self.estimates.values())
        self.completed = 0
        self.filename = ""

    def start_document(self, filename: str, pages: int) -> None:
        # Replace the pre-read estimate with the actual selected page count.
        self.total += pages - self.estimates.pop(filename, 0)
        self.filename = filename
        self._emit()

    def advance(self) -> None:
        self.completed += 1
        self._emit()

    def skip_document(self, filename: str) -> None:
        self.total -= self.estimates.pop(filename, 0)
        self.filename = filename
        self._emit()

    def finish(self) -> None:
        self.total = self.completed
        self._emit()

    def _emit(self) -> None:
        if self.callback is not None:
            self.callback(self.completed, max(self.total, self.completed), self.filename)
