"""PDF import and page-level preprocessing for GraphRAG input."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from pypdf import PdfReader

from .projects import ProjectError, ProjectStore


class PdfPage(Protocol):
    def extract_text(self) -> str | None: ...


class PdfDocument(Protocol):
    pages: Iterable[PdfPage]


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    filename: str
    pages: int | None
    size_bytes: int
    status: str
    empty_pages: int = 0
    error_pages: int = 0
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


class DocumentService:
    def __init__(
        self,
        projects: ProjectStore,
        reader_factory: Callable[[str | Path], PdfDocument] = PdfReader,
    ) -> None:
        self.projects = projects
        self.reader_factory = reader_factory

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

        next_status = "STALE" if project.status in {"INDEXED", "STALE"} else "UPLOADED"
        self.projects.update_status(project_id, next_status)
        return self.list_documents(project_id)

    def list_documents(self, project_id: str) -> list[DocumentInfo]:
        source_directory = self.projects.path_for(project_id) / "source"
        report_by_name = self._document_reports(project_id)
        documents: list[DocumentInfo] = []
        for path in sorted(source_directory.glob("*"), key=lambda item: item.name.casefold()):
            if not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            previous = report_by_name.get(path.name, {})
            pages: int | None = previous.get("pages")
            status = previous.get("status", "UPLOADED")
            error = previous.get("error")
            if pages is None:
                try:
                    pages = len(list(self.reader_factory(path).pages))
                except Exception as exc:
                    status = "ERROR"
                    error = str(exc)
            documents.append(
                DocumentInfo(
                    filename=path.name,
                    pages=pages,
                    size_bytes=path.stat().st_size,
                    status=status,
                    empty_pages=previous.get("empty_pages", 0),
                    error_pages=previous.get("error_pages", 0),
                    error=error,
                )
            )
        return documents

    def preprocess(self, project_id: str) -> ProcessingReport:
        project_path = self.projects.path_for(project_id)
        sources = sorted((project_path / "source").glob("*.pdf"), key=lambda item: item.name.casefold())
        if not sources:
            raise ProjectError("專案中沒有可處理的 PDF")

        self.projects.update_status(project_id, "PROCESSING")
        processed_directory = project_path / "processed"
        records: list[dict[str, Any]] = []
        document_reports: list[dict[str, Any]] = []
        totals = {"pages": 0, "successful_pages": 0, "empty_pages": 0, "error_pages": 0}

        try:
            for source in sources:
                document_report = self._process_document(project_id, source, records, totals)
                document_reports.append(document_report)
            report = ProcessingReport(
                project_id=project_id,
                documents=len(sources),
                documents_with_errors=sum(1 for item in document_reports if item["error"] or item["error_pages"]),
                output="input.jsonl",
                completed_at=datetime.now(timezone.utc).isoformat(),
                **totals,
            )
            self._write_jsonl(processed_directory / "input.jsonl", records)
            self._write_json(
                processed_directory / "report.json",
                {"summary": asdict(report), "documents": document_reports},
            )
            self.projects.update_status(project_id, "READY")
            return report
        except Exception:
            self.projects.update_status(project_id, "FAILED")
            raise

    def _process_document(
        self,
        project_id: str,
        source: Path,
        records: list[dict[str, Any]],
        totals: dict[str, int],
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "filename": source.name,
            "pages": 0,
            "status": "PROCESSED",
            "empty_pages": 0,
            "error_pages": 0,
            "error": None,
            "page_errors": [],
        }
        try:
            pages = list(self.reader_factory(source).pages)
        except Exception as exc:
            report.update(status="ERROR", error=str(exc))
            return report

        report["pages"] = len(pages)
        totals["pages"] += len(pages)
        document_stem = self._safe_identifier(source.stem)
        for page_number, page in enumerate(pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
            except Exception as exc:
                report["error_pages"] += 1
                totals["error_pages"] += 1
                report["page_errors"].append({"page": page_number, "error": str(exc)})
                continue
            if not text:
                report["empty_pages"] += 1
                totals["empty_pages"] += 1
                continue
            records.append(
                {
                    "id": f"{project_id}-{document_stem}-p{page_number:04d}",
                    "title": f"{source.name} - Page {page_number}",
                    "text": text,
                    "document_id": source.name,
                    "page": page_number,
                }
            )
            totals["successful_pages"] += 1
        if report["error_pages"]:
            report["status"] = "PROCESSED_WITH_ERRORS"
        elif report["empty_pages"]:
            report["status"] = "PROCESSED_WITH_WARNINGS"
        return report

    def _document_reports(self, project_id: str) -> dict[str, dict[str, Any]]:
        report_path = self.projects.path_for(project_id) / "processed" / "report.json"
        try:
            value = json.loads(report_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return {item["filename"]: item for item in value.get("documents", [])}

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
