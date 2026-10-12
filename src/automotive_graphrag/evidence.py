"""Resolve GraphRAG query context sources and reports to displayable evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .projects import ProjectError, ProjectStore
from .source_metadata import SourceMetadataService


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    rank: int
    context_id: str
    text_unit_id: str
    chunk_id: str
    section_id: str
    section_name: str
    document_id: str
    page: int | None
    block_id: str
    text: str
    score: float | None = None


class EvidenceService:
    def __init__(self, projects: ProjectStore, metadata: SourceMetadataService | None = None) -> None:
        self.projects = projects
        self.metadata = metadata or SourceMetadataService(projects)

    def from_context(self, project_id: str, context: dict[str, Any]) -> list[Evidence]:
        raw_sources = self.source_records(context)
        result: list[Evidence] = []
        seen: set[str] = set()
        if isinstance(raw_sources, list) and raw_sources:
            output = self.projects.path_for(project_id) / "graphrag" / "output"
            text_units_path = output / "text_units.parquet"
            if not text_units_path.is_file():
                raise ProjectError("找不到 GraphRAG text_units.parquet")
            text_units = pd.read_parquet(text_units_path, columns=["id", "human_readable_id"])
            full_id_by_short_id = {
                str(row["human_readable_id"]): str(row["id"])
                for row in text_units.to_dict(orient="records")
            }
            metadata_by_text_unit = {item.text_unit_id: item for item in self.metadata.load(project_id)}

            for source in raw_sources:
                if not isinstance(source, dict) or source.get("in_context") is False:
                    continue
                context_id = str(source.get("id", ""))
                text_unit_id = full_id_by_short_id.get(context_id, context_id)
                source_metadata = metadata_by_text_unit.get(text_unit_id)
                if source_metadata is None or text_unit_id in seen:
                    continue
                seen.add(text_unit_id)
                rank = len(result) + 1
                result.append(
                    Evidence(
                        evidence_id=f"E{rank}",
                        rank=rank,
                        context_id=context_id,
                        text_unit_id=text_unit_id,
                        chunk_id=source_metadata.chunk_id,
                        section_id=source_metadata.section_id,
                        section_name=source_metadata.section_name,
                        document_id=source_metadata.document_id,
                        page=source_metadata.page,
                        block_id=source_metadata.block_id,
                        text=source_metadata.text,
                    )
                )
        if result:
            return result

        # Global Search supplies community reports rather than source text units.
        # Preserve the exact report-level context without pretending it maps to a PDF page.
        raw_reports = context.get("reports", [])
        if not isinstance(raw_reports, list):
            return []
        for report in raw_reports:
            if not isinstance(report, dict) or report.get("in_context") is False:
                continue
            context_id = str(report.get("id") or report.get("human_readable_id") or "")
            text = str(report.get("summary") or report.get("content") or report.get("full_content") or "")
            if not context_id or not text or context_id in seen:
                continue
            seen.add(context_id)
            rank = len(result) + 1
            community = str(report.get("community") or "")
            title = str(report.get("title") or f"Community Report {context_id}")
            result.append(
                Evidence(
                    evidence_id=f"E{rank}",
                    rank=rank,
                    context_id=context_id,
                    text_unit_id="",
                    chunk_id="",
                    section_id=community,
                    section_name=title,
                    document_id="GraphRAG Community Report",
                    page=None,
                    block_id="",
                    text=text,
                )
            )
        return result

    @staticmethod
    def source_records(context: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect source text-unit rows from direct or nested DRIFT contexts."""
        records: list[dict[str, Any]] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                sources = value.get("sources")
                if isinstance(sources, list):
                    records.extend(item for item in sources if isinstance(item, dict))
                for key, nested in value.items():
                    if key != "sources":
                        visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(context)
        return records
