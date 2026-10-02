"""Resolve GraphRAG query context sources to original PDF evidence."""

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
    page: int
    block_id: str
    text: str
    score: float | None = None


class EvidenceService:
    def __init__(self, projects: ProjectStore, metadata: SourceMetadataService | None = None) -> None:
        self.projects = projects
        self.metadata = metadata or SourceMetadataService(projects)

    def from_context(self, project_id: str, context: dict[str, Any]) -> list[Evidence]:
        raw_sources = context.get("sources", [])
        if not isinstance(raw_sources, list):
            return []
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

        result: list[Evidence] = []
        seen: set[str] = set()
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
        return result
