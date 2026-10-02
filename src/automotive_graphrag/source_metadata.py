"""Map GraphRAG text units back to original PDF source metadata."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .projects import ProjectError, ProjectStore


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    text_unit_id: str
    chunk_id: str
    project_id: str
    section_id: str
    section_name: str
    document_id: str
    page: int
    block_id: str
    text: str


class SourceMetadataService:
    def __init__(self, projects: ProjectStore) -> None:
        self.projects = projects

    def build(self, project_id: str) -> list[SourceMetadata]:
        project_path = self.projects.path_for(project_id)
        output = project_path / "graphrag" / "output"
        documents_path = output / "documents.parquet"
        text_units_path = output / "text_units.parquet"
        if not documents_path.is_file() or not text_units_path.is_file():
            raise ProjectError("GraphRAG 索引缺少 documents.parquet 或 text_units.parquet")

        documents = pd.read_parquet(documents_path)
        text_units = pd.read_parquet(text_units_path)
        if not {"id", "raw_data"}.issubset(documents.columns):
            raise ProjectError("GraphRAG documents.parquet 缺少來源欄位")
        if not {"id", "document_id", "text"}.issubset(text_units.columns):
            raise ProjectError("GraphRAG text_units.parquet 缺少來源欄位")

        metadata_by_document: dict[str, dict[str, Any]] = {}
        for row in documents.to_dict(orient="records"):
            raw_data = row.get("raw_data")
            if isinstance(raw_data, str):
                try:
                    raw_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    raw_data = None
            if isinstance(raw_data, dict):
                metadata_by_document[str(row["id"])] = raw_data

        mappings: list[SourceMetadata] = []
        missing_documents: set[str] = set()
        required = (
            "chunk_id",
            "project_id",
            "section_id",
            "section_name",
            "document_id",
            "page",
            "block_id",
        )
        for row in text_units.to_dict(orient="records"):
            graph_document_id = str(row["document_id"])
            source = metadata_by_document.get(graph_document_id)
            if source is None or any(field not in source for field in required):
                missing_documents.add(graph_document_id)
                continue
            mappings.append(
                SourceMetadata(
                    text_unit_id=str(row["id"]),
                    chunk_id=str(source["chunk_id"]),
                    project_id=str(source["project_id"]),
                    section_id=str(source["section_id"]),
                    section_name=str(source["section_name"]),
                    document_id=str(source["document_id"]),
                    page=int(source["page"]),
                    block_id=str(source["block_id"]),
                    text=str(row["text"]),
                )
            )
        if missing_documents:
            examples = ", ".join(sorted(missing_documents)[:3])
            raise ProjectError(f"GraphRAG Text Unit 無法回連來源 Metadata：{examples}")

        destination = output / "source_metadata.jsonl"
        content = "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in mappings)
        self._atomic_text(destination, content)
        return mappings

    def load(self, project_id: str) -> list[SourceMetadata]:
        path = self.projects.path_for(project_id) / "graphrag" / "output" / "source_metadata.jsonl"
        try:
            return [
                SourceMetadata(**json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except FileNotFoundError as exc:
            raise ProjectError("找不到來源 Metadata 映射，請重新建圖") from exc
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProjectError("來源 Metadata 映射格式錯誤") from exc

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".source-metadata-", suffix=".jsonl")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
