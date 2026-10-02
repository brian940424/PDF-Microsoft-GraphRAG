"""Persist selected user-facing query results as reviewable cases."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evidence import Evidence
from .projects import ProjectError, ProjectStore
from .querying import QueryService


@dataclass(frozen=True, slots=True)
class CaseRecord:
    case_id: str
    project_id: str
    query_id: str
    question: str
    answer: str
    evidence: tuple[Evidence, ...]
    note: str
    created_at: str


class CaseService:
    def __init__(self, projects: ProjectStore, queries: QueryService | None = None) -> None:
        self.projects = projects
        self.queries = queries or QueryService(projects)

    def save_query(self, project_id: str, query_id: str, note: str = "") -> CaseRecord:
        query = next((item for item in self.queries.history(project_id) if item.query_id == query_id), None)
        if query is None:
            raise ProjectError(f"找不到此專案的問答紀錄：{query_id}")
        if query.status != "COMPLETED" or not query.answer:
            raise ProjectError("只有完成且具有回答的問答可以加入案例")
        existing = self.list(project_id)
        if any(item.query_id == query_id for item in existing):
            raise ProjectError("此問答已加入案例紀錄")
        record = CaseRecord(
            case_id=uuid.uuid4().hex,
            project_id=project_id,
            query_id=query.query_id,
            question=query.question,
            answer=query.answer,
            evidence=query.evidence,
            note=note.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write(project_id, [*existing, record])
        return record

    def list(self, project_id: str) -> list[CaseRecord]:
        self.projects.get(project_id)
        try:
            value = json.loads(self._path(project_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise ProjectError("案例紀錄格式錯誤") from exc
        if not isinstance(value, list):
            raise ProjectError("案例紀錄格式錯誤")
        records = []
        try:
            for item in value:
                item["evidence"] = tuple(Evidence(**evidence) for evidence in item.get("evidence", []))
                records.append(CaseRecord(**item))
        except (KeyError, TypeError) as exc:
            raise ProjectError("案例紀錄格式錯誤") from exc
        return records

    def export(self, project_id: str) -> tuple[Path, Path]:
        records = self.list(project_id)
        if not records:
            raise ProjectError("尚無案例紀錄可匯出")
        directory = self.projects.path_for(project_id) / "exports"
        json_path = directory / "query-cases.json"
        csv_path = directory / "query-cases.csv"
        self._atomic_text(json_path, json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(CaseRecord.__dataclass_fields__))
        writer.writeheader()
        for item in records:
            row = asdict(item)
            row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False)
            writer.writerow(row)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    def _path(self, project_id: str) -> Path:
        return self.projects.path_for(project_id) / "runs" / "query-cases.json"

    def _write(self, project_id: str, records: list[CaseRecord]) -> None:
        self._atomic_text(
            self._path(project_id),
            json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
