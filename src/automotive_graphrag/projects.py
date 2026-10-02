"""Persistent, filesystem-backed project management."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
PROJECT_DIRECTORIES = (
    "source",
    "processed",
    "graphrag",
    "question_sets",
    "runs",
    "exports",
)
PROJECT_STATUSES = {
    "EMPTY",
    "UPLOADED",
    "PROCESSING",
    "READY",
    "INDEXING",
    "INDEXED",
    "STALE",
    "FAILED",
}


class ProjectError(ValueError):
    """A project request cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class Project:
    project_id: str
    display_name: str
    vehicle_name: str
    manual_version: str
    description: str
    enabled: bool
    status: str
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Project":
        try:
            normalized = dict(value)
            normalized.setdefault("enabled", True)
            project = cls(**{field: normalized[field] for field in cls.__dataclass_fields__})
        except (KeyError, TypeError) as exc:
            raise ProjectError("project.json 缺少必要欄位或格式錯誤") from exc
        if not isinstance(project.enabled, bool):
            raise ProjectError("專案 enabled 欄位格式錯誤")
        if project.status not in PROJECT_STATUSES:
            raise ProjectError(f"未知的專案狀態：{project.status}")
        return project


class ProjectStore:
    """Create and discover isolated GraphRAG project workspaces."""

    def __init__(self, root: str | Path = "projects") -> None:
        self.root = Path(root)

    def create(
        self,
        *,
        project_id: str,
        display_name: str,
        vehicle_name: str,
        manual_version: str,
        description: str = "",
    ) -> Project:
        values = {
            "project_id": project_id.strip(),
            "display_name": display_name.strip(),
            "vehicle_name": vehicle_name.strip(),
            "manual_version": manual_version.strip(),
            "description": description.strip(),
        }
        self._validate(values)
        project_path = self.root / values["project_id"]

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            project_path.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise ProjectError(f"Project ID 已存在：{values['project_id']}") from exc

        try:
            for directory in PROJECT_DIRECTORIES:
                (project_path / directory).mkdir()
            now = datetime.now(timezone.utc).isoformat()
            project = Project(
                **values,
                enabled=True,
                status="EMPTY",
                created_at=now,
                updated_at=now,
            )
            self._write_metadata(project_path / "project.json", asdict(project))
        except Exception:
            self._remove_empty_project(project_path)
            raise
        return project

    def list(self) -> list[Project]:
        if not self.root.exists():
            return []
        projects: list[Project] = []
        for path in self.root.iterdir():
            metadata = path / "project.json"
            if path.is_dir() and metadata.is_file():
                projects.append(self.get(path.name))
        return sorted(projects, key=lambda project: project.project_id.casefold())

    def get(self, project_id: str) -> Project:
        if not PROJECT_ID_PATTERN.fullmatch(project_id):
            raise ProjectError("Project ID 格式不正確")
        metadata = self.root / project_id / "project.json"
        try:
            value = json.loads(metadata.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectError(f"找不到專案：{project_id}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError(f"專案資料無法解析：{project_id}") from exc
        return Project.from_dict(value)

    def document_count(self, project_id: str) -> int:
        self.get(project_id)
        source = self.root / project_id / "source"
        return sum(1 for path in source.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")

    def path_for(self, project_id: str) -> Path:
        """Return a registered project path without accepting arbitrary paths."""
        self.get(project_id)
        return self.root / project_id

    def update_status(self, project_id: str, status: str) -> Project:
        if status not in PROJECT_STATUSES:
            raise ProjectError(f"未知的專案狀態：{status}")
        project = self.get(project_id)
        value = asdict(project)
        value["status"] = status
        value["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata(self.root / project_id / "project.json", value)
        return Project.from_dict(value)

    def set_enabled(self, project_id: str, enabled: bool) -> Project:
        if not isinstance(enabled, bool):
            raise ProjectError("專案啟用狀態格式錯誤")
        project = self.get(project_id)
        value = asdict(project)
        value["enabled"] = enabled
        value["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_metadata(self.root / project_id / "project.json", value)
        return Project.from_dict(value)

    def delete(self, project_id: str) -> None:
        project_path = self.path_for(project_id)
        if project_path.is_symlink():
            raise ProjectError("拒絕刪除符號連結專案")
        if (project_path / "graphrag" / ".indexing.lock").exists():
            raise ProjectError("建圖工作執行中，無法刪除專案")
        shutil.rmtree(project_path)

    def table_rows(self) -> list[list[str | int]]:
        return [
            [
                project.project_id,
                project.display_name,
                self.document_count(project.project_id),
                project.status,
                project.enabled,
                project.updated_at,
            ]
            for project in self.list()
        ]

    @staticmethod
    def _validate(values: dict[str, str]) -> None:
        if not values["project_id"]:
            raise ProjectError("Project ID 為必填")
        if not PROJECT_ID_PATTERN.fullmatch(values["project_id"]):
            raise ProjectError("Project ID 只允許英數字、- 與 _")
        for field, label in (
            ("display_name", "顯示名稱"),
            ("vehicle_name", "車型"),
            ("manual_version", "手冊版本"),
        ):
            if not values[field]:
                raise ProjectError(f"{label}為必填")

    @staticmethod
    def _write_metadata(path: Path, value: dict[str, Any]) -> None:
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".project-", suffix=".json")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                json.dump(value, temporary, ensure_ascii=False, indent=2)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _remove_empty_project(project_path: Path) -> None:
        for directory in reversed(PROJECT_DIRECTORIES):
            path = project_path / directory
            if path.exists():
                path.rmdir()
        project_path.rmdir()
