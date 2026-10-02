"""Single-question GraphRAG query execution and persistence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from .connections import ConnectionSettings
from .projects import ProjectError, ProjectStore


QUERY_METHODS = {"local", "global", "drift", "basic"}
CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class QueryResult:
    query_id: str
    project_id: str
    question: str
    method: str
    status: str
    answer: str
    error: str | None
    started_at: str
    completed_at: str
    duration_seconds: float


class QueryService:
    def __init__(
        self,
        projects: ProjectStore,
        runner: CommandRunner | None = None,
        connection_settings: ConnectionSettings | None = None,
    ) -> None:
        self.projects = projects
        self.runner = runner or self._run
        self.connection_settings = connection_settings or ConnectionSettings(projects.root)

    def ask(self, project_id: str, question: str, method: str = "local") -> QueryResult:
        project = self.projects.get(project_id)
        prompt = question.strip()
        normalized_method = method.strip().lower()
        if project.status != "INDEXED":
            raise ProjectError(f"專案狀態 {project.status} 尚未完成建圖，無法查詢")
        if not prompt:
            raise ProjectError("問題為必填")
        if normalized_method not in QUERY_METHODS:
            raise ProjectError(f"不支援的查詢方法：{method}")

        graph_root = self.projects.path_for(project_id) / "graphrag"
        if not (graph_root / "output").is_dir():
            raise ProjectError("找不到此專案的 GraphRAG 索引輸出")
        self.connection_settings.apply_to_environment()
        self._configure_connection(
            graph_root / "settings.yaml",
            self.connection_settings.get_api_base_url(),
            self.connection_settings.get_chat_model(),
            self.connection_settings.get_embedding_model(),
        )
        started = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        result = self.runner(
            [
                sys.executable,
                "-m",
                "graphrag",
                "query",
                "--root",
                str(graph_root),
                "--method",
                normalized_method,
                prompt,
            ]
        )
        completed = datetime.now(timezone.utc)
        answer = (result.stdout or "").strip() if result.returncode == 0 else ""
        error = None if result.returncode == 0 else self._last_error(result)
        if result.returncode == 0 and not answer:
            error = "GraphRAG 未回傳回答"
        query_result = QueryResult(
            query_id=uuid.uuid4().hex,
            project_id=project_id,
            question=prompt,
            method=normalized_method,
            status="COMPLETED" if error is None else "FAILED",
            answer=answer,
            error=error,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=round(time.monotonic() - started_clock, 3),
        )
        self._append_record(self.projects.path_for(project_id) / "runs" / "queries.jsonl", query_result)
        return query_result

    def history(self, project_id: str) -> list[QueryResult]:
        path = self.projects.path_for(project_id) / "runs" / "queries.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        try:
            return [QueryResult(**json.loads(line)) for line in lines if line.strip()]
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProjectError("問答紀錄格式錯誤") from exc

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, text=True, capture_output=True, check=False)

    @staticmethod
    def _last_error(result: subprocess.CompletedProcess[str]) -> str:
        lines = ((result.stderr or "") + "\n" + (result.stdout or "")).strip().splitlines()
        return lines[-1] if lines else f"GraphRAG 查詢失敗（exit {result.returncode}）"

    @staticmethod
    def _configure_connection(
        settings_path: Path,
        api_base_url: str,
        chat_model: str,
        embedding_model: str,
    ) -> None:
        try:
            value = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ProjectError("找不到此專案的 GraphRAG settings.yaml") from exc
        for section in ("completion_models", "embedding_models"):
            for model in value.get(section, {}).values():
                model["api_base"] = api_base_url
        for model in value.get("completion_models", {}).values():
            model["model"] = chat_model
        for model in value.get("embedding_models", {}).values():
            model["model"] = embedding_model
        handle, temporary_name = tempfile.mkstemp(dir=settings_path.parent, prefix=".settings-", suffix=".yaml")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                yaml.safe_dump(value, temporary, allow_unicode=True, sort_keys=False)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, settings_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _append_record(path: Path, result: QueryResult) -> None:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        content = existing + json.dumps(asdict(result), ensure_ascii=False) + "\n"
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".queries-", suffix=".jsonl")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
