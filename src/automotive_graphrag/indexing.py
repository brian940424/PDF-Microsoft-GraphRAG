"""Microsoft GraphRAG workspace initialization and indexing orchestration."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

from .connections import ALLOWED_CHAT_MODELS, ConnectionSettings, configure_completion_model
from .projects import ProjectError, ProjectStore
from .source_metadata import SourceMetadataService


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
MetadataBuilder = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class IndexingResult:
    project_id: str
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    return_code: int
    last_message: str
    log_file: str


class IndexingService:
    def __init__(
        self,
        projects: ProjectStore,
        runner: CommandRunner | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        connection_settings: ConnectionSettings | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        self.projects = projects
        self.runner = runner or self._run
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.connection_settings = connection_settings or ConnectionSettings(projects.root)
        self.metadata_builder = metadata_builder or SourceMetadataService(projects).build
        self._active_log: Path | None = None
        self._log_callback: Callable[[str], None] | None = None

    def initialize(self, project_id: str) -> Path:
        return self._initialize(project_id, self.chat_model)

    def _initialize(self, project_id: str, chat_model_override: str | None) -> Path:
        self.connection_settings.apply_to_environment(project_id)
        chat_model = chat_model_override or self.chat_model or self.connection_settings.get_chat_model()
        embedding_model = self.embedding_model or self.connection_settings.get_embedding_model()
        project_path = self.projects.path_for(project_id)
        graph_root = project_path / "graphrag"
        settings = graph_root / "settings.yaml"
        if not settings.exists():
            result = self.runner(
                [
                    sys.executable,
                    "-m",
                    "graphrag",
                    "init",
                    "--root",
                    str(graph_root),
                    "--model",
                    chat_model,
                    "--embedding",
                    embedding_model,
                ]
            )
            if result.returncode:
                raise ProjectError(self._failure_message("GraphRAG 初始化失敗", result))
        if not settings.is_file():
            raise ProjectError("GraphRAG 初始化後未產生 settings.yaml")
        self._configure_jsonl_input(settings)
        self._configure_api_base(settings, self.connection_settings.get_api_base_url())
        self._configure_models(settings, chat_model, embedding_model)
        self._sync_input(project_path, graph_root)
        return settings

    def build(
        self,
        project_id: str,
        log_callback: Callable[[str], None] | None = None,
        chat_model: str | None = None,
    ) -> IndexingResult:
        project = self.projects.get(project_id)
        if project.status not in {"READY", "STALE", "FAILED", "INDEXED"}:
            raise ProjectError(f"專案狀態 {project.status} 不允許建圖")
        if chat_model is not None and chat_model not in ALLOWED_CHAT_MODELS:
            raise ProjectError(f"不支援的建圖模型：{chat_model}")
        project_path = self.projects.path_for(project_id)
        graph_root = project_path / "graphrag"
        lock_path = graph_root / ".indexing.lock"
        lock_descriptor = self._acquire_lock(lock_path)
        started = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        log_path = graph_root / "indexing.log"
        output = graph_root / "output"
        backup = graph_root / ".last-successful-output"
        previous_index = output.exists() and project.status in {"STALE", "FAILED", "INDEXED"}
        try:
            self._active_log = log_path
            self._active_log.write_text("", encoding="utf-8")
            self._log_callback = log_callback
            self._initialize(project_id, chat_model)
            self.projects.update_status(project_id, "INDEXING")
            if backup.exists():
                shutil.rmtree(backup)
            if previous_index:
                output.replace(backup)
            result = self.runner([sys.executable, "-m", "graphrag", "index", "--root", str(graph_root), "--verbose"])
            if self.runner != self._run:
                for line in ((result.stdout or "") + (result.stderr or "")).splitlines(keepends=True):
                    self._write_log(line)
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
            if not output.is_dir():
                raise ProjectError("GraphRAG 建圖完成但找不到 output 目錄")
            self.metadata_builder(project_id)
            if backup.exists():
                shutil.rmtree(backup)
            status = "INDEXED"
            self.projects.update_status(project_id, status)
            last_message = self._last_message(result) or "GraphRAG index completed"
            return_code = result.returncode
        except Exception as exc:
            if output.exists():
                shutil.rmtree(output)
            if backup.exists():
                backup.replace(output)
            self.projects.update_status(project_id, "FAILED")
            status = "FAILED"
            return_code = getattr(exc, "returncode", 1)
            if isinstance(exc, subprocess.CalledProcessError):
                failed_result = subprocess.CompletedProcess(
                    exc.cmd,
                    exc.returncode,
                    exc.output,
                    exc.stderr,
                )
                last_message = self._last_message(failed_result) or str(exc)
            else:
                last_message = str(exc)
        finally:
            self._active_log = None
            self._log_callback = None
            os.close(lock_descriptor)
            lock_path.unlink(missing_ok=True)

        completed = datetime.now(timezone.utc)
        indexing_result = IndexingResult(
            project_id=project_id,
            status=status,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=round(time.monotonic() - started_clock, 3),
            return_code=return_code,
            last_message=last_message,
            log_file=log_path.name,
        )
        self._write_result(graph_root / "last_index_run.json", indexing_result)
        return indexing_result

    def last_result(self, project_id: str) -> IndexingResult | None:
        path = self.projects.path_for(project_id) / "graphrag" / "last_index_run.json"
        try:
            return IndexingResult(**json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            self._write_log(line)
        return_code = process.wait()
        return subprocess.CompletedProcess(command, return_code, "".join(output), "")

    def _write_log(self, text: str) -> None:
        if self._active_log is not None and text:
            with self._active_log.open("a", encoding="utf-8") as log:
                log.write(text)
                log.flush()
        if self._log_callback is not None and text:
            self._log_callback(text)

    @staticmethod
    def _configure_jsonl_input(settings_path: Path) -> None:
        value = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        input_config = value.setdefault("input", {})
        storage = input_config.setdefault("storage", {})
        storage.update({"type": "file", "base_dir": "input"})
        input_config.update(
            {
                "type": "jsonl",
                # GraphRAG expands settings with string.Template before YAML parsing.
                "file_pattern": r".*\.jsonl$$",
                "id_column": "id",
                "title_column": "title",
                "text_column": "text",
            }
        )
        IndexingService._atomic_text(settings_path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))

    @staticmethod
    def _sync_input(project_path: Path, graph_root: Path) -> None:
        source = project_path / "processed" / "input.jsonl"
        if not source.is_file():
            raise ProjectError("找不到前處理輸出 input.jsonl")
        input_directory = graph_root / "input"
        input_directory.mkdir(exist_ok=True)
        destination = input_directory / "input.jsonl"
        handle, temporary_name = tempfile.mkstemp(dir=input_directory, prefix=".input-", suffix=".jsonl")
        os.close(handle)
        try:
            shutil.copy2(source, temporary_name)
            os.replace(temporary_name, destination)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _configure_api_base(settings_path: Path, api_base_url: str) -> None:
        value = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        for section in ("completion_models", "embedding_models"):
            for model in value.get(section, {}).values():
                model["api_base"] = api_base_url
        IndexingService._atomic_text(settings_path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))

    @staticmethod
    def _configure_models(settings_path: Path, chat_model: str, embedding_model: str) -> None:
        value = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        for model in value.get("completion_models", {}).values():
            configure_completion_model(model, chat_model)
        for model in value.get("embedding_models", {}).values():
            model["model"] = embedding_model
        IndexingService._atomic_text(settings_path, yaml.safe_dump(value, allow_unicode=True, sort_keys=False))

    @staticmethod
    def _acquire_lock(path: Path) -> int:
        try:
            return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ProjectError("此專案已有建圖工作正在執行") from exc

    @staticmethod
    def _failure_message(prefix: str, result: subprocess.CompletedProcess[str]) -> str:
        return f"{prefix}：{IndexingService._last_message(result) or f'exit {result.returncode}'}"

    @staticmethod
    def _last_message(result: subprocess.CompletedProcess[str]) -> str:
        lines = ((result.stderr or "") + "\n" + (result.stdout or "")).strip().splitlines()
        return lines[-1] if lines else ""

    @staticmethod
    def _write_result(path: Path, result: IndexingResult) -> None:
        IndexingService._atomic_text(path, json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
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
