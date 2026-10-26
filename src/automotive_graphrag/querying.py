"""Single-question GraphRAG query execution and persistence."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from .connections import ALLOWED_CHAT_MODELS, ConnectionSettings, configure_completion_model
from .agentic_search import AgenticSearch
from .evidence import Evidence, EvidenceService
from .projects import ProjectError, ProjectStore
from .strategy_routing import StrategyRouter


QUERY_METHODS = {"local", "global", "drift", "basic"}
AUTO_METHOD = "auto"
AGENTIC_METHOD = "agentic"


@dataclass(frozen=True, slots=True)
class QueryExecution:
    returncode: int
    stdout: str
    stderr: str
    context: dict[str, Any] = field(default_factory=dict)


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str] | QueryExecution]


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
    evidence: tuple[Evidence, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    response_type: str = "Single Paragraph"
    requested_method: str | None = None
    routing_reason: str | None = None
    source: str | None = None


class QueryService:
    def __init__(
        self,
        projects: ProjectStore,
        runner: CommandRunner | None = None,
        connection_settings: ConnectionSettings | None = None,
        evidence_service: EvidenceService | None = None,
        router: StrategyRouter | None = None,
        agentic_search: AgenticSearch | None = None,
    ) -> None:
        self.projects = projects
        self.runner = runner or self._run
        self._custom_runner = runner is not None
        self.connection_settings = connection_settings or ConnectionSettings(projects.root)
        self.evidence_service = evidence_service or EvidenceService(projects)
        self.router = router or StrategyRouter()
        self.agentic_search = agentic_search or AgenticSearch()
        self._history_lock = threading.Lock()

    def ask(
        self,
        project_id: str,
        question: str,
        method: str = "local",
        chat_model: str | None = None,
        response_type: str = "Single Paragraph",
        source: str | None = None,
        reasoning_effort: str = "medium",
    ) -> QueryResult:
        project = self.projects.get(project_id)
        project_path = self.projects.path_for(project_id)
        prompt = question.strip()
        normalized_method = method.strip().lower()
        if project.status != "INDEXED":
            raise ProjectError(f"專案狀態 {project.status} 尚未完成建圖，無法查詢")
        if not prompt:
            raise ProjectError("問題為必填")
        if normalized_method not in QUERY_METHODS | {AUTO_METHOD, AGENTIC_METHOD}:
            raise ProjectError(f"不支援的查詢方法：{method}")
        selected_model = chat_model or self.connection_settings.get_chat_model()
        if selected_model not in ALLOWED_CHAT_MODELS:
            raise ProjectError(f"不支援的 Chat 模型：{selected_model}")
        graph_root = project_path / "graphrag"
        if not (graph_root / "output").is_dir():
            raise ProjectError("找不到此專案的 GraphRAG 索引輸出")
        self.connection_settings.apply_to_environment(project_id)
        self._configure_connection(
            graph_root / "settings.yaml",
            self.connection_settings.get_api_base_url(),
            selected_model,
            self.connection_settings.get_embedding_model(),
            reasoning_effort,
        )
        started = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        requested_method = normalized_method
        routing_reason = None
        if normalized_method == AUTO_METHOD:
            decision = self.router.route(
                prompt,
                selected_model,
                self.connection_settings.get_api_base_url(),
                self.connection_settings.get_api_key(project_id),
                reasoning_effort,
            )
            normalized_method, routing_reason = decision.method, decision.reason
        command = [
            sys.executable,
            "-m",
            "graphrag",
            "query",
            "--root",
            str(graph_root),
            "--method",
            normalized_method,
            "--response-type",
            response_type,
            prompt,
        ]
        if normalized_method == AGENTIC_METHOD:
            def execute_search(search_method: str, search_question: str, timeout_seconds: float | None = None):
                search_command = [
                    sys.executable, "-m", "graphrag", "query", "--root", str(graph_root),
                    "--method", search_method, "--response-type", response_type, search_question,
                ]
                if not self._custom_runner:
                    return self._run(search_command, timeout_seconds=timeout_seconds)
                return self.runner(search_command)

            decision = self.agentic_search.run(
                prompt,
                selected_model,
                self.connection_settings.get_api_base_url(),
                self.connection_settings.get_api_key(project_id),
                execute_search,
                reasoning_effort,
            )
            result = QueryExecution(decision.returncode, decision.answer, decision.error or "", decision.context)
            routing_reason = decision.stop_reason
        else:
            result = self.runner(command)
        completed = datetime.now(timezone.utc)
        answer = (result.stdout or "").strip() if result.returncode == 0 else ""
        error = None if result.returncode == 0 else self._last_error(result)
        context = result.context if isinstance(result, QueryExecution) else {}
        evidence: tuple[Evidence, ...] = ()
        if context:
            try:
                evidence = tuple(self.evidence_service.from_context(project_id, context))
            except (ProjectError, OSError, ValueError) as exc:
                context = dict(context)
                context["_evidence_warning"] = str(exc)
        if requested_method == AGENTIC_METHOD and answer:
            answer = self._align_agentic_citations(answer, context, evidence)
        if result.returncode == 0 and not answer:
            error = "GraphRAG 未回傳回答"
        query_result = QueryResult(
            query_id=uuid.uuid4().hex,
            project_id=project_id,
            question=prompt,
            method=AGENTIC_METHOD if requested_method == AGENTIC_METHOD else normalized_method,
            status="COMPLETED" if error is None else "FAILED",
            answer=answer,
            error=error,
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=round(time.monotonic() - started_clock, 3),
            evidence=evidence,
            context=context,
            response_type=response_type,
            requested_method=requested_method,
            routing_reason=routing_reason,
            source=source,
        )
        try:
            self._append_record(project_path / "runs" / "queries.jsonl", query_result)
        except FileNotFoundError:
            # A separate admin process may remove the project while a long query is running.
            pass
        return query_result

    def history(self, project_id: str) -> list[QueryResult]:
        path = self.projects.path_for(project_id) / "runs" / "queries.jsonl"
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        try:
            return [self._result_from_dict(json.loads(line)) for line in lines if line.strip()]
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProjectError("問答紀錄格式錯誤") from exc

    @staticmethod
    def context_summary(context: dict[str, Any]) -> dict[str, Any]:
        """Return a bounded UI-safe summary instead of the potentially huge raw context."""
        sections: dict[str, dict[str, Any]] = {}
        for key, value in context.items():
            sections[str(key)] = {
                "type": type(value).__name__,
                "items": len(value) if isinstance(value, (list, dict, tuple)) else None,
            }
        return {
            "available": bool(context),
            "sections": sections,
            "message": "完整 Query Context 保留於後端查詢紀錄；前端僅顯示摘要以避免大型回應。",
        }

    @staticmethod
    def _align_agentic_citations(answer: str, context: dict[str, Any], evidence: tuple[Evidence, ...]) -> str:
        """Map planner-visible citations to the Evidence IDs actually shown by the UI."""
        source_by_citation: dict[str, str] = {}
        for step in context.get("agentic_steps", []):
            if not isinstance(step, dict):
                continue
            for item in step.get("evidence", []):
                if isinstance(item, dict) and item.get("citation_id") and item.get("source_id"):
                    source_by_citation[str(item["citation_id"])] = str(item["source_id"])
        visible_id_by_source = {item.context_id: item.evidence_id for item in evidence if item.context_id}

        def replace(match: re.Match[str]) -> str:
            citation = match.group(1)
            source_id = source_by_citation.get(citation)
            visible_id = visible_id_by_source.get(source_id or "")
            return f"[{visible_id}]" if visible_id else "[來源未映射]"

        return re.sub(r"\[(E\d+)\]", replace, answer)

    @staticmethod
    def _result_from_dict(value: dict[str, Any]) -> QueryResult:
        value["evidence"] = tuple(Evidence(**item) for item in value.get("evidence", []))
        value.setdefault("context", {})
        return QueryResult(**value)

    @staticmethod
    def _run(command: list[str], timeout_seconds: float | None = None) -> subprocess.CompletedProcess[str] | QueryExecution:
        method = command[command.index("--method") + 1]
        root = Path(command[command.index("--root") + 1])
        question = command[-1]
        model = ""
        settings_path = root / "settings.yaml"
        try:
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
            completion_models = settings.get("completion_models", {})
            model = str(next(iter(completion_models.values())).get("model", ""))
        except (OSError, StopIteration, AttributeError, TypeError, yaml.YAMLError):
            pass
        response_type = (
            command[command.index("--response-type") + 1]
            if "--response-type" in command
            else "Single Paragraph"
        )
        package_root = Path(__file__).resolve().parents[2]
        environment = os.environ.copy()
        source_path = str(package_root / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (source_path, environment.get("PYTHONPATH", "")) if part
        )
        worker_payload = json.dumps(
            {
                "root_dir": str(root),
                "question": question,
                "method": method,
                "model": model,
                "response_type": response_type,
            },
            ensure_ascii=False,
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "automotive_graphrag.graph_query_worker"],
                cwd=package_root,
                env=environment,
                input=worker_payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            response = json.loads(completed.stdout)
            if not isinstance(response, dict):
                raise ValueError("工作程序結果不是 JSON 物件")
            if completed.returncode or "error" in response:
                error = str(response.get("error") or completed.stderr or "GraphRAG 查詢失敗")
                return QueryExecution(completed.returncode or 1, "", error)
            context = response.get("context", {})
            return QueryExecution(
                0,
                str(response.get("answer", "")),
                completed.stderr or "",
                context if isinstance(context, dict) else {},
            )
        except subprocess.TimeoutExpired:
            return QueryExecution(124, "", "Agentic GraphRAG 搜尋超過單題時間上限")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            detail = (completed.stderr or completed.stdout).strip().splitlines() if "completed" in locals() else []
            error = detail[-1] if detail else str(exc)
            return QueryExecution(1, "", error)

    @staticmethod
    def _last_error(result: subprocess.CompletedProcess[str] | QueryExecution) -> str:
        lines = ((result.stderr or "") + "\n" + (result.stdout or "")).strip().splitlines()
        return lines[-1] if lines else f"GraphRAG 查詢失敗（exit {result.returncode}）"

    @staticmethod
    def _serialize_context(context: Any) -> dict[str, Any]:
        if not isinstance(context, dict):
            return {"raw": str(context)}
        def serialize(value: Any) -> Any:
            if isinstance(value, pd.DataFrame):
                return json.loads(value.to_json(orient="records", force_ascii=False))
            if isinstance(value, dict):
                return {str(key): serialize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serialize(item) for item in value]
            try:
                json.dumps(value)
                return value
            except (TypeError, ValueError):
                return str(value)

        return serialize(context)

    @staticmethod
    def _configure_connection(
        settings_path: Path,
        api_base_url: str,
        chat_model: str,
        embedding_model: str,
        reasoning_effort: str = "medium",
    ) -> None:
        try:
            value = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError as exc:
            raise ProjectError("找不到此專案的 GraphRAG settings.yaml") from exc
        for section in ("completion_models", "embedding_models"):
            for model in value.get(section, {}).values():
                model["api_base"] = api_base_url
        for model in value.get("completion_models", {}).values():
            configure_completion_model(model, chat_model, reasoning_effort)
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

    def _append_record(self, path: Path, result: QueryResult) -> None:
        line = json.dumps(asdict(result), ensure_ascii=False) + "\n"
        with self._history_lock, path.open("a", encoding="utf-8") as output:
            output.write(line)
            output.flush()
            os.fsync(output.fileno())
