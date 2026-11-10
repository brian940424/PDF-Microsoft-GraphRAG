"""Single-question GraphRAG query execution and persistence."""

from __future__ import annotations

import json
import hashlib
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

from .cancellation import JobStopped, ProcessStopWatcher, raise_if_stopped
from .connections import ConnectionSettings, configure_completion_model, configure_embedding_model
from .agentic_search import AgenticSearch
from .evidence import Evidence, EvidenceService
from .projects import ProjectError, ProjectStore
from .answer_prompts import ANSWER_PROMPT_VERSION, install_answer_prompts
from .strategy_routing import SearchOutcome, StrategyRouter, context_source_count, search_with_fallback


QUERY_METHODS = {"local", "global", "drift", "basic"}
AUTO_METHOD = "auto"
AGENTIC_METHOD = "agentic"

_MANUAL_QUERY_ALIASES = {
    "翻譯": ("懸浮翻譯", "翻譯助理", "即時翻譯"),
    "選擇語言": ("語言設定", "變更語言", "系統語言"),
    "未知號碼": ("封鎖未知號碼", "來電封鎖", "封鎖來電"),
    "指紋": ("指紋辨識", "註冊指紋", "指紋解鎖"),
    "解鎖碼": (
        "變更螢幕鎖定方式", "鎖定螢幕與 AOD", "安全鎖定設定", "預設螢幕鎖定方式",
        "連續輸入錯誤解鎖碼", "錯誤解鎖碼達到嘗試上限", "自動原廠重設", "自動原廠重設開關",
    ),
    "藍牙無法正常運作": ("重設 Wi-Fi 與藍牙設定", "已註冊的資訊可能會遺失"),
    "過熱": ("裝置使用時發生過熱情形", "裝置過熱時的處理步驟"),
    "懸浮翻譯": ("切換逐字或整句翻譯", "聆聽原文發音"),
    "螢幕錄影": ("S Pen 游標", "顯示游標圖示"),
    "遠端拍照": ("使用 S Pen 拍照", "變更拍攝模式", "切換相機"),
    "隨手便利貼": (
        "釘選至 Always On Display", "備忘錄尚未儲存", "儲存至 Samsung Notes",
        "將 S Pen 重新插入插槽",
    ),
    "具備超寬頻": (
        "設定 連接 藍牙 超寬頻 (UWB)", "使用此功能將增加電池用量",
    ),
}


def expand_manual_query(question: str) -> str:
    """Add a small deterministic alias set when a manual uses a related feature name."""
    additions: list[str] = []
    for term, aliases in _MANUAL_QUERY_ALIASES.items():
        if term in question:
            additions.extend(alias for alias in aliases if alias not in question and alias not in additions)
    if not additions:
        return question
    return f"{question}\nRelated manual terms: {'; '.join(additions)}"


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
    execution_metadata: dict[str, Any] = field(default_factory=dict)


def _query_worker_command() -> list[str]:
    return [sys.executable, "-m", "automotive_graphrag.graph_query_worker"]


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
        stop_event: threading.Event | None = None,
    ) -> QueryResult:
        """Answer one question; setting `stop_event` terminates the query worker and raises JobStopped."""
        raise_if_stopped(stop_event)
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
        if selected_model not in self.connection_settings.available_chat_models():
            raise ProjectError(f"不支援的 Chat 模型：{selected_model}")
        graph_root = project_path / "graphrag"
        if not (graph_root / "output").is_dir():
            raise ProjectError("找不到此專案的 GraphRAG 索引輸出")
        self.connection_settings.apply_to_environment(project_id)
        applied_effort = self.connection_settings.reasoning_effort_for(selected_model, reasoning_effort)
        self._configure_connection(
            graph_root / "settings.yaml",
            self.connection_settings.get_api_base_url(),
            selected_model,
            self.connection_settings.get_embedding_model(),
            applied_effort,
        )
        started = datetime.now(timezone.utc)
        started_clock = time.monotonic()
        requested_method = normalized_method
        routing_reason = None
        execution_metadata: dict[str, Any] = {
            "trace_schema_version": "1",
            "model": selected_model,
            "reasoning_effort": reasoning_effort,
            "response_type": response_type,
            "requested_method": requested_method,
            "prompt_version": ANSWER_PROMPT_VERSION,
            "model_call_count": None,
            "estimated_cost_usd": None,
            "cost_status": "unknown_usage_not_returned_by_query_adapter",
            "settings_sha256": hashlib.sha256((graph_root / "settings.yaml").read_bytes()).hexdigest(),
        }
        retrieval_query = expand_manual_query(prompt)
        if retrieval_query != prompt:
            execution_metadata["expanded_retrieval_query"] = retrieval_query
        def run_query(command: list[str]) -> subprocess.CompletedProcess[str] | QueryExecution:
            if self._custom_runner:
                raise_if_stopped(stop_event)
                return self.runner(command)
            return self._run(command, stop_event=stop_event)

        def query_command(search_method: str, search_question: str) -> list[str]:
            return [
                sys.executable, "-m", "graphrag", "query", "--root", str(graph_root),
                "--method", search_method, "--response-type", response_type, search_question,
            ]

        if normalized_method == AUTO_METHOD:
            api_base_url = self.connection_settings.get_api_base_url()
            api_key = self.connection_settings.get_api_key(project_id)

            def search(search_method: str) -> SearchOutcome:
                try:
                    execution = run_query(query_command(search_method, retrieval_query))
                except ProjectError as exc:
                    return SearchOutcome("", str(exc), QueryExecution(1, "", str(exc)))
                if execution.returncode != 0:
                    return SearchOutcome("", self._last_error(execution), execution)
                try:
                    section_titles = tuple(item.section_name for item in self.evidence_service.from_context(
                        project_id, getattr(execution, "context", {}) or {},
                    ))
                except ProjectError:
                    section_titles = ()
                return SearchOutcome(
                    (execution.stdout or "").strip(), None, execution,
                    context_source_count(getattr(execution, "context", None)),
                    section_titles,
                )

            fallback = search_with_fallback(
                self.router.route(prompt, selected_model, api_base_url, api_key, applied_effort),
                search,
                lambda answer: self.router.check_answer(
                    prompt, answer, selected_model, api_base_url, api_key, applied_effort,
                ),
                prompt,
            )
            normalized_method, routing_reason = fallback.method, fallback.reason
            execution_metadata["strategy_attempts"] = list(fallback.attempts)
            result = fallback.outcome.payload or QueryExecution(0, fallback.outcome.answer, "")
        elif normalized_method == AGENTIC_METHOD:
            def execute_search(search_method: str, search_question: str, timeout_seconds: float | None = None):
                search_command = query_command(search_method, search_question)
                if not self._custom_runner:
                    return self._run(search_command, timeout_seconds=timeout_seconds, stop_event=stop_event)
                raise_if_stopped(stop_event)
                return self.runner(search_command)

            decision = self.agentic_search.run(
                prompt,
                selected_model,
                self.connection_settings.get_api_base_url(),
                self.connection_settings.get_api_key(project_id),
                execute_search,
                applied_effort,
            )
            result = QueryExecution(decision.returncode, decision.answer, decision.error or "", decision.context)
            routing_reason = decision.stop_reason
            execution_metadata.update(decision.execution_metadata)
            execution_metadata["requested_method"] = requested_method
        else:
            result = run_query(query_command(normalized_method, retrieval_query))
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
            execution_metadata=execution_metadata,
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
        value.setdefault("execution_metadata", {})
        return QueryResult(**value)

    @staticmethod
    def _run(
        command: list[str],
        timeout_seconds: float | None = None,
        stop_event: threading.Event | None = None,
    ) -> subprocess.CompletedProcess[str] | QueryExecution:
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
            completed = QueryService._run_worker(environment, package_root, worker_payload, timeout_seconds, stop_event)
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
    def _run_worker(
        environment: dict[str, str],
        package_root: Path,
        payload: str,
        timeout_seconds: float | None,
        stop_event: threading.Event | None,
    ) -> subprocess.CompletedProcess[str]:
        command = _query_worker_command()
        if stop_event is None:
            return subprocess.run(
                command, cwd=package_root, env=environment, input=payload, text=True,
                capture_output=True, check=False, timeout=timeout_seconds,
            )
        raise_if_stopped(stop_event)
        process = subprocess.Popen(
            command,
            cwd=package_root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Its own process group, so a stop also ends any helper the worker started.
            start_new_session=True,
        )
        with ProcessStopWatcher(process, stop_event) as watcher:
            try:
                stdout, stderr = process.communicate(payload, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
        if watcher.stopped:
            raise JobStopped("查詢已由使用者停止")
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

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
        reasoning_effort: str | None = None,
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
            configure_embedding_model(model, embedding_model)
        install_answer_prompts(value, settings_path.parent)
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
