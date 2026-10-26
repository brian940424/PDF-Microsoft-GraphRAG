"""Persistent, cancellable comparisons of Microsoft GraphRAG query strategies."""

from __future__ import annotations

import json
import inspect
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import yaml

from .agentic_search import AgenticSearch
from .automatic_evaluation import AutomaticEvaluationService
from .connections import (
    ALLOWED_CHAT_MODELS,
    DEFAULT_REASONING_EFFORT,
    GPT6_LUNA_MODEL,
    REASONING_EFFORT_OPTIONS,
    ConnectionSettings,
    configure_completion_model,
)
from .evidence import Evidence, EvidenceService
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, QuestionSetService
from .retrieval_evaluation import compute_source_metrics
from .querying import QueryExecution
from .strategy_routing import StrategyRouter


EXPERIMENT_METHODS = ("local", "global", "drift", "basic")
EXPERIMENT_AUTO_METHOD = "auto"
EXPERIMENT_AGENTIC_METHOD = "agentic"
EXPERIMENT_GROUP_METHODS = (*EXPERIMENT_METHODS, EXPERIMENT_AGENTIC_METHOD, EXPERIMENT_AUTO_METHOD)
EVIDENCE_METHODS = frozenset({"local", "basic", "drift"})
DEFAULT_EXPERIMENT_RESPONSE_TYPE = "Single Paragraph"
EXPERIMENT_RESPONSE_TYPE_OPTIONS = (
    ("單句", "Single Sentence"),
    ("3–7 點條列", "List of 3-7 Points"),
    ("單段回答", "Single Paragraph"),
    ("多頁報告", "Multi-Page Report"),
    ("2–3 句精簡回答", "2-3 concise sentences"),
    ("精簡段落；流程題以編號列必要步驟", "A concise paragraph; use numbered steps only for procedures"),
    ("簡短回答，不要前言", "A short answer with no preamble"),
)
EXPERIMENT_RESPONSE_TYPES = tuple(value for _, value in EXPERIMENT_RESPONSE_TYPE_OPTIONS)
EXPERIMENT_REASONING_EFFORT_OPTIONS = (("低", "low"), ("中", "medium"), ("高", "high"))
EXPERIMENT_STRATEGY_PARAMETER_SPECS = {
    "local": (
        ("community_level", "Community 層級", "int", 2, 0, 10),
        ("text_unit_prop", "原文片段比例", "float", 0.5, 0, 1),
        ("community_prop", "社群報告比例", "float", 0.15, 0, 1),
        ("conversation_history_max_turns", "對話歷史輪數", "int", 5, 0, 50),
        ("top_k_entities", "相關實體數", "int", 10, 1, 100),
        ("top_k_relationships", "相關關係數", "int", 10, 1, 100),
        ("max_context_tokens", "最大上下文 Token", "int", 12000, 100, 100000),
    ),
    "global": (
        ("community_level", "Community 層級", "int", 2, 0, 10),
        ("dynamic_community_selection", "啟用動態社群選擇", "bool", False, None, None),
        ("max_context_tokens", "最大上下文 Token", "int", 12000, 100, 100000),
        ("data_max_tokens", "Map 階段資料 Token 上限", "int", 12000, 100, 100000),
        ("map_max_length", "Map 回答長度上限（字）", "int", 1000, 100, 10000),
        ("reduce_max_length", "Reduce 回答長度上限（字）", "int", 2000, 100, 10000),
        ("dynamic_search_threshold", "動態選擇相關門檻", "int", 1, 0, 100),
        ("dynamic_search_keep_parent", "保留父層社群", "bool", False, None, None),
        ("dynamic_search_num_repeats", "社群評估重複次數", "int", 1, 1, 10),
        ("dynamic_search_use_summary", "使用社群摘要", "bool", False, None, None),
        ("dynamic_search_max_level", "動態搜尋最高層級", "int", 2, 0, 10),
    ),
    "drift": (
        ("community_level", "Community 層級", "int", 2, 0, 10),
        ("n_depth", "探索深度（輪數）", "int", 3, 1, 10),
        ("drift_k_followups", "每輪 Follow-up 上限", "int", 20, 1, 100),
        ("primer_folds", "Primer 報告分組數", "int", 5, 1, 50),
        ("local_search_text_unit_prop", "原文片段比例", "float", 0.9, 0, 1),
        ("local_search_community_prop", "社群報告比例", "float", 0.1, 0, 1),
        ("local_search_top_k_mapped_entities", "局部搜尋實體數", "int", 10, 1, 100),
        ("local_search_top_k_relationships", "局部搜尋關係數", "int", 10, 1, 100),
        ("local_search_max_data_tokens", "每輪最大上下文 Token", "int", 12000, 100, 100000),
        ("local_search_temperature", "每輪回答 Temperature", "float", 0.0, 0, 2),
        ("local_search_top_p", "每輪回答 Top P", "float", 1.0, 0, 1),
        ("local_search_n", "每輪候選回答數", "int", 1, 1, 10),
    ),
    "basic": (
        ("k", "取回文字片段數（k）", "int", 10, 1, 100),
        ("max_context_tokens", "最大上下文 Token", "int", 12000, 100, 100000),
    ),
}
EXPERIMENT_STRATEGY_CONFIG_KEYS = {
    "local": {
        "text_unit_prop", "community_prop", "conversation_history_max_turns",
        "top_k_entities", "top_k_relationships", "max_context_tokens",
    },
    "global": {
        "max_context_tokens", "data_max_tokens", "map_max_length", "reduce_max_length",
        "dynamic_search_threshold", "dynamic_search_keep_parent", "dynamic_search_num_repeats",
        "dynamic_search_use_summary", "dynamic_search_max_level",
    },
    "drift": {
        "n_depth", "drift_k_followups", "primer_folds", "local_search_text_unit_prop",
        "local_search_community_prop", "local_search_top_k_mapped_entities",
        "local_search_top_k_relationships", "local_search_max_data_tokens", "local_search_n",
        "local_search_temperature", "local_search_top_p",
    },
    "basic": {"k", "max_context_tokens"},
}
UNSUPPORTED_RETRIEVAL_METRICS = "暫不可計算：Global 社群報告尚未可靠映射至原始文件"


def default_strategy_parameters() -> dict[str, dict[str, object]]:
    return {
        method: {name: default for name, _label, _kind, default, _minimum, _maximum in specs}
        for method, specs in EXPERIMENT_STRATEGY_PARAMETER_SPECS.items()
    }


@dataclass(frozen=True, slots=True)
class ExperimentGroup:
    group_id: str
    name: str
    answer_model: str
    method: str
    response_type: str = DEFAULT_EXPERIMENT_RESPONSE_TYPE
    answer_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    strategy_parameters: dict[str, dict[str, object]] = field(default_factory=default_strategy_parameters)


@dataclass(frozen=True, slots=True)
class ExperimentQuestionResult:
    group_id: str
    group_name: str
    answer_model: str
    judge_model: str
    method: str
    question_id: str
    question: str
    source_documents: tuple[str, ...]
    correct_answer: str
    actual_answer: str
    evaluation_result: int | str
    evaluation_reason: str
    answer_source_rank: int | None
    retrieval_metrics_status: str
    status: str
    error: str | None = None
    response_type: str = DEFAULT_EXPERIMENT_RESPONSE_TYPE
    retrieved_evidence: tuple[Evidence, ...] = ()
    dataset_id: str = ""
    dataset_name: str = ""
    answer_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    routed_method: str = ""
    routing_reason: str = ""
    execution_trace: dict[str, object] = field(default_factory=dict)
    duration_seconds: float | None = None
    answer_point_coverage_percent: float | None = None
    unsupported_claim_count: int | None = None
    refusal_correctness: str | None = None

    @property
    def effective_method(self) -> str:
        """The strategy GraphRAG actually ran; differs from ``method`` only for auto groups."""
        return self.routed_method or self.method


@dataclass(frozen=True, slots=True)
class ExperimentGroupSummary:
    group_id: str
    group_name: str
    answer_model: str
    judge_model: str
    method: str
    question_count: int
    completed_count: int
    correct_count: int
    partial_correct_count: int
    incorrect_count: int
    accuracy: float | None
    score_accuracy: float | None
    evidence_hit_rate: float | None
    evidence_precision: float | None
    document_hit_rate: float | None
    no_evidence_question_count: int
    retrieval_metrics_status: str
    response_type: str = DEFAULT_EXPERIMENT_RESPONSE_TYPE
    answer_reasoning_effort: str = DEFAULT_REASONING_EFFORT
    strategy_parameters: dict[str, object] = field(default_factory=dict)
    average_latency_seconds: float | None = None
    p95_latency_seconds: float | None = None
    average_search_steps: float | None = None
    average_agent_model_calls: float | None = None
    estimated_cost_usd: float | None = None
    cost_status: str = "此適配器未提供 Token 用量，成本未知"
    answer_point_coverage_percent: float | None = None
    unsupported_answer_rate: float | None = None
    average_unsupported_claim_count: float | None = None
    refusal_accuracy: float | None = None
    quality_reviewed_count: int = 0


@dataclass(frozen=True, slots=True)
class RetrievalExperimentRun:
    project_id: str
    project_name: str
    question_set_id: str
    question_set_name: str
    status: str
    max_concurrency: int
    question_count: int
    started_at: str
    completed_at: str | None
    groups: tuple[ExperimentGroupSummary, ...]
    results: tuple[ExperimentQuestionResult, ...]
    question_set_ids: tuple[str, ...] = ()
    question_set_names: tuple[str, ...] = ()


QueryFunction = Callable[..., tuple[str, dict[str, object]]]
UpdateFunction = Callable[[RetrievalExperimentRun], None]


class RetrievalExperimentService:
    def __init__(
        self,
        projects: ProjectStore,
        question_sets: QuestionSetService,
        connections: ConnectionSettings,
        judging: AutomaticEvaluationService,
        query_function: QueryFunction | None = None,
        router: StrategyRouter | None = None,
        agentic_search: AgenticSearch | None = None,
    ) -> None:
        self.projects = projects
        self.question_sets = question_sets
        self.connections = connections
        self.judging = judging
        self.query_function = query_function or self._graphrag_query
        self.router = router or StrategyRouter()
        self.agentic_search = agentic_search or AgenticSearch()
        self._lock = threading.RLock()
        self._stop_events: dict[str, threading.Event] = {}

    def load(self, project_id: str) -> dict[str, object]:
        self.projects.get(project_id)
        try:
            value = json.loads(self._path(project_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"groups": [], "question_set_id": "", "question_set_ids": [], "max_concurrency": 5,
                    "judge_model": self.connections.get_chat_model(), "evaluation_mode": "strict", "run": None}
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError("檢索實驗保存資料無法讀取") from exc
        value.setdefault("groups", [])
        value.setdefault("question_set_id", "")
        value["question_set_ids"] = self._normalize_question_set_ids(
            value.get("question_set_ids") or value.get("question_set_id")
        )
        value.setdefault("max_concurrency", 5)
        value.setdefault("evaluation_mode", "strict")
        if value["evaluation_mode"] not in {"strict", "lenient"}:
            value["evaluation_mode"] = "strict"
        value.setdefault("run", None)
        legacy_groups = value.get("groups", [])
        legacy_judge = next((
            item.get("judge_model") for item in legacy_groups
            if isinstance(item, dict) and item.get("judge_model")
        ), None)
        value.setdefault("judge_model", legacy_judge or self.connections.get_chat_model())
        value["groups"] = [
            {key: field for key, field in item.items() if key != "judge_model"}
            for item in legacy_groups if isinstance(item, dict)
        ]
        migrated = False
        for group in value["groups"]:
            saved = group.get("strategy_parameters")
            if not isinstance(saved, dict):
                saved = {}
                migrated = True
            defaults = default_strategy_parameters()
            normalized = {}
            for method, params in defaults.items():
                current_params = saved.get(method)
                if not isinstance(current_params, dict):
                    current_params = {}
                    migrated = migrated or method not in saved
                normalized[method] = {**params, **current_params}
            if normalized != saved:
                migrated = True
            group["strategy_parameters"] = normalized
        run = value.get("run")
        if isinstance(run, dict):
            for item in run.get("results", []):
                old = item.get("evaluation_result")
                grade = self._parse_grade(old)
                if grade is not None and old != grade:
                    item["evaluation_result"] = grade
                    migrated = True
            for group in run.get("groups", []):
                group_results = [item for item in run.get("results", []) if item.get("group_id") == group.get("group_id")]
                judged = [item for item in group_results if self._parse_grade(item.get("evaluation_result")) is not None]
                full = sum(self._parse_grade(item.get("evaluation_result")) == 2 for item in judged)
                partial = sum(self._parse_grade(item.get("evaluation_result")) == 1 for item in judged)
                incorrect = sum(self._parse_grade(item.get("evaluation_result")) == 0 for item in judged)
                total = int(group.get("question_count", len(group_results)))
                updated = {
                    "correct_count": full, "partial_correct_count": partial,
                    "incorrect_count": incorrect, "accuracy": full / total if total else None,
                    "score_accuracy": (2 * full + partial) / (2 * total) if total else None,
                }
                if any(group.get(key) != field for key, field in updated.items()):
                    group.update(updated)
                    migrated = True
        if migrated:
            self._write(project_id, value)
        return value

    def save_configuration(
        self,
        project_id: str,
        groups: Sequence[ExperimentGroup],
        question_set_id: str | Sequence[str],
        max_concurrency: int,
        judge_model: str | None = None,
    ) -> None:
        if not 1 <= max_concurrency <= 32:
            raise ProjectError("最大並行請求數必須介於 1 到 32")
        self._validate_groups(groups)
        current = self.load(project_id)
        current["groups"] = [asdict(group) for group in groups]
        if judge_model is not None:
            if judge_model not in ALLOWED_CHAT_MODELS:
                raise ProjectError("全域評測模型不支援")
            current["judge_model"] = judge_model
        question_set_ids = self._normalize_question_set_ids(question_set_id)
        current["question_set_ids"] = question_set_ids
        current["question_set_id"] = question_set_ids[0] if question_set_ids else ""
        current["max_concurrency"] = max_concurrency
        self._write(project_id, current)

    def set_evaluation_mode(self, project_id: str, evaluation_mode: str) -> dict[str, object]:
        if evaluation_mode not in {"strict", "lenient"}:
            raise ProjectError("不支援的評分方式")
        current = self.load(project_id)
        current["evaluation_mode"] = evaluation_mode
        self._write(project_id, current)
        return current

    def import_question_set(self, project_id: str, filepath: str | Path) -> object:
        return self.import_question_sets(project_id, [filepath])[0]

    def import_question_sets(self, project_id: str, filepaths: Sequence[str | Path]) -> list[object]:
        imported, errors = self.import_question_sets_with_errors(project_id, filepaths)
        if errors:
            raise ProjectError("題目集未取代，舊資料已保留：\n- " + "\n- ".join(errors))
        return imported

    def import_question_sets_with_errors(
        self, project_id: str, filepaths: Sequence[str | Path]
    ) -> tuple[list[object], list[str]]:
        if not filepaths:
            raise ProjectError("請至少選擇一份 JSON 題目集")
        current = self.load(project_id)
        old_ids = self._normalize_question_set_ids(
            current.get("question_set_ids") or current.get("question_set_id")
        )
        imported = []
        errors = []
        for filepath in filepaths:
            try:
                imported.append(self.question_sets.import_file(project_id, filepath))
            except (ProjectError, OSError, ValueError) as exc:
                errors.append(f"{Path(filepath).name}：{exc}")
        if errors or not imported:
            # Treat multi-file replacement atomically: a malformed file must not
            # replace or partially erase the currently loaded datasets.
            for item in imported:
                try:
                    self.question_sets.delete(project_id, item.question_set_id)
                except (ProjectError, OSError):
                    pass
            return [], errors or ["沒有可匯入的題目集"]

        ids = [item.question_set_id for item in imported]
        current["question_set_ids"] = ids
        current["question_set_id"] = ids[0]
        current["run"] = None
        self._write(project_id, current)
        for old_id in old_ids:
            if old_id not in ids:
                try:
                    self.question_sets.delete(project_id, old_id)
                except ProjectError:
                    # A stale ID may already have been removed; the replacement
                    # state is authoritative and remains usable.
                    pass
        return imported, errors

    def remove_question_set(self, project_id: str, question_set_id: str) -> dict[str, object]:
        current = self.load(project_id)
        ids = [item for item in self._normalize_question_set_ids(current.get("question_set_ids"))
               if item != question_set_id]
        if len(ids) == len(self._normalize_question_set_ids(current.get("question_set_ids"))):
            raise ProjectError(f"題目集不在檢索實驗清單中：{question_set_id}")
        current["question_set_ids"] = ids
        current["question_set_id"] = ids[0] if ids else ""
        current["run"] = None
        self._write(project_id, current)
        try:
            self.question_sets.delete(project_id, question_set_id)
        except ProjectError:
            # The active list is already updated; tolerate a stale missing file.
            pass
        return current

    @staticmethod
    def _normalize_question_set_ids(value: str | Sequence[str] | object) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        return []

    def add_group(self, project_id: str) -> dict[str, object]:
        current = self.load(project_id)
        raw_groups = current["groups"]
        used = {str(group.get("group_id", "")) for group in raw_groups if isinstance(group, dict)}
        index = max((int(group_id[1:]) for group_id in used
                     if group_id.startswith("G") and group_id[1:].isdigit()), default=0) + 1
        while f"G{index:02d}" in used:
            index += 1
        default_model = self.connections.get_chat_model()
        group = ExperimentGroup(f"G{index:02d}", f"實驗組{index}", default_model, "local")
        current["groups"] = [*raw_groups, asdict(group)]
        self._write(project_id, current)
        return current

    def update_group_fields(self, project_id: str, rows: object) -> dict[str, object]:
        """Merge non-empty cell edits; missing/blank rows never delete saved groups."""
        current = self.load(project_id)
        groups = [dict(group) for group in current["groups"] if isinstance(group, dict)]
        by_id = {str(group.get("group_id", "")): group for group in groups}
        if rows is None:
            return current
        for row in rows:
            if isinstance(row, (str, bytes)) or not hasattr(row, "__len__") or len(row) < 4:
                continue
            group = by_id.get(str(row[0] or ""))
            if group is None:
                continue
            for index, field_name in enumerate(("name", "answer_model"), start=1):
                raw_value = row[index]
                if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
                    continue
                value = str(raw_value).strip()
                if value:
                    if field_name == "answer_model" and value not in ALLOWED_CHAT_MODELS:
                        raise ProjectError(f"{group.get('group_id')} 的回答模型不支援")
                    group[field_name] = value
            if len(row) >= 4:
                method = str(row[3] or "").strip()
                if method in EXPERIMENT_GROUP_METHODS:
                    group["method"] = method
                elif method:
                    raise ProjectError(f"{group.get('group_id')} 的 GraphRAG 策略不支援")
        current["groups"] = groups
        self._write(project_id, current)
        return current

    def set_group_method(self, project_id: str, group_id: str, method: str) -> dict[str, object]:
        if method not in EXPERIMENT_GROUP_METHODS:
            raise ProjectError("不支援的 GraphRAG 檢索策略")
        current = self.load(project_id)
        matched = False
        for group in current["groups"]:
            if group.get("group_id") == group_id:
                group["method"] = method
                matched = True
        if matched:
            self._write(project_id, current)
        return current

    def set_group_strategy_parameter(
        self, project_id: str, group_id: str, method: str, name: str, value: object
    ) -> dict[str, object]:
        specs = {item[0]: item for item in EXPERIMENT_STRATEGY_PARAMETER_SPECS.get(method, ())}
        spec = specs.get(name)
        if spec is None:
            raise ProjectError("此策略沒有該設定參數")
        _name, label, kind, _default, minimum, maximum = spec
        if kind == "bool":
            if not isinstance(value, bool):
                raise ProjectError(f"{label}必須為開啟或關閉")
            normalized: object = value
        elif kind == "int":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or int(value) != float(value):
                raise ProjectError(f"{label}必須是整數")
            normalized = int(value)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ProjectError(f"{label}必須是有限數值")
            normalized = float(value)
        if minimum is not None and normalized < minimum or maximum is not None and normalized > maximum:
            raise ProjectError(f"{label}必須介於 {minimum} 與 {maximum} 之間")
        current = self.load(project_id)
        matched = False
        for group in current["groups"]:
            if group.get("group_id") != group_id:
                continue
            params = group.setdefault("strategy_parameters", default_strategy_parameters())
            params.setdefault(method, {})[name] = normalized
            matched = True
        if matched:
            self._write(project_id, current)
        return current

    def set_group_answer_model(self, project_id: str, group_id: str, answer_model: str) -> dict[str, object]:
        if answer_model not in ALLOWED_CHAT_MODELS:
            raise ProjectError("不支援的回答模型")
        current = self.load(project_id)
        matched = False
        for group in current["groups"]:
            if group.get("group_id") == group_id:
                group["answer_model"] = answer_model
                matched = True
        if matched:
            self._write(project_id, current)
        return current

    def set_group_response_type(self, project_id: str, group_id: str, response_type: str) -> dict[str, object]:
        if response_type not in EXPERIMENT_RESPONSE_TYPES:
            raise ProjectError("不支援的 GraphRAG 回答格式")
        current = self.load(project_id)
        matched = False
        for group in current["groups"]:
            if group.get("group_id") == group_id:
                group["response_type"] = response_type
                matched = True
        if matched:
            self._write(project_id, current)
        return current

    def set_group_reasoning_effort(
        self, project_id: str, group_id: str, reasoning_effort: str
    ) -> dict[str, object]:
        if reasoning_effort not in REASONING_EFFORT_OPTIONS:
            raise ProjectError("不支援的 Luna 推理深度")
        current = self.load(project_id)
        matched = False
        for group in current["groups"]:
            if group.get("group_id") == group_id:
                if group.get("answer_model") != GPT6_LUNA_MODEL:
                    raise ProjectError("只有回答模型為 gpt-6-luna 時可以設定推理深度")
                group["answer_reasoning_effort"] = reasoning_effort
                matched = True
        if matched:
            self._write(project_id, current)
        return current

    def remove_group(self, project_id: str, group_id: str) -> dict[str, object]:
        current = self.load(project_id)
        current["groups"] = [g for g in current["groups"] if g.get("group_id") != group_id]
        self._write(project_id, current)
        return current

    def run(
        self,
        project_id: str,
        question_set_id: str | Sequence[str],
        groups: Sequence[ExperimentGroup],
        max_concurrency: int = 5,
        update_callback: UpdateFunction | None = None,
        judge_model: str | None = None,
        evaluate: bool = True,
    ) -> RetrievalExperimentRun:
        if not groups:
            raise ProjectError("請先新增至少一個實驗組")
        if not 1 <= max_concurrency <= 32:
            raise ProjectError("最大並行請求數必須介於 1 到 32")
        self._validate_groups(groups)
        state = self.load(project_id)
        judge_model = judge_model or str(state.get("judge_model", self.connections.get_chat_model()))
        if judge_model not in ALLOWED_CHAT_MODELS:
            raise ProjectError("全域評測模型不支援")
        project = self.projects.get(project_id)
        question_set_ids = self._normalize_question_set_ids(question_set_id)
        if not question_set_ids:
            raise ProjectError("此專案尚未匯入題目集，請先匯入 JSON 題目集")
        question_sets = [self.question_sets.get(project_id, item) for item in question_set_ids]
        if project.status != "INDEXED":
            raise ProjectError("專案尚未完成 GraphRAG 建圖")
        if not any(question_set.questions for question_set in question_sets):
            raise ProjectError("匯入題目集沒有題目")
        missing_answers = [
            f"{question_set.name}/{item.question_id}"
            for question_set in question_sets for item in question_set.questions
            if not item.reference_answer.strip()
        ]
        if missing_answers:
            raise ProjectError("以下題目缺少正確答案，無法評測：" + ", ".join(missing_answers))
        question_count = sum(len(item.questions) for item in question_sets)
        stop_event = threading.Event()
        with self._lock:
            self._stop_events[project_id] = stop_event

        start = datetime.now(timezone.utc).isoformat()
        results: dict[tuple[str, str, str], ExperimentQuestionResult] = {}
        run_status = "running"
        initial_status = "generating_answers" if not evaluate else run_status
        self._persist_run(project_id, question_set_ids, groups, judge_model, max_concurrency, start, initial_status, results)
        work = [
            (group, question_set, question)
            for group in groups for question_set in question_sets for question in question_set.questions
        ]
        executor = ThreadPoolExecutor(max_workers=max_concurrency)
        work_iter = iter(work)
        futures: dict[Future[ExperimentQuestionResult], tuple[ExperimentGroup, object, BatchQuestion]] = {}

        def fill_available_workers() -> None:
            while not stop_event.is_set() and len(futures) < max_concurrency:
                try:
                    group, question_set, question = next(work_iter)
                except StopIteration:
                    return
                worker = self._run_case if evaluate else self._generate_case
                futures[executor.submit(
                    worker, project_id, group, question, judge_model,
                    question_set.question_set_id, question_set.name,
                )] = (group, question_set, question)

        fill_available_workers()
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    group, question_set, question = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # Preserve an individual failure instead of losing the run.
                        result = self._failed_result(
                            group, question, judge_model, str(exc), question_set.question_set_id, question_set.name
                        )
                    results[(group.group_id, question_set.question_set_id, question.question_id)] = result
                    run_status = "stopping" if stop_event.is_set() else (
                        "evaluating" if evaluate else "generating_answers"
                    )
                    run = self._persist_run(
                        project_id, question_set_ids, groups, judge_model, max_concurrency, start, run_status, results
                    )
                    if update_callback:
                        update_callback(run)
                fill_available_workers()
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self._persist_run(project_id, question_set_ids, groups, judge_model, max_concurrency, start, "stopped", results)
            raise
        finally:
            with self._lock:
                self._stop_events.pop(project_id, None)

        if stop_event.is_set():
            run_status = "stopped"
        elif evaluate:
            run_status = "completed"
        else:
            expected_count = len(groups) * question_count
            run_status = "answers_completed" if len(results) == expected_count and all(
                item.status == "answered" and item.actual_answer.strip()
                for item in results.values()
            ) else "answers_partial"
        completed_at = datetime.now(timezone.utc).isoformat()
        return self._persist_run(
            project_id, question_set_ids, groups, judge_model, max_concurrency, start, run_status, results, completed_at
        )

    def generate_answers(
        self,
        project_id: str,
        question_set_id: str | Sequence[str],
        groups: Sequence[ExperimentGroup],
        max_concurrency: int = 5,
        update_callback: UpdateFunction | None = None,
        judge_model: str | None = None,
    ) -> RetrievalExperimentRun:
        """Run retrieval and answer generation only; judging is a separate action."""
        return self.run(
            project_id,
            question_set_id,
            groups,
            max_concurrency,
            update_callback,
            judge_model,
            evaluate=False,
        )

    def evaluate_answers(
        self,
        project_id: str,
        question_set_id: str | Sequence[str],
        judge_model: str | None = None,
        max_concurrency: int | None = None,
        update_callback: UpdateFunction | None = None,
        evaluation_mode: str | None = None,
        judge_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> RetrievalExperimentRun:
        """Evaluate previously saved answers without repeating GraphRAG retrieval."""
        state = self.load(project_id)
        evaluation_mode = evaluation_mode or str(state.get("evaluation_mode", "strict"))
        if evaluation_mode not in {"strict", "lenient"}:
            raise ProjectError("不支援的評分方式")
        run_state = state.get("run")
        question_set_ids = self._normalize_question_set_ids(question_set_id)
        run_question_set_ids = self._normalize_question_set_ids(
            run_state.get("question_set_ids", run_state.get("question_set_id", ""))
            if isinstance(run_state, dict) else ""
        )
        if not isinstance(run_state, dict) or run_question_set_ids != question_set_ids:
            raise ProjectError("請先執行「檢索並生成答案」")
        groups = [ExperimentGroup(**item) for item in state.get("groups", [])]
        if not groups:
            raise ProjectError("請先新增至少一個實驗組")
        judge_model = judge_model or str(state.get("judge_model", self.connections.get_chat_model()))
        if judge_model not in ALLOWED_CHAT_MODELS:
            raise ProjectError("全域評測模型不支援")
        concurrency = int(max_concurrency or state.get("max_concurrency", 5))
        if not 1 <= concurrency <= 32:
            raise ProjectError("最大並行請求數必須介於 1 到 32")
        state["evaluation_mode"] = evaluation_mode
        self._write(project_id, state)
        question_sets = [self.question_sets.get(project_id, item) for item in question_set_ids]
        dataset_by_id = {item.question_set_id: item for item in question_sets}
        results: dict[tuple[str, str, str], ExperimentQuestionResult] = {}
        for item in run_state.get("results", []):
            result = self._result_from_dict(item)
            dataset_id = result.dataset_id or question_set_ids[0]
            results[(result.group_id, dataset_id, result.question_id)] = result
        work: list[tuple[ExperimentGroup, object, BatchQuestion, ExperimentQuestionResult]] = []
        for group in groups:
            for question_set in question_sets:
                for question in question_set.questions:
                    result = results.get((group.group_id, question_set.question_set_id, question.question_id))
                    if result is None or not result.actual_answer.strip():
                        raise ProjectError(
                            f"{group.name}／{question_set.name} 的題目 {question.question_id} 尚未完成回答；請先生成所有答案"
                        )
                    if (
                        result.answer_model != group.answer_model
                        or result.method != group.method
                        or result.response_type != group.response_type
                        or result.answer_reasoning_effort != group.answer_reasoning_effort
                    ):
                        raise ProjectError(f"{group.name} 的模型、推理深度、檢索策略或回答格式已變更；請重新生成答案")
                    if result.question != question.question or result.correct_answer != question.reference_answer:
                        raise ProjectError(f"{question_set.name} 的題目 {question.question_id} 已變更；請重新生成答案")
                    work.append((group, question_set, question, result))

        stop_event = threading.Event()
        with self._lock:
            self._stop_events[project_id] = stop_event
        started_at = str(run_state.get("started_at") or datetime.now(timezone.utc).isoformat())
        self._persist_run(
            project_id, question_set_ids, groups, judge_model, concurrency,
            started_at, "evaluating", results, evaluation_mode=evaluation_mode,
        )
        futures: dict[Future[ExperimentQuestionResult], tuple[ExperimentGroup, object, BatchQuestion]] = {}
        work_iter = iter(work)
        executor = ThreadPoolExecutor(max_workers=concurrency)

        def fill_available_workers() -> None:
            while not stop_event.is_set() and len(futures) < concurrency:
                try:
                    group, question_set, question, result = next(work_iter)
                except StopIteration:
                    return
                future = executor.submit(
                    self._evaluate_case, project_id, group, question, judge_model, result, evaluation_mode,
                    judge_reasoning_effort,
                )
                futures[future] = (group, question_set, question)

        fill_available_workers()
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    group, question_set, question = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = self._failed_result(
                            group, question, judge_model, str(exc), question_set.question_set_id, question_set.name
                        )
                    results[(group.group_id, question_set.question_set_id, question.question_id)] = result
                    status = "stopping" if stop_event.is_set() else "evaluating"
                    updated = self._persist_run(
                        project_id, question_set_ids, groups, judge_model, concurrency,
                        started_at, status, results, evaluation_mode=evaluation_mode,
                    )
                    if update_callback:
                        update_callback(updated)
                fill_available_workers()
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self._persist_run(
                project_id, question_set_ids, groups, judge_model, concurrency,
                started_at, "stopped", results, evaluation_mode=evaluation_mode,
            )
            raise
        finally:
            with self._lock:
                self._stop_events.pop(project_id, None)

        run_status = "stopped" if stop_event.is_set() else "completed"
        return self._persist_run(
            project_id, question_set_ids, groups, judge_model, concurrency,
            started_at, run_status, results, datetime.now(timezone.utc).isoformat(),
            evaluation_mode=evaluation_mode,
        )

    def stop(self, project_id: str) -> bool:
        with self._lock:
            event = self._stop_events.get(project_id)
        if event is None:
            return False
        event.set()
        return True

    def update_manual_judgments(
        self, project_id: str, judgments: Sequence[tuple[str, ...]]
    ) -> dict[str, object]:
        """Persist user-edited 0/1/2 grades and refresh per-group statistics."""
        state = self.load(project_id)
        run = state.get("run")
        if not isinstance(run, dict):
            raise ProjectError("目前沒有可修改的檢索實驗結果")
        results = run.get("results", [])
        changed = False
        for judgment in judgments:
            if len(judgment) == 4:
                group_name, dataset_id, question_id, raw_correct = judgment
            elif len(judgment) == 3:
                group_name, question_id, raw_correct = judgment
                dataset_id = ""
            else:
                continue
            score = self._parse_grade(raw_correct)
            if score is None:
                continue
            for item in results:
                if (
                    item.get("group_name") != group_name or item.get("question_id") != question_id
                    or (dataset_id and item.get("dataset_id") != dataset_id)
                ):
                    continue
                next_result = score
                if item.get("evaluation_result") == "評判失敗":
                    continue
                if item.get("evaluation_result") != next_result:
                    item["evaluation_result"] = next_result
                    item["evaluation_reason"] = ""
                    item["error"] = None
                    item["status"] = "completed"
                    changed = True
        if changed:
            for group in run.get("groups", []):
                group_results = [item for item in results if item.get("group_id") == group.get("group_id")]
                judged = [item for item in group_results if self._parse_grade(item.get("evaluation_result")) is not None]
                full = sum(self._parse_grade(item.get("evaluation_result")) == 2 for item in judged)
                partial = sum(self._parse_grade(item.get("evaluation_result")) == 1 for item in judged)
                incorrect = sum(self._parse_grade(item.get("evaluation_result")) == 0 for item in judged)
                total = int(group.get("question_count", len(group_results)))
                group.update(correct_count=full, partial_correct_count=partial, incorrect_count=incorrect,
                             accuracy=full / total if total else None,
                             score_accuracy=(2 * full + partial) / (2 * total) if total else None)
            self._write(project_id, state)
        return run

    def update_quality_annotations(
        self, project_id: str, annotations: Sequence[tuple[str, ...]]
    ) -> dict[str, object]:
        """Persist human review metrics without invoking any model."""
        normalized = []
        for annotation in annotations:
            if len(annotation) != 6:
                continue
            group_name, dataset_id, question_id, raw_coverage, raw_unsupported, raw_refusal = annotation
            coverage = self._optional_annotation_number(raw_coverage, "答案要點覆蓋率", 0, 100)
            unsupported = self._optional_annotation_number(raw_unsupported, "無根據主張數", 0, None, integer=True)
            refusal_text = str(raw_refusal or "").strip().casefold()
            refusal_values = {
                "正確": "correct", "correct": "correct",
                "錯誤": "incorrect", "incorrect": "incorrect",
                "不適用": "not_applicable", "not_applicable": "not_applicable",
                "": None, "nan": None,
            }
            if refusal_text not in refusal_values:
                raise ProjectError("拒答判定只接受：正確、錯誤、不適用或留白")
            normalized.append((str(group_name), str(dataset_id), str(question_id), coverage, unsupported, refusal_values[refusal_text]))

        state = self.load(project_id)
        run = state.get("run")
        if not isinstance(run, dict):
            raise ProjectError("目前沒有可標註的檢索實驗結果")
        results = run.get("results", [])
        changed = False
        for group_name, dataset_id, question_id, coverage, unsupported, refusal in normalized:
            for item in results:
                if (
                    item.get("group_name") != group_name
                    or item.get("question_id") != question_id
                    or (dataset_id and item.get("dataset_id") != dataset_id)
                ):
                    continue
                values = {
                    "answer_point_coverage_percent": coverage,
                    "unsupported_claim_count": unsupported,
                    "refusal_correctness": refusal,
                }
                for key, value in values.items():
                    if item.get(key) != value:
                        item[key] = value
                        changed = True
        for summary in run.get("groups", []):
            selected = [item for item in results if item.get("group_id") == summary.get("group_id")]
            summary.update(self._quality_annotation_summary(selected))
        if changed or run.get("groups"):
            self._write(project_id, state)
        return run

    @staticmethod
    def _optional_annotation_number(
        value: object, label: str, minimum: int | float, maximum: int | float | None,
        integer: bool = False,
    ) -> int | float | None:
        if value is None or str(value).strip().casefold() in {"", "nan", "none"}:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"{label}必須是數字或留白") from exc
        if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
            raise ProjectError(f"{label}超出允許範圍")
        if integer and not number.is_integer():
            raise ProjectError(f"{label}必須是整數")
        return int(number) if integer else number

    @staticmethod
    def _quality_annotation_summary(items: Sequence[dict[str, object]]) -> dict[str, object]:
        coverages = [float(item["answer_point_coverage_percent"]) for item in items
                     if isinstance(item.get("answer_point_coverage_percent"), (int, float))]
        unsupported_counts = [int(item["unsupported_claim_count"]) for item in items
                              if isinstance(item.get("unsupported_claim_count"), int)]
        refusal_labels = [item.get("refusal_correctness") for item in items]
        refusal_scored = [item for item in refusal_labels if item in {"correct", "incorrect"}]
        reviewed = sum(
            item.get("answer_point_coverage_percent") is not None
            or item.get("unsupported_claim_count") is not None
            or item.get("refusal_correctness") is not None
            for item in items
        )
        return {
            "answer_point_coverage_percent": sum(coverages) / len(coverages) if coverages else None,
            "unsupported_answer_rate": (
                sum(count > 0 for count in unsupported_counts) / len(unsupported_counts)
                if unsupported_counts else None
            ),
            "average_unsupported_claim_count": (
                sum(unsupported_counts) / len(unsupported_counts) if unsupported_counts else None
            ),
            "refusal_accuracy": (
                sum(label == "correct" for label in refusal_scored) / len(refusal_scored)
                if refusal_scored else None
            ),
            "quality_reviewed_count": reviewed,
        }

    def export(self, project_id: str) -> tuple[Path, Path]:
        state = self.load(project_id)
        if state.get("run") is None:
            raise ProjectError("目前沒有可匯出的檢索實驗結果")
        export_dir = self.projects.path_for(project_id) / "exports"
        summary_path = export_dir / "retrieval-experiment-summary.json"
        details_path = export_dir / "retrieval-experiment-details.json"
        run = state["run"]
        run_groups = run.get("groups", [])
        run_results = run.get("results", [])
        group_rows = []
        evaluated_count = full_count = partial_count = incorrect_count = 0
        for summary in run_groups:
            group_results = [item for item in run_results if item.get("group_id") == summary.get("group_id")]
            judged = [item for item in group_results if self._parse_grade(item.get("evaluation_result")) is not None]
            group_correct = sum(self._parse_grade(item.get("evaluation_result")) == 2 for item in judged)
            group_partial = sum(self._parse_grade(item.get("evaluation_result")) == 1 for item in judged)
            group_incorrect = sum(self._parse_grade(item.get("evaluation_result")) == 0 for item in judged)
            group_evaluated = len(judged)
            evaluated_count += group_evaluated
            full_count += group_correct
            partial_count += group_partial
            incorrect_count += group_incorrect
            group_total = int(summary.get("question_count", len(group_results)))
            group_rows.append({
                "name": summary.get("group_name"),
                "parameters": {
                    "answer_model": summary.get("answer_model"),
                    "retrieval_mode": self._retrieval_mode_label(summary.get("method")),
                    "response_type": summary.get("response_type", DEFAULT_EXPERIMENT_RESPONSE_TYPE),
                    "answer_reasoning_effort": (
                        summary.get("answer_reasoning_effort", DEFAULT_REASONING_EFFORT)
                        if summary.get("answer_model") == GPT6_LUNA_MODEL
                        else None
                    ),
                    "strategy_parameters": summary.get("strategy_parameters", {}),
                },
                "summary": {
                    "answer_model": summary.get("answer_model"),
                    "judge_model": summary.get("judge_model"),
                    "question_count": summary.get("question_count", 0),
                    "fully_correct_count": group_correct,
                    "correct_count": group_correct,
                    "partial_correct_count": group_partial,
                    "incorrect_count": group_incorrect,
                    "fully_correct_total": f"{group_correct} / {group_total}",
                    "accuracy": group_correct / group_total if group_total else None,
                    "score_accuracy": (2 * group_correct + group_partial) / (2 * group_total) if group_total else None,
                    "evidence_hit_rate": summary.get("evidence_hit_rate"),
                    "evidence_precision": summary.get("evidence_precision"),
                    "document_hit_rate": summary.get("document_hit_rate"),
                    "no_evidence_question_count": summary.get("no_evidence_question_count", 0),
                    "average_latency_seconds": summary.get("average_latency_seconds"),
                    "p95_latency_seconds": summary.get("p95_latency_seconds"),
                    "average_search_steps": summary.get("average_search_steps"),
                    "average_agent_model_calls": summary.get("average_agent_model_calls"),
                    "estimated_cost_usd": summary.get("estimated_cost_usd"),
                    "cost_status": summary.get("cost_status"),
                    "answer_point_coverage_percent": summary.get("answer_point_coverage_percent"),
                    "unsupported_answer_rate": summary.get("unsupported_answer_rate"),
                    "average_unsupported_claim_count": summary.get("average_unsupported_claim_count"),
                    "refusal_accuracy": summary.get("refusal_accuracy"),
                    "quality_reviewed_count": summary.get("quality_reviewed_count", 0),
                },
            })
        expected_count = sum(int(item.get("question_count", 0)) for item in run_groups)
        payload = {
            "schema_version": 1,
            "format": "manual-graphrag-experiment-summary",
            "project": {"project_id": project_id, "name": self.projects.get(project_id).display_name},
            "max_concurrent_requests": state.get("max_concurrency", 5),
            "evaluation": {
                "judge_model": state.get("judge_model", ""),
                "mode": run.get("evaluation_mode", state.get("evaluation_mode", "strict")),
                "judge_reasoning_effort": (
                    DEFAULT_REASONING_EFFORT
                    if state.get("judge_model") == GPT6_LUNA_MODEL
                    else None
                ),
            },
            "summary": {
                "question_count": expected_count,
                "evaluated_count": evaluated_count,
                "fully_correct_count": full_count,
                "correct_count": full_count,
                "partial_correct_count": partial_count,
                "incorrect_count": incorrect_count,
                "fully_correct_total": f"{full_count} / {expected_count}",
                "accuracy": full_count / expected_count if expected_count else None,
                "score_accuracy": (2 * full_count + partial_count) / (2 * expected_count) if expected_count else None,
                "evidence_hit_rate": self._average_metric(run_groups, "evidence_hit_rate"),
                "evidence_precision": self._average_metric(run_groups, "evidence_precision"),
                "document_hit_rate": self._average_metric(run_groups, "document_hit_rate"),
                "average_latency_seconds": self._average_metric(run_groups, "average_latency_seconds"),
                "estimated_cost_usd": None,
                "cost_status": "GraphRAG／Chat adapter 未提供 Token 用量；成本未知",
            },
            "groups": group_rows,
            "datasets": [
                {"question_set_id": item, "name": name}
                for item, name in zip(
                    run.get("question_set_ids", [run.get("question_set_id", "")]),
                    run.get("question_set_names", [run.get("question_set_name", "")]),
                )
            ],
        }
        details_payload = {
            "schema_version": 1,
            "format": "manual-graphrag-experiment-question-results",
            "project": payload["project"],
            "execution_status": run.get("status"),
            "evaluation_mode": run.get("evaluation_mode", state.get("evaluation_mode", "strict")),
            "question_set_id": state.get("question_set_id", ""),
            "question_set_ids": run.get("question_set_ids", [run.get("question_set_id", "")]),
            "datasets": payload["datasets"],
            "summary": payload["summary"],
            "question_set_name": run.get("question_set_name", ""),
            "question_count": expected_count,
            "groups": [
                {
                    "name": group.get("name"),
                    "answer_model": group.get("parameters", {}).get("answer_model"),
                    "retrieval_mode": group.get("parameters", {}).get("retrieval_mode"),
                    "response_type": group.get("parameters", {}).get("response_type"),
                    "strategy_parameters": group.get("parameters", {}).get("strategy_parameters", {}),
                    "judge_model": group.get("summary", {}).get("judge_model"),
                }
                for group in group_rows
            ],
            "question_results": run_results,
        }
        self._atomic_write(summary_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        self._atomic_write(details_path, json.dumps(details_payload, ensure_ascii=False, indent=2) + "\n")
        return summary_path, details_path

    @staticmethod
    def _retrieval_mode_label(method: str | None) -> str | None:
        return {
            "local": "Microsoft GraphRAG Local",
            "global": "Microsoft GraphRAG Global",
            "drift": "Microsoft GraphRAG DRIFT",
            "basic": "Microsoft GraphRAG Basic",
            "agentic": "Agentic Search（實驗）",
            "auto": "自動（回答模型逐題選擇 Microsoft GraphRAG 策略）",
        }.get(method)

    @staticmethod
    def _average_metric(groups: Sequence[dict[str, object]], field: str) -> float | None:
        values = [
            float(group[field]) for group in groups
            if isinstance(group.get(field), (int, float)) and math.isfinite(float(group[field]))
        ]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _parse_grade(value: object) -> int | None:
        if isinstance(value, bool):
            return 2 if value else 0
        if isinstance(value, int) and value in {0, 1, 2}:
            return value
        text = str(value or "").strip()
        legacy = {"正確": 2, "全對": 2, "錯誤": 0, "部分對": 1}
        if text in legacy:
            return legacy[text]
        if text in {"0", "1", "2"}:
            return int(text)
        for prefix, score in (("0 ", 0), ("1 ", 1), ("2 ", 2)):
            if text.startswith(prefix):
                return score
        return None

    def _run_case(
        self, project_id: str, group: ExperimentGroup, question: BatchQuestion, judge_model: str,
        dataset_id: str = "", dataset_name: str = "",
    ) -> ExperimentQuestionResult:
        generated = self._generate_case(project_id, group, question, judge_model, dataset_id, dataset_name)
        if generated.status != "answered":
            return generated
        return self._evaluate_case(project_id, group, question, judge_model, generated)

    def _generate_case(
        self, project_id: str, group: ExperimentGroup, question: BatchQuestion, judge_model: str,
        dataset_id: str = "", dataset_name: str = "",
    ) -> ExperimentQuestionResult:
        started_clock = time.monotonic()
        routed_method = routing_reason = ""
        method = group.method
        execution_trace: dict[str, object] = {}
        try:
            if group.method == EXPERIMENT_AUTO_METHOD:
                decision = self.router.route(
                    question.question,
                    group.answer_model,
                    self.connections.get_api_base_url(),
                    self.connections.get_api_key(project_id),
                )
                method = routed_method = decision.method
                routing_reason = decision.reason
            if group.method == EXPERIMENT_AGENTIC_METHOD:
                api_base = self.connections.get_api_base_url()
                api_key = self.connections.get_api_key(project_id)

                def execute_search(search_method: str, search_question: str, timeout_seconds: float | None = None):
                    arguments = (
                        project_id, search_question, search_method, group.answer_model,
                        group.response_type, group.answer_reasoning_effort,
                    )
                    signature = inspect.signature(self.query_function)
                    accepts_parameters = any(
                        item.kind == inspect.Parameter.VAR_POSITIONAL
                        for item in signature.parameters.values()
                    ) or len(signature.parameters) >= 7
                    kwargs = {}
                    if "timeout_seconds" in signature.parameters:
                        kwargs["timeout_seconds"] = timeout_seconds
                    if accepts_parameters:
                        answer_value, context_value = self.query_function(
                            *arguments, group.strategy_parameters.get(search_method, {}), **kwargs
                        )
                    else:
                        answer_value, context_value = self.query_function(*arguments, **kwargs)
                    return QueryExecution(0, str(answer_value), "", context_value)

                decision = self.agentic_search.run(
                    question.question, group.answer_model, api_base, api_key,
                    execute_search, group.answer_reasoning_effort,
                )
                method = EXPERIMENT_AGENTIC_METHOD
                answer = decision.answer
                context = dict(decision.context)
                execution_trace = context
                if decision.returncode != 0 and not answer.strip():
                    failed = self._failed_result(
                        group, question, judge_model,
                        decision.error or decision.stop_reason, dataset_id, dataset_name,
                    )
                    return replace(
                        failed,
                        routing_reason=decision.stop_reason,
                        execution_trace=execution_trace,
                        duration_seconds=round(time.monotonic() - started_clock, 3),
                    )
                routing_reason = decision.stop_reason
                step_contexts = [
                    step.get("context", {}) for step in context.get("agentic_steps", [])
                    if isinstance(step, dict) and isinstance(step.get("context"), dict)
                ]
                context["sources"] = [
                    source for step_context in step_contexts
                    for source in (step_context.get("sources") if isinstance(step_context.get("sources"), list) else [])
                ]
                context["reports"] = [
                    report for step_context in step_contexts
                    for report in (step_context.get("reports") if isinstance(step_context.get("reports"), list) else [])
                ]
                execution_trace = context
            else:
                query_arguments = (
                    project_id, question.question, method, group.answer_model, group.response_type,
                    group.answer_reasoning_effort,
                )
                parameters = inspect.signature(self.query_function).parameters.values()
                accepts_strategy_parameters = any(
                    item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters
                ) or len(inspect.signature(self.query_function).parameters) >= 7
                if accepts_strategy_parameters:
                    answer, context = self.query_function(
                        *query_arguments, group.strategy_parameters.get(method, {})
                    )
                else:
                    answer, context = self.query_function(*query_arguments)
        except Exception as exc:
            return replace(
                self._failed_result(group, question, judge_model, str(exc), dataset_id, dataset_name),
                routed_method=routed_method, routing_reason=routing_reason,
            )
        try:
            retrieved_evidence: tuple[Evidence, ...] = ()
            mapping_error = None
            if method in EVIDENCE_METHODS or method == EXPERIMENT_AGENTIC_METHOD:
                try:
                    retrieved_evidence = self._resolved_evidence(project_id, context)
                    raw_sources = EvidenceService.source_records(context)
                    if raw_sources and not retrieved_evidence:
                        mapping_error = "GraphRAG 回傳了來源，但無法映射至原始文件"
                except (ProjectError, OSError, ValueError, TypeError) as exc:
                    mapping_error = f"來源映射失敗：{exc}"
            source_evidence = tuple(dict.fromkeys((
                *question.question_source_evidence,
                *question.answer_source_evidence,
                *question.gold_evidence,
            )))
            source_docs = tuple(dict.fromkeys(
                [item.document_name or item.document_id for item in source_evidence]
                + [item.document_id for item in retrieved_evidence]
                + list(question.source_documents)
            ))
            retrieval_status = (
                f"不可計算：{mapping_error}" if mapping_error else
                "已映射至原始文件／頁碼；不使用 context 順序作排名"
                if method in EVIDENCE_METHODS or (
                    method == EXPERIMENT_AGENTIC_METHOD
                    and any(getattr(item, "source_type", "text_unit") == "text_unit" for item in retrieved_evidence)
                )
                else UNSUPPORTED_RETRIEVAL_METRICS
            )
            return ExperimentQuestionResult(
                group.group_id, group.name, group.answer_model, judge_model, group.method,
                question.question_id, question.question, source_docs, question.reference_answer,
                answer, "待評測", "", None, retrieval_status, "answered",
                response_type=group.response_type,
                answer_reasoning_effort=group.answer_reasoning_effort,
                retrieved_evidence=tuple(retrieved_evidence),
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                routed_method=routed_method,
                routing_reason=routing_reason,
                execution_trace=execution_trace,
                duration_seconds=round(time.monotonic() - started_clock, 3),
            )
        except Exception as exc:
            return replace(
                self._failed_result(group, question, judge_model, str(exc), dataset_id, dataset_name),
                routed_method=routed_method, routing_reason=routing_reason,
            )

    def _evaluate_case(
        self,
        project_id: str,
        group: ExperimentGroup,
        question: BatchQuestion,
        judge_model: str,
        generated: ExperimentQuestionResult,
        evaluation_mode: str = "strict",
        judge_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> ExperimentQuestionResult:
        try:
            judged = self.judging.evaluate_single_answer_grade(
                project_id, question.question_id, question.question,
                question.reference_answer, generated.actual_answer, judge_model,
                evaluation_mode=evaluation_mode,
                reasoning_effort=judge_reasoning_effort,
            )
            return replace(
                generated,
                judge_model=judge_model,
                evaluation_result=judged.score,
                evaluation_reason=judged.judge_reason,
                status="completed",
                error=None,
            )
        except Exception as exc:
            return replace(
                generated,
                judge_model=judge_model,
                evaluation_result="評判失敗",
                evaluation_reason=str(exc),
                status="evaluation_failed",
                error=str(exc),
            )

    def _failed_result(
        self, group: ExperimentGroup, question: BatchQuestion, judge_model: str, error: str,
        dataset_id: str = "", dataset_name: str = "",
    ) -> ExperimentQuestionResult:
        source_docs = tuple(dict.fromkeys(
            item.document_name or item.document_id
            for item in (
                *question.question_source_evidence,
                *question.answer_source_evidence,
                *question.gold_evidence,
            )
        ))
        return ExperimentQuestionResult(
            group.group_id, group.name, group.answer_model, judge_model, group.method,
            question.question_id, question.question, source_docs, question.reference_answer,
            "", "未評判", error, None,
            (
                UNSUPPORTED_RETRIEVAL_METRICS if group.method == "global"
                else "不可計算：查詢失敗，未取得檢索證據"
            ),
            "failed", error,
            response_type=group.response_type,
            answer_reasoning_effort=group.answer_reasoning_effort,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )

    def _persist_run(
        self,
        project_id: str,
        question_set_id: str | Sequence[str],
        groups: Sequence[ExperimentGroup],
        judge_model: str,
        max_concurrency: int,
        started_at: str,
        status: str,
        results: dict[tuple[str, ...], ExperimentQuestionResult],
        completed_at: str | None = None,
        evaluation_mode: str | None = None,
    ) -> RetrievalExperimentRun:
        question_set_ids = self._normalize_question_set_ids(question_set_id)
        question_sets = [self.question_sets.get(project_id, item) for item in question_set_ids]
        dataset_by_id = {item.question_set_id: item for item in question_sets}
        project = self.projects.get(project_id)
        summaries = []
        result_items = tuple(results.values())
        for group in groups:
            group_items = [item for item in result_items if item.group_id == group.group_id]
            judged = [item for item in group_items if self._parse_grade(item.evaluation_result) is not None]
            correct = sum(self._parse_grade(item.evaluation_result) == 2 for item in judged)
            partial = sum(self._parse_grade(item.evaluation_result) == 1 for item in judged)
            incorrect = sum(self._parse_grade(item.evaluation_result) == 0 for item in judged)
            total = sum(len(question_set.questions) for question_set in question_sets)
            answered = sum(bool(item.actual_answer.strip()) for item in group_items)
            if group.method in EVIDENCE_METHODS or group.method in {EXPERIMENT_AUTO_METHOD, EXPERIMENT_AGENTIC_METHOD}:
                # Auto groups only score questions that were routed to an evidence-mapping strategy.
                metric_results = [
                    item for item in group_items
                    if item.status != "failed"
                    and (
                        item.effective_method in EVIDENCE_METHODS
                        if group.method == EXPERIMENT_AUTO_METHOD
                        else group.method != EXPERIMENT_AGENTIC_METHOD or any(
                            getattr(evidence, "source_type", "text_unit") == "text_unit"
                            for evidence in item.retrieved_evidence
                        )
                    )
                    and not item.retrieval_metrics_status.startswith("不可計算：")
                ]
                metric_samples = []
                for item in metric_results:
                    dataset = dataset_by_id.get(item.dataset_id) or (question_sets[0] if question_sets else None)
                    if dataset is not None:
                        metric_samples.append((
                            item.retrieved_evidence, self._gold_evidence_for(dataset, item.question_id)
                        ))
                metric_values = compute_source_metrics(tuple(metric_samples))
                evidence_hit_rate, evidence_precision, document_hit_rate, no_evidence_count = metric_values
                metrics_status = (
                    "尚無成功完成來源解析的題目"
                    if not metric_results else "已計算：依文件 ID／頁碼比對 Gold Evidence；逐題 precision 平均（無標註頁碼者排除），不使用排名"
                )
                if metric_results and group.method == EXPERIMENT_AUTO_METHOD:
                    metrics_status += "；自動選擇 Global 的題目不計入"
                elif group.method == EXPERIMENT_AGENTIC_METHOD and not metric_results:
                    metrics_status = "不可計算：本組未取得可映射頁碼的 Text Unit Evidence；Community Report 不計入 Gold Evidence 指標"
            else:
                evidence_hit_rate = evidence_precision = document_hit_rate = None
                no_evidence_count = 0
                metrics_status = UNSUPPORTED_RETRIEVAL_METRICS
            durations = [item.duration_seconds for item in group_items if item.duration_seconds is not None]
            sorted_durations = sorted(durations)
            p95_latency = (
                sorted_durations[max(0, math.ceil(0.95 * len(sorted_durations)) - 1)]
                if sorted_durations else None
            )
            agentic_results = [item for item in group_items if item.execution_trace]
            step_counts = [
                float(item.execution_trace.get("agentic_metadata", {}).get("search_step_count", 0))
                for item in agentic_results
                if isinstance(item.execution_trace.get("agentic_metadata"), dict)
            ]
            model_calls = [
                float(item.execution_trace.get("agentic_metadata", {}).get("model_call_count", 0))
                for item in agentic_results
                if isinstance(item.execution_trace.get("agentic_metadata"), dict)
            ]
            quality_summary = self._quality_annotation_summary([
                {
                    "answer_point_coverage_percent": item.answer_point_coverage_percent,
                    "unsupported_claim_count": item.unsupported_claim_count,
                    "refusal_correctness": item.refusal_correctness,
                }
                for item in group_items
            ])
            summaries.append(ExperimentGroupSummary(
                group.group_id, group.name, group.answer_model, judge_model, group.method,
                total, answered, correct, partial, incorrect,
                correct / total if total else None,
                (2 * correct + partial) / (2 * total) if total else None,
                evidence_hit_rate, evidence_precision, document_hit_rate,
                no_evidence_count, metrics_status, group.response_type, group.answer_reasoning_effort,
                (
                    {method: dict(params) for method, params in group.strategy_parameters.items()}
                    if group.method == EXPERIMENT_AUTO_METHOD
                    else dict(group.strategy_parameters.get(group.method, {}))
                ),
                sum(durations) / len(durations) if durations else None,
                p95_latency,
                sum(step_counts) / len(step_counts) if step_counts else None,
                sum(model_calls) / len(model_calls) if model_calls else None,
                None,
                "GraphRAG／Chat adapter 未提供 Token 用量；成本未知",
                **quality_summary,
            ))
        question_set_name = "、".join(item.name for item in question_sets)
        run = RetrievalExperimentRun(
            project_id, project.display_name,
            question_set_ids[0] if question_set_ids else "", question_set_name, status,
            max_concurrency, sum(len(item.questions) for item in question_sets), started_at, completed_at,
            tuple(summaries), result_items, tuple(question_set_ids),
            tuple(item.name for item in question_sets),
        )
        state = self.load(project_id)
        state["question_set_ids"] = question_set_ids
        state["question_set_id"] = question_set_ids[0] if question_set_ids else ""
        state["judge_model"] = judge_model
        run_value = asdict(run)
        if evaluation_mode is not None:
            run_value["evaluation_mode"] = evaluation_mode
        state["run"] = run_value
        self._write(project_id, state)
        return run

    def _resolved_evidence(self, project_id: str, context: dict[str, object]) -> tuple[Evidence, ...]:
        raw = context.get("sources")
        if not isinstance(raw, list) or not raw:
            return ()
        evidence = EvidenceService(self.projects).from_context(project_id, context)
        return tuple(evidence)

    def _resolved_sources(self, project_id: str, context: dict[str, object]) -> tuple[str, ...]:
        service = EvidenceService(self.projects)
        evidence = service.from_context(project_id, context)
        document_by_unit = {
            item.text_unit_id: item.document_id for item in service.metadata.load(project_id)
        }
        return tuple(dict.fromkeys(
            document_by_unit.get(item.text_unit_id, item.document_id) for item in evidence
        ))

    @staticmethod
    def _gold_evidence_for(question_set: object, question_id: str):
        question = next((item for item in question_set.questions if item.question_id == question_id), None)
        if question is None:
            return ()
        return question.gold_evidence or question.answer_source_evidence

    @staticmethod
    def _result_from_dict(value: dict[str, object]) -> ExperimentQuestionResult:
        data = dict(value)
        data.setdefault("retrieved_evidence", [])
        data["retrieved_evidence"] = tuple(
            item if isinstance(item, Evidence) else Evidence(**item)
            for item in data["retrieved_evidence"]
        )
        return ExperimentQuestionResult(**data)

    def _graphrag_query(
        self,
        project_id: str,
        question: str,
        method: str,
        model: str,
        response_type: str = DEFAULT_EXPERIMENT_RESPONSE_TYPE,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        strategy_parameters: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[str, dict[str, object]]:
        if method not in EXPERIMENT_METHODS:
            raise ProjectError("不支援的 GraphRAG 檢索策略")
        if response_type not in EXPERIMENT_RESPONSE_TYPES:
            raise ProjectError("不支援的 GraphRAG 回答格式")
        if reasoning_effort not in REASONING_EFFORT_OPTIONS:
            raise ProjectError("不支援的 Luna 推理深度")
        defaults = default_strategy_parameters()[method]
        selected_parameters = {**defaults, **(strategy_parameters or {})}
        allowed_names = {item[0] for item in EXPERIMENT_STRATEGY_PARAMETER_SPECS[method]}
        if set(selected_parameters) - allowed_names:
            raise ProjectError("策略參數包含不支援的欄位")
        project_path = self.projects.path_for(project_id) / "graphrag"
        settings_path = project_path / "settings.yaml"
        try:
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ProjectError("無法讀取 GraphRAG 設定") from exc
        for section in ("completion_models", "embedding_models"):
            for config in settings.get(section, {}).values():
                config["api_base"] = self.connections.get_api_base_url()
        for config in settings.get("completion_models", {}).values():
            configure_completion_model(config, model, reasoning_effort)
        settings_section = {"local": "local_search", "global": "global_search", "drift": "drift_search", "basic": "basic_search"}[method]
        section = settings.setdefault(settings_section, {})
        for key in EXPERIMENT_STRATEGY_CONFIG_KEYS[method]:
            if key in selected_parameters:
                section[key] = selected_parameters[key]
        api_key = self.connections.apply_to_environment(project_id)
        del api_key  # GraphRAG reads the standard API key environment variable.

        with tempfile.TemporaryDirectory(prefix="graphrag-experiment-") as temporary:
            root = Path(temporary)
            (root / "settings.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
            (root / "output").symlink_to(project_path / "output", target_is_directory=True)
            worker_payload = json.dumps(
                {
                    "root_dir": str(root),
                    "question": question,
                    "method": method,
                    "model": model,
                    "response_type": response_type,
                    "community_level": selected_parameters.get("community_level", 2),
                    "dynamic_community_selection": selected_parameters.get("dynamic_community_selection", False),
                },
                ensure_ascii=False,
            )
            package_root = Path(__file__).resolve().parents[2]
            environment = os.environ.copy()
            source_path = str(package_root / "src")
            environment["PYTHONPATH"] = os.pathsep.join(
                part for part in (source_path, environment.get("PYTHONPATH", "")) if part
            )
            result = subprocess.run(
                [sys.executable, "-m", "automotive_graphrag.graph_query_worker"],
                cwd=package_root,
                env=environment,
                input=worker_payload,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise ProjectError(
                f"GraphRAG 查詢工作程序未回傳有效結果：{detail[-1] if detail else result.returncode}"
            ) from exc
        if result.returncode or "error" in response:
            raise ProjectError(str(response.get("error") or result.stderr or "GraphRAG 查詢失敗"))
        context = response.get("context", {})
        return str(response.get("answer", "")), context if isinstance(context, dict) else {}

    def _validate_groups(self, groups: Sequence[ExperimentGroup]) -> None:
        seen: set[str] = set()
        for group in groups:
            if not group.group_id or group.group_id in seen:
                raise ProjectError("實驗組 ID 空白或重複")
            seen.add(group.group_id)
            if not group.name.strip():
                raise ProjectError(f"{group.group_id} 的實驗組名稱不可空白")
            if group.answer_model not in ALLOWED_CHAT_MODELS:
                raise ProjectError(f"{group.group_id} 的回答模型不支援")
            if group.method not in EXPERIMENT_GROUP_METHODS:
                raise ProjectError(f"{group.group_id} 的 GraphRAG 策略不支援")
            if group.response_type not in EXPERIMENT_RESPONSE_TYPES:
                raise ProjectError(f"{group.group_id} 的 GraphRAG 回答格式不支援")
            if group.answer_reasoning_effort not in REASONING_EFFORT_OPTIONS:
                raise ProjectError(f"{group.group_id} 的 Luna 推理深度不支援")
            strategy_parameters = group.strategy_parameters or default_strategy_parameters()
            for method, parameters in strategy_parameters.items():
                specs = {item[0]: item for item in EXPERIMENT_STRATEGY_PARAMETER_SPECS.get(method, ())}
                if method not in EXPERIMENT_METHODS or not isinstance(parameters, dict) or set(parameters) - set(specs):
                    raise ProjectError(f"{group.group_id} 的策略參數不支援")

    def _path(self, project_id: str) -> Path:
        return self.projects.path_for(project_id) / "runs" / "retrieval-experiment-state.json"

    def _write(self, project_id: str, value: dict[str, object]) -> None:
        self._atomic_write(self._path(project_id), json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
