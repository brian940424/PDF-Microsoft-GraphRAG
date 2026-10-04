"""Persistent, cancellable comparisons of Microsoft GraphRAG query strategies."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import yaml

from .automatic_evaluation import AutomaticEvaluationService
from .connections import ALLOWED_CHAT_MODELS, ConnectionSettings
from .evidence import EvidenceService
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, QuestionSetService
from .querying import QueryService


EXPERIMENT_METHODS = ("local", "global", "drift", "basic")
UNRANKED_REASON = (
    "不可計算：GraphRAG API context 未提供可驗證的來源排序分數；context 列表順序不保證是檢索排名。"
)


@dataclass(frozen=True, slots=True)
class ExperimentGroup:
    group_id: str
    name: str
    answer_model: str
    method: str


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
    evaluation_result: str
    evaluation_reason: str
    answer_source_rank: int | None
    retrieval_metrics_status: str
    status: str
    error: str | None = None


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
    accuracy: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr: float | None
    retrieval_metrics_status: str


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


QueryFunction = Callable[[str, str, str, str], tuple[str, dict[str, object]]]
UpdateFunction = Callable[[RetrievalExperimentRun], None]


class RetrievalExperimentService:
    def __init__(
        self,
        projects: ProjectStore,
        question_sets: QuestionSetService,
        connections: ConnectionSettings,
        judging: AutomaticEvaluationService,
        query_function: QueryFunction | None = None,
    ) -> None:
        self.projects = projects
        self.question_sets = question_sets
        self.connections = connections
        self.judging = judging
        self.query_function = query_function or self._graphrag_query
        self._lock = threading.RLock()
        self._stop_events: dict[str, threading.Event] = {}

    def load(self, project_id: str) -> dict[str, object]:
        self.projects.get(project_id)
        try:
            value = json.loads(self._path(project_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"groups": [], "question_set_id": "", "max_concurrency": 5,
                    "judge_model": self.connections.get_chat_model(), "run": None}
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectError("檢索實驗保存資料無法讀取") from exc
        value.setdefault("groups", [])
        value.setdefault("question_set_id", "")
        value.setdefault("max_concurrency", 5)
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
        return value

    def save_configuration(
        self,
        project_id: str,
        groups: Sequence[ExperimentGroup],
        question_set_id: str,
        max_concurrency: int,
        judge_model: str | None = None,
    ) -> None:
        if not 1 <= max_concurrency <= 32:
            raise ProjectError("最大並行請求數必須介於 1 到 32")
        current = self.load(project_id)
        current["groups"] = [asdict(group) for group in groups]
        if judge_model is not None:
            if judge_model not in ALLOWED_CHAT_MODELS:
                raise ProjectError("全域評測模型不支援")
            current["judge_model"] = judge_model
        current["question_set_id"] = question_set_id
        current["max_concurrency"] = max_concurrency
        self._write(project_id, current)

    def import_question_set(self, project_id: str, filepath: str | Path) -> object:
        question_set = self.question_sets.import_file(project_id, filepath)
        current = self.load(project_id)
        current["question_set_id"] = question_set.question_set_id
        current["run"] = None
        self._write(project_id, current)
        return question_set

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
                if method in EXPERIMENT_METHODS:
                    group["method"] = method
                elif method:
                    raise ProjectError(f"{group.get('group_id')} 的 GraphRAG 策略不支援")
        current["groups"] = groups
        self._write(project_id, current)
        return current

    def set_group_method(self, project_id: str, group_id: str, method: str) -> dict[str, object]:
        if method not in EXPERIMENT_METHODS:
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

    def remove_group(self, project_id: str, group_id: str) -> dict[str, object]:
        current = self.load(project_id)
        current["groups"] = [g for g in current["groups"] if g.get("group_id") != group_id]
        self._write(project_id, current)
        return current

    def run(
        self,
        project_id: str,
        question_set_id: str,
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
        question_set = self.question_sets.get(project_id, question_set_id)
        if project.status != "INDEXED":
            raise ProjectError("專案尚未完成 GraphRAG 建圖")
        if not question_set.questions:
            raise ProjectError("匯入題目集沒有題目")
        missing_answers = [item.question_id for item in question_set.questions if not item.reference_answer.strip()]
        if missing_answers:
            raise ProjectError("以下題目缺少正確答案，無法評測：" + ", ".join(missing_answers))
        stop_event = threading.Event()
        with self._lock:
            self._stop_events[project_id] = stop_event

        start = datetime.now(timezone.utc).isoformat()
        results: dict[tuple[str, str], ExperimentQuestionResult] = {}
        run_status = "running"
        initial_status = "generating_answers" if not evaluate else run_status
        self._persist_run(project_id, question_set_id, groups, judge_model, max_concurrency, start, initial_status, results)
        work = [(group, question) for group in groups for question in question_set.questions]
        executor = ThreadPoolExecutor(max_workers=max_concurrency)
        work_iter = iter(work)
        futures: dict[Future[ExperimentQuestionResult], tuple[ExperimentGroup, BatchQuestion]] = {}

        def fill_available_workers() -> None:
            while not stop_event.is_set() and len(futures) < max_concurrency:
                try:
                    group, question = next(work_iter)
                except StopIteration:
                    return
                worker = self._run_case if evaluate else self._generate_case
                futures[executor.submit(worker, project_id, group, question, judge_model)] = (group, question)

        fill_available_workers()
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    group, question = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # Preserve an individual failure instead of losing the run.
                        result = self._failed_result(group, question, judge_model, str(exc))
                    results[(group.group_id, question.question_id)] = result
                    run_status = "stopping" if stop_event.is_set() else (
                        "evaluating" if evaluate else "generating_answers"
                    )
                    run = self._persist_run(
                        project_id, question_set_id, groups, judge_model, max_concurrency, start, run_status, results
                    )
                    if update_callback:
                        update_callback(run)
                fill_available_workers()
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self._persist_run(project_id, question_set_id, groups, judge_model, max_concurrency, start, "stopped", results)
            raise
        finally:
            with self._lock:
                self._stop_events.pop(project_id, None)

        if stop_event.is_set():
            run_status = "stopped"
        elif evaluate:
            run_status = "completed"
        else:
            expected_count = len(groups) * len(question_set.questions)
            run_status = "answers_completed" if len(results) == expected_count and all(
                item.status == "answered" and item.actual_answer.strip()
                for item in results.values()
            ) else "answers_partial"
        completed_at = datetime.now(timezone.utc).isoformat()
        return self._persist_run(
            project_id, question_set_id, groups, judge_model, max_concurrency, start, run_status, results, completed_at
        )

    def generate_answers(
        self,
        project_id: str,
        question_set_id: str,
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
        question_set_id: str,
        judge_model: str | None = None,
        max_concurrency: int | None = None,
        update_callback: UpdateFunction | None = None,
    ) -> RetrievalExperimentRun:
        """Evaluate previously saved answers without repeating GraphRAG retrieval."""
        state = self.load(project_id)
        run_state = state.get("run")
        if not isinstance(run_state, dict) or run_state.get("question_set_id") != question_set_id:
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
        question_set = self.question_sets.get(project_id, question_set_id)
        question_by_id = {item.question_id: item for item in question_set.questions}
        results: dict[tuple[str, str], ExperimentQuestionResult] = {}
        for item in run_state.get("results", []):
            result = ExperimentQuestionResult(**item)
            results[(result.group_id, result.question_id)] = result
        work: list[tuple[ExperimentGroup, BatchQuestion, ExperimentQuestionResult]] = []
        for group in groups:
            for question in question_set.questions:
                result = results.get((group.group_id, question.question_id))
                if result is None or not result.actual_answer.strip():
                    raise ProjectError(
                        f"{group.name} 的題目 {question.question_id} 尚未完成回答；請先生成所有答案"
                    )
                if result.answer_model != group.answer_model or result.method != group.method:
                    raise ProjectError(f"{group.name} 的模型或檢索策略已變更；請重新生成答案")
                if result.question != question.question or result.correct_answer != question.reference_answer:
                    raise ProjectError(f"題目 {question.question_id} 已變更；請重新生成答案")
                work.append((group, question_by_id[question.question_id], result))

        stop_event = threading.Event()
        with self._lock:
            self._stop_events[project_id] = stop_event
        started_at = str(run_state.get("started_at") or datetime.now(timezone.utc).isoformat())
        self._persist_run(
            project_id, question_set_id, groups, judge_model, concurrency,
            started_at, "evaluating", results,
        )
        futures: dict[Future[ExperimentQuestionResult], tuple[ExperimentGroup, BatchQuestion]] = {}
        work_iter = iter(work)
        executor = ThreadPoolExecutor(max_workers=concurrency)

        def fill_available_workers() -> None:
            while not stop_event.is_set() and len(futures) < concurrency:
                try:
                    group, question, result = next(work_iter)
                except StopIteration:
                    return
                future = executor.submit(self._evaluate_case, project_id, group, question, judge_model, result)
                futures[future] = (group, question)

        fill_available_workers()
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    group, question = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = self._failed_result(group, question, judge_model, str(exc))
                    results[(group.group_id, question.question_id)] = result
                    status = "stopping" if stop_event.is_set() else "evaluating"
                    updated = self._persist_run(
                        project_id, question_set_id, groups, judge_model, concurrency,
                        started_at, status, results,
                    )
                    if update_callback:
                        update_callback(updated)
                fill_available_workers()
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            stop_event.set()
            executor.shutdown(wait=True, cancel_futures=True)
            self._persist_run(
                project_id, question_set_id, groups, judge_model, concurrency,
                started_at, "stopped", results,
            )
            raise
        finally:
            with self._lock:
                self._stop_events.pop(project_id, None)

        run_status = "stopped" if stop_event.is_set() else "completed"
        return self._persist_run(
            project_id, question_set_id, groups, judge_model, concurrency,
            started_at, run_status, results, datetime.now(timezone.utc).isoformat(),
        )

    def stop(self, project_id: str) -> bool:
        with self._lock:
            event = self._stop_events.get(project_id)
        if event is None:
            return False
        event.set()
        return True

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
        evaluated_count = 0
        correct_count = 0
        for summary in run_groups:
            group_results = [item for item in run_results if item.get("group_id") == summary.get("group_id")]
            judged = [item for item in group_results if item.get("evaluation_result") in {"正確", "錯誤"}]
            group_correct = sum(item.get("evaluation_result") == "正確" for item in judged)
            group_evaluated = len(judged)
            evaluated_count += group_evaluated
            correct_count += group_correct
            group_rows.append({
                "name": summary.get("group_name"),
                "parameters": {
                    "answer_model": summary.get("answer_model"),
                    "retrieval_mode": self._retrieval_mode_label(summary.get("method")),
                },
                "summary": {
                    "answer_model": summary.get("answer_model"),
                    "judge_model": summary.get("judge_model"),
                    "question_count": summary.get("question_count", 0),
                    "correct_count": group_correct,
                    "correct_total": f"{group_correct} / {group_evaluated}",
                    "accuracy": group_correct / group_evaluated if group_evaluated else None,
                    "recall_at_5": summary.get("recall_at_5"),
                    "recall_at_10": summary.get("recall_at_10"),
                    "mrr": summary.get("mrr"),
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
                "judge_reasoning_effort": None,
            },
            "summary": {
                "question_count": expected_count,
                "correct_count": correct_count,
                "correct_total": f"{correct_count} / {evaluated_count}",
                "accuracy": correct_count / evaluated_count if evaluated_count else None,
                "recall_at_5": self._average_metric(run_groups, "recall_at_5"),
                "recall_at_10": self._average_metric(run_groups, "recall_at_10"),
                "mrr": self._average_metric(run_groups, "mrr"),
            },
            "groups": group_rows,
        }
        details_payload = {
            "schema_version": 1,
            "format": "manual-graphrag-experiment-question-results",
            "project": payload["project"],
            "execution_status": run.get("status"),
            "question_set_id": state.get("question_set_id", ""),
            "question_set_name": run.get("question_set_name", ""),
            "question_count": expected_count,
            "groups": [
                {
                    "name": group.get("name"),
                    "answer_model": group.get("parameters", {}).get("answer_model"),
                    "retrieval_mode": group.get("parameters", {}).get("retrieval_mode"),
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
        }.get(method)

    @staticmethod
    def _average_metric(groups: Sequence[dict[str, object]], field: str) -> float | None:
        values = [
            float(group[field]) for group in groups
            if isinstance(group.get(field), (int, float)) and math.isfinite(float(group[field]))
        ]
        return sum(values) / len(values) if values else None

    def _run_case(
        self, project_id: str, group: ExperimentGroup, question: BatchQuestion, judge_model: str
    ) -> ExperimentQuestionResult:
        generated = self._generate_case(project_id, group, question, judge_model)
        if generated.status != "answered":
            return generated
        return self._evaluate_case(project_id, group, question, judge_model, generated)

    def _generate_case(
        self, project_id: str, group: ExperimentGroup, question: BatchQuestion, judge_model: str
    ) -> ExperimentQuestionResult:
        try:
            answer, context = self.query_function(project_id, question.question, group.method, group.answer_model)
        except Exception as exc:
            return self._failed_result(group, question, judge_model, str(exc))
        try:
            try:
                retrieved = self._resolved_sources(project_id, context)
            except (ProjectError, OSError, ValueError, TypeError):
                retrieved = ()
            source_evidence = tuple(dict.fromkeys((
                *question.question_source_evidence,
                *question.answer_source_evidence,
                *question.gold_evidence,
            )))
            source_docs = tuple(dict.fromkeys(
                [item.document_name or item.document_id for item in source_evidence]
                + [item for item in retrieved]
                + list(question.source_documents)
            ))
            return ExperimentQuestionResult(
                group.group_id, group.name, group.answer_model, judge_model, group.method,
                question.question_id, question.question, source_docs, question.reference_answer,
                answer, "待評測", "", None, UNRANKED_REASON, "answered",
            )
        except Exception as exc:
            return self._failed_result(group, question, judge_model, str(exc))

    def _evaluate_case(
        self,
        project_id: str,
        group: ExperimentGroup,
        question: BatchQuestion,
        judge_model: str,
        generated: ExperimentQuestionResult,
    ) -> ExperimentQuestionResult:
        try:
            judged = self.judging.evaluate_single_answer(
                project_id, question.question_id, question.question,
                question.reference_answer, generated.actual_answer, judge_model,
            )
            return replace(
                generated,
                judge_model=judge_model,
                evaluation_result="正確" if judged.is_correct else "錯誤",
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
        self, group: ExperimentGroup, question: BatchQuestion, judge_model: str, error: str
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
            "", "未評判", error, None, UNRANKED_REASON, "failed", error,
        )

    def _persist_run(
        self,
        project_id: str,
        question_set_id: str,
        groups: Sequence[ExperimentGroup],
        judge_model: str,
        max_concurrency: int,
        started_at: str,
        status: str,
        results: dict[tuple[str, str], ExperimentQuestionResult],
        completed_at: str | None = None,
    ) -> RetrievalExperimentRun:
        question_set = self.question_sets.get(project_id, question_set_id)
        project = self.projects.get(project_id)
        summaries = []
        result_items = tuple(results.values())
        for group in groups:
            group_items = [item for item in result_items if item.group_id == group.group_id]
            judged = [item for item in group_items if item.evaluation_result in {"正確", "錯誤"}]
            correct = sum(item.evaluation_result == "正確" for item in judged)
            answered = sum(bool(item.actual_answer.strip()) for item in group_items)
            summaries.append(ExperimentGroupSummary(
                group.group_id, group.name, group.answer_model, judge_model, group.method,
                len(question_set.questions), answered, correct,
                correct / len(judged) if judged else None,
                None, None, None, UNRANKED_REASON,
            ))
        run = RetrievalExperimentRun(
            project_id, project.display_name, question_set_id, question_set.name, status,
            max_concurrency, len(question_set.questions), started_at, completed_at,
            tuple(summaries), result_items,
        )
        state = self.load(project_id)
        state["judge_model"] = judge_model
        state["run"] = asdict(run)
        self._write(project_id, state)
        return run

    def _resolved_sources(self, project_id: str, context: dict[str, object]) -> tuple[str, ...]:
        raw = context.get("sources")
        if not isinstance(raw, list) or not raw:
            return ()
        evidence = EvidenceService(self.projects).from_context(project_id, context)
        metadata = {
            item.text_unit_id: getattr(item, "document_name", None) or item.document_id
            for item in EvidenceService(self.projects).metadata.load(project_id)
        }
        return tuple(dict.fromkeys(metadata.get(item.text_unit_id, item.document_id) for item in evidence))

    def _graphrag_query(self, project_id: str, question: str, method: str, model: str) -> tuple[str, dict[str, object]]:
        if method not in EXPERIMENT_METHODS:
            raise ProjectError("不支援的 GraphRAG 檢索策略")
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
            config["model"] = model
        api_key = self.connections.apply_to_environment(project_id)
        del api_key  # GraphRAG reads the standard API key environment variable.

        from graphrag.cli.query import (
            run_basic_search, run_drift_search, run_global_search, run_local_search,
        )

        with tempfile.TemporaryDirectory(prefix="graphrag-experiment-") as temporary:
            root = Path(temporary)
            (root / "settings.yaml").write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
            (root / "output").symlink_to(project_path / "output", target_is_directory=True)
            common = dict(data_dir=None, root_dir=root, response_type="Multiple Paragraphs", streaming=False,
                          query=question, verbose=False)
            if method == "local":
                answer, context = run_local_search(community_level=2, **common)
            elif method == "global":
                answer, context = run_global_search(
                    community_level=2, dynamic_community_selection=False, **common
                )
            elif method == "drift":
                answer, context = run_drift_search(community_level=2, **common)
            else:
                answer, context = run_basic_search(**common)
        serialized = QueryService._serialize_context(context)
        return str(answer), serialized

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
            if group.method not in EXPERIMENT_METHODS:
                raise ProjectError(f"{group.group_id} 的 GraphRAG 策略不支援")

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
