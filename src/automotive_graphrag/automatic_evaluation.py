"""Cost-conscious LLM judging for completed grounded answers."""

from __future__ import annotations

import csv
import inspect
import io
import json
import os
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .evaluation.classic_metrics import (
    answer_term_scores,
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
)
from .evaluation.source_scoring import (
    DEFAULT_SOURCE_WEIGHT,
    combined_score,
    source_score,
    validate_source_weight,
)
from .connections import (
    ALLOWED_CHAT_MODELS,
    GPT6_LUNA_MODEL,
    ConnectionSettings,
    JudgeConnectionSettings,
    JudgeEndpoint,
    MAX_JUDGE_MODELS,
    format_judge_models,
    is_litellm_model,
    judge_model_label,
    parse_judge_models,
    judge_completion_api_params,
    litellm_alias,
)
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, QuestionSetService
from .reviews import ReviewService


PROMPT_VERSION = "answer-match-judge-v2"
JUDGMENT_RESULTS = {"correct", "incorrect"}
JudgeClient = Callable[..., object]


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """One judge model's verdict for one question; ``score`` is None when the request failed."""

    model: str
    score: float | None
    reason: str = ""
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationItem:
    question_id: str
    is_correct: bool
    judge_reason: str
    judges: tuple[JudgeScore, ...] = ()
    average_score: float | None = None
    manual: bool = False
    source_score: float | None = None
    """Cited-source vs gold answer-source score (0–1); None when not applicable."""
    combined_score: float | None = None
    """Weighted judge score and source score (0–1); equals ``score`` when the source score is N/A."""

    @property
    def score(self) -> float:
        """Question score (0–1): judge average, or the manual override when edited."""
        if self.average_score is not None:
            return self.average_score
        return 1.0 if self.is_correct else 0.0

    @property
    def final_score(self) -> float:
        """Combined score, falling back to the judge score for records without one."""
        return self.combined_score if self.combined_score is not None else self.score


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationGrade:
    question_id: str
    score: int
    judge_reason: str


@dataclass(frozen=True, slots=True)
class ClassicMetricsItem:
    question_id: str
    answer_em: float
    answer_f1: float
    answer_term_hit: float | None = None
    answer_term_coverage: float | None = None


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationResult:
    project_id: str
    question_set_id: str
    evaluated_at: str
    model: str
    prompt_version: str
    judge_prompt: str
    question_count: int
    correct_count: int
    items: tuple[AutomaticEvaluationItem, ...]
    classic_items: tuple[ClassicMetricsItem, ...] = ()
    average_answer_em: float | None = None
    average_answer_f1: float | None = None
    average_answer_term_hit: float | None = None
    average_answer_term_coverage: float | None = None
    enable_llm_judge: bool = True
    enable_classic_metrics: bool = False
    judge_error: str | None = None
    models: tuple[str, ...] = ()
    average_score: float | None = None
    judge_average_scores: dict[str, float] = field(default_factory=dict)
    disagreement_count: int = 0
    source_weight: float = DEFAULT_SOURCE_WEIGHT
    average_source_score: float | None = None
    average_combined_score: float | None = None


def average_judge_score(judges: Sequence[JudgeScore]) -> float | None:
    scores = [item.score for item in judges if item.score is not None]
    return round(sum(scores) / len(scores), 4) if scores else None


def combined_judge_reason(judges: Sequence[JudgeScore], score_format: Callable[[float], str] = lambda value: f"{value:g}") -> str:
    """Single judge keeps its plain reason; a panel lists each model's score and reason."""
    if len(judges) == 1 and judges[0].score is not None:
        return judges[0].reason
    return "\n".join(
        f"{judge_model_label(item.model)}：{score_format(item.score)}｜{item.reason}"
        if item.score is not None
        else f"{judge_model_label(item.model)}：失敗｜{item.error or ''}"
        for item in judges
    )


def judge_panel_summary(items: Sequence[object], judges_of: Callable[[object], Sequence[JudgeScore]]) -> tuple[dict[str, float], int]:
    """Per-judge average score and the number of questions where successful judges disagree."""
    totals: dict[str, list[float]] = {}
    disagreements = 0
    for item in items:
        scores = []
        for judge in judges_of(item):
            if judge.score is None:
                continue
            totals.setdefault(judge.model, []).append(judge.score)
            scores.append(judge.score)
        if len(set(scores)) > 1:
            disagreements += 1
    return {model: round(sum(values) / len(values), 4) for model, values in totals.items()}, disagreements


def judge_scores_from_dicts(values: object) -> tuple[JudgeScore, ...]:
    if not isinstance(values, (list, tuple)):
        return ()
    result = []
    for value in values:
        if isinstance(value, JudgeScore):
            result.append(value)
        elif isinstance(value, dict) and isinstance(value.get("model"), str):
            score = value.get("score")
            result.append(JudgeScore(
                value["model"],
                float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                str(value.get("reason") or ""),
                value.get("error") if isinstance(value.get("error"), str) else None,
            ))
    return tuple(result)


class AutomaticEvaluationService:
    def __init__(
        self,
        projects: ProjectStore,
        question_sets: QuestionSetService | None = None,
        connections: ConnectionSettings | None = None,
        client: JudgeClient | None = None,
        judge_connections: JudgeConnectionSettings | None = None,
    ) -> None:
        self.projects = projects
        self.question_sets = question_sets or QuestionSetService(projects)
        self.connections = connections or ConnectionSettings(projects.root)
        self.judge_connections = judge_connections or JudgeConnectionSettings(projects.root)
        self.client = client or self._openai_chat

    def judge_model_choices(self) -> tuple[str, ...]:
        """OpenAI judge models plus LiteLLM proxy aliases once the judge connection is saved."""
        return (*ALLOWED_CHAT_MODELS, *self.judge_connections.available_models())

    def supports_judge_model(self, model: object) -> bool:
        if model in ALLOWED_CHAT_MODELS:
            return True
        return is_litellm_model(model) and model in self.judge_connections.available_models()

    def supports_judge_reasoning(self, model: object) -> bool:
        """Whether a reasoning_effort selector applies to this judge model."""
        return model == GPT6_LUNA_MODEL or self.judge_connections.supports_reasoning(model)

    def _judge_endpoint(self, project_id: str, model: str) -> JudgeEndpoint:
        if model in ALLOWED_CHAT_MODELS:
            api_key = self.connections.apply_to_environment(project_id)
            return JudgeEndpoint(self.connections.get_api_base_url(), api_key, model == GPT6_LUNA_MODEL)
        return self.judge_connections.endpoint_for(model)

    def _call_judge(self, project_id: str, model: str, prompt: str, reasoning_effort: str) -> object:
        endpoint = self._judge_endpoint(project_id, model)
        # LiteLLM models that do not report reasoning support never receive reasoning_effort.
        effort = reasoning_effort if endpoint.supports_reasoning or model in ALLOWED_CHAT_MODELS else None
        try:
            args = (endpoint.api_base_url, endpoint.api_key, model, prompt)
            parameters = inspect.signature(self.client).parameters
            kwargs = {}
            if "reasoning_effort" in parameters:
                kwargs["reasoning_effort"] = effort
            if endpoint.extra_headers and "extra_headers" in parameters:
                kwargs["extra_headers"] = endpoint.extra_headers
            return self.client(*args, **kwargs)
        except ProjectError:
            raise
        except Exception as exc:
            raise ProjectError(f"答案評判 API 呼叫失敗：{exc}") from exc

    def evaluate(
        self,
        project_id: str,
        question_set_id: str,
        rerun_answers: bool = False,
        only_previous_failures: bool = False,
        model: str | Sequence[str] | None = None,
        concurrency: int = 3,
        progress_callback: Callable[[int, int], None] | None = None,
        reasoning_effort: str = "medium",
        enable_llm_judge: bool = True,
        enable_classic_metrics: bool = False,
        source_weight: float = DEFAULT_SOURCE_WEIGHT,
    ) -> AutomaticEvaluationResult:
        """Judge saved answers; with several models every judge scores each question and the
        question score is the average of the judges that succeeded. The combined score mixes
        that judge score with the cited-source score by ``source_weight``."""
        if not enable_llm_judge and not enable_classic_metrics:
            raise ProjectError("至少啟用一種評測方式")
        try:
            source_weight = validate_source_weight(source_weight)
        except ValueError as exc:
            raise ProjectError(str(exc)) from exc
        if concurrency < 1 or concurrency > 32:
            raise ProjectError("評測並行數必須介於 1 到 32")
        previous = self.last_result(project_id, question_set_id)
        selected_ids: set[str] | None = None
        if only_previous_failures and enable_llm_judge:
            if previous is None:
                raise ProjectError("尚無自動評測結果，無法只重跑失敗題")
            selected_ids = {item.question_id for item in previous.items if not item.is_correct}
            if not selected_ids:
                raise ProjectError("前次評測沒有答錯的題目")

        question_set = self.question_sets.get(project_id, question_set_id)
        eligible = [
            item
            for item in question_set.questions
            if item.reference_answer
            and (selected_ids is None or item.question_id in selected_ids)
        ]
        if not eligible:
            raise ProjectError("題目集沒有包含可供評判的正確答案")
        if rerun_answers:
            target_ids = [item.question_id for item in eligible]
            question_set = self.question_sets.run(project_id, question_set_id, selected_question_ids=target_ids)
            eligible = [item for item in question_set.questions if item.question_id in set(target_ids)]
        eligible = [item for item in eligible if item.status == "COMPLETED" and item.answer]
        if not eligible:
            raise ProjectError("沒有已完成回答且包含正確答案的題目可供評判")

        models = self.validate_judge_models(model) if enable_llm_judge else ()
        prompt = (
            "\n\n---\n\n".join(self._build_prompt([item]) for item in eligible)
            if enable_llm_judge else ""
        )
        judged_by_id: dict[str, AutomaticEvaluationItem] = {}
        failures: list[str] = []
        partial_failures: list[str] = []
        total_calls = len(eligible) * len(models)
        if progress_callback and enable_llm_judge:
            progress_callback(0, total_calls)
        if enable_llm_judge:
            verdicts: dict[str, dict[str, JudgeScore]] = {item.question_id: {} for item in eligible}
            finished = 0
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        self.evaluate_single_answer,
                        project_id,
                        item.question_id,
                        item.question,
                        item.reference_answer,
                        item.answer,
                        judge_model,
                        reasoning_effort=reasoning_effort,
                    ): (item, judge_model)
                    for item in eligible
                    for judge_model in models
                }
                for future in as_completed(futures):
                    question, judge_model = futures[future]
                    try:
                        single = future.result()
                        verdicts[question.question_id][judge_model] = JudgeScore(
                            judge_model, 1.0 if single.is_correct else 0.0, single.judge_reason
                        )
                    except Exception as exc:
                        verdicts[question.question_id][judge_model] = JudgeScore(judge_model, None, "", str(exc))
                    finished += 1
                    if progress_callback:
                        progress_callback(finished, total_calls)
            for question in eligible:
                judges = tuple(verdicts[question.question_id][judge_model] for judge_model in models)
                failed = [item for item in judges if item.score is None]
                if len(failed) == len(judges):
                    failures.append(f"{question.question_id}: " + "；".join(
                        (f"{judge_model_label(item.model)} " if len(judges) > 1 else "") + str(item.error)
                        for item in failed
                    ))
                    continue
                if failed:
                    partial_failures.append(
                        f"{question.question_id}（{'、'.join(judge_model_label(item.model) for item in failed)}）"
                    )
                average = average_judge_score(judges)
                source = self._question_source_score(question)
                judged_by_id[question.question_id] = AutomaticEvaluationItem(
                    question_id=question.question_id,
                    is_correct=average >= 0.5,
                    judge_reason=combined_judge_reason(judges),
                    judges=judges,
                    average_score=average,
                    source_score=source,
                    combined_score=combined_score(average, source, source_weight),
                )
        if failures and not enable_classic_metrics:
            raise ProjectError("自動評測部分請求失敗：" + "；".join(failures))
        if enable_llm_judge and selected_ids is not None and previous is not None:
            items_by_id = {item.question_id: item for item in previous.items}
            items_by_id.update(judged_by_id)
            # Carried-over items are re-weighted so every question uses this run's source weight.
            items = tuple(self._with_combined(item, source_weight) for item in items_by_id.values())
        else:
            items = tuple(
                judged_by_id[item.question_id]
                for item in eligible
                if item.question_id in judged_by_id
            ) if enable_llm_judge else ()
        classic_items = tuple(
            ClassicMetricsItem(
                question_id=item.question_id,
                answer_em=metric_max_over_ground_truths(
                    exact_match_score, item.answer, [item.reference_answer, *item.answer_aliases]
                ),
                answer_f1=metric_max_over_ground_truths(
                    f1_score, item.answer, [item.reference_answer, *item.answer_aliases]
                ),
                answer_term_hit=metric_max_over_ground_truths(
                    lambda prediction, truth: answer_term_scores(prediction, truth)[0],
                    item.answer, [item.reference_answer, *item.answer_aliases],
                ),
                answer_term_coverage=metric_max_over_ground_truths(
                    lambda prediction, truth: answer_term_scores(prediction, truth)[1],
                    item.answer, [item.reference_answer, *item.answer_aliases],
                ),
            )
            for item in eligible
        ) if enable_classic_metrics else ()
        if progress_callback and not enable_llm_judge:
            progress_callback(len(eligible), len(eligible))
        judge_messages = []
        if failures:
            judge_messages.append("自動評測部分請求失敗：" + "；".join(failures))
        if partial_failures:
            judge_messages.append("部分評審失敗，該題只平均成功的評審：" + "、".join(partial_failures))
        judge_averages, disagreements = judge_panel_summary(items, lambda item: item.judges)
        result = AutomaticEvaluationResult(
            project_id=project_id,
            question_set_id=question_set_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            model=format_judge_models(models) if enable_llm_judge else "—",
            prompt_version=PROMPT_VERSION,
            judge_prompt=prompt,
            question_count=len(items),
            correct_count=sum(item.is_correct for item in items),
            items=items,
            classic_items=classic_items,
            average_answer_em=(sum(item.answer_em for item in classic_items) / len(classic_items) if classic_items else None),
            average_answer_f1=(sum(item.answer_f1 for item in classic_items) / len(classic_items) if classic_items else None),
            average_answer_term_hit=(sum(item.answer_term_hit for item in classic_items) / len(classic_items) if classic_items else None),
            average_answer_term_coverage=(sum(item.answer_term_coverage for item in classic_items) / len(classic_items) if classic_items else None),
            enable_llm_judge=enable_llm_judge,
            enable_classic_metrics=enable_classic_metrics,
            judge_error="\n".join(judge_messages) or None,
            models=models,
            average_score=self._average_item_score(items),
            judge_average_scores=judge_averages,
            disagreement_count=disagreements,
            source_weight=source_weight,
            average_source_score=self._mean_present(item.source_score for item in items),
            average_combined_score=self._average_final_score(items),
        )
        self._write(result)
        return result

    @staticmethod
    def _question_source_score(question: BatchQuestion) -> float | None:
        gold = question.answer_source_evidence or question.gold_evidence
        return source_score(question.answer, question.retrieved_evidence, gold)

    @staticmethod
    def _with_combined(item: AutomaticEvaluationItem, weight: float) -> AutomaticEvaluationItem:
        return replace(item, combined_score=combined_score(item.score, item.source_score, weight))

    @staticmethod
    def _average_final_score(items: Sequence[AutomaticEvaluationItem]) -> float | None:
        return round(sum(item.final_score for item in items) / len(items), 4) if items else None

    def validate_judge_models(self, value: object) -> tuple[str, ...]:
        models = parse_judge_models(value) or (self.connections.get_chat_model(),)
        if len(models) > MAX_JUDGE_MODELS:
            raise ProjectError(f"評測模型最多選擇 {MAX_JUDGE_MODELS} 個")
        for item in models:
            if not self.supports_judge_model(item):
                raise ProjectError(self._unsupported_model_message(item))
        return models

    @staticmethod
    def _average_item_score(items: Sequence[AutomaticEvaluationItem]) -> float | None:
        return round(sum(item.score for item in items) / len(items), 4) if items else None

    def evaluate_single_answer(
        self,
        project_id: str,
        question_id: str,
        question: str,
        correct_answer: str,
        system_answer: str,
        model: str,
        evaluation_mode: str = "strict",
        reasoning_effort: str = "medium",
    ) -> AutomaticEvaluationItem:
        """Judge one answer without retrieval evidence or shared question-set mutation."""
        if not self.supports_judge_model(model):
            raise ProjectError(self._unsupported_model_message(model))
        if evaluation_mode not in {"strict", "lenient"}:
            raise ProjectError("不支援的評分方式")
        case = BatchQuestion(
            question_id=question_id,
            question=question,
            reference_answer=correct_answer,
            answer=system_answer,
        )
        prompt = self._build_prompt([case], evaluation_mode)
        response = self._call_judge(project_id, model, prompt, reasoning_effort)
        return self._parse_response(response, [case])[0]

    def evaluate_single_answer_grade(
        self, project_id: str, question_id: str, question: str, correct_answer: str,
        system_answer: str, model: str, evaluation_mode: str = "strict",
        reasoning_effort: str = "medium",
    ) -> AutomaticEvaluationGrade:
        """Three-level evaluation for retrieval experiments only (0 incorrect, 1 partial, 2 full)."""
        if not self.supports_judge_model(model):
            raise ProjectError(self._unsupported_model_message(model))
        if evaluation_mode not in {"strict", "lenient"}:
            raise ProjectError("不支援的評分方式")
        case = BatchQuestion(question_id, question, correct_answer, answer=system_answer)
        prompt = self._build_grade_prompt(case, evaluation_mode)
        response = self._call_judge(project_id, model, prompt, reasoning_effort)
        if isinstance(response, str):
            response = extract_json_object(response)
            if response is None:
                raise ProjectError("三段評測結果不是有效 JSON")
        raw = response.get("item") if isinstance(response, dict) else None
        if not isinstance(raw, dict) or raw.get("question_id") != question_id:
            raise ProjectError("三段評測結果缺少正確的 question_id")
        score = raw.get("result")
        if isinstance(score, str) and score.isdigit():
            score = int(score)
        if isinstance(score, bool) or score not in {0, 1, 2}:
            raise ProjectError(f"{question_id} 的 result 必須是 0、1 或 2")
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ProjectError(f"{question_id} 缺少評判理由")
        return AutomaticEvaluationGrade(question_id, score, reason.strip())

    @staticmethod
    def _build_grade_prompt(case: BatchQuestion, evaluation_mode: str) -> str:
        mode = "採較寬鬆語意理解，同義改寫及不改變正解的合理補充可接受。" if evaluation_mode == "lenient" else "採嚴格比對，不可缺少正解的必要資訊或步驟，也不可增加會導致錯誤的實質內容。"
        return (
            "你是答案比對員，只根據題目、正確答案和系統回答評判，不得使用外部知識或檢索證據。"
            f"{mode}請分三等級：0=錯誤（核心內容錯誤、矛盾，或幾乎未答對）；"
            "1=部分對（有正確核心內容，但仍缺少重要資訊／必要步驟，或有局部錯誤）；"
            "2=全對（正確答案的所有必要資訊、條件與步驟均完整且無實質錯誤）。"
            "流程題若漏任何必要步驟，不可評 2；若已答對部分步驟評 1。"
            '只輸出 JSON：{"item":{"question_id":"...","result":0或1或2,"reason":"..."}}。'
            f"理由簡短說明評級依據。案例：{json.dumps({'question_id':case.question_id,'question':case.question,'correct_answer':case.reference_answer,'system_answer':case.answer}, ensure_ascii=False)}"
        )

    def last_result(self, project_id: str, question_set_id: str) -> AutomaticEvaluationResult | None:
        try:
            value = json.loads(self._path(project_id, question_set_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError as exc:
            raise ProjectError("自動評測紀錄格式錯誤") from exc
        # Migrate records written by the former scored/evidence-based judge.
        raw_items = value.pop("items", [])
        migrated_items = []
        for item in raw_items:
            if "is_correct" not in item:
                item = {
                    "question_id": item["question_id"],
                    "is_correct": item.get("answer_score", 0) >= 4,
                    "judge_reason": item.get("judge_reason", "舊版評測結果；請重新評判"),
                }
            item = {key: value for key, value in item.items() if key in AutomaticEvaluationItem.__dataclass_fields__}
            item["judges"] = judge_scores_from_dicts(item.get("judges"))
            migrated_items.append(AutomaticEvaluationItem(**item))
        raw_classic_items = value.pop("classic_items", [])
        classic_items = tuple(ClassicMetricsItem(**item) for item in raw_classic_items)
        allowed_fields = set(AutomaticEvaluationResult.__dataclass_fields__) - {"items"}
        value = {key: field for key, field in value.items() if key in allowed_fields}
        value.setdefault("correct_count", sum(item.is_correct for item in migrated_items))
        value["classic_items"] = classic_items
        value.setdefault(
            "average_answer_em",
            sum(item.answer_em for item in classic_items) / len(classic_items) if classic_items else None,
        )
        value.setdefault(
            "average_answer_f1",
            sum(item.answer_f1 for item in classic_items) / len(classic_items) if classic_items else None,
        )
        value.setdefault(
            "average_answer_term_hit",
            self._mean_present(item.answer_term_hit for item in classic_items),
        )
        value.setdefault(
            "average_answer_term_coverage",
            self._mean_present(item.answer_term_coverage for item in classic_items),
        )
        value["items"] = tuple(migrated_items)
        value["models"] = tuple(value.get("models") or parse_judge_models(value.get("model")))
        if value.get("average_score") is None and value.get("enable_llm_judge", True):
            value["average_score"] = self._average_item_score(migrated_items)
        if value.get("average_combined_score") is None and value.get("enable_llm_judge", True):
            # Records written before source scoring: combined equals the judge score.
            value["average_combined_score"] = self._average_final_score(migrated_items)
        if "judge_average_scores" not in value:
            value["judge_average_scores"], value["disagreement_count"] = judge_panel_summary(
                migrated_items, lambda item: item.judges
            )
        return AutomaticEvaluationResult(**value)

    def update_manual_results(
        self, project_id: str, question_set_id: str, decisions: dict[str, object]
    ) -> AutomaticEvaluationResult:
        """Save manual question scores (0–1) as an override; each judge's own scores stay intact."""
        result = self.last_result(project_id, question_set_id)
        if result is None:
            raise ProjectError("尚無評測結果可供修改")
        unknown = set(decisions) - {item.question_id for item in result.items}
        if unknown:
            raise ProjectError("評測結果找不到題號：" + ", ".join(sorted(unknown)))
        items = []
        for item in result.items:
            if item.question_id not in decisions:
                items.append(item)
                continue
            score = self._parse_manual_score(item.question_id, decisions[item.question_id])
            if abs(score - item.score) < 0.005:
                items.append(item)
                continue
            items.append(self._with_combined(
                replace(item, is_correct=score >= 0.5, average_score=score, manual=True), result.source_weight
            ))
        updated = replace(
            result,
            question_count=len(items),
            correct_count=sum(item.is_correct for item in items),
            items=tuple(items),
            average_score=self._average_item_score(items),
            average_combined_score=self._average_final_score(items),
        )
        self._write(updated)
        return updated

    @staticmethod
    def _parse_manual_score(question_id: str, value: object) -> float:
        if value in {"正確", True}:
            return 1.0
        if value in {"錯誤", False}:
            return 0.0
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ProjectError(f"{question_id} 的平均分數必須是 0 到 1 的數字") from exc
        if not 0 <= score <= 1:
            raise ProjectError(f"{question_id} 的平均分數必須介於 0 到 1")
        return round(score, 4)

    @staticmethod
    def _mean_present(values) -> float | None:
        present = [value for value in values if value is not None]
        return sum(present) / len(present) if present else None

    def clear_result(self, project_id: str, question_set_id: str) -> None:
        path = self._path(project_id, question_set_id)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise ProjectError("無法清除舊評測結果") from exc

    def export(self, project_id: str, question_set_id: str) -> tuple[Path, Path]:
        result = self.last_result(project_id, question_set_id)
        if result is None:
            raise ProjectError("尚未執行自動評測")
        directory = self.projects.path_for(project_id) / "exports"
        json_path = directory / f"{question_set_id}-automatic-evaluation.json"
        csv_path = directory / f"{question_set_id}-automatic-evaluation.csv"
        reviews = ReviewService(self.projects, self.question_sets).records(project_id, question_set_id)
        review_by_id = {item.question_id: item for item in reviews}
        payload = asdict(result)
        payload["human_reviews"] = [asdict(item) for item in reviews if item.human_label is not None]
        judge_by_id = {item.question_id: item for item in result.items}
        classic_by_id = {item.question_id: item for item in result.classic_items}
        payload["question_results"] = [
            {
                "question_id": question_id,
                "is_correct": judge_by_id[question_id].is_correct if question_id in judge_by_id else None,
                "average_score": judge_by_id[question_id].score if question_id in judge_by_id else None,
                "source_score": judge_by_id[question_id].source_score if question_id in judge_by_id else None,
                "combined_score": judge_by_id[question_id].final_score if question_id in judge_by_id else None,
                "manual": judge_by_id[question_id].manual if question_id in judge_by_id else False,
                "judges": [asdict(judge) for judge in judge_by_id[question_id].judges] if question_id in judge_by_id else [],
                "judge_reason": judge_by_id[question_id].judge_reason if question_id in judge_by_id else "",
                "answer_em": classic_by_id[question_id].answer_em if question_id in classic_by_id else None,
                "answer_f1": classic_by_id[question_id].answer_f1 if question_id in classic_by_id else None,
                "answer_term_hit": classic_by_id[question_id].answer_term_hit if question_id in classic_by_id else None,
                "answer_term_coverage": classic_by_id[question_id].answer_term_coverage if question_id in classic_by_id else None,
            }
            for question_id in dict.fromkeys([*judge_by_id, *classic_by_id])
        ]
        payload["summary"] = {
            "question_count": result.question_count,
            "judge_evaluated_count": len(result.items),
            "correct_count": result.correct_count,
            "judge_models": list(result.models),
            "average_score": result.average_score,
            "source_weight": result.source_weight,
            "average_source_score": result.average_source_score,
            "average_combined_score": result.average_combined_score,
            "judge_average_scores": result.judge_average_scores,
            "disagreement_count": result.disagreement_count,
            "average_answer_em": result.average_answer_em,
            "average_answer_f1": result.average_answer_f1,
            "average_answer_term_hit": result.average_answer_term_hit,
            "average_answer_term_coverage": result.average_answer_term_coverage,
            "enable_llm_judge": result.enable_llm_judge,
            "enable_classic_metrics": result.enable_classic_metrics,
        }
        self._atomic_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        judge_columns = [
            column for judge_model in result.models
            for column in (f"judge_score:{judge_model}", f"judge_reason:{judge_model}")
        ]
        fieldnames = [
            "question_id", "is_correct", "average_score", "source_score", "combined_score", "manual", "judge_reason",
            *judge_columns,
            "answer_em",
            "answer_f1",
            "answer_term_hit",
            "answer_term_coverage",
            "human_label",
            "reviewer_note",
            "reviewed_at",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        judge_by_id = {item.question_id: item for item in result.items}
        classic_by_id = {item.question_id: item for item in result.classic_items}
        for question_id in dict.fromkeys([*judge_by_id, *classic_by_id]):
            item = judge_by_id.get(question_id)
            classic = classic_by_id.get(question_id)
            row = {
                "question_id": question_id,
                "is_correct": item.is_correct if item else None,
                "average_score": item.score if item else None,
                "source_score": item.source_score if item else None,
                "combined_score": item.final_score if item else None,
                "manual": item.manual if item else False,
                "judge_reason": item.judge_reason if item else "",
            }
            for judge in (item.judges if item else ()):
                if f"judge_score:{judge.model}" in judge_columns:
                    row[f"judge_score:{judge.model}"] = judge.score
                    row[f"judge_reason:{judge.model}"] = judge.reason or judge.error or ""
            row["answer_em"] = classic.answer_em if classic else None
            row["answer_f1"] = classic.answer_f1 if classic else None
            row["answer_term_hit"] = classic.answer_term_hit if classic else None
            row["answer_term_coverage"] = classic.answer_term_coverage if classic else None
            review = review_by_id.get(question_id)
            row.update(
                {
                    "human_label": review.human_label if review else None,
                    "reviewer_note": review.reviewer_note if review else "",
                    "reviewed_at": review.reviewed_at if review else None,
                }
            )
            writer.writerow(row)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    @staticmethod
    def _build_prompt(questions: list[BatchQuestion], evaluation_mode: str = "strict") -> str:
        cases = [
            {
                "question_id": item.question_id,
                "question": item.question,
                "correct_answer": item.reference_answer,
                "system_answer": item.answer,
            }
            for item in questions
        ]
        if evaluation_mode == "lenient":
            criteria = (
                "使用寬鬆的語意比對：若系統回答已表達正確答案的全部核心資訊與必要步驟，判為 correct。"
                "必須接受同義詞、語序或格式差異、簡潔改寫，以及不改變正解含義的合理補充。\n"
                "不要要求逐字包含正確答案，也不要只因回答比正確答案更詳細、包含正確答案未提到的背景或補充資訊，就判為 incorrect。"
                "只有在回答漏掉核心資訊／必要步驟、與正確答案的核心內容矛盾，或額外內容明確使答案變成錯誤時，才判為 incorrect。\n"
                "流程題仍須涵蓋所有必要步驟；只有當順序會影響結果時才要求相同順序。非流程題以是否傳達正確答案的核心意思為準。\n"
            )
        else:
            criteria = (
                "若系統回答與正確答案表達的內容一致，且完整包含所有必要步驟、條件、順序與要求，判為 correct。\n"
                "只要缺少必要內容／步驟、步驟順序錯誤、與正確答案矛盾，或加入正確答案未支持的額外步驟／實質資訊，判為 incorrect。"
                "不要求字面完全相同；同義改寫可接受，但不能因此省略或新增實質內容。非流程型答案也不得缺漏或添加實質主張。\n"
            )
        return (
            "你是答案比對員。只根據每筆資料中的題目、正確答案和系統回答判斷，不得使用外部知識或任何檢索證據。\n"
            f"{criteria}"
            "只輸出 JSON object，格式為 "
            '{"items":[{"question_id":"...","result":"correct或incorrect","reason":"..."}]}。'
            "每個輸入題號必須恰好出現一次。reason 簡短指出符合之處，或具體缺漏、矛盾、錯誤。\n"
            f"評測案例：{json.dumps(cases, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_response(
        response: object, questions: list[BatchQuestion]
    ) -> list[AutomaticEvaluationItem]:
        if isinstance(response, str):
            response = extract_json_object(response)
            if response is None:
                raise ProjectError("自動評測結果不是有效 JSON")
        if not isinstance(response, dict) or not isinstance(response.get("items"), list):
            raise ProjectError("自動評測結果缺少 items 陣列")
        raw_items = response["items"]
        question_by_id = {item.question_id: item for item in questions}
        if len(raw_items) != len(question_by_id):
            raise ProjectError("Judge 回傳題數與預期不符")
        result: list[AutomaticEvaluationItem] = []
        seen: set[str] = set()
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ProjectError("Judge 回傳項目格式錯誤")
            question_id = raw.get("question_id")
            judgment = raw.get("result")
            reason = raw.get("reason")
            if not isinstance(question_id, str) or question_id not in question_by_id or question_id in seen:
                raise ProjectError("Judge 回傳未知或重複的 question_id")
            if not isinstance(judgment, str) or judgment not in JUDGMENT_RESULTS:
                raise ProjectError(f"{question_id} 的 result 必須是 correct 或 incorrect")
            if not isinstance(reason, str) or not reason.strip():
                raise ProjectError(f"{question_id} 缺少 Judge reason")
            result.append(
                AutomaticEvaluationItem(
                    question_id=question_id,
                    is_correct=judgment == "correct",
                    judge_reason=reason.strip(),
                )
            )
            seen.add(question_id)
        return result

    def _unsupported_model_message(self, model: object) -> str:
        if is_litellm_model(model) and not self.judge_connections.is_configured():
            return "尚未完成 LiteLLM 評測模型設定，請先到「0 專案與連線設定」儲存評測模型"
        if is_litellm_model(model):
            return f"評測模型不在 LiteLLM 評測模型設定的清單：{litellm_alias(str(model))}"
        return f"不支援的評判模型：{model}"

    @staticmethod
    def _openai_chat(
        api_base_url: str, api_key: str, model: str, prompt: str, reasoning_effort: str | None = "medium",
        extra_headers: dict[str, str] | None = None,
    ) -> object:
        litellm = is_litellm_model(model)
        payload: dict[str, object] = {
            "model": litellm_alias(model) if litellm else model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        payload.update(judge_completion_api_params(model, 4000, reasoning_effort))
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if litellm:
            headers.update(extra_headers or {})

        def post(body: dict[str, object]) -> dict[str, object]:
            request = urllib.request.Request(
                f"{api_base_url}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))

        try:
            try:
                value = post(payload)
            except urllib.error.HTTPError as exc:
                # Some providers behind LiteLLM reject JSON mode; retry once without it
                # and recover the JSON object from the plain-text reply.
                if not litellm or exc.code != 400:
                    raise
                value = post({key: item for key, item in payload.items() if key != "response_format"})
        except urllib.error.HTTPError as exc:
            raise ProjectError(f"自動評測 API 回應 HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProjectError("自動評測 API 無法連線或回應格式錯誤") from exc
        try:
            content = value["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProjectError("自動評測 API 回應缺少有效 JSON 內容") from exc
        parsed = extract_json_object(content) if isinstance(content, str) else None
        if parsed is None:
            raise ProjectError("自動評測 API 回應缺少有效 JSON 內容")
        return parsed

    def _path(self, project_id: str, question_set_id: str) -> Path:
        self.question_sets.get(project_id, question_set_id)
        return self.projects.path_for(project_id) / "runs" / f"automatic-evaluation-{question_set_id}.json"

    def _write(self, result: AutomaticEvaluationResult) -> None:
        self._atomic_text(
            self._path(result.project_id, result.question_set_id),
            json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n",
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


def extract_json_object(text: str) -> dict[str, object] | None:
    """Parse a JSON object, falling back to the first object embedded in plain text.

    Models without JSON mode may wrap the verdict in prose or Markdown code fences.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None
