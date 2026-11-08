"""Cost-conscious LLM judging for completed grounded answers."""

from __future__ import annotations

import csv
import inspect
import io
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit
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
    DOCUMENT_ONLY_SCORE,
    combined_score,
    cited_sources,
    source_score,
    strip_citations,
    validate_source_weight,
)
from .connections import (
    ConnectionSettings,
    MAX_JUDGE_MODELS,
    chat_completion_api_params,
    extract_json_object,
    format_judge_models,
    judge_model_label,
    llm_request_headers,
    parse_judge_models,
)
from .documents import DocumentService
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, QuestionSetService
from .reviews import ReviewService


PROMPT_VERSION = "answer-rubric-judge-v2"
RUBRIC_MAX_LEVEL = 5
CORRECT_SCORE = 0.8
"""Question score (0–1) from which an answer counts as correct: rubric level 4."""
EXTRA_CONTENT_LEVEL = 3
"""The rubric level that holds answers covering the correct answer but wrong through added content."""
RUBRIC_PROMPT = """Grade the system answer against the correct answer with this rubric. Pick the highest level whose description fully applies, then apply the caps.

Level 5 (100%): Contains all required information, steps, and conditions, with no substantive error. Paraphrases and format differences are acceptable.
Level 4 (80%): All core content is correct. Only a minor detail is missing, or a small deviation does not affect the outcome.
Level 3 (60%): Either (a) the answer states all of the correct answer accurately, and its only fault is content the correct answer does not mention, or (b) the core is correct but one important item is missing.
Level 2 (40%): About half of the required content is correct.
Level 1 (20%): Related to the question, but only scattered content is correct.
Level 0 (0%): Contradicts the correct answer throughout, is off-topic, or gives no answer.

Caps override the levels above. A cap only limits the level and never raises it:
- A procedure missing a required step scores at most level 3.
- Steps in the wrong order, where the order affects the outcome, score at most level 2.
- Core content that contradicts the correct answer scores at most level 1. Changing a value, condition, or step that the correct answer states is a contradiction, not added content, so it can never be level 3 case (a).
"""
STRICT_RUBRIC_MODE = (
    "Mode: strict. Substantive steps or claims that the correct answer does not support "
    "score at most level 3, even when everything else is correct.\n"
)
LENIENT_RUBRIC_MODE = (
    "Mode: lenient. Accept synonyms, concise rewording, format changes, and a different order "
    "when order does not affect the outcome. Do not deduct for detail the correct answer does "
    "not mention unless that detail makes the answer wrong.\n"
)
RUBRIC_OUTPUT = """Set "extra_content_error" to true only for level 3 case (a). Otherwise set it to false.
Write "reason" in Traditional Chinese. Begin it with the level number, then justify that level: cite the specific content that matches the correct answer, cite what is missing, wrong, or added, and name any cap you applied. Do not only restate the rubric.
Output only a JSON object in this format:
{"item":{"question_id":"...","level":0,"extra_content_error":false,"reason":"..."}}
"""
logger = logging.getLogger(__name__)
JudgeClient = Callable[..., object]


@dataclass(frozen=True, slots=True)
class JudgeScore:
    """One judge model's verdict for one question; ``score`` is None when the request failed."""

    model: str
    score: float | None
    reason: str = ""
    error: str | None = None
    extra_content_error: bool = False
    """The answer covers the correct answer and lost credit only because of added content."""


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
    """Judge score plus the source bonus (0–1); equals ``score`` when no bonus applies."""
    extra_content_error: bool = False

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

    @property
    def source_bonus_share(self) -> float:
        """Share of judges granting the source bonus; a manual score is final and gets none."""
        return 0.0 if self.manual else extra_error_share(self.judges)


@dataclass(frozen=True, slots=True)
class AutomaticEvaluationGrade:
    """One judge's rubric verdict for one answer."""

    question_id: str
    level: int
    judge_reason: str
    extra_content_error: bool = False

    @property
    def rubric_score(self) -> float:
        """Rubric level on the 0–1 scale."""
        return round(self.level / RUBRIC_MAX_LEVEL, 4)

    @property
    def score(self) -> float:
        """Rubric level on the 0–2 scale that retrieval experiments store."""
        return round(self.rubric_score * 2, 4)


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


def extra_error_share(judges: Sequence[JudgeScore]) -> float:
    """Share of successful judges reporting that only added content made the answer wrong."""
    scored = [item for item in judges if item.score is not None]
    return sum(item.extra_content_error for item in scored) / len(scored) if scored else 0.0


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


def project_document_only_score(projects: ProjectStore, project_id: str) -> float:
    """Credit for citing the right document on the wrong page: none when the graph has one document."""
    return 0.0 if len(DocumentService(projects).indexed_documents(project_id)) == 1 else DOCUMENT_ONLY_SCORE


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
                value.get("extra_content_error") is True,
            ))
    return tuple(result)


class AutomaticEvaluationService:
    def __init__(
        self,
        projects: ProjectStore,
        question_sets: QuestionSetService | None = None,
        connections: ConnectionSettings | None = None,
        client: JudgeClient | None = None,
    ) -> None:
        self.projects = projects
        self.question_sets = question_sets or QuestionSetService(projects)
        self.connections = connections or ConnectionSettings(projects.root)
        self.client = client or self._openai_chat

    def judge_model_choices(self) -> tuple[str, ...]:
        return self.connections.available_chat_models()

    def supports_judge_model(self, model: object) -> bool:
        return model in self.connections.available_chat_models()

    def supports_judge_reasoning(self, model: object) -> bool:
        """Whether a reasoning_effort selector applies to this judge model."""
        return self.connections.supports_reasoning(model)

    def _call_judge(self, project_id: str, model: str, prompt: str, reasoning_effort: str) -> object:
        api_key = self.connections.apply_to_environment(project_id)
        effort = self.connections.reasoning_effort_for(model, reasoning_effort)
        try:
            args = (self.connections.get_api_base_url(), api_key, model, prompt)
            if "reasoning_effort" in inspect.signature(self.client).parameters:
                return self.client(*args, reasoning_effort=effort)
            return self.client(*args)
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
            "\n\n---\n\n".join(self._build_prompt(item) for item in eligible)
            if enable_llm_judge else ""
        )
        judged_by_id: dict[str, AutomaticEvaluationItem] = {}
        document_only_score = project_document_only_score(self.projects, project_id)
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
                            judge_model, single.score, single.judge_reason,
                            extra_content_error=single.extra_content_error,
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
                source = self._question_source_score(question, document_only_score)
                judged_by_id[question.question_id] = AutomaticEvaluationItem(
                    question_id=question.question_id,
                    is_correct=average >= CORRECT_SCORE,
                    judge_reason=combined_judge_reason(judges),
                    judges=judges,
                    average_score=average,
                    source_score=source,
                    combined_score=combined_score(average, source, source_weight, extra_error_share(judges)),
                )
        if failures and not enable_classic_metrics:
            raise ProjectError("自動評測部分請求失敗：" + "；".join(failures))
        if enable_llm_judge and selected_ids is not None and previous is not None:
            items_by_id = {item.question_id: item for item in previous.items}
            items_by_id.update(judged_by_id)
            # Carried-over items are rescored so every question uses this run's source weight.
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
    def _question_source_score(question: BatchQuestion, document_only_score: float) -> float | None:
        gold = question.answer_source_evidence or question.gold_evidence
        return source_score(question.answer, question.retrieved_evidence, gold, document_only_score)

    @staticmethod
    def _with_combined(item: AutomaticEvaluationItem, weight: float) -> AutomaticEvaluationItem:
        return replace(
            item, combined_score=combined_score(item.score, item.source_score, weight, item.source_bonus_share)
        )

    @staticmethod
    def _average_final_score(items: Sequence[AutomaticEvaluationItem]) -> float | None:
        return round(sum(item.final_score for item in items) / len(items), 4) if items else None

    def validate_judge_models(self, value: object) -> tuple[str, ...]:
        models = parse_judge_models(value)
        if not models:
            if value is not None:
                raise ProjectError("請至少選擇一個評測模型")
            models = (self.connections.get_chat_model(),)
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
        verdict = self._judge_answer(
            project_id,
            BatchQuestion(
                question_id=question_id,
                question=question,
                reference_answer=correct_answer,
                answer=system_answer,
            ),
            model, evaluation_mode, reasoning_effort,
        )
        return AutomaticEvaluationItem(
            question_id=question_id,
            is_correct=verdict.rubric_score >= CORRECT_SCORE,
            judge_reason=verdict.judge_reason,
            average_score=verdict.rubric_score,
            extra_content_error=verdict.extra_content_error,
        )

    def evaluate_single_answer_grade(
        self, project_id: str, question_id: str, question: str, correct_answer: str,
        system_answer: str, model: str, evaluation_mode: str = "strict",
        reasoning_effort: str = "medium",
    ) -> AutomaticEvaluationGrade:
        """Rubric verdict for retrieval experiments, which store its 0–2 ``score``."""
        if not self.supports_judge_model(model):
            raise ProjectError(self._unsupported_model_message(model))
        case = BatchQuestion(question_id, question, correct_answer, answer=system_answer)
        return self._judge_answer(project_id, case, model, evaluation_mode, reasoning_effort)

    def _judge_answer(
        self, project_id: str, case: BatchQuestion, model: str, evaluation_mode: str, reasoning_effort: str,
    ) -> AutomaticEvaluationGrade:
        if evaluation_mode not in {"strict", "lenient"}:
            raise ProjectError("不支援的評分方式")
        response = self._call_judge(project_id, model, self._build_prompt(case, evaluation_mode), reasoning_effort)
        return self._parse_verdict(response, case.question_id)

    @staticmethod
    def _build_prompt(case: BatchQuestion, evaluation_mode: str = "strict") -> str:
        evaluation_case = {
            "question_id": case.question_id,
            "question": case.question,
            "correct_answer": case.reference_answer,
            "system_answer": strip_citations(case.answer),
        }
        return (
            "You are an answer grader. Judge only from the question, correct answer, and system answer "
            "below. Do not use outside knowledge or retrieval evidence.\n\n"
            f"{RUBRIC_PROMPT}\n"
            f"{LENIENT_RUBRIC_MODE if evaluation_mode == 'lenient' else STRICT_RUBRIC_MODE}\n"
            f"{RUBRIC_OUTPUT}\n"
            f"Evaluation case: {json.dumps(evaluation_case, ensure_ascii=False)}"
        )

    @staticmethod
    def _parse_verdict(response: object, question_id: str) -> AutomaticEvaluationGrade:
        if isinstance(response, str):
            response = extract_json_object(response)
            if response is None:
                raise ProjectError("自動評測結果不是有效 JSON")
        raw = response.get("item") if isinstance(response, dict) else None
        if not isinstance(raw, dict) or raw.get("question_id") != question_id:
            raise ProjectError("自動評測結果缺少正確的 question_id")
        level = raw.get("level")
        if isinstance(level, str) and level.isdigit():
            level = int(level)
        if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= RUBRIC_MAX_LEVEL:
            raise ProjectError(f"{question_id} 的 level 必須是 0 到 {RUBRIC_MAX_LEVEL} 的整數")
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ProjectError(f"{question_id} 缺少評分理由")
        return AutomaticEvaluationGrade(
            question_id, level, reason.strip(),
            level == EXTRA_CONTENT_LEVEL and raw.get("extra_content_error") is True,
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
        source_weight = float(value.get("source_weight", DEFAULT_SOURCE_WEIGHT))
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
            # Rescored on load so records from the former weighted formula match the current rule.
            migrated_items.append(self._with_combined(AutomaticEvaluationItem(**item), source_weight))
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
        if value.get("enable_llm_judge", True):
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
                replace(item, is_correct=score >= CORRECT_SCORE, average_score=score, manual=True), result.source_weight
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
        question_set = self.question_sets.get(project_id, question_set_id)
        question_by_id = {item.question_id: item for item in question_set.questions}
        judge_by_id = {item.question_id: item for item in result.items}
        classic_by_id = {item.question_id: item for item in result.classic_items}
        payload["question_results"] = [
            {
                "question_id": question_id,
                "question": question_by_id[question_id].question if question_id in question_by_id else "",
                "expected_answer": question_by_id[question_id].reference_answer if question_id in question_by_id else "",
                "system_answer": strip_citations(question_by_id[question_id].answer) if question_id in question_by_id else "",
                "raw_system_answer": question_by_id[question_id].answer if question_id in question_by_id else "",
                "system_answer_sources": cited_sources(
                    question_by_id[question_id].answer, question_by_id[question_id].retrieved_evidence,
                ) if question_id in question_by_id else [],
                "answer_sources": [
                    asdict(source)
                    for source in (
                        question_by_id[question_id].answer_source_evidence
                        or question_by_id[question_id].gold_evidence
                    )
                ] if question_id in question_by_id else [],
                "retrieved_evidence": [
                    asdict(evidence) for evidence in question_by_id[question_id].retrieved_evidence
                ] if question_id in question_by_id else [],
                "routed_method": question_by_id[question_id].routed_method if question_id in question_by_id else "",
                "routing_reason": question_by_id[question_id].routing_reason if question_id in question_by_id else "",
                "duration_seconds": question_by_id[question_id].duration_seconds if question_id in question_by_id else None,
                "is_correct": judge_by_id[question_id].is_correct if question_id in judge_by_id else None,
                "average_score": judge_by_id[question_id].score if question_id in judge_by_id else None,
                "source_score": judge_by_id[question_id].source_score if question_id in judge_by_id else None,
                "combined_score": judge_by_id[question_id].final_score if question_id in judge_by_id else None,
                "source_bonus_share": judge_by_id[question_id].source_bonus_share if question_id in judge_by_id else 0.0,
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
            "question_id", "question", "expected_answer", "system_answer", "raw_system_answer",
            "system_answer_sources", "answer_sources",
            "retrieved_evidence", "routed_method", "routing_reason", "duration_seconds",
            "is_correct", "average_score", "source_score", "combined_score", "source_bonus_share", "manual", "judge_reason",
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
                "question": question_by_id[question_id].question if question_id in question_by_id else "",
                "expected_answer": question_by_id[question_id].reference_answer if question_id in question_by_id else "",
                "system_answer": strip_citations(question_by_id[question_id].answer) if question_id in question_by_id else "",
                "raw_system_answer": question_by_id[question_id].answer if question_id in question_by_id else "",
                "system_answer_sources": json.dumps(
                    cited_sources(
                        question_by_id[question_id].answer, question_by_id[question_id].retrieved_evidence,
                    ) if question_id in question_by_id else [],
                    ensure_ascii=False,
                ),
                "answer_sources": json.dumps(
                    [
                        asdict(source)
                        for source in (
                            question_by_id[question_id].answer_source_evidence
                            or question_by_id[question_id].gold_evidence
                        )
                    ] if question_id in question_by_id else [],
                    ensure_ascii=False,
                ),
                "retrieved_evidence": json.dumps(
                    [asdict(evidence) for evidence in question_by_id[question_id].retrieved_evidence]
                    if question_id in question_by_id else [],
                    ensure_ascii=False,
                ),
                "routed_method": question_by_id[question_id].routed_method if question_id in question_by_id else "",
                "routing_reason": question_by_id[question_id].routing_reason if question_id in question_by_id else "",
                "duration_seconds": question_by_id[question_id].duration_seconds if question_id in question_by_id else None,
                "is_correct": item.is_correct if item else None,
                "average_score": item.score if item else None,
                "source_score": item.source_score if item else None,
                "combined_score": item.final_score if item else None,
                "source_bonus_share": item.source_bonus_share if item else 0.0,
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
    def _unsupported_model_message(model: object) -> str:
        return f"評測模型不在連線設定的可用清單：{model}"

    @staticmethod
    def _openai_chat(
        api_base_url: str, api_key: str, model: str, prompt: str, reasoning_effort: str | None = None,
    ) -> object:
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        payload.update(chat_completion_api_params(4000, reasoning_effort))
        headers = llm_request_headers(api_key)
        chat_url = f"{api_base_url.rstrip('/')}/chat/completions"

        def post(body: dict[str, object]) -> dict[str, object]:
            request = urllib.request.Request(
                chat_url,
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
                if exc.code != 400:
                    raise
                value = post({key: item for key, item in payload.items() if key != "response_format"})
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                response_headers = exc.headers or {}
                server = str(response_headers.get("server", "")).lower()
                cf_ray = str(response_headers.get("cf-ray", "")).strip()
                try:
                    raw_error_body = exc.read(8193).decode("utf-8", errors="replace")
                except (OSError, AttributeError):
                    raw_error_body = ""
                response_body_truncated = len(raw_error_body) > 8192
                error_body = raw_error_body[:8192]
                for secret in (api_key, headers.get("CF-Access-Client-Id"), headers.get("CF-Access-Client-Secret")):
                    if secret:
                        error_body = error_body.replace(secret, "[redacted]")
                parsed_error: object = None
                try:
                    parsed_error = json.loads(error_body)
                except json.JSONDecodeError:
                    pass
                error_message = ""
                if isinstance(parsed_error, dict):
                    error_value = parsed_error.get("error", parsed_error)
                    if isinstance(error_value, dict):
                        error_value = error_value.get("message") or error_value.get("detail")
                    if isinstance(error_value, str):
                        error_message = error_value.strip().replace("\n", " ")[:400]
                if "error code: 1010" in error_body.lower():
                    error_message = (
                        "Cloudflare Error 1010：請求的 User-Agent 簽章遭封鎖；"
                        "評測用戶端已設定 automotive-graphrag/0.1，請重新載入新版後重試"
                    )
                endpoint = urlsplit(chat_url)
                safe_endpoint = f"{endpoint.scheme}://{endpoint.netloc}{endpoint.path}"
                logger.error(
                    "LiteLLM judge HTTP 403 endpoint=%s model=%s api_key_suffix=%s "
                    "cf_access_headers_present=%s cf_ray=%s server=%s response_body_truncated=%s response_body=%r",
                    safe_endpoint,
                    payload.get("model", ""),
                    api_key[-4:] if api_key else "(missing)",
                    "CF-Access-Client-Id" in headers and "CF-Access-Client-Secret" in headers,
                    cf_ray or "(missing)",
                    server or "(missing)",
                    response_body_truncated,
                    error_body or "(empty)",
                )
                if cf_ray or "cloudflare" in server:
                    detail = f"回應本文：{error_message}。" if error_message else "回應未提供 JSON 錯誤說明。"
                    has_cf_credentials = (
                        "CF-Access-Client-Id" in headers and "CF-Access-Client-Secret" in headers
                    )
                    credential_status = (
                        "本次請求已帶入 CF-Access-Client-Id 與 CF-Access-Client-Secret。"
                        if has_cf_credentials else
                        "本次請求未帶齊 CF-Access-Client-Id／CF-Access-Client-Secret。"
                    )
                    key_status = (
                        f"本次 LiteLLM Bearer Key 尾碼：••••{api_key[-4:]}。"
                        if api_key else "本次請求沒有設定 LiteLLM Bearer Key。"
                    )
                    ray_status = f"CF-Ray：{cf_ray}。" if cf_ray else "回應未提供 CF-Ray。"
                    raise ProjectError(
                        "LiteLLM 評測端點回傳 HTTP 403（回應經過 Cloudflare；僅憑此狀態無法確認是 Access policy 拒絕）；"
                        f"{credential_status}{key_status}{ray_status}{detail}請以同一時間及 CF-Ray 對照 Access 與 LiteLLM proxy 記錄。"
                    ) from exc
                raise ProjectError(
                    "LiteLLM proxy 拒絕評測請求（HTTP 403）；"
                    f"{f'回應本文：{error_message}。' if error_message else '請查 server log 的安全遮蔽診斷記錄。'}"
                ) from exc
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
