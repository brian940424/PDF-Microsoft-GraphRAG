"""End-to-end question generation, answering, judging, and retrieval evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .automatic_evaluation import AutomaticEvaluationResult, AutomaticEvaluationService
from .connections import ALLOWED_CHAT_MODELS, GPT6_LUNA_MODEL
from .projects import ProjectError, ProjectStore
from .question_generation import QuestionGenerationService, normalize_question
from .question_sets import BatchQuestion, QuestionSet, QuestionSetService
from .querying import QueryResult, QueryService
from .retrieval_evaluation import RetrievalEvaluationResult, RetrievalEvaluationService
from .source_sampling import SourceSamplingService


@dataclass(frozen=True, slots=True)
class AutomaticQAReport:
    question_set: QuestionSet
    generation_model: str
    answer_model: str
    judge_model: str
    generation_errors: tuple[str, ...]
    answers: tuple[QueryResult, ...]
    judge: AutomaticEvaluationResult | None
    judge_error: str | None
    retrieval: RetrievalEvaluationResult | None
    retrieval_error: str | None


class AutomaticQATestService:
    def __init__(
        self,
        projects: ProjectStore,
        sampling: SourceSamplingService,
        generation: QuestionGenerationService,
        querying: QueryService,
        question_sets: QuestionSetService,
        judging: AutomaticEvaluationService,
        retrieval: RetrievalEvaluationService,
    ) -> None:
        self.projects = projects
        self.sampling = sampling
        self.generation = generation
        self.querying = querying
        self.question_sets = question_sets
        self.judging = judging
        self.retrieval = retrieval

    def generate_question_set(
        self,
        project_id: str,
        questions_per_pdf: int,
        parallel_pdf_generation: bool,
        generation_model: str,
        method: str = "local",
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> QuestionSet:
        if questions_per_pdf < 1:
            raise ProjectError("每份 PDF 題數必須是正整數")
        if generation_model not in ALLOWED_CHAT_MODELS:
            raise ProjectError(f"不支援的生題模型：{generation_model}")
        project = self.projects.get(project_id)
        if project.status != "INDEXED":
            raise ProjectError(f"專案狀態 {project.status} 尚未完成建圖")

        documents = self.projects.path_for(project_id) / "source"
        pdf_ids = sorted(
            (path.name for path in documents.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
            key=str.casefold,
        )
        if not pdf_ids:
            raise ProjectError("此專案尚未匯入 PDF")
        samples = self.sampling.scan(project_id, minimum_characters=1)
        accepted: dict[str, list] = {document_id: [] for document_id in pdf_ids}
        seen: set[str] = set()
        errors: list[str] = []
        if progress_callback:
            progress_callback(0, f"準備為 {len(pdf_ids)} 份 PDF 生成題目")
        # Each wave submits at most one request for any PDF. Waves only advance
        # after all previous requests finish, so retries can safely see all results.
        for _attempt in range(3):
            pending = [item for item in pdf_ids if len(accepted[item]) < questions_per_pdf]
            if not pending:
                break
            excluded = [question.question for questions in accepted.values() for question in questions]
            def generate_one(document_id: str):
                needed = questions_per_pdf - len(accepted[document_id])
                return self.generation.generate_for_document(
                    project_id,
                    document_id,
                    samples,
                    needed,
                    generation_model,
                    excluded_questions=excluded,
                )
            if parallel_pdf_generation:
                with ThreadPoolExecutor(max_workers=min(3, len(pending))) as pool:
                    futures = {pool.submit(generate_one, item): item for item in pending}
                    generated = {}
                    finished = 0
                    for future in as_completed(futures):
                        doc = futures[future]
                        finished += 1
                        try:
                            generated[doc] = future.result()
                        except Exception as exc:
                            errors.append(f"{doc}：{exc}")
                        if progress_callback:
                            fraction = min(0.95, ((_attempt * len(pdf_ids)) + finished) / (3 * len(pdf_ids)))
                            progress_callback(fraction, f"第 {_attempt + 1} 輪：已完成 {finished}/{len(pending)} 份 PDF")
            else:
                generated = {}
                for position, doc in enumerate(pending, start=1):
                    if progress_callback:
                        progress_callback(0, f"第 {_attempt + 1} 輪：正在生成 {position}/{len(pending)} — {doc}")
                    try:
                        generated[doc] = generate_one(doc)
                    except Exception as exc:
                        errors.append(f"{doc}：{exc}")
                    if progress_callback:
                        fraction = min(0.95, ((_attempt * len(pdf_ids)) + position) / (3 * len(pdf_ids)))
                        progress_callback(fraction, f"第 {_attempt + 1} 輪：已完成 {position}/{len(pending)} 份 PDF")
            for doc in pending:
                for question in generated.get(doc, ()):
                    normalized = normalize_question(question.question)
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    accepted[doc].append(question)

        if progress_callback:
            progress_callback(1, "題目生成完成，正在儲存題目集")

        batch_questions: list[BatchQuestion] = []
        for questions in accepted.values():
            for question in questions:
                batch_questions.append(
                    BatchQuestion(
                        question_id=f"Q{len(batch_questions) + 1:04d}",
                        question=question.question,
                        reference_answer=question.reference_answer,
                        gold_evidence=question.gold_evidence,
                        question_source_evidence=question.question_source_evidence,
                        answer_source_evidence=question.answer_source_evidence or question.gold_evidence,
                    )
                )
        if not batch_questions:
            detail = "；".join(errors[:4])
            raise ProjectError("所有 PDF 均未能生成有效題目" + (f"：{detail}" if detail else ""))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        question_set = self.question_sets.create_with_questions(
            project_id,
            f"自動問答測試-{stamp}",
            batch_questions,
            description=(
                f"每份 PDF 目標 {questions_per_pdf} 題；生題模型 {generation_model}。"
                + ("未完成項目：" + "；".join(errors) if errors else "")
            ),
            method=method,
        )
        return question_set

    def generate_and_run(
        self,
        project_id: str,
        questions_per_pdf: int,
        parallel_pdf_generation: bool,
        generation_model: str,
        answer_model: str,
        judge_model: str,
        method: str,
        answer_concurrency: int,
        top_k: int = 5,
    ) -> AutomaticQAReport:
        question_set = self.generate_question_set(
            project_id, questions_per_pdf, parallel_pdf_generation, generation_model, method
        )
        return self._run_existing(
            question_set, generation_model, answer_model, judge_model,
            method, answer_concurrency, top_k, (),
        )

    def run_existing(
        self,
        project_id: str,
        question_set_id: str,
        answer_model: str,
        judge_model: str,
        method: str,
        answer_concurrency: int,
        top_k: int = 5,
    ) -> AutomaticQAReport:
        question_set = self.question_sets.get(project_id, question_set_id)
        return self._run_existing(
            question_set,
            "—",
            answer_model,
            judge_model,
            method,
            answer_concurrency,
            top_k,
            (),
        )

    def answer_existing(
        self,
        project_id: str,
        question_set_id: str,
        answer_model: str,
        method: str,
        answer_concurrency: int,
    ) -> AutomaticQAReport:
        """Generate and persist answers only; judging is a separate user action."""
        report = self._run_existing(
            self.question_sets.get(project_id, question_set_id),
            "—",
            answer_model,
            "—",
            method,
            answer_concurrency,
            5,
            (),
            evaluate=False,
        )
        self.judging.clear_result(project_id, question_set_id)
        return report

    def evaluate_existing(
        self,
        project_id: str,
        question_set_id: str,
        answer_model: str,
        judge_model: str,
        method: str,
        top_k: int = 5,
        concurrency: int = 3,
    ) -> AutomaticQAReport:
        self._validate_model_method(answer_model, method)
        question_set = self.question_sets.get(project_id, question_set_id)
        if not question_set.questions or not all(
            item.status == "COMPLETED" and item.answer.strip()
            for item in question_set.questions
        ):
            raise ProjectError("請先按「檢索並生成回答」完成所有題目的系統回答")
        judge = None
        judge_error = None
        try:
            judge = self.judging.evaluate(
                project_id, question_set_id, model=judge_model, concurrency=concurrency
            )
        except ProjectError as exc:
            judge_error = str(exc)
        retrieval = None
        retrieval_error = None
        try:
            retrieval = self.retrieval.evaluate(
                project_id, question_set_id, top_k=top_k, method=method
            )
        except ProjectError as exc:
            retrieval_error = str(exc)
        return AutomaticQAReport(
            question_set=question_set,
            generation_model="—",
            answer_model=answer_model,
            judge_model=judge_model,
            generation_errors=(),
            answers=(),
            judge=judge,
            judge_error=judge_error,
            retrieval=retrieval,
            retrieval_error=retrieval_error,
        )

    def _run_existing(
        self,
        question_set: QuestionSet,
        generation_model: str,
        answer_model: str,
        judge_model: str,
        method: str,
        concurrency: int,
        top_k: int,
        generation_errors: tuple[str, ...],
        evaluate: bool = True,
    ) -> AutomaticQAReport:
        self._validate_model_method(answer_model, method)
        if concurrency < 1 or concurrency > 32:
            raise ProjectError("回答並行數必須介於 1 到 32")
        pending = [item for item in question_set.questions]
        results: dict[str, QueryResult] = {}
        def answer(item: BatchQuestion) -> QueryResult:
            return self.querying.ask(question_set.project_id, item.question, method, chat_model=answer_model)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(answer, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results[item.question_id] = future.result()
                except Exception as exc:
                    # QueryService normally converts GraphRAG process failures to QueryResult;
                    # unexpected setup errors remain visible through the report.
                    results[item.question_id] = QueryResult(
                        query_id="", project_id=question_set.project_id, question=item.question,
                        method=method, status="FAILED", answer="", error=str(exc), started_at="",
                        completed_at="", duration_seconds=0,
                    )
        updated = self.question_sets.update_answers(
            question_set.project_id, question_set.question_set_id, results, method
        )
        if not evaluate:
            return AutomaticQAReport(
                question_set=updated,
                generation_model=generation_model,
                answer_model=answer_model,
                judge_model=judge_model,
                generation_errors=generation_errors,
                answers=tuple(results[item.question_id] for item in updated.questions),
                judge=None,
                judge_error=None,
                retrieval=None,
                retrieval_error=None,
            )
        judge = None
        judge_error = None
        try:
            judge = self.judging.evaluate(
                updated.project_id, updated.question_set_id, model=judge_model,
            )
        except ProjectError as exc:
            judge_error = str(exc)
        retrieval = None
        retrieval_error = None
        try:
            retrieval = self.retrieval.evaluate(
                updated.project_id, updated.question_set_id, top_k=top_k, method=method
            )
        except ProjectError as exc:
            retrieval_error = str(exc)
        return AutomaticQAReport(
            question_set=updated,
            generation_model=generation_model,
            answer_model=answer_model,
            judge_model=judge_model,
            generation_errors=generation_errors,
            answers=tuple(results[item.question_id] for item in updated.questions),
            judge=judge,
            judge_error=judge_error,
            retrieval=retrieval,
            retrieval_error=retrieval_error,
        )

    @staticmethod
    def _validate_model_method(answer_model: str, method: str) -> None:
        if method.strip().lower() == "drift" and answer_model == GPT6_LUNA_MODEL:
            raise ProjectError("DRIFT 不支援 GPT-6 Luna；請改用 Local、Global 或 Basic，或選擇其他回答模型")
