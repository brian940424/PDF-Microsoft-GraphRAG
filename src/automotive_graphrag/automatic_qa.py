"""End-to-end question generation, answering, judging, and retrieval evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

from .automatic_evaluation import AutomaticEvaluationResult, AutomaticEvaluationService
from .connections import ALLOWED_CHAT_MODELS
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
        if questions_per_pdf < 1:
            raise ProjectError("每份 PDF 題數必須是正整數")
        if answer_concurrency < 1 or answer_concurrency > 32:
            raise ProjectError("回答並行數必須介於 1 到 32")
        for label, model in (("生題", generation_model), ("回答", answer_model), ("評判", judge_model)):
            if model not in ALLOWED_CHAT_MODELS:
                raise ProjectError(f"不支援的{label}模型：{model}")
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
                with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                    futures = {pool.submit(generate_one, item): item for item in pending}
                    generated = {}
                    for future in as_completed(futures):
                        doc = futures[future]
                        try:
                            generated[doc] = future.result()
                        except Exception as exc:
                            errors.append(f"{doc}：{exc}")
            else:
                generated = {}
                for doc in pending:
                    try:
                        generated[doc] = generate_one(doc)
                    except Exception as exc:
                        errors.append(f"{doc}：{exc}")
            for doc in pending:
                for question in generated.get(doc, ()):
                    normalized = normalize_question(question.question)
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    accepted[doc].append(question)

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
            description=f"每份 PDF 目標 {questions_per_pdf} 題；生題模型 {generation_model}。",
            method=method,
        )
        return self._run_existing(
            question_set,
            generation_model,
            answer_model,
            judge_model,
            method,
            answer_concurrency,
            top_k,
            tuple(errors),
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
    ) -> AutomaticQAReport:
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
        judge = None
        judge_error = None
        try:
            judge = self.judging.evaluate(
                updated.project_id, updated.question_set_id, top_k=top_k,
                model=judge_model, allow_missing_gold=True,
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
