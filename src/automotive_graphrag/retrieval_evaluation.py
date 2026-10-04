"""Compute retrieval metrics against human-labelled Gold Evidence."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evidence import Evidence
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, GoldEvidence, QuestionSetService


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationItem:
    question_id: str
    question: str
    gold_evidence: tuple[GoldEvidence, ...]
    retrieved_evidence: tuple[Evidence, ...]
    first_relevant_rank: int | None
    recall_at_5: bool
    recall_at_10: bool
    passed_at_k: bool
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    project_id: str
    question_set_id: str
    evaluated_at: str
    top_k: int
    question_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    average_first_relevant_rank: float | None
    evidence_source_accuracy: float
    average_latency_seconds: float | None
    items: tuple[RetrievalEvaluationItem, ...]


class RetrievalEvaluationService:
    def __init__(self, projects: ProjectStore, question_sets: QuestionSetService | None = None) -> None:
        self.projects = projects
        self.question_sets = question_sets or QuestionSetService(projects)

    def evaluate(
        self,
        project_id: str,
        question_set_id: str,
        top_k: int = 5,
        rerun_queries: bool = False,
        method: str = "local",
    ) -> RetrievalEvaluationResult:
        if not isinstance(top_k, int) or top_k < 1:
            raise ProjectError("Top-K 必須是正整數")
        question_set = self.question_sets.get(project_id, question_set_id)
        eligible_ids = [item.question_id for item in question_set.questions if item.gold_evidence]
        if not eligible_ids:
            raise ProjectError("題目集沒有可評估的 Gold Evidence")
        if rerun_queries:
            question_set = self.question_sets.run(
                project_id,
                question_set_id,
                method=method,
                selected_question_ids=eligible_ids,
            )

        eligible = [item for item in question_set.questions if item.gold_evidence]
        items = tuple(self._evaluate_question(item, top_k) for item in eligible)
        count = len(items)
        ranks = [item.first_relevant_rank for item in items if item.first_relevant_rank is not None]
        latencies = [item.duration_seconds for item in items if item.duration_seconds is not None]
        considered_evidence = [
            evidence
            for item in eligible
            for evidence in item.retrieved_evidence[:top_k]
        ]
        relevant_evidence = sum(
            self._is_relevant(evidence, item.gold_evidence)
            for item in eligible
            for evidence in item.retrieved_evidence[:top_k]
        )
        result = RetrievalEvaluationResult(
            project_id=project_id,
            question_set_id=question_set_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            top_k=top_k,
            question_count=count,
            recall_at_5=sum(item.recall_at_5 for item in items) / count,
            recall_at_10=sum(item.recall_at_10 for item in items) / count,
            mrr=sum(1 / rank for rank in ranks) / count,
            average_first_relevant_rank=sum(ranks) / len(ranks) if ranks else None,
            evidence_source_accuracy=(relevant_evidence / len(considered_evidence) if considered_evidence else 0.0),
            average_latency_seconds=sum(latencies) / len(latencies) if latencies else None,
            items=items,
        )
        self._write_result(result)
        return result

    def last_result(self, project_id: str, question_set_id: str) -> RetrievalEvaluationResult | None:
        path = self._result_path(project_id, question_set_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        items = []
        for item in value.pop("items"):
            item.pop("recall_at_1", None)
            item.pop("recall_at_3", None)
            item.setdefault(
                "recall_at_10",
                item.get("first_relevant_rank") is not None and item["first_relevant_rank"] <= 10,
            )
            item["gold_evidence"] = tuple(
                GoldEvidence(
                    document_id=gold["document_id"],
                    pages=tuple(gold.get("pages", [])),
                    chunk_ids=tuple(gold.get("chunk_ids", [])),
                )
                for gold in item["gold_evidence"]
            )
            item["retrieved_evidence"] = tuple(Evidence(**evidence) for evidence in item["retrieved_evidence"])
            items.append(RetrievalEvaluationItem(**item))
        value.pop("recall_at_1", None)
        value.pop("recall_at_3", None)
        value.setdefault("recall_at_10", sum(item.recall_at_10 for item in items) / len(items))
        return RetrievalEvaluationResult(**value, items=tuple(items))

    def export(self, project_id: str, question_set_id: str) -> tuple[Path, Path]:
        result = self.last_result(project_id, question_set_id)
        if result is None:
            raise ProjectError("尚未執行 Retrieval 評估")
        directory = self.projects.path_for(project_id) / "exports"
        json_path = directory / f"{question_set_id}-retrieval-evaluation.json"
        csv_path = directory / f"{question_set_id}-retrieval-evaluation.csv"
        self._atomic_text(json_path, json.dumps(asdict(result), ensure_ascii=False, indent=2) + "\n")
        buffer = io.StringIO(newline="")
        fieldnames = [
            "question_id",
            "question",
            "gold_evidence",
            "retrieved_evidence",
            "first_relevant_rank",
            "recall_at_5",
            "recall_at_10",
            "passed_at_k",
            "duration_seconds",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.items:
            row = asdict(item)
            row["gold_evidence"] = json.dumps(row["gold_evidence"], ensure_ascii=False)
            row["retrieved_evidence"] = json.dumps(row["retrieved_evidence"], ensure_ascii=False)
            writer.writerow(row)
        self._atomic_text(csv_path, buffer.getvalue())
        return json_path, csv_path

    @classmethod
    def _evaluate_question(cls, item: BatchQuestion, top_k: int) -> RetrievalEvaluationItem:
        first_rank = next(
            (
                rank
                for rank, evidence in enumerate(item.retrieved_evidence, start=1)
                if cls._is_relevant(evidence, item.gold_evidence)
            ),
            None,
        )
        return RetrievalEvaluationItem(
            question_id=item.question_id,
            question=item.question,
            gold_evidence=item.gold_evidence,
            retrieved_evidence=item.retrieved_evidence,
            first_relevant_rank=first_rank,
            recall_at_5=first_rank is not None and first_rank <= 5,
            recall_at_10=first_rank is not None and first_rank <= 10,
            passed_at_k=first_rank is not None and first_rank <= top_k,
            duration_seconds=item.duration_seconds,
        )

    @staticmethod
    def _is_relevant(evidence: Evidence, gold_evidence: tuple[GoldEvidence, ...]) -> bool:
        return any(
            evidence.document_id == gold.document_id
            and (
                (bool(gold.chunk_ids) and evidence.chunk_id in gold.chunk_ids)
                or (bool(gold.pages) and evidence.page in gold.pages)
            )
            for gold in gold_evidence
        )

    def _result_path(self, project_id: str, question_set_id: str) -> Path:
        return self.projects.path_for(project_id) / "runs" / f"retrieval-evaluation-{question_set_id}.json"

    def _write_result(self, result: RetrievalEvaluationResult) -> None:
        self._atomic_text(
            self._result_path(result.project_id, result.question_set_id),
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
