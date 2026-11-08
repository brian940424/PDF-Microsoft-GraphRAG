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
from typing import Sequence

from .evidence import Evidence
from .projects import ProjectError, ProjectStore
from .question_sets import BatchQuestion, GoldEvidence, QuestionSetService


def compute_source_metrics(
    samples: Sequence[tuple[tuple[Evidence, ...], tuple[GoldEvidence, ...]]],
) -> tuple[float | None, float | None, float | None, int]:
    """Return question hit rate, macro page precision, document-hit rate, and no-evidence count."""
    eligible = [(retrieved, gold) for retrieved, gold in samples if gold]
    if not eligible:
        return None, None, None, 0
    hits = 0
    document_hits = 0
    no_evidence = 0
    question_precisions = []
    for retrieved, gold in eligible:
        unique_retrieved = RetrievalEvaluationService._dedupe_pages(retrieved)
        relevant = sum(RetrievalEvaluationService._is_relevant(item, gold) for item in unique_retrieved)
        hits += relevant > 0
        gold_documents = {item.document_id for item in gold}
        document_hits += bool(gold_documents & {item.document_id for item in unique_retrieved})
        no_evidence += not bool(unique_retrieved)
        if RetrievalEvaluationService._has_page_gold(gold):
            question_precisions.append(relevant / len(unique_retrieved) if unique_retrieved else 0.0)
    return (
        hits / len(eligible),
        sum(question_precisions) / len(question_precisions) if question_precisions else None,
        document_hits / len(eligible),
        no_evidence,
    )


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationItem:
    question_id: str
    question: str
    gold_evidence: tuple[GoldEvidence, ...]
    retrieved_evidence: tuple[Evidence, ...]
    evidence_hit: bool | None
    evidence_precision: float | None
    document_hit: bool | None
    retrieved_evidence_count: int
    relevant_evidence_count: int
    duration_seconds: float | None


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    project_id: str
    question_set_id: str
    evaluated_at: str
    method: str
    question_count: int
    evidence_hit_rate: float | None
    evidence_precision: float | None
    document_hit_rate: float | None
    no_evidence_question_count: int
    retrieval_metrics_status: str
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

        # Context source lists are not guaranteed to represent GraphRAG ranking.
        # DRIFT, Local and Basic return text-unit sources that can be mapped back
        # to PDF metadata; Global community reports cannot be reliably matched.
        # Also reject stale evidence from another strategy if queries were not rerun.
        strategy_matches = rerun_queries or question_set.method == method
        if method == "auto" and strategy_matches:
            # Each question ran its own routed strategy; only evidence-mapping strategies are scored.
            routed = [item for item in eligible if item.routed_method in {"local", "basic", "drift"}]
            items = tuple(
                self._evaluate_question(item, supported=item.routed_method in {"local", "basic", "drift"})
                for item in eligible
            )
            metrics = (
                compute_source_metrics(tuple((item.retrieved_evidence, item.gold_evidence) for item in routed))[:3]
                if routed else (None, None, None)
            )
            status = (
                f"已計算：自動模式只計入被選到 Local／DRIFT／Basic 的 {len(routed)}/{len(eligible)} 題；"
                "依文件 ID／頁碼比對 Gold Evidence，不使用檢索排名"
                if routed else "不可計算：自動模式未選到可映射來源的策略（Local／DRIFT／Basic）"
            )
        elif method not in {"local", "basic", "drift", "agentic"} or not strategy_matches:
            items = tuple(self._evaluate_question(item, supported=False) for item in eligible)
            metrics = (None, None, None)
            status = (
                f"暫不可計算：{method.upper()} 社群報告無法可靠映射至原始文件"
                if method == "global"
                else f"不可計算：題目集保存的是 {question_set.method} 策略證據，與所選 {method} 不同"
            )
        else:
            items = tuple(self._evaluate_question(item, supported=True) for item in eligible)
            metrics = compute_source_metrics(tuple((item.retrieved_evidence, item.gold_evidence) for item in eligible))[:3]
            status = "已計算：依文件 ID／頁碼比對 Gold Evidence；逐題 precision 平均（無標註頁碼者排除），不使用檢索排名"

        count = len(items)
        latencies = [item.duration_seconds for item in items if item.duration_seconds is not None]
        result = RetrievalEvaluationResult(
            project_id=project_id,
            question_set_id=question_set_id,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            method=method,
            question_count=count,
            evidence_hit_rate=metrics[0],
            evidence_precision=metrics[1],
            document_hit_rate=metrics[2],
            no_evidence_question_count=sum(item.retrieved_evidence_count == 0 for item in items),
            retrieval_metrics_status=status,
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
            # Older saved reports can still be opened; old rank metrics are
            # intentionally discarded because list order is not a rank signal.
            for key in ("first_relevant_rank", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "passed_at_k"):
                item.pop(key, None)
            item.setdefault("evidence_hit", None)
            item.setdefault("evidence_precision", None)
            item.setdefault("document_hit", None)
            item.setdefault("retrieved_evidence_count", len(item.get("retrieved_evidence", [])))
            item.setdefault("relevant_evidence_count", 0)
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
        for key in ("top_k", "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr", "average_first_relevant_rank", "evidence_source_accuracy"):
            value.pop(key, None)
        value.setdefault("method", "local")
        value.setdefault("evidence_hit_rate", None)
        value.setdefault("evidence_precision", None)
        value.setdefault("document_hit_rate", None)
        value.setdefault("no_evidence_question_count", 0)
        value.setdefault("retrieval_metrics_status", "舊版保存結果；請重新評估以計算新指標")
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
            "evidence_hit",
            "evidence_precision",
            "document_hit",
            "retrieved_evidence_count",
            "relevant_evidence_count",
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
    def _evaluate_question(cls, item: BatchQuestion, supported: bool = True) -> RetrievalEvaluationItem:
        unique_retrieved = cls._dedupe_pages(item.retrieved_evidence)
        relevant_count = sum(cls._is_relevant(evidence, item.gold_evidence) for evidence in unique_retrieved)
        docs = {gold.document_id for gold in item.gold_evidence}
        retrieved_docs = {evidence.document_id for evidence in unique_retrieved}
        count = len(unique_retrieved)
        precision = (
            (relevant_count / count if count else 0.0)
            if cls._has_page_gold(item.gold_evidence) else None
        )
        return RetrievalEvaluationItem(
            question_id=item.question_id,
            question=item.question,
            gold_evidence=item.gold_evidence,
            retrieved_evidence=item.retrieved_evidence,
            evidence_hit=(relevant_count > 0) if supported else None,
            evidence_precision=precision if supported else None,
            document_hit=(bool(docs & retrieved_docs)) if supported else None,
            retrieved_evidence_count=count,
            relevant_evidence_count=relevant_count,
            duration_seconds=item.duration_seconds,
        )

    @staticmethod
    def _is_relevant(evidence: Evidence, gold_evidence: tuple[GoldEvidence, ...]) -> bool:
        return any(
            evidence.document_id == gold.document_id
            and evidence.page is not None
            and evidence.page in gold.pages
            for gold in gold_evidence
        )

    @staticmethod
    def _has_page_gold(gold_evidence: Sequence[GoldEvidence]) -> bool:
        return any(item.document_id and item.pages for item in gold_evidence)

    @staticmethod
    def _dedupe_pages(evidence: Sequence[Evidence]) -> tuple[Evidence, ...]:
        """Deduplicate by document/page; unpaged evidence stays distinct and cannot match gold."""
        unique = []
        seen = set()
        for item in evidence:
            key = (
                (item.document_id, "page", item.page)
                if item.page is not None
                else (item.document_id, "unpaged", item.evidence_id or item.text_unit_id or item.rank)
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return tuple(unique)

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
