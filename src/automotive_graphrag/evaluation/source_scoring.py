"""Score the sources an answer actually cites against the question set's answer sources.

The answer text carries GraphRAG citations such as ``[Data: Sources (12, 34, +more)]`` (the
number is the text unit's ``human_readable_id``, kept as ``Evidence.context_id``) or the
Agentic Search form ``[E3]`` (``Evidence.evidence_id``). Only cited evidence that resolves to
a retrieved text unit with a document/page is compared with the gold answer sources.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_SOURCE_WEIGHT = 0.1
"""Share of the combined score taken by the source score; the LLM judge takes the rest."""

PAGE_TOLERANCE = 1
"""Cited pages within this distance of a gold page count as a hit (PDF vs printed page offsets)."""

DOCUMENT_ONLY_SCORE = 0.5
"""Score when the cited sources reach the right document but none of the gold pages."""

_DATA_BLOCK = re.compile(r"\[\s*Data\s*:([^\]]*)\]", re.IGNORECASE)
_SOURCES_GROUP = re.compile(r"Sources\s*\(([^)]*)\)", re.IGNORECASE)
_EVIDENCE_CITATION = re.compile(r"\[(E\d+)\]")


def cited_source_ids(answer: str) -> tuple[set[str], set[str]]:
    """Return (GraphRAG ``Sources`` ids, Agentic ``E`` ids) cited in the answer."""
    source_ids: set[str] = set()
    for block in _DATA_BLOCK.findall(answer or ""):
        for group in _SOURCES_GROUP.findall(block):
            source_ids.update(part.strip() for part in group.split(",") if part.strip().isdigit())
    evidence_ids = set(_EVIDENCE_CITATION.findall(answer or ""))
    return source_ids, evidence_ids


def cited_evidence(answer: str, retrieved: Sequence[object]) -> list[object]:
    """Retrieved text-unit evidence the answer cites, in retrieval order."""
    source_ids, evidence_ids = cited_source_ids(answer)
    return [
        item for item in retrieved
        if getattr(item, "source_type", "text_unit") == "text_unit"
        and (
            str(getattr(item, "context_id", "")) in source_ids
            or str(getattr(item, "evidence_id", "")) in evidence_ids
        )
    ]


def source_score(answer: str, retrieved: Sequence[object], gold: Sequence[object]) -> float | None:
    """0–1 score of the cited sources against the gold answer sources; None when not applicable.

    * No gold answer sources, or no retrieved text units (e.g. Global Search reports): None.
    * Gold pages: recall of gold pages hit by a cited page (±``PAGE_TOLERANCE``) in the same
      document; when no page is hit but a cited source is in a gold document, ``DOCUMENT_ONLY_SCORE``.
    * Gold documents without pages: 1.0 when any cited source is in that document.
    * An answer that cites nothing resolvable scores 0.
    """
    if not gold:
        return None
    if not any(getattr(item, "source_type", "text_unit") == "text_unit" for item in retrieved):
        return None
    cited = cited_evidence(answer, retrieved)
    cited_pages: dict[str, set[int]] = {}
    for item in cited:
        pages = cited_pages.setdefault(str(getattr(item, "document_id", "")), set())
        page = getattr(item, "page", None)
        if isinstance(page, int):
            pages.add(page)
    gold_documents = {str(getattr(item, "document_id", "")) for item in gold}
    document_hit = bool(gold_documents & set(cited_pages))
    gold_pages = {
        (str(getattr(item, "document_id", "")), page)
        for item in gold
        for page in getattr(item, "pages", ()) or ()
    }
    if not gold_pages:
        return 1.0 if document_hit else 0.0
    hits = sum(
        any(abs(page - cited) <= PAGE_TOLERANCE for cited in cited_pages.get(document, ()))
        for document, page in gold_pages
    )
    recall = hits / len(gold_pages)
    if recall == 0 and document_hit:
        return DOCUMENT_ONLY_SCORE
    return round(recall, 4)


def combined_score(judge_score: float | None, source: float | None, weight: float = DEFAULT_SOURCE_WEIGHT) -> float | None:
    """Weighted 0–1 score; falls back to the judge score alone when the source score is N/A."""
    if judge_score is None:
        return None
    if source is None:
        return round(judge_score, 4)
    return round((1 - weight) * judge_score + weight * source, 4)


def validate_source_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("來源分數權重必須是 0 到 1 的數字") from exc
    if not 0 <= weight <= 1:
        raise ValueError("來源分數權重必須介於 0 到 1")
    return weight
