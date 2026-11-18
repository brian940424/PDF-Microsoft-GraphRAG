"""Score the sources an answer actually cites against the question set's answer sources.

The answer text carries GraphRAG citations such as ``[Data: Sources (12, 34, +more)]`` (the
number is the text unit's ``human_readable_id``, kept as ``Evidence.context_id``) or the
Agentic Search form ``[E3]`` (``Evidence.evidence_id``). Only cited evidence that resolves to
a retrieved text unit with a document/page is compared with the gold answer sources.

Saved answers keep their citations. Evaluation splits them with ``strip_citations`` (the text
judges and answer metrics see) and ``cited_sources`` (the sources in question-set format).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_SOURCE_WEIGHT = 0.1
"""Largest bonus (on the 0–1 scale) a fully matching source can add to an eligible answer."""

PAGE_TOLERANCE = 1
"""Cited pages within this distance of a gold page count as a hit (PDF vs printed page offsets)."""

DOCUMENT_ONLY_SCORE = 0.5
"""Score when the cited sources reach the right document but none of the gold pages.

Only meaningful with several documents to choose from; single-document projects pass 0.
"""

_DATA_BLOCK = re.compile(r"\[\s*Data\s*:([^\]]*)\]", re.IGNORECASE)
_SOURCES_GROUP = re.compile(r"Sources\s*\(([^)]*)\)", re.IGNORECASE)
_EVIDENCE_CITATION = re.compile(r"\[(E\d+)\]")
_ANY_CITATION = re.compile(r"\s*\[\s*Data\s*:[^\]]*\]|\s*\[E\d+\]|\s*\[來源未映射\]", re.IGNORECASE)


def strip_citations(answer: str) -> str:
    """Answer text without GraphRAG, Agentic, or unmapped citation markers."""
    return _ANY_CITATION.sub("", answer or "").strip()


def cited_source_ids(answer: str) -> tuple[set[str], set[str]]:
    """Return (GraphRAG ``Sources`` ids, Agentic ``E`` ids) cited in the answer."""
    source_ids: set[str] = set()
    for block in _DATA_BLOCK.findall(answer or ""):
        for group in _SOURCES_GROUP.findall(block):
            source_ids.update(part.strip() for part in group.split(",") if part.strip().isdigit())
    evidence_ids = set(_EVIDENCE_CITATION.findall(answer or ""))
    return source_ids, evidence_ids


def cited_evidence(answer: str, retrieved: Sequence[object]) -> list[object]:
    """Retrieved text-unit evidence the answer cites, in retrieval order.

    Models sometimes cite the page number printed in the source text instead of its record id,
    so a ``Sources`` number that matches no retrieved record falls back to the retrieved page.
    """
    source_ids, evidence_ids = cited_source_ids(answer)
    text_units = [item for item in retrieved if getattr(item, "source_type", "text_unit") == "text_unit"]
    page_citations = source_ids - {str(getattr(item, "context_id", "")) for item in text_units}
    return [
        item for item in text_units
        if str(getattr(item, "context_id", "")) in source_ids
        or str(getattr(item, "evidence_id", "")) in evidence_ids
        or bool(set(str(page) for page in (getattr(item, "pages", ()) or (getattr(item, "page", None),))) & page_citations)
    ]


def cited_sources(answer: str, retrieved: Sequence[object]) -> list[dict[str, object]]:
    """Cited evidence grouped by document, in the question set's ``answer_sources`` shape."""
    pages_by_document: dict[str, set[int]] = {}
    for item in cited_evidence(answer, retrieved):
        pages = pages_by_document.setdefault(str(getattr(item, "document_id", "")), set())
        item_pages = getattr(item, "pages", ()) or (getattr(item, "page", None),)
        pages.update(page for page in item_pages if isinstance(page, int))
    return [
        {"document_id": document, "document_name": document, "pages": sorted(pages)}
        for document, pages in pages_by_document.items()
    ]


def source_score(
    answer: str,
    retrieved: Sequence[object],
    gold: Sequence[object],
    document_only_score: float = DOCUMENT_ONLY_SCORE,
) -> float | None:
    """0–1 score of the cited sources against the gold answer sources; None when not applicable.

    * No gold answer sources, or no retrieved text units (e.g. Global Search reports): None.
    * Gold pages: recall of gold pages hit by a cited page (±``PAGE_TOLERANCE``) in the same
      document; when no page is hit but a cited source is in a gold document, ``document_only_score``.
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
        item_pages = getattr(item, "pages", ()) or (getattr(item, "page", None),)
        pages.update(page for page in item_pages if isinstance(page, int))
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
        return document_only_score
    return round(recall, 4)


def combined_score(
    judge_score: float | None,
    source: float | None,
    weight: float = DEFAULT_SOURCE_WEIGHT,
    bonus_share: float = 0.0,
) -> float | None:
    """Judge score (0–1) plus a source bonus, capped at 1.

    Sources never reduce the score. ``bonus_share`` is the share of judges that found the answer
    covers the correct answer and is wrong only because of added content; with no such judge the
    result is the judge score alone.
    """
    if judge_score is None:
        return None
    if source is None:
        return round(judge_score, 4)
    return round(min(1.0, judge_score + bonus_share * weight * source), 4)


def validate_source_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("來源分數權重必須是 0 到 1 的數字") from exc
    if not 0 <= weight <= 1:
        raise ValueError("來源分數權重必須介於 0 到 1")
    return weight
