"""Keep the community reports closest to the question before Global maps them (P-05).

Global maps every report of the selected levels in batches, so its cost grows with the number of
communities (SM3E: 1,072 reports at level <= 2, about 820k tokens per question). The index already
stores an embedding of each report's full content (LanceDB ``community_full_content``, keyed by the
report ``id``); ranking those against the question costs one embedding call. Offline on SM3E
03_global, the top 200 reports kept 83% of the reports a full map cited, at about 19% of the map.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .global_sources import Embedder, _normalized, config_embedder, text_unit_vectors

GLOBAL_REPORT_PREFILTER_VERSION = 1
DEFAULT_GLOBAL_REPORT_TOP_N = 200
"""Reports kept for the map stage; 0 maps every report. SM3E 03_global: 300 cost 44% more tokens without a higher score."""

REPORT_VECTOR_TABLE = "community_full_content"


def prefilter_signature(top_n: int = DEFAULT_GLOBAL_REPORT_TOP_N) -> str:
    """Cache-key value for answers produced with this prefilter setting."""
    return f"v{GLOBAL_REPORT_PREFILTER_VERSION}-top{int(top_n)}"


def select_reports(
    reports: Any,
    question_vector: Any,
    report_vectors: dict[str, np.ndarray],
    top_n: int,
    max_level: int | None,
) -> tuple[Any, dict[str, Any]]:
    """Keep the ``top_n`` reports at or under ``max_level`` most similar to the question.

    Reports without a stored vector rank last. Reports above ``max_level`` are dropped too, as
    GraphRAG would drop them anyway; ``communities`` stays whole so its level roll-up still works.
    """
    candidates = reports if max_level is None else reports[reports["level"].astype(int) <= max_level]
    query = _normalized(np.asarray(question_vector, dtype=np.float32))
    ids = [str(value) for value in candidates["id"]]
    scores = np.array([
        float(report_vectors[item] @ query) if item in report_vectors else -np.inf for item in ids
    ])
    order = np.argsort(-scores, kind="stable")[:top_n]
    kept = {ids[position] for position in order}
    selected = candidates[candidates["id"].astype(str).isin(kept)]
    finite = scores[order][np.isfinite(scores[order])]
    return selected, {
        "candidates": len(ids),
        "kept": len(selected),
        "missing_vectors": sum(item not in report_vectors for item in ids),
        "min_kept_score": round(float(finite.min()), 4) if finite.size else None,
    }


def prefilter_reports(
    reports: Any,
    question: str,
    root_dir: Path,
    top_n: int,
    community_level: int | None,
    embed: Embedder | None = None,
    report_vectors: dict[str, np.ndarray] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return the reports Global should map and a report of what was kept.

    Leaves the reports unchanged when the filter is off, when there are no more candidates than
    ``top_n`` (no embedding call), or when the index lacks report vectors.
    """
    report: dict[str, Any] = {
        "method": "question-embedding", "version": GLOBAL_REPORT_PREFILTER_VERSION, "top_n": int(top_n),
    }
    if top_n <= 0:
        return reports, {**report, "applied": False, "reason": "disabled"}
    if not {"id", "level"} <= set(getattr(reports, "columns", ())):
        return reports, {**report, "applied": False, "reason": "reports lack id/level columns"}
    levels = reports["level"].astype(int)
    candidate_count = int((levels <= community_level).sum()) if community_level is not None else len(reports)
    if candidate_count <= top_n:
        return reports, {**report, "applied": False, "reason": "few reports", "candidates": candidate_count}
    if report_vectors is None:
        report_vectors = text_unit_vectors(Path(root_dir) / "output" / "lancedb", REPORT_VECTOR_TABLE)
    if not report_vectors:
        return reports, {**report, "applied": False, "reason": "no report vectors"}
    question_vector = (embed or config_embedder(root_dir))([question])[0]
    selected, stats = select_reports(reports, question_vector, report_vectors, top_n, community_level)
    return selected, {**report, "applied": True, **stats}
