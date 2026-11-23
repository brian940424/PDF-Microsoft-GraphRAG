"""Ground Global answer sentences in source passages (F-70).

A Global answer cites community reports, and a report covers every page of its community
(a level-0 report spans over a hundred pages). For each cited statement this module ranks
the text units of the cited communities by embedding similarity to the statement and appends
the best ones as ``Sources (…)`` to the citation, so evidence, display and source scoring use
the same text-unit path as Local and Basic. It costs one embedding call per answer, no LLM call.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# "[Data: Reports (2, 7, +more)]" or "[Data: Reports (2); Entities (5)]".
_REPORT_CITATION = re.compile(r"\[\s*Data\s*:[^\]]*?Reports\s*\(([^)]*)\)[^\]]*\]", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"\n+|(?<=[。！？；])|(?<=[.!?;])\s+(?=\S)")
_CITATION = re.compile(r"\[[^\]]*\]")
_MARKUP = re.compile(r"[*#>`|]+")

SOURCES_PER_STATEMENT = 2
"""Text units kept for the cited text as a whole."""

MAX_SOURCES_PER_CITATION = 4
"""Upper bound per citation after adding the best unit of each sentence."""

MIN_SENTENCE_CHARS = 12
"""Shorter fragments (list numbers, headings) are not embedded on their own."""

MAX_STATEMENT_CHARS = 2000
MAX_CANDIDATE_UNITS = 400
"""Citations covering more text units than this (several level-0 reports) are not grounded."""

Embedder = Callable[[list[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True, slots=True)
class GroundedCitation:
    start: int
    end: int
    statement: str
    report_ids: tuple[str, ...]
    text_unit_ids: tuple[str, ...]


def _clean(text: str) -> str:
    return " ".join(_MARKUP.sub(" ", _CITATION.sub(" ", text)).split()).strip(" -:：")


def cited_statements(answer: str) -> list[tuple[int, int, str, tuple[str, ...]]]:
    """(start, end, cited text, report ids) for each Reports citation in the answer.

    The cited text is everything after the previous citation: a model often writes a whole
    paragraph or list and cites it once at the end (SM3E 03_global Q01).
    """
    result: list[tuple[int, int, str, tuple[str, ...]]] = []
    previous_end = 0
    for match in _REPORT_CITATION.finditer(answer or ""):
        statement = _clean(answer[previous_end:match.start()])[-MAX_STATEMENT_CHARS:]
        ids = tuple(dict.fromkeys(
            part.strip() for part in match.group(1).split(",") if part.strip().isdigit()
        ))
        result.append((match.start(), match.end(), statement, ids))
        previous_end = match.end()
    return result


def _sentences(statement: str) -> list[str]:
    parts = [_clean(part) for part in _SENTENCE_BREAK.split(statement)]
    return [part for part in parts if len(part) >= MIN_SENTENCE_CHARS]


def ground_citations(
    answer: str,
    community_units: dict[str, Sequence[str]],
    unit_vectors: dict[str, np.ndarray],
    embed: Embedder,
    per_statement: int = SOURCES_PER_STATEMENT,
    max_sources: int = MAX_SOURCES_PER_CITATION,
) -> list[GroundedCitation]:
    """Pick the text units of the cited communities closest to each cited text.

    Keeps the ``per_statement`` best units for the whole cited text, then adds the best unit of
    each sentence in it, up to ``max_sources``; one embedding call covers the whole answer.
    """
    statements = [item for item in cited_statements(answer) if item[2] and item[3]]
    jobs: list[tuple[int, list[str], list[str]]] = []
    for index, (_, _, statement, report_ids) in enumerate(statements):
        units = list(dict.fromkeys(
            str(unit) for report_id in report_ids for unit in community_units.get(report_id, ())
            if str(unit) in unit_vectors
        ))
        if units and len(units) <= MAX_CANDIDATE_UNITS:
            sentences = _sentences(statement)
            jobs.append((index, units, [statement, *(sentences if len(sentences) > 1 else [])]))
    if not jobs:
        return []
    texts = [text for _, _, job_texts in jobs for text in job_texts]
    vectors = iter(embed(texts))
    grounded: list[GroundedCitation] = []
    for index, units, job_texts in jobs:
        matrix = np.stack([unit_vectors[unit] for unit in units])
        rankings = [
            np.argsort(-(matrix @ _normalized(np.asarray(next(vectors), dtype=np.float32))))
            for _ in job_texts
        ]
        chosen = [units[position] for position in rankings[0][:per_statement]]
        for ranking in rankings[1:]:
            best = units[ranking[0]]
            if best not in chosen and len(chosen) < max_sources:
                chosen.append(best)
        start, end, statement, report_ids = statements[index]
        grounded.append(GroundedCitation(start, end, statement, report_ids, tuple(chosen)))
    return grounded


def add_source_citations(
    answer: str, grounded: Sequence[GroundedCitation], short_id_by_unit: dict[str, str],
) -> str:
    """Append "; Sources (…)" with text-unit record ids inside each grounded citation."""
    pieces: list[str] = []
    position = 0
    for item in sorted(grounded, key=lambda value: value.start):
        short_ids = [short_id_by_unit[unit] for unit in item.text_unit_ids if unit in short_id_by_unit]
        if not short_ids:
            continue
        pieces.append(answer[position:item.end - 1])
        pieces.append(f"; Sources ({', '.join(short_ids)})]")
        position = item.end
    pieces.append(answer[position:])
    return "".join(pieces)


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


_vector_lock = threading.Lock()
_vector_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def text_unit_vectors(lancedb_dir: Path, table: str = "text_unit_text") -> dict[str, np.ndarray]:
    """Normalized vectors by id from a table of the index's LanceDB store, cached per table version."""
    import lancedb

    path = Path(lancedb_dir) / f"{table}.lance"
    if not path.is_dir():
        return {}
    stamp = max((item.stat().st_mtime_ns for item in path.rglob("*") if item.is_file()), default=0)
    key = (str(path.resolve()), stamp)
    with _vector_lock:
        cached = _vector_cache.get(key)
    if cached is not None:
        return cached
    rows = lancedb.connect(str(lancedb_dir)).open_table(table).to_arrow().to_pylist()
    vectors = {
        str(row["id"]): _normalized(np.asarray(row["vector"], dtype=np.float32))
        for row in rows if row.get("vector") is not None
    }
    with _vector_lock:
        # One version per table: Global keeps both report and text-unit vectors loaded.
        for stale in [item for item in _vector_cache if item[0] == key[0]]:
            del _vector_cache[stale]
        _vector_cache[key] = vectors
    return vectors


def config_embedder(root_dir: Path) -> Embedder:
    """Batch embedder built from the query root's GraphRAG settings (the index's text-unit model)."""
    from graphrag.config.load_config import load_config
    from graphrag_llm.embedding import create_embedding

    config = load_config(root_dir=Path(root_dir))
    model = create_embedding(config.get_embedding_model_config(config.basic_search.embedding_model_id))

    def embed(texts: list[str]) -> list[list[float]]:
        return model.embedding(input=texts).embeddings

    return embed


def ground_global_answer(
    answer: str,
    communities: object,
    root_dir: Path,
    text_units: object | None = None,
    embed: Embedder | None = None,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    """Return the answer with ``Sources (…)`` added, the source rows for the context, and a report."""
    import pandas as pd

    output = Path(root_dir) / "output"
    community_units = {
        str(int(community)): [str(unit) for unit in units] if units is not None else []
        for community, units in zip(communities["community"], communities["text_unit_ids"])
    }
    if text_units is None:
        text_units = pd.read_parquet(output / "text_units.parquet", columns=["id", "human_readable_id", "text"])
    short_id_by_unit = {str(unit): str(short) for unit, short in zip(text_units["id"], text_units["human_readable_id"])}
    text_by_unit = dict(zip(text_units["id"].astype(str), text_units["text"].astype(str)))
    grounded = ground_citations(
        answer, community_units, text_unit_vectors(output / "lancedb"), embed or config_embedder(root_dir),
    )
    chosen = list(dict.fromkeys(unit for item in grounded for unit in item.text_unit_ids if unit in short_id_by_unit))
    sources = [
        {"id": short_id_by_unit[unit], "text": text_by_unit.get(unit, ""), "in_context": True}
        for unit in chosen
    ]
    report = {
        "method": "statement-embedding",
        "citations": len(cited_statements(answer)),
        "grounded_citations": len(grounded),
        "text_units": len(chosen),
    }
    return add_source_citations(answer, grounded, short_id_by_unit), sources, report
