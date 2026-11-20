"""Add pages that a retrieved Basic passage refers to ("Refer to BCS-79") when they almost made the top-k.

Service manuals split one task across pages and link them with "Refer to <section>-<page>". After
the F-65 re-chunking, SM3E Q1 Q26 lost BCS p.79 (re-register all keyfobs when a new BCM has none),
which p.63 refers to, and Q04 lost ACC p.3, which BR p.18 refers to; both pages ranked just below
the 10th passage. A referenced page is added only when its best passage is within
`SIMILARITY_MARGIN` cosine of the k-th retrieved passage, at most `MAX_ADDED_PAGES` pages per
question (offline over SM3E Q1+Q2, 60 questions: 42 pages added, both gold pages recovered).

A reference "XX-n" is followed only when page n of XX.pdf prints the label "XX-n" in its footer,
so manuals whose printed page labels differ from the PDF page index are left unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_PATTERN = re.compile(r"\bRefer to ([A-Z]{2,4})-(\d{1,4})\b")
SIMILARITY_MARGIN = 0.03
MAX_ADDED_PAGES = 2
FOOTER_FRACTION = 0.07


def capture_basic_retrieval(capture: dict[str, Any]) -> None:
    """Record the query vector, the top-k results and the vector store of Basic's similarity search."""
    from graphrag_vectors.vector_store import VectorStore

    original = VectorStore.similarity_search_by_text
    if getattr(original, "_capture_patch", False):
        original = original.__wrapped__

    def similarity_search_by_text(self: Any, text: str, text_embedder: Any, k: int = 10, **kwargs: Any):
        vector = text_embedder(text)
        if not vector:
            return []
        results = self.similarity_search_by_vector(query_embedding=vector, k=k, **kwargs)
        capture.update(store=self, vector=list(vector), results=results)
        return results

    similarity_search_by_text._capture_patch = True  # type: ignore[attr-defined]
    similarity_search_by_text.__wrapped__ = original  # type: ignore[attr-defined]
    VectorStore.similarity_search_by_text = similarity_search_by_text  # type: ignore[method-assign]


def referenced_pages(texts: Iterable[str], documents: Iterable[str]) -> list[tuple[str, int]]:
    """Pages named by "Refer to XX-n" in `texts`, in order of first mention, limited to known documents."""
    known = set(documents)
    pages: dict[tuple[str, int], None] = {}
    for text in texts:
        for code, number in REFERENCE_PATTERN.findall(text):
            document_id = f"{code}.pdf"
            if document_id in known:
                pages.setdefault((document_id, int(number)), None)
    return list(pages)


def page_label_matches(pdf: Path, page: int, label: str) -> bool:
    """True when the footer of `page` (1-based) in `pdf` prints `label`, for example "BCS-79"."""
    import pymupdf

    try:
        with pymupdf.open(pdf) as document:
            if not 1 <= page <= document.page_count:
                return False
            sheet = document[page - 1]
            height = sheet.rect.height
            footer = sheet.get_text(clip=pymupdf.Rect(0, height * (1 - FOOTER_FRACTION), sheet.rect.width, height))
    except Exception:
        return False
    return re.search(rf"(?<![A-Z]){re.escape(label)}(?!\d)", footer) is not None


def _cosine(left: Any, right: Any) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm else 0.0


def select_referenced_units(
    query_vector: Any,
    top_vectors: Mapping[str, Any],
    top_texts: Iterable[str],
    locations: Mapping[str, tuple[str, int]],
    unit_vector: Any,
    label_matches: Any,
    margin: float = SIMILARITY_MARGIN,
    max_pages: int = MAX_ADDED_PAGES,
) -> list[dict[str, Any]]:
    """Choose referenced pages close to the k-th result; return their passages ordered by gap.

    `top_vectors` maps the full ids of the retrieved text units to their vectors, `locations` maps
    every full text-unit id to (document_id, page), `unit_vector(full_id)` returns a vector or None
    and `label_matches(document_id, page)` checks the printed page label.
    """
    if not top_vectors:
        return []
    kth = min(_cosine(query_vector, vector) for vector in top_vectors.values())
    present = {locations[unit] for unit in top_vectors if unit in locations}
    units_by_page: dict[tuple[str, int], list[str]] = {}
    for unit, location in locations.items():
        units_by_page.setdefault(location, []).append(unit)
    candidates = []
    for location in referenced_pages(top_texts, {document for document, _ in locations.values()}):
        if location in present or location not in units_by_page:
            continue
        scored = []
        for unit in units_by_page[location]:
            vector = unit_vector(unit)
            if vector is not None:
                scored.append((_cosine(query_vector, vector), unit))
        if not scored:
            continue
        best = max(score for score, _ in scored)
        if kth - best > margin:
            continue
        document_id, page = location
        if not label_matches(document_id, page):
            continue
        units = [unit for score, unit in sorted(scored, reverse=True) if kth - score <= margin]
        candidates.append({"document_id": document_id, "page": page, "gap": round(kth - best, 4), "units": units})
    candidates.sort(key=lambda item: item["gap"])
    return candidates[:max_pages]


def _locations(output_directory: Path) -> dict[str, tuple[str, int]]:
    locations: dict[str, tuple[str, int]] = {}
    for line in (output_directory / "source_metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("page") is not None:
            locations[str(item["text_unit_id"])] = (str(item.get("document_id", "")), int(item["page"]))
    return locations


def follow_references(result: Any, capture: Mapping[str, Any], root: Path) -> tuple[Any, dict[str, Any]]:
    """Append the passages of nearly-retrieved referenced pages to a GraphRAG Basic context."""
    import pandas as pd

    frame = result.context_records.get("sources") if isinstance(result.context_records, dict) else None
    results = capture.get("results") or []
    store, query_vector = capture.get("store"), capture.get("vector")
    if not isinstance(frame, pd.DataFrame) or frame.empty or not results or store is None or query_vector is None:
        return result, {}
    output_directory = root / "output"
    top_vectors = {str(item.document.id): item.document.vector for item in results if item.document.vector is not None}
    locations = _locations(output_directory)
    from .variants import project_path_for_index_root

    project_sources = project_path_for_index_root(root) / "source"

    def unit_vector(unit: str) -> Any:
        try:
            return store.search_by_id(unit).vector
        except Exception:
            return None

    def label_matches(document_id: str, page: int) -> bool:
        return page_label_matches(project_sources / document_id, page, f"{Path(document_id).stem}-{page}")

    selected = select_referenced_units(
        query_vector, top_vectors, [str(text) for text in frame["text"]], locations, unit_vector, label_matches,
    )
    if not selected:
        return result, {}
    wanted = {unit for item in selected for unit in item["units"]}
    text_units = pd.read_parquet(output_directory / "text_units.parquet", columns=["id", "human_readable_id", "text"])
    rows = {str(row["id"]): row for row in text_units.to_dict(orient="records") if str(row["id"]) in wanted}
    known = set(frame["id"].astype(str))
    added = []
    for item in selected:
        for unit in item["units"]:
            row = rows.get(unit)
            if row is not None and str(row["human_readable_id"]) not in known:
                known.add(str(row["human_readable_id"]))
                added.append({"id": str(row["human_readable_id"]), "text": str(row["text"])})
    if not added:
        return result, {}
    frame = pd.concat([frame, pd.DataFrame(added, columns=["id", "text"])], ignore_index=True)
    records = {**result.context_records, "sources": frame}
    chunks = frame.to_csv(index=False, escapechar="\\", sep="|")
    report = {"added": [{key: item[key] for key in ("document_id", "page", "gap")} for item in selected],
              "added_ids": [item["id"] for item in added]}
    return dataclasses.replace(result, context_chunks=chunks, context_records=records), report
