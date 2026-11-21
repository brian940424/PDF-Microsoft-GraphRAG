"""Passages repeated on several pages or in several manuals (F-59).

Service manuals print the same caution (for example the SRS power-tool warning or the
CONSULT disconnect caution) in many sections and manuals. The answer model cites one or
two of them. Here every repeated passage is found from the PDF text layers; when an answer
sentence uses a repeated passage on a cited page, all other pages with that passage are
added to the citations by code, marked ``added_by="duplicate_passage"``.

Pages of other variant chapters of a multi-variant manual (F-58) are not added: only the
variants the question named or, without a named variant, the first chapter that has the
passage. Scoring treats pages with the same page text as equivalent (`PageEquivalence`).
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .evaluation.source_scoring import cited_evidence, strip_citations
from .variants import (
    BODY_BOTTOM,
    BODY_TOP,
    DUPLICATE_SIMILARITY,
    _atomic_json,
    _comparable,
    _trigrams,
    project_variants,
    read_page_bodies,
)

ADDED_BY = "duplicate_passage"
CACHE_FILE = "duplicate_passages.json"
# Text blocks shorter than this (normalized characters) are headings or labels, not passages.
PASSAGE_MIN_CHARS = 120
# Two blocks are the same passage when this share of the smaller block's word trigrams is in
# the other (containment, so a block merged with the next heading still matches). SM3E: the
# SRS power-tool warning ("three" vs "3", one block merged with a heading) scores 0.95.
PASSAGE_CONTAINMENT = 0.9
PASSAGE_SIZE_RATIO = 0.6
# An answer sentence uses a passage when this share of its content words appear in it.
# SM3E: sentences built on a passage score 0.82-1.0, other passages on the same page <= 0.58.
ANSWER_MATCH = 0.7
_STOPWORDS = frozenset(
    "the and for with from that this when then than are was were not all any can may will must into "
    "onto each other only also have has had been use used using before after while which what does "
    "should would there their they them its you your".split()
)


@dataclass(frozen=True, slots=True)
class PassageGroup:
    locations: tuple[tuple[str, int], ...]
    text: str


def content_words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z][a-z0-9]{2,}", text.lower()) if word not in _STOPWORDS}


def group_passages(blocks: Sequence[tuple[str, int, str]]) -> list[PassageGroup]:
    """Group (document, page, text) blocks that hold the same passage; only groups on 2+ pages."""
    items = []
    for document_id, page, text in blocks:
        normalized = _comparable(text, "")
        if len(normalized) >= PASSAGE_MIN_CHARS:
            items.append((document_id, page, text, _trigrams(normalized)))
    parent = list(range(len(items)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    order = sorted(range(len(items)), key=lambda index: len(items[index][3]))
    for position, first in enumerate(order):
        small = items[first][3]
        if not small:
            continue
        for second in order[position + 1:]:
            large = items[second][3]
            # Compare blocks of similar size only: a passage inside a much larger block is a
            # different passage with extra content, not a repeat.
            if len(small) < PASSAGE_SIZE_RATIO * len(large):
                break
            if items[first][:2] != items[second][:2] and len(small & large) / len(small) >= PASSAGE_CONTAINMENT:
                parent[root(first)] = root(second)
    members: dict[int, list[int]] = {}
    for index in range(len(items)):
        members.setdefault(root(index), []).append(index)
    groups = []
    for indexes in members.values():
        locations = tuple(sorted({items[index][:2] for index in indexes}))
        if len(locations) > 1:
            first = min(indexes, key=lambda index: items[index][:2])
            groups.append(PassageGroup(locations, items[first][2]))
    return sorted(groups, key=lambda group: group.locations)


def _pdf_blocks(path: Path) -> list[tuple[int, str]]:
    import pymupdf

    document = pymupdf.open(path)
    try:
        result = []
        for number, page in enumerate(document, start=1):
            rect = page.rect
            band = pymupdf.Rect(rect.x0, rect.y0 + rect.height * BODY_TOP, rect.x1, rect.y0 + rect.height * BODY_BOTTOM)
            result.extend((number, str(block[4])) for block in page.get_text("blocks", clip=band))
        return result
    finally:
        document.close()


def project_passage_groups(project_path: Path, output_directory: Path) -> list[PassageGroup]:
    """Repeated passages across the project's PDFs, cached in the index output by PDF stamps."""
    source_directory = project_path / "source"
    pdfs = sorted(source_directory.glob("*.pdf")) if source_directory.is_dir() else []
    stamps = {pdf.name: f"{pdf.stat().st_size}-{pdf.stat().st_mtime_ns}" for pdf in pdfs}
    cache_path = output_directory / CACHE_FILE
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("stamps") == stamps:
            return [PassageGroup(tuple((str(d), int(p)) for d, p in item["locations"]), str(item["text"]))
                    for item in cached.get("groups", [])]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    blocks = []
    for pdf in pdfs:
        try:
            blocks.extend((pdf.name, page, text) for page, text in _pdf_blocks(pdf))
        except Exception:
            continue
    groups = group_passages(blocks)
    _atomic_json(cache_path, {
        "stamps": stamps,
        "groups": [{"locations": [list(location) for location in group.locations], "text": group.text} for group in groups],
    })
    return groups


def used_groups(answer: str, cited_pages: Iterable[tuple[str, int]], groups: Sequence[PassageGroup]) -> list[PassageGroup]:
    """Repeated passages on a cited page that an answer sentence is built on."""
    cited = set(cited_pages)
    sentences = [
        words for sentence in re.split(r"(?<=[.!?。])\s+|\n+", strip_citations(answer))
        if len(words := content_words(sentence)) >= 3
    ]
    result = []
    for group in groups:
        if not cited.intersection(group.locations):
            continue
        passage = content_words(group.text)
        if any(len(words & passage) / len(words) >= ANSWER_MATCH for words in sentences):
            result.append(group)
    return result


def expansion_locations(
    groups: Sequence[PassageGroup], cited_pages: Iterable[tuple[str, int]],
    variants: Mapping[str, Mapping[int, str]], named: Mapping[str, Iterable[str]] | None = None,
) -> list[tuple[str, int]]:
    """Locations to add: every page of each group, limited to one variant chapter per F-58 rule."""
    cited = set(cited_pages)
    named_sets = {document: set(values) for document, values in (named or {}).items()}
    added: list[tuple[str, int]] = []
    for group in groups:
        by_document: dict[str, list[int]] = {}
        for document_id, page in group.locations:
            by_document.setdefault(document_id, []).append(page)
        for document_id, pages in by_document.items():
            tags = variants.get(document_id, {})
            if len(set(tags.values())) >= 2:
                allowed = named_sets.get(document_id)
                if allowed:
                    pages = [page for page in pages if tags.get(page) in allowed]
                else:
                    first = tags.get(min(pages), "")
                    pages = [page for page in pages if tags.get(page, "") == first]
            for page in sorted(pages):
                location = (document_id, page)
                if location not in cited and location not in added:
                    added.append(location)
    return added


def with_duplicate_citations(
    projects: Any, project_id: str, answer: str, evidence: Sequence[Any], context: Mapping[str, Any] | None = None,
    text_units: Any = None,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Append Evidence for the other pages of repeated passages the answer used.

    A page already retrieved but not cited is marked instead of added again. Returns the
    evidence (original items first) and a record of each page whose citation was added.
    """
    cited = [item for item in cited_evidence(answer, evidence) if isinstance(getattr(item, "page", None), int)]
    if not cited:
        return tuple(evidence), []
    project_path = projects.path_for(project_id)
    output_directory = project_path / "graphrag" / "output"
    groups = project_passage_groups(project_path, output_directory)
    cited_pages = {(item.document_id, item.page) for item in cited}
    groups_used = used_groups(answer, cited_pages, groups)
    if not groups_used:
        return tuple(evidence), []
    named = ((context or {}).get("variant_selection") or {}).get("named_variants") or {}
    variants = project_variants(project_path, output_directory)
    locations = expansion_locations(groups_used, cited_pages, variants, named)
    if not locations:
        return tuple(evidence), []
    result = list(evidence)
    records = []
    wanted = set(locations)
    for index, item in enumerate(result):
        location = (item.document_id, item.page)
        if location in wanted and getattr(item, "source_type", "text_unit") == "text_unit":
            # Retrieved but not cited: the citation is added by code.
            result[index] = dataclasses.replace(item, added_by=ADDED_BY)
            wanted.discard(location)
            records.append({"document_id": location[0], "page": location[1], "retrieved": True})
    locations = [location for location in locations if location in wanted]
    if not locations:
        return tuple(result), records

    from .evidence import Evidence, section_path_with_variant
    from .source_metadata import SourceMetadataService

    import pandas as pd

    metadata_by_page: dict[tuple[str, int], list[Any]] = {}
    for item in SourceMetadataService(projects).load(project_id):
        metadata_by_page.setdefault((item.document_id, item.page), []).append(item)
    if text_units is None:
        text_units = pd.read_parquet(output_directory / "text_units.parquet", columns=["id", "human_readable_id"])
    else:
        text_units = text_units[["id", "human_readable_id"]]
    short_id = {str(row["id"]): str(row["human_readable_id"]) for row in text_units.to_dict(orient="records")}
    for document_id, page in locations:
        units = metadata_by_page.get((document_id, page))
        if not units:
            continue
        group = next(group for group in groups_used if (document_id, page) in group.locations)
        passage = content_words(group.text)
        unit = max(units, key=lambda item: len(content_words(item.text) & passage))
        variant = variants.get(document_id, {}).get(page, "")
        rank = len(result) + 1
        result.append(Evidence(
            evidence_id=f"E{rank}", rank=rank, context_id=short_id.get(unit.text_unit_id, ""),
            text_unit_id=unit.text_unit_id, chunk_id=unit.chunk_id, section_id=unit.section_id,
            section_name=unit.section_name, document_id=document_id, page=page, block_id=unit.block_id,
            text=unit.text, section_path=section_path_with_variant(unit.section_path, variant),
            variant=variant, added_by=ADDED_BY,
        ))
        records.append({"document_id": document_id, "page": page, "retrieved": False})
    return tuple(result), records


class PageEquivalence:
    """Pages of one document whose PDF text layers are the same (F-58 standard), for scoring."""

    def __init__(self, project_path: Path) -> None:
        self.source_directory = project_path / "source"
        self._trigrams: dict[tuple[str, int], set[str]] = {}

    def _page(self, document_id: str, page: int) -> set[str]:
        key = (document_id, page)
        if key not in self._trigrams:
            path = self.source_directory / document_id
            try:
                text = read_page_bodies(path, [page]).get(page, "") if path.is_file() else ""
            except Exception:
                text = ""
            self._trigrams[key] = _trigrams(_comparable(text, ""))
        return self._trigrams[key]

    def __call__(self, document_id: str, first: int, second: int) -> bool:
        if first == second:
            return True
        a, b = self._page(document_id, first), self._page(document_id, second)
        return bool(a and b) and len(a & b) / len(a | b) >= DUPLICATE_SIMILARITY

