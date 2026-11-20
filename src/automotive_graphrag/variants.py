"""Manual variant tags (vehicle model or equipment version) read from PDF page headers.

Service manuals repeat whole chapters for each equipment version (for example
``[BASE AUDIO]`` and ``[NAVIGATION WITH BOSE]``) with nearly identical text. The tag is
printed in the page header, which preprocessing leaves out, so it is read from the PDF
here and cached next to the index. Basic search uses it to keep only the version named
in the question, to keep only the first chapter when the same text is retrieved from
several versions, and to show the tag to the answer model (F-58).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# Header band searched for the tag, as a share of the page height.
HEADER_BAND = 0.1
# Pages whose PDF text layers share at least this share of word trigrams (Jaccard) are the
# same passage. Measured on the SM3E AV manual: repeated chapters score 0.96-1.0, while the
# chunk text is unusable for this because Vision describes the same figure differently.
DUPLICATE_SIMILARITY = 0.9
# Body band compared for duplicates: below the running header, above the footer.
BODY_TOP = 0.08
BODY_BOTTOM = 0.95
CACHE_FILE = "page_variants.json"

_VARIANT_LINE = re.compile(r"^\[([A-Z0-9][A-Z0-9 &/+.\-]{0,46})\]$")
_CHAPTER_LINE = re.compile(r"^<\s*(.+?)\s*>$")
_FAMILY_SPLIT = re.compile(r"\s+(?:WITH|WITHOUT)\s+")
# Question words in other languages for common variant families.
VARIANT_ALIASES = {"導航": "navigation", "导航": "navigation"}


def header_tags(header_text: str) -> tuple[str, str]:
    """Return (variant, chapter) found on standalone header lines; empty strings when absent."""
    variant = chapter = ""
    for line in header_text.splitlines():
        value = " ".join(line.split())
        if not variant and (match := _VARIANT_LINE.match(value)):
            variant = match.group(1)
        elif not chapter and (match := _CHAPTER_LINE.match(value)):
            chapter = match.group(1)
    return variant, chapter


def assign_variants(headers: Iterable[tuple[str, str]]) -> list[str]:
    """Give each page its tag; an untagged page inherits the previous tag only within the same chapter."""
    result: list[str] = []
    previous_variant = previous_chapter = ""
    for variant, chapter in headers:
        if not variant and chapter and chapter == previous_chapter:
            variant = previous_variant
        result.append(variant)
        previous_variant, previous_chapter = variant, chapter
    return result


def read_pdf_variants(path: Path) -> dict[int, str]:
    """Map 1-based page numbers to their variant tag (pages without a tag are left out)."""
    import pymupdf

    document = pymupdf.open(path)
    try:
        headers = []
        for page in document:
            rect = page.rect
            band = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * HEADER_BAND)
            headers.append(header_tags(page.get_text("text", clip=band) or ""))
    finally:
        document.close()
    return {number: tag for number, tag in enumerate(assign_variants(headers), start=1) if tag}


def project_variants(project_path: Path, output_directory: Path) -> dict[str, dict[int, str]]:
    """Return {PDF filename: {page: variant}} for the project's PDFs, cached in the index output."""
    source_directory = project_path / "source"
    cache_path = output_directory / CACHE_FILE
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        cached = {}
    documents: dict[str, Any] = {}
    changed = False
    for pdf in sorted(source_directory.glob("*.pdf")) if source_directory.is_dir() else []:
        stamp = f"{pdf.stat().st_size}-{pdf.stat().st_mtime_ns}"
        entry = cached.get("documents", {}).get(pdf.name)
        if not isinstance(entry, dict) or entry.get("stamp") != stamp:
            try:
                pages = read_pdf_variants(pdf)
            except Exception:
                pages = {}
            entry = {"stamp": stamp, "pages": {str(page): tag for page, tag in pages.items()}}
            changed = True
        documents[pdf.name] = entry
    if changed or set(documents) != set(cached.get("documents", {})):
        _atomic_json(cache_path, {"documents": documents})
    return {
        name: {int(page): str(tag) for page, tag in entry.get("pages", {}).items()}
        for name, entry in documents.items()
    }


def project_path_for_index_root(root: Path) -> Path:
    """Project folder for a GraphRAG root; experiment roots link `output` to the project index."""
    return (root / "output").resolve().parent.parent


def _normalize_name(value: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z㐀-鿿]+", " ", value.lower()).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def question_variants(question: str, variants: Iterable[str]) -> set[str]:
    """Variants named in the question: full names first, else a family such as "navigation"."""
    text = question
    for alias, replacement in VARIANT_ALIASES.items():
        text = text.replace(alias, f" {replacement} ")
    normalized = _normalize_name(text)
    names = set(variants)
    full = {name for name in names if _contains_phrase(normalized, _normalize_name(name))}
    if full:
        return full
    return {
        name for name in names
        if _FAMILY_SPLIT.search(name) and _contains_phrase(normalized, _normalize_name(_FAMILY_SPLIT.split(name)[0]))
    }


@dataclass(frozen=True, slots=True)
class VariantSource:
    """One retrieved text unit with its location and variant."""

    key: str
    document_id: str
    page: int | None
    variant: str
    text: str
    compare_text: str = ""


def _comparable(text: str, variant: str) -> str:
    value = text.replace(f"[{variant}]", " ") if variant else text
    value = re.sub(r"INFOID:\s*\d+", " ", value)
    value = re.sub(r"\b[A-Z]{2,4}-\d+\b", " ", value)  # page cross references such as AV-204
    return " ".join(value.lower().split())


def _trigrams(text: str) -> set[str]:
    words = re.findall(r"[0-9a-z\u3400-\u9fff]+", text)
    return {" ".join(words[index:index + 3]) for index in range(max(len(words) - 2, 0))}


def read_page_bodies(path: Path, pages: Iterable[int]) -> dict[int, str]:
    """PDF text layer of the given pages without header and footer (no Vision text)."""
    import pymupdf

    document = pymupdf.open(path)
    try:
        result = {}
        for page_number in sorted(set(pages)):
            if 1 <= page_number <= document.page_count:
                page = document[page_number - 1]
                rect = page.rect
                band = pymupdf.Rect(rect.x0, rect.y0 + rect.height * BODY_TOP, rect.x1, rect.y0 + rect.height * BODY_BOTTOM)
                result[page_number] = page.get_text("text", clip=band) or ""
        return result
    finally:
        document.close()


def select_sources(
    sources: list[VariantSource], question: str, catalog: Mapping[str, Mapping[int, str]],
) -> tuple[list[VariantSource], dict[str, Any]]:
    """Drop retrieved sources of other variants than the question names, then repeated passages.

    Only documents with two or more variants are touched. When the question names a
    variant but none of the retrieved sources of that document carry it, nothing is
    dropped for that document. Returns the kept sources (original order) and a report.
    """
    dropped: dict[str, str] = {}
    named_by_document: dict[str, set[str]] = {}
    for document_id in {source.document_id for source in sources}:
        variants = set(catalog.get(document_id, {}).values())
        if len(variants) < 2:
            continue
        named = question_variants(question, variants)
        if not named:
            continue
        document_sources = [source for source in sources if source.document_id == document_id and source.variant]
        if not any(source.variant in named for source in document_sources):
            continue
        named_by_document[document_id] = named
        for source in document_sources:
            if source.variant not in named:
                dropped[source.key] = f"question names {', '.join(sorted(named))}"

    kept = [source for source in sources if source.key not in dropped]
    multi_variant = {name for name, pages in catalog.items() if len(set(pages.values())) >= 2}
    candidates = sorted(
        (source for source in kept if source.document_id in multi_variant and source.variant),
        key=lambda source: (source.document_id, source.page if source.page is not None else 10**9),
    )
    comparable = {source.key: _trigrams(_comparable(source.compare_text or source.text, source.variant)) for source in candidates}
    for index, first in enumerate(candidates):
        if first.key in dropped:
            continue
        for later in candidates[index + 1:]:
            if later.key in dropped or later.document_id != first.document_id or later.variant == first.variant:
                continue
            a, b = comparable[first.key], comparable[later.key]
            if a and b and len(a & b) / len(a | b) >= DUPLICATE_SIMILARITY:
                dropped[later.key] = f"same passage as {first.document_id} p.{first.page} [{first.variant}]"

    kept = [source for source in sources if source.key not in dropped]
    report = {
        "named_variants": {name: sorted(values) for name, values in named_by_document.items()},
        "dropped": [
            {"id": source.key, "document_id": source.document_id, "page": source.page,
             "variant": source.variant, "reason": dropped[source.key]}
            for source in sources if source.key in dropped
        ],
    }
    return kept, report


def earliest_equivalent_pages(
    sources: Iterable[VariantSource], catalog: Mapping[str, Mapping[int, str]],
    bodies: Mapping[str, Mapping[int, str]], named: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, int]:
    """For sources from a later variant chapter, the first page of another variant with the same text.

    `bodies` holds the PDF text layer of every page of each multi-variant document. When the
    question named variants of a document (`named`), only those variants are candidates.
    Returns {source key: earlier page}; sources without an earlier equivalent are left out.
    """
    named = {document: set(values) for document, values in (named or {}).items()}
    trigrams: dict[tuple[str, int], set[str]] = {}

    def page_trigrams(document_id: str, page: int) -> set[str]:
        key = (document_id, page)
        if key not in trigrams:
            pages = catalog.get(document_id, {})
            trigrams[key] = _trigrams(_comparable(bodies.get(document_id, {}).get(page, ""), pages.get(page, "")))
        return trigrams[key]

    result: dict[str, int] = {}
    for source in sources:
        pages = catalog.get(source.document_id, {})
        if source.page is None or not source.variant or len(set(pages.values())) < 2:
            continue
        own = page_trigrams(source.document_id, source.page)
        if not own:
            continue
        allowed = named.get(source.document_id)
        for candidate in sorted(
            page for page, variant in pages.items()
            if page < source.page and variant != source.variant and (allowed is None or variant in allowed)
        ):
            other = page_trigrams(source.document_id, candidate)
            if other and len(own & other) / len(own | other) >= DUPLICATE_SIMILARITY:
                result[source.key] = candidate
                break
    return result


def with_variant_prefix(text: str, variant: str) -> str:
    """Show the variant tag to the answer model at the start of a source text."""
    if not variant or text.lstrip().startswith(f"[{variant}]"):
        return text
    return f"[{variant}] {text}"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".page-variants-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _unit_locations(output_directory: Path, text_units: Any = None) -> dict[str, tuple[str, int | None]]:
    """Map GraphRAG short text-unit ids to (PDF filename, page)."""
    import pandas as pd

    if text_units is None:
        text_units = pd.read_parquet(output_directory / "text_units.parquet", columns=["id", "human_readable_id"])
    else:
        text_units = text_units[["id", "human_readable_id"]]
    short_by_id = {str(row["id"]): str(row["human_readable_id"]) for row in text_units.to_dict(orient="records")}
    locations: dict[str, tuple[str, int | None]] = {}
    for line in (output_directory / "source_metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        short_id = short_by_id.get(str(item.get("text_unit_id")))
        if short_id is not None:
            page = item.get("page")
            locations[short_id] = (str(item.get("document_id", "")), int(page) if page is not None else None)
    return locations


def _replace_with_earliest(
    kept: list[VariantSource], catalog: Mapping[str, Mapping[int, str]], project_path: Path,
    output_directory: Path, locations: Mapping[str, tuple[str, int | None]],
    named: Mapping[str, Iterable[str]] | None = None,
    text_units: Any = None,
) -> tuple[list[VariantSource], list[dict[str, Any]]]:
    """Swap sources of a later variant chapter for the text units of the first chapter with the same page."""
    import pandas as pd

    documents = {
        source.document_id for source in kept
        if source.variant and len(set(catalog.get(source.document_id, {}).values())) >= 2
    }
    bodies = {}
    for document_id in documents:
        pdf = project_path / "source" / document_id
        if pdf.is_file():
            bodies[document_id] = read_page_bodies(pdf, catalog[document_id].keys())
    earlier = earliest_equivalent_pages(kept, catalog, bodies, named)
    if not earlier:
        return kept, []
    units_by_page: dict[tuple[str, int], list[str]] = {}
    for key, (document_id, page) in locations.items():
        if page is not None:
            units_by_page.setdefault((document_id, page), []).append(key)
    wanted = {unit for source in kept if source.key in earlier
              for unit in units_by_page.get((source.document_id, earlier[source.key]), [])}
    texts: dict[str, str] = {}
    if wanted:
        if text_units is None:
            text_units = pd.read_parquet(output_directory / "text_units.parquet", columns=["human_readable_id", "text"])
        else:
            text_units = text_units[["human_readable_id", "text"]]
        texts = {str(row["human_readable_id"]): str(row["text"]) for row in text_units.to_dict(orient="records")
                 if str(row["human_readable_id"]) in wanted}
    result: list[VariantSource] = []
    seen: set[str] = set()
    replaced: list[dict[str, Any]] = []
    for source in kept:
        page = earlier.get(source.key)
        units = sorted(units_by_page.get((source.document_id, page), []), key=lambda unit: (len(unit), unit)) if page else []
        units = [unit for unit in units if unit in texts]
        if not units:
            if source.key not in seen:
                seen.add(source.key)
                result.append(source)
            continue
        replaced.append({"id": source.key, "document_id": source.document_id, "page": source.page,
                         "variant": source.variant, "replaced_by_page": page,
                         "replaced_by_variant": catalog[source.document_id].get(page, "")})
        for unit in units:
            if unit not in seen:
                seen.add(unit)
                result.append(VariantSource(unit, source.document_id, page, catalog[source.document_id].get(page, ""), texts[unit]))
    return result, replaced


def select_basic_context(
    result: Any, question: str, root: Path, text_units: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply variant selection to a GraphRAG Basic context and tag each kept source with its variant."""
    import dataclasses

    import pandas as pd

    frame = result.context_records.get("sources") if isinstance(result.context_records, dict) else None
    if not isinstance(frame, pd.DataFrame) or frame.empty or not {"id", "text"}.issubset(frame.columns):
        return result, {}
    output_directory = root / "output"
    catalog = project_variants(project_path_for_index_root(root), output_directory.resolve())
    if not any(len(set(pages.values())) >= 2 for pages in catalog.values()):
        return result, {}
    locations = _unit_locations(output_directory, text_units)
    sources = []
    for row in frame.to_dict(orient="records"):
        key = str(row["id"])
        document_id, page = locations.get(key, ("", None))
        variant = catalog.get(document_id, {}).get(page, "") if page is not None else ""
        sources.append(VariantSource(key, document_id, page, variant, str(row["text"])))
    bodies: dict[tuple[str, int], str] = {}
    project_path = project_path_for_index_root(root)
    for document_id in {source.document_id for source in sources if source.variant}:
        pages = [source.page for source in sources if source.document_id == document_id and source.page is not None]
        pdf = project_path / "source" / document_id
        if pdf.is_file():
            bodies.update({(document_id, page): text for page, text in read_page_bodies(pdf, pages).items()})
    sources = [
        dataclasses.replace(source, compare_text=bodies.get((source.document_id, source.page), ""))
        if source.page is not None else source
        for source in sources
    ]
    kept, report = select_sources(sources, question, catalog)
    kept, replaced = _replace_with_earliest(
        kept, catalog, project_path, output_directory, locations, report["named_variants"], text_units,
    )
    report["replaced"] = replaced
    kept_rows = {source.key: source for source in kept}
    known = set(frame["id"].astype(str))
    added = [{"id": source.key, "text": source.text} for source in kept if source.key not in known]
    frame = pd.concat([frame, pd.DataFrame(added, columns=["id", "text"])], ignore_index=True) if added else frame
    order = {source.key: index for index, source in enumerate(kept)}
    kept_frame = frame[frame["id"].astype(str).isin(kept_rows)].copy()
    kept_frame = kept_frame.iloc[sorted(range(len(kept_frame)), key=lambda i: order[str(kept_frame["id"].iloc[i])])]
    kept_frame = kept_frame.reset_index(drop=True)
    variant_by_key = {source.key: source.variant for source in kept}
    kept_frame["text"] = [
        with_variant_prefix(str(text), variant_by_key.get(str(key), ""))
        for key, text in zip(kept_frame["id"], kept_frame["text"])
    ]
    records = {**result.context_records, "sources": kept_frame}
    chunks = kept_frame.to_csv(index=False, escapechar="\\", sep="|")
    report["tagged"] = sum(1 for variant in variant_by_key.values() if variant)
    return dataclasses.replace(result, context_chunks=chunks, context_records=records), report
