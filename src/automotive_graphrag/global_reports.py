"""Map Global Search community reports to the manuals and pages they summarize (F-70).

A community report is written at index time from the text units of its community, so it
does not name the manual. Global answers therefore could not say which manuals contain
something, and its evidence had no PDF page. The community → text unit → source metadata
chain gives both without rebuilding the index.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

MANUALS_LINE_PREFIX = "Manuals:"
# GraphRAG citation format: "[Data: Reports (2, 7, 64, +more)]"; "+more" drops ids that cannot be recovered.
_REPORTS_REFERENCE = re.compile(r"Reports\s*\(([^)]*)\)", re.IGNORECASE)

_cache_lock = threading.Lock()
_pages_cache: dict[tuple[str, int, int], dict[str, dict[str, tuple[int, ...]]]] = {}


def _manual_name(document_id: str) -> str:
    return document_id[:-4] if document_id.lower().endswith(".pdf") else document_id


# A level-0 community spans a hundred pages; listing every range cost up to 333 tokens per report.
MAX_PAGE_RANGES = 6


def _page_ranges(pages: tuple[int, ...]) -> str:
    """Compact page ranges, "3–5, 9"; only the span "3–140" when there are many ranges."""
    parts: list[str] = []
    start = previous = None
    for page in pages:
        if start is None:
            start = previous = page
        elif page == previous + 1:
            previous = page
        else:
            parts.append(str(start) if start == previous else f"{start}–{previous}")
            start = previous = page
    if start is not None:
        parts.append(str(start) if start == previous else f"{start}–{previous}")
    if len(parts) > MAX_PAGE_RANGES:
        return f"{pages[0]}–{pages[-1]}"
    return ", ".join(parts)


def _text_unit_pages(output_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    path = output_dir / "source_metadata.jsonl"
    result: dict[str, tuple[str, tuple[int, ...]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pages = tuple(int(page) for page in (row.get("pages") or [row["page"]]))
            result[str(row["text_unit_id"])] = (str(row["document_id"]), pages)
    return result


def community_manual_pages(
    output_dir: Path, communities: pd.DataFrame,
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return {community id as text: {document_id: sorted pages}} for every community.

    Manuals are ordered by how many of the community's text units they hold, most first, so the
    manual a report is mainly about leads its "Manuals:" line.
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "source_metadata.jsonl"
    stat = metadata_path.stat()
    key = (str(metadata_path.resolve()), stat.st_mtime_ns, len(communities))
    with _cache_lock:
        cached = _pages_cache.get(key)
    if cached is not None:
        return cached
    text_units = _text_unit_pages(output_dir)
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for community, unit_ids in zip(communities["community"], communities["text_unit_ids"]):
        pages_by_document: dict[str, set[int]] = {}
        units_by_document: dict[str, int] = {}
        for unit_id in unit_ids if unit_ids is not None else ():
            found = text_units.get(str(unit_id))
            if found is not None:
                pages_by_document.setdefault(found[0], set()).update(found[1])
                units_by_document[found[0]] = units_by_document.get(found[0], 0) + 1
        order = sorted(pages_by_document, key=lambda document: (-units_by_document[document], document))
        result[str(int(community))] = {document: tuple(sorted(pages_by_document[document])) for document in order}
    with _cache_lock:
        _pages_cache.clear()
        _pages_cache[key] = result
    return result


def manuals_line(pages_by_document: dict[str, tuple[int, ...]]) -> str:
    """"Manuals: BR (pages 3–5, 9); BRC (pages 56)"; empty when nothing maps."""
    if not pages_by_document:
        return ""
    items = [
        f"{_manual_name(document)} (pages {_page_ranges(pages)})"
        for document, pages in pages_by_document.items()
    ]
    return f"{MANUALS_LINE_PREFIX} " + "; ".join(items)


def label_reports(
    reports: pd.DataFrame, communities: pd.DataFrame, output_dir: Path,
) -> pd.DataFrame:
    """Return a copy of the reports with a "Manuals:" line before each summary and full content."""
    pages = community_manual_pages(output_dir, communities)
    labelled = reports.copy()
    lines = [manuals_line(pages.get(str(int(community)), {})) for community in labelled["community"]]

    def prefix(column: str) -> list[str]:
        return [
            f"{line}\n\n{text}" if line else str(text)
            for line, text in zip(lines, labelled[column].fillna("").astype(str))
        ]

    for column in ("full_content", "summary"):
        if column in labelled.columns:
            labelled[column] = prefix(column)
    return labelled


def cited_report_ids(answer: str) -> list[str]:
    """Report ids cited in a Global answer, in first-citation order."""
    cited: list[str] = []
    for match in _REPORTS_REFERENCE.finditer(answer or ""):
        for part in match.group(1).split(","):
            value = part.strip()
            if value.isdigit() and value not in cited:
                cited.append(value)
    return cited


def keep_cited_reports(
    context: dict[str, Any], answer: str, pages: dict[str, dict[str, tuple[int, ...]]] | None,
) -> dict[str, Any]:
    """Keep only cited reports in the serialized context and attach their manual pages.

    The map step sees every report of the selected level (1,072 on SM3E), which made each
    question's stored evidence megabytes long while the answer used a handful.
    """
    reports = context.get("reports")
    if not isinstance(reports, list):
        return context
    cited = cited_report_ids(answer)
    by_id = {
        str(row.get("id")): row for row in reports
        if isinstance(row, dict) and row.get("id") is not None
    }
    kept: list[dict[str, Any]] = []
    for report_id in cited:
        row = by_id.get(report_id)
        if row is None:
            continue
        row = dict(row)
        community = str(row.get("community") or report_id)
        if pages is not None:
            row["manual_pages"] = {
                document: list(values) for document, values in pages.get(community, {}).items()
            }
        kept.append(row)
    context["reports"] = kept
    context["reports_in_context_count"] = len(reports)
    return context
