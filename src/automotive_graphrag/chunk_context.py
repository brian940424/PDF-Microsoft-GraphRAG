"""Give graph input records their manual heading context and split oversized pages (F-65).

Service manuals print a heading once (for example ``AV CONTROL UNIT`` then ``Reference
Value``) and continue its terminal table over several pages without repeating it. A
continuation page embedded alone has no words linking it to the component, so vector
retrieval ranks it far below unrelated pages. Long pages that hold several topics have
the opposite problem: one topic's sentences are diluted by the others.

This module walks each document's records in page order, carries a short heading trail
across pages, splits a long page at its headings and a long Markdown table into row groups
that repeat the table header, and prefixes every resulting text with the trail.

PDF extraction often loses the heading of the next diagnostic code (for example BRC p.88
starts at the ``C1155`` detection table without its ``C1155 BR FLUID LEVEL LOW`` heading),
so the trail kept the previous code and C1155 steps were labeled ``C1154 PNP SWITCH`` (F-69).
A DTC detection table (``| DTC | Display item | ...``) whose codes are not in the trail
therefore starts a new code: the page is cut there and the trail restarts at the code.
"""

from __future__ import annotations

import re
from typing import Any

from .sectioning import looks_like_section_heading

CHUNK_CONTEXT_VERSION = 2
# Pages at or below this size stay whole; the retrieval probe showed a heading prefix is enough.
SPLIT_MIN_CHARS = 2500
# Row groups and heading segments aim for this size.
TARGET_CHARS = 1500
# A heading starts a new segment only after this much body text, keeping short notes together.
MIN_SEGMENT_CHARS = 400

_INFOID = re.compile(r"^INFOID:\S+$")
_UPPER_HEADING = re.compile(r"^[A-Z][A-Z0-9&/.,'()\- ]*[A-Z0-9)]$")
_ACRONYM_EXPANSION = re.compile(r"^[A-Z][A-Z0-9-]{1,7}\s+\([A-Z][A-Za-z0-9 ,/&-]+\)$")
_TABLE_SEPARATOR = re.compile(r"^\|\s*:?-{3,}")
_DTC_CODE = re.compile(r"\b[BCPU][0-9A-F][0-9A-F]{3}\b")
# The detection-logic table of one code: "| DTC | Display item | ..." (CONSULT display column).
_DTC_TABLE_HEADER = re.compile(r"^\|\s*DTC\s*\|\s*Display\b", re.IGNORECASE)
DTC_TITLE = "DTC Logic"


_TITLE_WORD = re.compile(r"^(?:[A-Z][A-Za-z0-9/-]*|for|and|of|the|on|in|with|to|a|an)$")
CAPS = "caps"
TITLE = "title"


def _is_caps_heading(value: str) -> bool:
    if len(value) > 60 or value.startswith(("|", "•", "-", "*")) or ".." in value:
        return False
    if _ACRONYM_EXPANSION.match(value):
        return True
    if not _UPPER_HEADING.match(value):
        return False
    # Two lettered words rule out figure identifiers such as "AWMIA1310ZZ" and labels such as "NOTE".
    return sum(1 for word in value.split() if len(re.sub(r"[^A-Za-z]", "", word)) >= 2) >= 2


def _is_short_title(value: str) -> bool:
    words = value.split()
    return (
        1 <= len(words) <= 5 and len(value) <= 40 and words[0][:1].isupper()
        and all(_TITLE_WORD.match(word) for word in words) and not value.isupper()
    )


def heading_lines(lines: list[str]) -> dict[int, str]:
    """Heading line indexes and their kind.

    ``caps`` is an all-caps chapter, component or sub-topic heading; ``title`` is a subsection
    title, recognized by the INFOID printed under it or by directly following a caps heading.
    """
    found: dict[int, str] = {}
    previous: int | None = None
    for index, line in enumerate(lines):
        value = " ".join(line.split())
        if not value or _INFOID.match(value):
            continue
        following = next((other.strip() for other in lines[index + 1:] if other.strip()), "")
        if _INFOID.match(following):
            found[index] = TITLE
        elif _is_caps_heading(value):
            found[index] = CAPS
        elif previous is not None and found.get(previous) == CAPS and _is_short_title(value):
            found[index] = TITLE
        elif looks_like_section_heading(value):
            found[index] = TITLE
        previous = index
    return found


class _Trail:
    """Heading levels carried across pages.

    Caps headings stacked directly above a subsection title name the chapter and component;
    a caps heading after a title's body is a sub-topic of that title.
    """

    def __init__(self) -> None:
        self.outer: list[str] = []
        self.title = ""
        self.topic = ""
        self.pending: list[str] = []
        self.after_caps = False

    def heading(self, value: str, kind: str) -> None:
        if kind == TITLE:
            if self.pending and self.after_caps:
                self.outer = self.pending[-2:]
            self.title, self.topic, self.pending = value, "", []
            self.after_caps = False
            return
        self.pending = [*self.pending, value] if self.after_caps else [value]
        self.topic = value
        self.after_caps = True

    def restart(self, component: str, title: str) -> None:
        """Start a new component (a diagnostic code) whose own heading the page lost."""
        self.outer, self.title, self.topic, self.pending = [component], title, "", []
        self.after_caps = False

    def body(self) -> None:
        self.after_caps = False

    @property
    def items(self) -> list[str]:
        return [item for item in (*self.outer, self.title, self.topic) if item]


def dtc_switches(lines: list[str]) -> dict[int, str]:
    """Line indexes where a DTC detection table starts a code, with that code's heading.

    The index is the caps heading directly above the table (such as ``DTC DETECTION LOGIC``)
    when there is one, so the new code's context covers it. One code keeps its display item
    (``C1155 BR FLUID LEVEL LOW``); several codes are listed (``C1105, C1106``).
    """
    switches: dict[int, str] = {}
    for index, line in enumerate(lines[:-1]):
        if not _DTC_TABLE_HEADER.match(line.strip()) or not _TABLE_SEPARATOR.match(lines[index + 1].strip()):
            continue
        rows: list[tuple[str, str]] = []
        for row in lines[index + 2:]:
            if not row.lstrip().startswith("|"):
                break
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            code = _DTC_CODE.fullmatch(cells[0]) if cells else None
            if code:
                rows.append((code.group(), cells[1] if len(cells) > 1 else ""))
        codes = list(dict.fromkeys(code for code, _ in rows))
        if not codes:
            continue
        label = f"{codes[0]} {rows[0][1]}".strip() if len(codes) == 1 else ", ".join(codes)
        start = index
        previous = next((other for other in range(index - 1, -1, -1) if lines[other].strip()), None)
        if previous is not None and _is_caps_heading(" ".join(lines[previous].split())):
            start = previous
        switches[start] = label
    return switches


def _starts_new_code(label: str, trail: list[str]) -> bool:
    known = set(_DTC_CODE.findall(" ".join(trail)))
    return not known.intersection(_DTC_CODE.findall(label))


def _page_cuts(lines: list[str], trail: list[str]) -> set[int]:
    """Switch indexes whose code is neither in the trail nor in a heading above it on the page."""
    headings = heading_lines(lines)
    cuts: set[int] = set()
    for index, label in dtc_switches(lines).items():
        above = [" ".join(lines[other].split()) for other in headings if other < index]
        if _starts_new_code(label, [*trail, *above]):
            cuts.add(index)
    return cuts


def _cut_at(lines: list[str], cuts: set[int]) -> list[list[str]]:
    """Split page lines before the given indexes; a cut needs text above it."""
    result: list[list[str]] = [[]]
    for index, line in enumerate(lines):
        if index in cuts and any(item.strip() for item in result[-1]):
            result.append([])
        result[-1].append(line)
    return result


def _table_row_groups(lines: list[str]) -> list[list[str]] | None:
    """Split one segment holding a Markdown table into row groups that repeat the header."""
    try:
        separator = next(index for index, line in enumerate(lines) if _TABLE_SEPARATOR.match(line.strip()))
    except StopIteration:
        return None
    if separator == 0 or not lines[separator - 1].lstrip().startswith("|"):
        return None
    header_start = separator - 1
    end = separator + 1
    while end < len(lines) and lines[end].lstrip().startswith("|"):
        end += 1
    before, header, rows, after = lines[:header_start], lines[header_start:separator + 1], lines[separator + 1:end], lines[end:]
    # A row whose first cell is empty continues the row above (another condition of one terminal).
    logical: list[list[str]] = []
    for row in rows:
        cells = row.strip().strip("|").split("|")
        if logical and (not cells or not cells[0].strip()):
            logical[-1].append(row)
        else:
            logical.append([row])
    groups: list[list[str]] = []
    current: list[str] = []
    for row in logical:
        if current and sum(len(line) for line in current) + sum(len(line) for line in row) > TARGET_CHARS:
            groups.append(current)
            current = []
        current.extend(row)
    if current:
        groups.append(current)
    if len(groups) <= 1:
        return None
    pieces = [header + group for group in groups]
    pieces[0] = before + pieces[0]
    pieces[-1] = pieces[-1] + after
    return pieces


def _split_page(lines: list[str], headings: dict[int, str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    body_chars = 0
    for index, line in enumerate(lines):
        if index in headings and body_chars >= MIN_SEGMENT_CHARS:
            segments.append([])
            body_chars = 0
        segments[-1].append(line)
        if index not in headings:
            body_chars += len(line.strip())
    result: list[list[str]] = []
    for segment in segments:
        groups = _table_row_groups(segment) if sum(len(line) for line in segment) > TARGET_CHARS else None
        result.extend(groups or [segment])
    return [segment for segment in result if any(line.strip() for line in segment)]


def _context(record: dict[str, Any], trail: list[str]) -> str:
    path = [part.strip() for part in str(record.get("section_path") or record.get("document_id") or "").split(" > ")]
    path = [part for part in path if part]
    seen = {part.casefold() for part in path}
    extra = [item for item in trail if item.casefold() not in seen]
    if not extra:
        return ""
    return " > ".join(path + extra)


def contextualize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefix heading context and split oversized pages; records keep their page and chunk ids."""
    output: list[dict[str, Any]] = []
    trails: dict[str, _Trail] = {}
    for record in records:
        text = str(record.get("text", ""))
        document = str(record.get("document_id", ""))
        trail = trails.setdefault(document, _Trail())
        lines = text.splitlines()
        segments: list[list[str]] = []
        for part in _cut_at(lines, _page_cuts(lines, trail.items)):
            if sum(len(line) + 1 for line in part) > SPLIT_MIN_CHARS:
                segments.extend(_split_page(part, heading_lines(part)))
            elif any(line.strip() for line in part):
                segments.append(part)
        segments = segments or [lines]
        pieces: list[tuple[str, str]] = []
        for segment in segments:
            # Headings that open the segment are part of its own context.
            opening = True
            context = ""
            headings = heading_lines(segment)
            switches = dtc_switches(segment)
            for offset, line in enumerate(segment):
                value = " ".join(line.split())
                switch = switches.get(offset)
                if switch and _starts_new_code(switch, trail.items):
                    trail.restart(switch, DTC_TITLE)
                if not value or _INFOID.match(value):
                    continue
                if offset in headings:
                    trail.heading(value, headings[offset])
                    continue
                if opening:
                    context = _context(record, trail.items)
                    opening = False
                trail.body()
            if opening:
                context = _context(record, trail.items)
            pieces.append((context, "\n".join(segment).strip()))
        for number, (context, body) in enumerate(pieces, start=1):
            item = dict(record)
            if len(pieces) > 1:
                item["id"] = f'{record.get("id")}-c{number:02d}'
            item["text"] = f"{context}\n\n{body}" if context else body
            item["chunk_context_version"] = CHUNK_CONTEXT_VERSION
            output.append(item)
    return output
