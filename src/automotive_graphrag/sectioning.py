"""Conservative heading-based section splitting for preprocessed manual pages."""

from __future__ import annotations

import re
import unicodedata


_EXPLICIT_HEADING = re.compile(
    r"^(?:第\s*[\d一二三四五六七八九十百]+\s*[章節篇]|"
    r"[\d一二三四五六七八九十]+(?:[.、．][\d一二三四五六七八九十]+)*[、.．)）])\s*\S+"
)
_ACTION_HEADING = re.compile(
    r"^(?:變更|更改|設定|強制|註冊|開啟|關閉|使用|新增|刪除|移除|選擇|設定|切換|"
    r"建立|管理|封鎖|解除|還原|移動|複製|傳送|拍攝|錄製|連接|中斷|啟用|停用|"
    r"將.*設為|如何)"
)
_BULLET_OR_STEP = re.compile(r"^(?:[•·▪◦※*]|[-–—]|\d+[.、．)]|[（(]\d+[）)])")


def looks_like_section_heading(line: str) -> bool:
    """Return true only for standalone, short heading forms common in manuals.

    Ambiguous lines remain in the current section. This deliberately favors a larger
    chunk over splitting a step or note away from its procedure.
    """
    value = " ".join(line.strip().split())
    if not value or len(value) > 48 or _BULLET_OR_STEP.match(value):
        return False
    if value.endswith(("。", "，", "、", ",", ";", "；", "：", ":", "!", "！", "?", "？")):
        return False
    if _EXPLICIT_HEADING.match(value):
        return True
    return bool(_ACTION_HEADING.match(value)) and not any(
        marker in value for marker in ("。", "，", "；", ";")
    )


def remove_leading_printed_page_number(text: str) -> str:
    """Drop an isolated folio near the page's beginning, retaining its page metadata."""
    lines = text.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return text
    for position, index in enumerate(nonempty[:3]):
        value = lines[index].strip()
        if not re.fullmatch(r"\d{1,4}", value):
            continue
        # A folio is usually surrounded by a running title or body text. Avoid
        # removing a lone numbered list item at the start of a page.
        neighbors = nonempty[max(0, position - 1):position] + nonempty[position + 1:position + 2]
        if position > 0 and neighbors and any(
            not re.fullmatch(r"\d{1,4}", lines[item].strip()) for item in neighbors
        ):
            lines[index] = ""
            break
    return "\n".join(lines).strip()


def split_into_sections(text: str, previous_heading: str = "") -> tuple[list[tuple[str, str]], str]:
    """Split text at confident headings and return (heading, body) blocks.

    A page with no recognizable heading is returned whole and inherits the previous
    section label when available, preserving procedures that continue across pages.
    """
    lines = text.splitlines()
    heading = previous_heading
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] = []
    for line in lines:
        candidate = line.strip()
        if looks_like_section_heading(candidate):
            if current:
                blocks.append((heading, current))
            heading = candidate
            current = [candidate]
        else:
            current.append(line)
    if current:
        blocks.append((heading, current))
    if not blocks:
        return [(previous_heading, text)], previous_heading

    compact = [(name, "\n".join(body).strip()) for name, body in blocks]
    compact = [(name, body) for name, body in compact if body]
    if len(compact) <= 1:
        # Do not claim a section title when the only candidate was an isolated line.
        return compact or [(previous_heading, text)], heading
    return compact, heading


def section_identifier(heading: str, fallback: str) -> str:
    value = unicodedata.normalize("NFKC", heading or fallback)
    value = re.sub(r"[^\w\u3400-\u9fff-]+", "-", value, flags=re.UNICODE).strip("-")
    return value[:64] or fallback
