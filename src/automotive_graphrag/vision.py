"""Vision-model transcription of PDF pages for preprocessing.

Pages are rendered to PNG and sent to an OpenAI-compatible Chat Completions
endpoint, which returns the page text with tables as Markdown and figures
described in words, so GraphRAG can index content that the text layer misses.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import pymupdf

from .connections import ALLOWED_CHAT_MODELS, GPT6_LUNA_MODEL, REASONING_EFFORT_OPTIONS
from .projects import ProjectError


ALLOWED_VISION_MODELS = ALLOWED_CHAT_MODELS
DEFAULT_VISION_CONCURRENCY = 8
RENDER_SCALE = 2.0
REQUEST_TIMEOUT_SECONDS = 180
MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2.0
# HTTP errors that fail identically on every page (bad key, no access, unknown
# model); stop sending requests instead of retrying each page.
ABORT_HTTP_CODES = {401, 403, 404}
RETRY_HTTP_CODES = {429, 500, 502, 503, 504}

# Which pages are sent to the vision model: every page, or only pages with a
# detected figure/table, only tables, or only diagrams (images and vector
# illustrations). Pages without a text layer, such as scans, are always sent.
SCOPE_ALL = "all"
SCOPE_FIGURES = "figures"
SCOPE_TABLES = "tables"
SCOPE_DIAGRAMS = "diagrams"
VISION_SCOPES = (SCOPE_FIGURES, SCOPE_TABLES, SCOPE_DIAGRAMS, SCOPE_ALL)
DEFAULT_VISION_SCOPE = SCOPE_FIGURES
VISION_REASONING_EFFORT_OPTIONS = ("none", *REASONING_EFFORT_OPTIONS)
FIGURE_KINDS = ("table", "image", "drawing")
DIAGRAM_KINDS = ("image", "drawing")
SCOPE_KINDS: dict[str, tuple[str, ...]] = {
    SCOPE_ALL: FIGURE_KINDS,
    SCOPE_FIGURES: FIGURE_KINDS,
    SCOPE_TABLES: ("table",),
    SCOPE_DIAGRAMS: DIAGRAM_KINDS,
}

SYSTEM_PROMPT = """You are an expert assistant for transcribing technical document pages. Convert the supplied page image into a JSON object with page_number and content fields.
Follow these rules strictly:
1. Transcribe all visible text completely, in its original language and script, without translating, summarizing, or omitting anything, except the page furniture listed in rule 2.
2. Do not transcribe page numbers (for example "12", "- 12 -", "Page 12", "第 12 頁", or "12 / 300"), or running headers and footers repeated on every page, such as the document title, chapter name, or copyright line printed at the top or bottom edge of the page. Numbers that belong to the content, such as step numbers, callout numbers, part numbers, and values in tables, must still be transcribed.
3. Convert every table, including specification tables, into Markdown table syntax.
4. For each diagram, figure, photo, exploded view, wiring diagram, flowchart, or screen illustration, write a description in the same language as the page text. State what the figure shows and which view it is (for example front, back, side, top, bottom, or screen), then go through its labeled components, callouts, or steps one by one, copying every label exactly as printed; never translate a label. For every labeled component, state where it is on the object: its position (top, bottom, left, right, center, corner, edge, front or back face) and its position relative to nearby labeled components (above, below, beside, between, next to). For a flowchart or sequence of steps, state the order and the connections between them. Do not invent labels that are not visible, and do not guess positions that the figure does not show.
5. Add content items in visual top-to-bottom order. Each item must have a type of text, table, or diagram and a string data field.
6. Return exactly one valid JSON object and no Markdown fences or explanation."""

# Characters that differ between Traditional and Simplified Chinese, in pairs.
_TRADITIONAL_CHARS = set("這個們來時為說國後發會對過還沒樣經開關門問間見話讓認請設車機長東書業電動網線圖頁點擊選單裝")
_SIMPLIFIED_CHARS = set("这个们来时为说国后发会对过还没样经开关门问间见话让认请设车机长东书业电动网线图页点击选单装")

# Vector paths closer than this many points belong to the same illustration.
FIGURE_CLUSTER_TOLERANCE = 12.0
FIGURE_MIN_SIZE = 36.0
# A table must have ruled top and bottom edges covering this share of its width;
# callout leader lines around an illustration are often mistaken for a grid.
TABLE_BORDER_COVERAGE = 0.9
TABLE_BORDER_TOLERANCE = 2.0
# Horizontal rules at least this share of the page width can bound an open-sided table.
TABLE_RULE_MIN_WIDTH = 0.3


@dataclass(frozen=True, slots=True)
class VisionSettings:
    model: str
    scope: str = DEFAULT_VISION_SCOPE
    api_base_url: str = ""
    api_key: str = ""
    max_workers: int = DEFAULT_VISION_CONCURRENCY
    reasoning_effort: str = "none"

    def __post_init__(self) -> None:
        if self.model not in ALLOWED_VISION_MODELS:
            raise ProjectError(f"Vision 模型不在允許清單：{', '.join(ALLOWED_VISION_MODELS)}")
        if self.scope not in VISION_SCOPES:
            raise ProjectError(f"不支援的 Vision 解析範圍：{self.scope}")
        if self.max_workers < 1:
            raise ProjectError("Vision 並行數必須大於 0")
        if self.reasoning_effort not in VISION_REASONING_EFFORT_OPTIONS:
            raise ProjectError("不支援的 Vision 推理深度")


class VisionAbort(Exception):
    """A vision error that will repeat on every page, so the run stops calling the API."""


# Called with (PNG Base64, 1-based page number, document language, settings); returns page text.
PageAnalyzer = Callable[[str, int, "str | None", VisionSettings], str]


def render_page(page: pymupdf.Page, clip: pymupdf.Rect | None = None, scale: float = RENDER_SCALE) -> str:
    """Render a page, or the clipped body area, as a Base64-encoded PNG."""
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
    return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


def detect_document_language(texts: Iterable[str]) -> str | None:
    """Guess the main language of a document from its PDF text layer.

    Pages that hold only figure labels give the vision model too little text
    to tell Traditional Chinese from Japanese or Simplified Chinese, so the
    document-wide language is stated in the prompt instead.
    """
    kana = hangul = han = latin = traditional = simplified = 0
    for text in texts:
        for char in text:
            if "぀" <= char <= "ヿ":
                kana += 1
            elif "가" <= char <= "힣":
                hangul += 1
            elif "一" <= char <= "鿿":
                han += 1
                traditional += char in _TRADITIONAL_CHARS
                simplified += char in _SIMPLIFIED_CHARS
            elif char.isascii() and char.isalpha():
                latin += 1
    if hangul > max(han, latin) * 0.5 and hangul >= 20:
        return "Korean"
    if kana >= 20 and kana > han * 0.1:
        return "Japanese"
    if han >= 20 and han * 3 >= latin:
        if traditional > simplified:
            return "Traditional Chinese"
        if simplified > traditional:
            return "Simplified Chinese"
        return None
    if latin >= 100 and not (han or kana or hangul):
        return "English"
    return None


def system_prompt(language: str | None = None) -> str:
    if not language:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT}\n7. This document is written in {language}. Write every diagram "
        f"description in {language}, even when the page contains only figure labels; "
        "never switch to another language or script."
    )


def analyze_page(image_base64: str, page_number: int, language: str | None, settings: VisionSettings) -> str:
    """Send one rendered page to the vision model and return its serialized text."""
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt(language)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Analyze page {page_number}."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
        # Transcription rarely benefits from reasoning; GPT-6 reasoning requests
        # only accept temperature when the effort is none.
    }
    if settings.model == GPT6_LUNA_MODEL:
        payload["reasoning_effort"] = settings.reasoning_effort
    if settings.model != GPT6_LUNA_MODEL or settings.reasoning_effort == "none":
        payload["temperature"] = 0
    response = _post_json(f"{settings.api_base_url.rstrip('/')}/chat/completions", payload, settings.api_key)
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Vision API 回傳缺少 choices/message/content") from exc
    try:
        result = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Vision 模型回傳的不是有效 JSON") from exc
    return serialize_result(validate_result(result, page_number))


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "automotive-graphrag/0.1",
    }
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            message = f"Vision API 回傳 HTTP {exc.code}：{detail}"
            if exc.code in ABORT_HTTP_CODES:
                raise VisionAbort(message) from exc
            if exc.code not in RETRY_HTTP_CODES or attempt == MAX_RETRIES:
                raise ValueError(message) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise ValueError(f"無法連線至 Vision API：{exc}") from exc
        time.sleep(RETRY_BASE_DELAY_SECONDS * 2**attempt)
    raise AssertionError("unreachable")


def validate_result(value: Any, page_number: int) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("content"), list):
        raise ValueError("Vision 模型回傳的 JSON 缺少 content 陣列")
    if value.get("page_number") not in (None, page_number):
        raise ValueError("Vision 模型回傳的頁碼不正確")
    validated: list[dict[str, str]] = []
    for item in value["content"]:
        if (
            not isinstance(item, dict)
            or item.get("type") not in {"text", "table", "diagram"}
            or not isinstance(item.get("data"), str)
        ):
            raise ValueError("Vision 模型回傳的 content 項目格式不正確")
        validated.append({"type": item["type"], "data": item["data"]})
    return {"page_number": page_number, "content": validated}


def serialize_result(result: dict[str, Any]) -> str:
    """Join content items, separating tables from neighbouring items with a blank line."""
    blocks = [(item["type"], item["data"].strip()) for item in result["content"] if item["data"].strip()]
    serialized: list[str] = []
    for index, (item_type, data) in enumerate(blocks):
        if index:
            serialized.append("\n\n" if "table" in {blocks[index - 1][0], item_type} else "\n")
        serialized.append(data)
    return "".join(serialized)


def page_needs_vision(page: pymupdf.Page, scope: str, text: str, body: pymupdf.Rect | None = None) -> bool:
    """Whether a page goes to the vision model under ``scope``.

    Pages without a text layer always do. Detection errors count as a figure so
    the page is still transcribed. ``find_tables()`` is by far the slowest
    check, so it only runs on pages without images or vector illustrations.
    """
    if scope == SCOPE_ALL or not text:
        return True
    kinds = SCOPE_KINDS[scope]
    diagram_kinds = tuple(kind for kind in DIAGRAM_KINDS if kind in kinds)
    try:
        return bool(
            (diagram_kinds and _figure_candidates(page, diagram_kinds, body))
            or ("table" in kinds and _figure_candidates(page, ("table",), body))
        )
    except Exception:
        return True


def _figure_candidates(
    page: pymupdf.Page, kinds: tuple[str, ...], body: pymupdf.Rect | None = None,
) -> list[tuple[str, pymupdf.Rect]]:
    page_rect = page.rect
    body = body or page_rect
    found: dict[str, list[pymupdf.Rect]] = {}
    drawings = page.get_drawings() if {"table", "drawing"} & set(kinds) else []
    if "table" in kinds:
        segments = _horizontal_segments(drawings)
        tables = page.find_tables(add_lines=_table_side_lines(segments, page_rect.width) or None)
        found["table"] = [
            rect for rect in (pymupdf.Rect(table.bbox) for table in tables.tables)
            if _has_table_border(rect, segments)
        ]
    if "image" in kinds:
        found["image"] = [pymupdf.Rect(info["bbox"]) & page_rect for info in page.get_image_info()]
    if "drawing" in kinds:
        found["drawing"] = _merge_nearby(
            [
                drawing["rect"] for drawing in drawings
                if not _is_rule_drawing(drawing, page_rect.width)
                and (drawing["rect"].width > 0 or drawing["rect"].height > 0)
            ],
            FIGURE_CLUSTER_TOLERANCE,
        )
    return [
        (kind, rect)
        for kind, rects in found.items()
        for rect in rects
        if not rect.is_empty and body.contains((rect.tl + rect.br) / 2)
        and (kind == "table" or min(rect.width, rect.height) >= FIGURE_MIN_SIZE)
    ]


def _is_rule_drawing(drawing: dict[str, Any], page_width: float) -> bool:
    """Long horizontal/vertical lines and rectangles: frames, rules and table borders."""
    rect = drawing["rect"]
    if max(rect.width, rect.height) < page_width * 0.2:
        return False
    for item in drawing["items"]:
        if item[0] == "l":
            start, end = item[1], item[2]
            if abs(start.x - end.x) > 1 and abs(start.y - end.y) > 1:
                return False
        elif item[0] == "qu":
            if not item[1].is_rectangular:
                return False
        elif item[0] != "re":
            return False
    return True


def _horizontal_segments(drawings: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Horizontal lines and rectangle edges as ``(x0, x1, y)``."""
    segments: list[tuple[float, float, float]] = []
    for drawing in drawings:
        for item in drawing["items"]:
            if item[0] == "l":
                start, end = item[1], item[2]
                if abs(start.y - end.y) <= 1:
                    segments.append((min(start.x, end.x), max(start.x, end.x), (start.y + end.y) / 2))
            elif item[0] == "re" or (item[0] == "qu" and item[1].is_rectangular):
                rect = item[1] if item[0] == "re" else item[1].rect
                segments.extend(((rect.x0, rect.x1, rect.y0), (rect.x0, rect.x1, rect.y1)))
    return segments


def _edge_coverage(segments: list[tuple[float, float, float]], x0: float, x1: float, y: float) -> float:
    spans = sorted(
        (max(start, x0), min(end, x1)) for start, end, line_y in segments
        if abs(line_y - y) <= TABLE_BORDER_TOLERANCE and end > x0 and start < x1
    )
    covered, reached = 0.0, x0
    for start, end in spans:
        if end > reached:
            covered += end - max(start, reached)
            reached = end
    return covered / (x1 - x0) if x1 > x0 else 0.0


def _has_table_border(rect: pymupdf.Rect, segments: list[tuple[float, float, float]]) -> bool:
    return all(
        _edge_coverage(segments, rect.x0, rect.x1, y) >= TABLE_BORDER_COVERAGE
        for y in (rect.y0, rect.y1)
    )


def _table_side_lines(
    segments: list[tuple[float, float, float]], page_width: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Left and right edges for tables ruled only horizontally.

    Many manuals draw tables with horizontal rules and inner column lines but
    no outer vertical border, which ``find_tables()`` cannot bound. Touching
    segments on one line are joined into rules; two or more long rules that
    share both ends get vertical lines at those ends, spanning first to last.
    """
    rules: list[list[float]] = []
    for x0, x1, y in sorted(segments, key=lambda segment: (segment[2], segment[0])):
        for rule in rules:
            if abs(rule[2] - y) <= TABLE_BORDER_TOLERANCE and x0 <= rule[1] + 1 and x1 >= rule[0] - 1:
                rule[0], rule[1] = min(rule[0], x0), max(rule[1], x1)
                break
        else:
            rules.append([x0, x1, y])
    groups: dict[tuple[int, int], list[float]] = {}
    for x0, x1, y in rules:
        if x1 - x0 >= page_width * TABLE_RULE_MIN_WIDTH:
            groups.setdefault((round(x0), round(x1)), []).append(y)
    lines = []
    for (x0, x1), ys in groups.items():
        if len(ys) >= 2:
            top, bottom = min(ys), max(ys)
            lines += [((x0, top), (x0, bottom)), ((x1, top), (x1, bottom))]
    return lines


def _merge_nearby(rects: list[pymupdf.Rect], tolerance: float) -> list[pymupdf.Rect]:
    merged = [pymupdf.Rect(rect) for rect in rects]
    changed = True
    while changed:
        changed = False
        result: list[pymupdf.Rect] = []
        for rect in merged:
            for index, existing in enumerate(result):
                if (existing + (-tolerance, -tolerance, tolerance, tolerance)).intersects(rect):
                    result[index] = existing | rect
                    changed = True
                    break
            else:
                result.append(rect)
        merged = result
    return merged
