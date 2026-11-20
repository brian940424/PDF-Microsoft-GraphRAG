"""Restore qualifiers that a Basic answer dropped from the sentences it cites (F-56).

Currently off: the query worker does not pass `revise=True`. With it on, the SM3E Q2 rerun
(2026-11-20, three strict judges) scored 16/30 against 17/30, because the restored test
conditions and explanations were judged as content the reference answer does not have.

The answer call keeps the main action and the numbers of a source sentence but often drops
the rest of that same sentence: its condition, a joined action, the end point, the result or
what happens in the other cases. Prompt rules alone did not fix this (v5), and copying the
source sentences before answering made single-object questions over-answer (v6, F-55).

After the answer is written, one extra call sees only the draft and the sources it cites and
may only put back what was dropped inside a sentence or step the draft already uses. A
deterministic guard rejects a revision that changes the cited sources or grows the answer
beyond a small margin, so the revision cannot turn into a longer, broader answer.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REVISION_PROMPT = """---Role---

You check a draft answer against the source passages it cites and put back qualifiers that the draft dropped. You do not write a new answer.

---Task---

Go through the draft sentence by sentence and step by step. For each one, find the source sentence or step it was taken from in the cited sources below.
- If the draft dropped anything that is inside that same source sentence or step, put it back into the draft sentence: a condition ("while ...", "when ...", "if ...", "with ... kept at ..."), an action joined in the same sentence or step ("..., and wrap ...", "then ..."), an end point ("until ..."), a result or how the result is shown ("shown as ..."), what the same statement gives for the other cases ("0 V in any other position"), or a fallback stated in the same sentence ("using the previous ...").
- An action joined in the same sentence or step is put back only when the question asks how to do the task (a procedure, steps or method), or when the asked fact (a limit, value, order or part) is attached to that action. When the question asks only which parts, values or rules apply, do not add the other actions of that step.
- A reason or consequence clause ("because ...", "since ...", "which may damage ...", "or ... will be damaged") is put back only when the question asks why, or asks for precautions, cautions or warnings.
- When the draft takes a value from a table, the same item's values for the other conditions in that table are part of the same statement (for example "Selector lever in R: battery voltage" with "Other than above: 0 V" on the next row). Put them back next to the value.
- If the draft weakened "will" to "can" or "may", restore "will".

Do not:
- add a sentence, step or item about something the draft does not already cover, such as another step of the procedure, another CAUTION, NOTE or WARNING, a check or inspection, a reason, a specification, an alternative or an example, even when it is in the same source;
- take anything from text that describes an illustration or figure (labels, callouts, "the illustration shows ..."); it is not part of any sentence the draft uses;
- expand a reference such as "repeat the above steps" or "bleed the other calipers" into the steps it refers to; only put back the qualifiers of that sentence itself;
- remove, reorder or change any fact, value or step that the draft already gives;
- add, remove or change citation identifiers such as [Data: Sources (12, 34)].

Keep the draft's language, wording, formatting and citations, and change only what you put back. Translate a restored qualifier into the draft's language. If nothing needs to be put back, output the draft exactly as it is.

---Output format---

Output only the full revised answer, with no comments before or after it.

---Question---

{question}

---Cited sources---

{sources}

---Draft answer---

{draft}
"""

# A sanity bound, not the main control: restoring "0 V in any other position" for each terminal
# more than doubled a short three-sentence answer (SM3E Q2 Q22), so the bound is loose and the
# prompt is what keeps the revision to qualifiers.
MAX_GROWTH_RATIO = 1.5
GROWTH_MARGIN_CHARS = 80

_DATA_CITATION = re.compile(r"\[\s*Data\s*:([^\]]*)\]", re.IGNORECASE)
_SOURCES_CITATION = re.compile(r"Sources\s*\(([^)]*)\)", re.IGNORECASE)


def cited_source_ids(answer: str) -> list[str]:
    """Return the Sources identifiers cited in `answer`, in first-seen order."""
    ids: list[str] = []
    for block in _DATA_CITATION.findall(answer):
        for group in _SOURCES_CITATION.findall(block):
            for part in group.split(","):
                value = part.strip()
                if value and value.lstrip("+").lower() != "more" and value not in ids:
                    ids.append(value)
    return ids


def cited_source_texts(answer: str, context_records: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Return (id, text) for each cited source found in the Basic context records."""
    frame = context_records.get("sources") if isinstance(context_records, Mapping) else None
    if frame is None or getattr(frame, "empty", True) or "id" not in frame or "text" not in frame:
        return []
    texts = {str(row_id): str(text) for row_id, text in zip(frame["id"], frame["text"])}
    return [(source_id, texts[source_id]) for source_id in cited_source_ids(answer) if source_id in texts]


def _paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n\s*\n", text.strip()) if part.strip()])


def check_revision(draft: str, revised: str) -> str | None:
    """Return why `revised` must not replace `draft`, or None when it may."""
    if not revised.strip():
        return "empty"
    if set(cited_source_ids(revised)) != set(cited_source_ids(draft)):
        return "citations_changed"
    limit = len(draft) * (1 + MAX_GROWTH_RATIO) + GROWTH_MARGIN_CHARS
    if len(revised) > limit:
        return "too_long"
    if len(revised) < len(draft) * 0.9:
        return "shortened"
    if _paragraph_count(revised) > _paragraph_count(draft):
        return "paragraph_added"
    return None


async def revise_answer(
    model: Any,
    question: str,
    draft: str,
    context_records: Mapping[str, Any] | None,
    model_params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return the answer to use and a report; the draft is kept whenever the revision is skipped or rejected."""
    sources = cited_source_texts(draft, context_records)
    if not sources:
        return draft, {"status": "skipped", "reason": "no_cited_sources"}
    prompt = REVISION_PROMPT.format(
        question=question,
        sources="\n\n".join(f"[{source_id}]\n{text}" for source_id, text in sources),
        draft=draft,
    )
    report: dict[str, Any] = {"cited_sources": [source_id for source_id, _ in sources], "draft_chars": len(draft)}
    try:
        response = await model.completion_async(
            messages=[{"role": "user", "content": prompt}], stream=False, **model_params,
        )
    except Exception as exc:  # the draft is still a valid answer
        return draft, {**report, "status": "error", "reason": str(exc)}
    revised = str(getattr(response, "content", "") or "").strip()
    report["revised_chars"] = len(revised)
    if revised == draft.strip():
        return draft, {**report, "status": "unchanged"}
    reason = check_revision(draft, revised)
    if reason is not None:
        return draft, {**report, "status": "rejected", "reason": reason, "revised": revised}
    return revised, {**report, "status": "applied", "draft": draft}
