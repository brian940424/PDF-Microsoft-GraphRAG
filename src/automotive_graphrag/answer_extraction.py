"""Two-stage Basic answers: copy the relevant source sentences first, then answer.

A single answer call tends to answer from the first relevant source and to shorten the
sentences it uses, dropping conditions, joined actions and results (F-55, F-56). Before
GraphRAG's Basic answer call, one extra call copies verbatim, per source, every sentence
the question needs. The copy is appended to the context so the answer call still sees the
full sources and their citation identifiers.

The extra call only runs for comparison questions (`needs_two_stage`). On single-object
questions the copied CAUTION, NOTE and check sentences made the answer call add items the
question did not ask for, and the exclusion rules dropped closing steps such as how to exit
a mode (SM3E Q1 rerun, 2026-11-17: Q2, Q6, Q13 over-answered, Q27 lost its exit step).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import AsyncGenerator, Callable
from typing import Any

from .answer_revision import revise_answer

EXTRACTION_HEADING = "-----Extracted source sentences-----"
NO_EXTRACTION = "NONE"

COMPARISON_PATTERN = re.compile(
    r"\b(?:compare[sd]?|comparing|comparison|differen(?:ce|ces|t)|differ|versus|vs\.?|both|in common)\b"
    r"|what is common|比較|差異|差別|區別|異同|不同|相同|共同|兩者",
    re.IGNORECASE,
)


def needs_two_stage(question: str) -> bool:
    """Run the extraction call only for comparison questions, where facts for each object are split across sources."""
    return bool(COMPARISON_PATTERN.search(question))

EXTRACTION_PROMPT = """---Role---

You prepare evidence for another assistant that will answer a question from service manual and user manual sources. You do not answer the question yourself.

---Task---

Go through every source in the data below, one by one. For each source that is about the object or task in the question, copy word for word every sentence or step that answers the question or that the asked task needs.
- Copy whole sentences and steps. Keep every condition ("while ...", "when ...", "if ...", "with ... filled to ..."), every action joined in the same sentence or step, every end point ("until ..."), every result or how it is shown, and any bracketed limit attached to a step. When the asked fact (a limit, value, order or reason) sits inside an instruction, copy the whole instruction.
- Copy conditions that trigger extra work, such as what to do when a new part does not come with something, and how a repair must be finished.
- Several sources about the same object often split the information, for example a description page and a later removal-and-installation page. Check all of them; a requirement that appears only in a later source, such as in a CAUTION at the end of an installation page, must be copied too.
- For a comparison question or a question about two or more objects, copy the sentences for every object.
- Do not copy separate items that are not part of the answer: CAUTION, NOTE or WARNING items about other components or situations, checks or inspections before or after the asked step, reasons why something happens, when or when not to do the task, alternative part or bolt styles, and text that only describes an illustration. Copy such an item only when the question asks for it or the asked task cannot be done correctly without it. This applies to separate items only; inside a sentence you copy, keep everything.
- Do not paraphrase, shorten, translate, merge or reorder the copied text, and do not add anything.

---Output format---

One line per copied sentence or step, starting with the source identifier exactly as given in the data:
[<source id>] <copied text>
Keep the lines of one source together. If no source is relevant, output only NONE.

---Data---

{context_data}
"""


def context_with_extraction(context_chunks: str | list[str], extraction: str) -> str | list[str]:
    """Append the extracted sentences to the Basic context; leave it unchanged when nothing was found."""
    text = extraction.strip()
    if not text or text.upper() == NO_EXTRACTION:
        return context_chunks
    block = f"\n\n{EXTRACTION_HEADING}\n{text}\n"
    if isinstance(context_chunks, list):
        return [*context_chunks, block]
    return f"{context_chunks}{block}"


def _context_text(context_chunks: str | list[str]) -> str:
    return "\n".join(context_chunks) if isinstance(context_chunks, list) else context_chunks


async def extract_source_sentences(model: Any, question: str, context_chunks: str | list[str], model_params: dict[str, Any]) -> str:
    """Ask the answer model to copy the sentences the question needs; empty string on failure."""
    messages = [
        {"role": "system", "content": EXTRACTION_PROMPT.format(context_data=_context_text(context_chunks))},
        {"role": "user", "content": question},
    ]
    try:
        response = await model.completion_async(messages=messages, stream=False, **model_params)
    except Exception:  # the answer call still runs on the full context
        return ""
    return str(getattr(response, "content", "") or "")


ContextTransform = Callable[[str, Any], Any]


def install_two_stage_basic(
    context_transform: ContextTransform | None = None,
    report: dict[str, Any] | None = None,
    revise: bool = False,
) -> None:
    """Patch GraphRAG BasicSearch.stream_search (used by the query API) to answer in two stages.

    `context_transform(query, context_result)` may replace the retrieved context first, for
    example to drop sources of another manual variant; it applies to both stages. Questions
    that are not comparisons skip the extraction call and answer in one stage. `report`, when
    given, receives `two_stage` (whether the extraction call ran) and, with `revise`,
    `qualifier_revision`. With `revise`, the whole answer is collected and checked once more
    against its cited sources (`answer_revision`), then yielded as a single chunk.
    """
    from graphrag.query.structured_search.basic_search.search import BasicSearch

    original = BasicSearch.stream_search
    if getattr(original, "_two_stage_patch", False):
        original = original.__wrapped__

    async def stream_search(self: Any, query: str, conversation_history: Any = None) -> AsyncGenerator[str, None]:
        builder = self.context_builder
        result = builder.build_context(
            query=query, conversation_history=conversation_history, **self.context_builder_params,
        )
        if context_transform is not None:
            result = context_transform(query, result)
        two_stage = needs_two_stage(query)
        if report is not None:
            report["two_stage"] = two_stage
        if two_stage:
            extraction = await extract_source_sentences(self.model, query, result.context_chunks, self.model_params)
            augmented = dataclasses.replace(result, context_chunks=context_with_extraction(result.context_chunks, extraction))
        else:
            augmented = result
        builder.build_context = lambda *args, **kwargs: augmented
        try:
            if not revise:
                async for token in original(self, query, conversation_history):
                    yield token
                return
            draft = "".join([token async for token in original(self, query, conversation_history)])
        finally:
            builder.__dict__.pop("build_context", None)
        answer, revision = await revise_answer(self.model, query, draft, augmented.context_records, self.model_params)
        if report is not None:
            report["qualifier_revision"] = revision
        yield answer

    stream_search._two_stage_patch = True  # type: ignore[attr-defined]
    stream_search.__wrapped__ = original  # type: ignore[attr-defined]
    BasicSearch.stream_search = stream_search  # type: ignore[method-assign]


def reset_two_stage_basic() -> None:
    """Remove the request-scoped Basic search wrapper installed for one GraphRAG query."""
    from graphrag.query.structured_search.basic_search.search import BasicSearch

    current = BasicSearch.stream_search
    if getattr(current, "_two_stage_patch", False):
        BasicSearch.stream_search = current.__wrapped__  # type: ignore[method-assign]
