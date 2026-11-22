"""Answer prompts for GraphRAG Local, Basic and DRIFT queries.

The GraphRAG default prompts are kept as the base and an answer-requirements section is
inserted before the final response-format section. The prompts are written next to the
settings used by each query and referenced by absolute path, because GraphRAG resolves
relative prompt paths against the working directory and silently falls back to its
defaults when a file is missing (as in the temporary roots of retrieval experiments).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ANSWER_PROMPT_VERSION = "answer-requirements-v10"
ANSWER_PROMPT_DIRECTORY = Path("prompts") / "automotive_answer"
_FORMAT_HEADING = "---Target response length and format---"

ANSWER_REQUIREMENTS = """---Answer requirements---

Answer only from the data.
- These requirements replace any earlier instruction to incorporate general knowledge. Do not use general knowledge, and never write a "General Knowledge" citation.
- Every statement in the answer must come from the data and cite its source identifiers. If the data does not contain the answer, say only that the provided data does not state it; do not answer from what is commonly known about the topic.

Scope: answer only what the question asks.
- Answer about the exact feature, app, setting, component or procedure in the question. When the data describes similarly named functions (for example the same feature name in another app, or another way to do a similar task), use only the passages about the one asked and do not mix in the others.
- Match the conditions in the question. When the data gives several examples, cases or patterns (for example different switch positions, lamps, modes or models), answer with the one whose conditions match the question, and do not substitute a case with different conditions.
- Keep each answer within the single section that directly answers the question. Do not combine adjacent sections just because they share a page.
- A passage often contains CAUTION, NOTE or WARNING items and background explanations next to the asked content. Leave out every such item that is not part of the answer: warnings about other components or situations, checks or inspections before or after the asked step, reasons why something happens, when or when not to do the task, and alternative part or bolt styles. Include a caution or warning only when the question asks for precautions or warnings, or when the asked task cannot be done correctly without it.
- For a specification question (torque, dimension, clearance, pressure, capacity, voltage and similar), give only the requested values with their units for the items asked, and no extra sentences.
- Write only the items the question asks about. Do not write terminals or pin numbers, values (voltages, resistances, torques, dimensions and similar) or operation steps that the question does not ask about, even when they appear in the same passage. This does not remove the steps or conditions of a task the question does ask about; those follow the completeness rules below.
- Do not add anything the data does not state, such as extra steps, conditions, costs, side effects or general knowledge.

Completeness: do not drop what the asked task needs.
- Include every step, condition and note that is required to complete the asked task, even when it appears later in the passage, in a separate note, or on another page. Examples: a condition that triggers an extra step (such as re-registering parts when a new unit does not come with them), or how a repair must be finished (such as soldering and taping).
- When the question asks for a list of steps, precautions, warnings, conditions or items, include every item of that list, not just the first few.
- Include the follow-up branches that the passage gives for the asked task, such as "To ...", "If ..." or "After ..." instructions (for example how to restore, undo, continue or set up next), unless the question clearly asks for only one part.
- Preserve the source order of procedure steps. Do not move a later registration or setup step before the instruction that introduces it.

Use every relevant source, not only the first one.
- Before writing, go through the sources in the data one by one. For each source that is about the asked object or task, take every step, condition, caution and note in it that the asked task needs. Do this silently; output only the answer.
- Several sources about the same object often split the information, for example an overview or description page and a later removal-and-installation page. Combine them. Never answer from only the first-ranked source or the lowest page number: a requirement that appears only in a later source, such as in a CAUTION at the end of an installation page, belongs in the answer.
- For a comparison question (the question asks to compare, or asks about differences, similarities, what is common or what is specific), or a question about two or more objects: first sort, silently, the points from every relevant source into what all objects share and what is specific to each object, then write the answer in that structure. Each object's part must cover the points from all sources about that object, not just one of them.
- Cite every source whose content you use, including each additional page that contributed a point.

Keep the qualifiers inside each sentence or step you use.
- A sentence or step you take from the data must keep its conditions ("while ...", "when ...", "if ...", "with X filled to / kept at ..."), the other actions joined to it in the same sentence or step ("..., and wrap ...", "then ..."), its end point ("until ...") and its result or how the result is shown ("indicated on the screen as ...", "shown as ..."). You may reword it, but keeping only the main action and the numbers is not enough.
- When the asked fact (a limit, value, order or reason) is attached to an instruction, for example a bracketed limit after a repair step or an order given inside a "Repeat ... with ..." step, state that instruction and its conditions together with the fact.
- Boundary with the scope rule above: qualifiers inside the same sentence or step as the asked content are kept. A separate CAUTION, NOTE or WARNING item, or text that describes an illustration, is a different item; include it only under the scope rule.
- Example to keep: the data says "Seal the joint and wrap it with heat-shrink tube. [Exposed conductor must be within 5 mm.]" For a question about the exposed-conductor limit, answer "Seal the joint and wrap it with heat-shrink tube, keeping exposed conductor within 5 mm", not only "within 5 mm".
- Example to leave out: the same passage also has a separate NOTE, "Inspect the connector housing for cracks." The question does not ask about the housing, so do not add that note.
- Final check before you answer: compare each of your sentences with the source sentence it came from. Put back any condition, joined action, end point or result you dropped, and restore "will" if you weakened it to "can" or "may".

Extracted source sentences.
- The data may end with a section "-----Extracted source sentences-----": sentences copied word for word from the sources above, each starting with its source identifier. Build the answer from these lines: cover every copied line that the question asks about, from every source listed, and keep each line's conditions, joined actions, end points and results. Leave a copied line out only when it is clearly outside what the question asks. Check the full sources above for anything the copy missed, and cite the source identifiers of the lines you use.

Wording and citations.
- Keep the certainty of the source wording: if the data says something will happen, do not soften it to "may"; if it says "may", do not make it definite.
- When the data gives different information for different device models, state which model each piece applies to.
- Use citation identifiers exactly as supplied in the source context. Printed page numbers inside source text are content, not citation identifiers.
- Cite sentence by sentence: end each sentence or step with the identifiers of every source that supports it. When one sentence combines several sources, list all of them, not only the main one.
- Also cite the source that establishes the situation in the question. When the question describes a setting (for example "while removing part A, connector B was disconnected"), cite the source that describes that setting, such as the removal procedure that says to remove part A, together with the source that gives the answer, even if the answer itself comes from another page.

"""

_UNSTATED_RULE = (
    "If the data does not contain the answer, say only that the provided data does not state it; "
    "do not answer from what is commonly known about the topic."
)
# Local sees only the passages linked to a few matched entities, so a missing point usually means
# the passage was not retrieved, not that the manual lacks it. Claiming the manual does not state
# it contradicted the reference answers (SM3E v4 Local: 22 of 30 answers, F-69).
INSUFFICIENT_DATA_PHRASE = "目前取得的資料不足以回答此部分"
INSUFFICIENT_DATA_PHRASE_EN = "The retrieved data is not sufficient to answer this part"
LOCAL_UNSTATED_RULE = (
    "The data holds only some passages of the manual, so a point missing from it may still be in the manual. "
    "Answer the parts of the question the data supports. For each part it does not support, write exactly "
    f'"{INSUFFICIENT_DATA_PHRASE}" (or "{INSUFFICIENT_DATA_PHRASE_EN}." when answering in English), '
    'and never write that the manual, the data or the sources do not state, mention or cover it. '
    "Do not answer from what is commonly known about the topic."
)
LOCAL_ANSWER_REQUIREMENTS = ANSWER_REQUIREMENTS.replace(_UNSTATED_RULE, LOCAL_UNSTATED_RULE)
if LOCAL_ANSWER_REQUIREMENTS == ANSWER_REQUIREMENTS:  # pragma: no cover - guards an edit of the shared rule
    raise RuntimeError("Local answer requirements did not replace the unstated-answer rule")

# settings section, settings key, file name
_PROMPT_TARGETS = (
    ("local_search", "prompt", "local_search_system_prompt.txt"),
    ("basic_search", "prompt", "basic_search_system_prompt.txt"),
    ("drift_search", "prompt", "drift_search_system_prompt.txt"),
    ("drift_search", "reduce_prompt", "drift_search_reduce_prompt.txt"),
)


def with_answer_requirements(prompt: str, requirements: str = ANSWER_REQUIREMENTS) -> str:
    """Insert the answer requirements before the last response-format section."""
    position = prompt.rfind(_FORMAT_HEADING)
    if position < 0:
        raise ValueError("GraphRAG prompt has no response-format section")
    return prompt[:position] + requirements + prompt[position:]


def answer_prompts() -> dict[str, str]:
    """Return the answer prompt text for each file name."""
    from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT
    from graphrag.prompts.query.drift_search_system_prompt import (
        DRIFT_LOCAL_SYSTEM_PROMPT,
        DRIFT_REDUCE_PROMPT,
    )
    from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT

    bases = {
        "local_search_system_prompt.txt": LOCAL_SEARCH_SYSTEM_PROMPT,
        "basic_search_system_prompt.txt": BASIC_SEARCH_SYSTEM_PROMPT,
        "drift_search_system_prompt.txt": DRIFT_LOCAL_SYSTEM_PROMPT,
        "drift_search_reduce_prompt.txt": DRIFT_REDUCE_PROMPT,
    }
    requirements = {"local_search_system_prompt.txt": LOCAL_ANSWER_REQUIREMENTS}
    return {
        name: with_answer_requirements(text, requirements.get(name, ANSWER_REQUIREMENTS))
        for name, text in bases.items()
    }


def install_answer_prompts(settings: dict[str, Any], root: Path) -> None:
    """Write the answer prompts under ``root`` and point ``settings`` at them by absolute path."""
    directory = (root / ANSWER_PROMPT_DIRECTORY).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prompts = answer_prompts()
    for name, text in prompts.items():
        path = directory / name
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    for section_name, key, name in _PROMPT_TARGETS:
        section = settings.get(section_name)
        if not isinstance(section, dict):
            section = settings[section_name] = {}
        section[key] = str(directory / name)
