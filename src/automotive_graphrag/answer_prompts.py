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

ANSWER_PROMPT_VERSION = "answer-requirements-v3"
ANSWER_PROMPT_DIRECTORY = Path("prompts") / "automotive_answer"
_FORMAT_HEADING = "---Target response length and format---"

ANSWER_REQUIREMENTS = """---Answer requirements---

Scope: answer only what the question asks.
- Answer about the exact feature, app, setting, component or procedure in the question. When the data describes similarly named functions (for example the same feature name in another app, or another way to do a similar task), use only the passages about the one asked and do not mix in the others.
- Match the conditions in the question. When the data gives several examples, cases or patterns (for example different switch positions, lamps, modes or models), answer with the one whose conditions match the question, and do not substitute a case with different conditions.
- Keep each answer within the single section that directly answers the question. Do not combine adjacent sections just because they share a page.
- A passage often contains CAUTION, NOTE or WARNING items and background explanations next to the asked content. Leave out every such item that is not part of the answer: warnings about other components or situations, checks or inspections before or after the asked step, reasons why something happens, when or when not to do the task, and alternative part or bolt styles. Include a caution or warning only when the question asks for precautions or warnings, or when the asked task cannot be done correctly without it.
- For a specification question (torque, dimension, clearance, pressure, capacity, voltage and similar), give only the requested values with their units for the items asked, and no extra sentences.
- Do not add anything the data does not state, such as extra steps, conditions, costs, side effects or general knowledge.

Completeness: do not drop what the asked task needs.
- Include every step, condition and note that is required to complete the asked task, even when it appears later in the passage, in a separate note, or on another page. Examples: a condition that triggers an extra step (such as re-registering parts when a new unit does not come with them), or how a repair must be finished (such as soldering and taping).
- When the question asks for a list of steps, precautions, warnings, conditions or items, include every item of that list, not just the first few.
- Include the follow-up branches that the passage gives for the asked task, such as "To ...", "If ..." or "After ..." instructions (for example how to restore, undo, continue or set up next), unless the question clearly asks for only one part.
- Preserve the source order of procedure steps. Do not move a later registration or setup step before the instruction that introduces it.

Wording and citations.
- Keep the certainty of the source wording: if the data says something will happen, do not soften it to "may"; if it says "may", do not make it definite.
- When the data gives different information for different device models, state which model each piece applies to.
- Use citation identifiers exactly as supplied in the source context. Printed page numbers inside source text are content, not citation identifiers.

"""

# settings section, settings key, file name
_PROMPT_TARGETS = (
    ("local_search", "prompt", "local_search_system_prompt.txt"),
    ("basic_search", "prompt", "basic_search_system_prompt.txt"),
    ("drift_search", "prompt", "drift_search_system_prompt.txt"),
    ("drift_search", "reduce_prompt", "drift_search_reduce_prompt.txt"),
)


def with_answer_requirements(prompt: str) -> str:
    """Insert the answer requirements before the last response-format section."""
    position = prompt.rfind(_FORMAT_HEADING)
    if position < 0:
        raise ValueError("GraphRAG prompt has no response-format section")
    return prompt[:position] + ANSWER_REQUIREMENTS + prompt[position:]


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
    return {name: with_answer_requirements(text) for name, text in bases.items()}


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
