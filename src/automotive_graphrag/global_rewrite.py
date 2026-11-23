"""Rewrite a Global answer from the source passages of the reports it cites (F-70 step 4).

Community reports are summaries made at index time, so a Global answer misses the details the
manual pages give (SM3E 03_global: "Before Replace ECU", the FAST Type ID check, plastic
material codes). After the reduce answer is grounded (``global_sources``), this module gives
the answer model the grounded passages plus the passages closest to the question, from the cited
communities and from the whole index (v3: a manual the draft never cited can still supply a point), with the Basic answer requirements (Local's rule for unsupported parts), and asks it to rewrite the draft from them.
It costs one embedding call and one LLM call; any failure keeps the grounded answer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .answer_prompts import LOCAL_ANSWER_REQUIREMENTS, with_answer_requirements
from .global_sources import Embedder, _normalized, cited_statements, config_embedder, text_unit_vectors

GLOBAL_SOURCE_REWRITE_VERSION = 3
"""v2: no remarks about missing data and only the asked points (v1 on SM3E 03_global: judge 0.556,
Q06 wrote "the data does not give a full item-by-item comparison", Q03 and Q06 lost points for
procedures the question did not ask about).
v3 (F-43): also the closest passages of the whole index, and rules to keep the cautions inside an
asked procedure and to write only real differences (v2: Q04 had no BRM passage because no cited
report covered BRM; Q03 dropped "Never start the engine" and the FAST last five digits from a
passage it had; Q06 listed outer sill work that is the same for both production periods)."""
MAX_CONTEXT_TOKENS = 12000
"""Same budget as the Basic and Local context (``max_context_tokens``)."""

MAX_COMMUNITY_UNITS = 2000
"""Cited communities covering more text units than this are not searched for extra passages."""
GLOBAL_SEARCH_UNITS = 15
"""Passages of the whole index closest to the question, added to the candidates (SM3E 03_global
Q04: the BRM plastics table ranks 7th of 935 passages but sits in no cited community)."""

REWRITE_REQUIREMENTS = """---Rewriting the draft answer---

The data ends with a draft answer written earlier from summaries of the manuals, not from the passages. Rewrite it into the final answer using the source passages above:
- Use the draft to see which points, objects and manuals the answer should cover.
- For each point, find the passages about it and take their concrete details: names of procedures, modes, screen items and parts, the steps in order, conditions, values with units, and which models they apply to. Cite those passages as [Data: Sources (ids)] with the ids from the "id" column.
- Add the points the question asks for that the passages give and the draft missed.
- When a passage contradicts the draft, follow the passage.
- When no passage covers a draft point the question asks for, keep that point as the draft states it, with the draft's citation.
- Keep only the points the question asks for. Leave out draft points and passage details about procedures, parts, systems or work the question does not name, even when the draft includes them.
- For a procedure the question asks about, keep what its passages say inside the steps: what must not be done during a step, what must not be done while checking, which value or which part of an identifier to use, and when an alternative applies. For example, if a passage says to check a warning lamp "with the engine not running", keep that condition.
- When the question asks how two things, versions or periods differ, write only what differs between them: parts, steps or conditions that exist for only one side or change between them, including list entries marked as applying to only one version. Do not present work that is the same for both as a difference.
- Each passage starts with its manual and section path, for example "BRC > ...": the first code is the manual. Name the manual for each point, for example "In BRC, ...". When the question asks which manuals cover something, name every manual whose passages or draft points support it.
- Never write that the manuals, the data, the passages or the draft do not state, confirm, verify, list or fully describe something, and do not add remarks about what is unconfirmed, incomplete or possibly different elsewhere.
- Output only the final answer. Do not mention the draft, the summaries or the rewriting.

"""

_TABLE_HEADER = "id|text"


def rewrite_prompt(response_type: str, context_data: str) -> str:
    """Basic system prompt with the answer and rewrite requirements, filled in."""
    from graphrag.prompts.query.basic_search_system_prompt import BASIC_SEARCH_SYSTEM_PROMPT

    # Local's rule for unsupported parts: the passages are only some of the manual (F-69).
    template = with_answer_requirements(BASIC_SEARCH_SYSTEM_PROMPT, LOCAL_ANSWER_REQUIREMENTS + REWRITE_REQUIREMENTS)
    return template.format(response_type=response_type, context_data=context_data)


def _row(short_id: str, text: str) -> str:
    return f"{short_id}|{' '.join(str(text).split())}"


def select_units(
    grounded_units: Sequence[str],
    candidate_units: Sequence[str],
    question_vector: np.ndarray | None,
    unit_vectors: dict[str, np.ndarray],
    token_count: Callable[[str], int],
    text_by_unit: dict[str, str],
    short_id_by_unit: dict[str, str],
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> list[str]:
    """Grounded units first, then candidate units closest to the question, within ``max_tokens``."""
    ranked = [unit for unit in candidate_units if unit in unit_vectors and unit not in grounded_units]
    if question_vector is not None and ranked:
        scores = np.stack([unit_vectors[unit] for unit in ranked]) @ question_vector
        ranked = [ranked[position] for position in np.argsort(-scores, kind="stable")]
    chosen: list[str] = []
    used = token_count(_TABLE_HEADER)
    for unit in dict.fromkeys([*grounded_units, *ranked]):
        if unit not in short_id_by_unit:
            continue
        tokens = token_count(_row(short_id_by_unit[unit], text_by_unit.get(unit, "")))
        if used + tokens > max_tokens:
            if unit in grounded_units:
                continue  # a long grounded passage should not stop shorter ones after it
            break
        chosen.append(unit)
        used += tokens
    return chosen


def closest_units(question_vector: np.ndarray, unit_vectors: dict[str, np.ndarray], limit: int) -> list[str]:
    """The ``limit`` units of the whole index closest to the question."""
    if limit <= 0 or not unit_vectors:
        return []
    ids = list(unit_vectors)
    scores = np.stack([unit_vectors[unit] for unit in ids]) @ question_vector
    return [ids[position] for position in np.argsort(-scores, kind="stable")[:limit]]


def build_context(units: Sequence[str], text_by_unit: dict[str, str], short_id_by_unit: dict[str, str], draft: str) -> str:
    rows = "\n".join(_row(short_id_by_unit[unit], text_by_unit.get(unit, "")) for unit in units)
    return f"-----Sources-----\n{_TABLE_HEADER}\n{rows}\n\n-----Draft answer-----\n{draft.strip()}\n"


def rewrite_global_answer(
    question: str,
    answer: str,
    grounded_short_ids: Sequence[str],
    communities: Any,
    root_dir: Path,
    response_type: str,
    text_units: Any | None = None,
    embed: Embedder | None = None,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
    token_count: Callable[[str], int] | None = None,
    unit_vectors: dict[str, np.ndarray] | None = None,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    """Return the rewritten answer, the source rows the model saw, and a report.

    Returns the answer unchanged with an empty source list when there is nothing to rewrite from.
    """
    import pandas as pd

    report: dict[str, object] = {"version": GLOBAL_SOURCE_REWRITE_VERSION, "applied": False}
    root_dir = Path(root_dir).resolve()  # GraphRAG's load_config changes the working directory
    output = root_dir / "output"
    if text_units is None:
        text_units = pd.read_parquet(output / "text_units.parquet", columns=["id", "human_readable_id", "text"])
    short_id_by_unit = {str(unit): str(short) for unit, short in zip(text_units["id"], text_units["human_readable_id"])}
    unit_by_short_id = {short: unit for unit, short in short_id_by_unit.items()}
    text_by_unit = dict(zip(text_units["id"].astype(str), text_units["text"].astype(str)))
    community_units = {
        str(int(community)): [str(unit) for unit in units] if units is not None else []
        for community, units in zip(communities["community"], communities["text_unit_ids"])
    }
    cited = list(dict.fromkeys(report_id for *_, ids in cited_statements(answer) for report_id in ids))
    candidates = list(dict.fromkeys(unit for report_id in cited for unit in community_units.get(report_id, ())))
    grounded = [unit_by_short_id[short] for short in grounded_short_ids if short in unit_by_short_id]
    if not grounded and not candidates:
        return answer, [], {**report, "reason": "no cited passages"}
    if len(candidates) > MAX_COMMUNITY_UNITS:
        candidates = []
    if unit_vectors is None:
        unit_vectors = text_unit_vectors(output / "lancedb")
    model = None
    if complete is None or token_count is None:
        model, model_params = _answer_model(root_dir)
        if complete is None:
            def complete(messages: list[dict[str, str]]) -> str:
                response = model.completion(messages=messages, stream=False, **model_params)
                return str(getattr(response, "content", "") or "")
        if token_count is None:
            token_count = lambda text: len(model.tokenizer.encode(text))  # noqa: E731
    question_vector = None
    searched: list[str] = []
    if unit_vectors:
        question_vector = _normalized(np.asarray((embed or config_embedder(root_dir))([question])[0], dtype=np.float32))
        searched = closest_units(question_vector, unit_vectors, GLOBAL_SEARCH_UNITS)
    community_candidates = set(candidates)
    candidates = list(dict.fromkeys([*candidates, *searched]))
    units = select_units(grounded, candidates, question_vector, unit_vectors, token_count, text_by_unit, short_id_by_unit)
    if not units:
        return answer, [], {**report, "reason": "no passages within the token budget"}
    messages = [
        {"role": "system", "content": rewrite_prompt(response_type, build_context(units, text_by_unit, short_id_by_unit, answer))},
        {"role": "user", "content": question},
    ]
    rewritten = complete(messages).strip()
    sources = [
        {"id": short_id_by_unit[unit], "text": text_by_unit.get(unit, ""), "in_context": True} for unit in units
    ]
    report.update({
        "grounded_units": sum(unit in grounded for unit in units),
        "added_units": sum(unit not in grounded for unit in units),
        "searched_units": sum(unit not in grounded and unit not in community_candidates for unit in units),
        "cited_reports": len(cited),
    })
    if not rewritten:
        return answer, [], {**report, "reason": "empty rewrite"}
    return rewritten, sources, {**report, "applied": True}


def _answer_model(root_dir: Path) -> tuple[Any, dict[str, Any]]:
    """The Global search chat model and call arguments from the query root's settings."""
    from graphrag.config.load_config import load_config
    from graphrag_llm.completion import create_completion

    config = load_config(root_dir=Path(root_dir))
    settings = config.get_completion_model_config(config.global_search.completion_model_id)
    return create_completion(settings), dict(settings.call_args)
