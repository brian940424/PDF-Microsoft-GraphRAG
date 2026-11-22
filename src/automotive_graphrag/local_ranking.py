"""Choose the text units Local search puts in its context (F-69).

GraphRAG Local walks the matched entities in order and takes every text unit each one links to,
so a hub entity (BOSE links 81 SM3E units) fills the text-unit budget, and a question naming
C1154 got the neighbouring C1155 steps. This module replaces that ordering:

- Entities whose title contains a code the question names (DTC, connector, part number) or whose
  full title appears in the question are added to the matched entities first.
- Units whose heading carries a named code come first, then units whose text does, then the rest.
- Within each tier, units more similar to the question (text-unit vectors) come first.
- A hub entity (a generic title, or at least ``hub_threshold`` units: above the graph's P99 and
  never below ``HUB_MIN_TEXT_UNITS``) brings only its most similar unit to the front; its other
  units, unless another entity links them, go last. Entities the question names are never hubs.

Only units linked to a matched entity are used; Local does not borrow Basic's unlinked passages.

Offline probe on SM3E v4 (30 questions, 95 gold pages, ``local_retrieval_probe``): gold pages in
context 23.2% → 27.4%; the matched entities only reach 28.4% of gold pages, so ranking cannot do
much more. Capping every entity at 1–3 units or preferring units linked by several entities
scored lower (24.2–26.3%), because dense DTC tables link many entities.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from .strategy_routing import question_identifiers

LOCAL_RANKING_VERSION = 2
# Floor of the hub threshold. SM3E (default entity types): median 1 linked unit per entity, P99 8.
# Manual entity types link more units (temp: P99 18), so the threshold follows each graph's P99.
HUB_MIN_TEXT_UNITS = 9
HUB_PERCENTILE = 99
GENERIC_ENTITY_TITLES = frozenset({
    "DIAGNOSIS PROCEDURE", "DTC CONFIRMATION PROCEDURE", "DTC LOGIC", "INSPECTION END",
    "COMPONENT INSPECTION", "REMOVAL AND INSTALLATION", "NISSAN", "CONSULT",
})
SIMILARITY_CANDIDATES = 200
_MIN_TITLE_CHARS = 6
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9/-]+")


def exact_entity_titles(question: str, titles: list[str]) -> list[str]:
    """Entity titles that contain a code named in the question or appear whole in the question."""
    codes = [code.casefold() for code in question_identifiers(question)]
    folded_question = " ".join(_LATIN_WORD.findall(question)).casefold()
    matched: list[str] = []
    for title in titles:
        folded = title.casefold()
        words = set(re.findall(r"[a-z0-9-]+", folded))
        if any(code in words for code in codes) or (
            len(title) >= _MIN_TITLE_CHARS and " " in title.strip()
            and re.search(rf"(?<![a-z0-9]){re.escape(folded)}(?![a-z0-9])", folded_question)
        ):
            matched.append(title)
    return matched


def hub_threshold(link_counts: list[int]) -> int:
    """Smallest linked-unit count that makes an entity a hub: above the graph's P99, at least the floor."""
    counts = sorted(link_counts)
    if not counts:
        return HUB_MIN_TEXT_UNITS
    rank = (len(counts) - 1) * HUB_PERCENTILE / 100
    low = math.floor(rank)
    high = min(low + 1, len(counts) - 1)
    percentile = counts[low] + (counts[high] - counts[low]) * (rank - low)
    return max(HUB_MIN_TEXT_UNITS, math.floor(percentile) + 1)


def is_hub(title: str, linked_units: int, threshold: int = HUB_MIN_TEXT_UNITS) -> bool:
    return linked_units >= threshold or title.strip().upper() in GENERIC_ENTITY_TITLES


def rank_text_units(
    entities: list[tuple[str, list[str]]],
    texts: dict[str, str],
    codes: tuple[str, ...] = (),
    similarity: dict[str, float] | None = None,
    hub_min_units: int = HUB_MIN_TEXT_UNITS,
    named_titles: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Order the text unit ids linked to the matched entities.

    ``entities`` holds each matched entity's title and linked unit ids, in match order.
    ``named_titles`` are entities the question names; they are never treated as hubs.
    """
    similarity = similarity or {}
    named = {title.casefold() for title in named_titles}
    folded_codes = [code.casefold() for code in codes]

    def code_tier(unit_id: str) -> int:
        if not folded_codes:
            return 0
        text = texts.get(unit_id, "").casefold()
        heading = text.split("\n", 1)[0]
        if any(code in heading for code in folded_codes):
            return 0
        return 1 if any(code in text for code in folded_codes) else 2

    first_entity: dict[str, int] = {}
    regular: set[str] = set()
    hubs: list[tuple[str, list[str]]] = []
    for order, (title, unit_ids) in enumerate(entities):
        own = [unit for unit in dict.fromkeys(unit_ids) if unit in texts]
        for unit_id in own:
            first_entity.setdefault(unit_id, order)
        if title.casefold() not in named and is_hub(title, len(unit_ids), hub_min_units):
            hubs.append((title, own))
        else:
            regular.update(own)

    def relevance(unit_id: str) -> tuple[Any, ...]:
        return (code_tier(unit_id), -similarity.get(unit_id, -math.inf), first_entity[unit_id])

    for _, own in hubs:
        if own:
            regular.add(min(own, key=relevance))
    ordered = sorted(first_entity, key=lambda unit: (unit not in regular, *relevance(unit)))
    report = {
        "version": LOCAL_RANKING_VERSION,
        "hub_threshold": hub_min_units,
        "hub_entities": [title for title, _ in hubs],
        "candidate_units": len(first_entity),
        "demoted_units": len(first_entity) - len(regular & set(first_entity)),
        "similarity_units": sum(1 for unit in first_entity if unit in similarity),
    }
    if codes:
        report["named_codes"] = list(codes)
    return ordered, report


SimilarityFunction = Callable[[Any, str], dict[str, float]]


def install_local_ranking(
    report: dict[str, Any], similarity_function: SimilarityFunction | None = None,
) -> None:
    """Patch GraphRAG Local's context builder for the queries until ``reset_local_ranking``.

    ``similarity_function(context_builder, query)`` returns text-unit id → similarity to the query.
    """
    from graphrag.query.context_builder.source_context import build_text_unit_context as format_text_units
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext

    reset_local_ranking()
    original_build = LocalSearchMixedContext.build_context
    original_units = LocalSearchMixedContext._build_text_unit_context

    def build_context(self, query: str, *args: Any, **kwargs: Any):
        titles = [entity.title for entity in self.entities.values()]
        exact = exact_entity_titles(query, titles)
        report["exact_entities"] = exact
        if getattr(self, "_f69_hub_threshold", None) is None:
            self._f69_hub_threshold = hub_threshold(
                [len(entity.text_unit_ids or []) for entity in self.entities.values()]
            )
        self._f69_exact = tuple(exact)
        self._f69_query = query
        included = list(kwargs.get("include_entity_names") or [])
        kwargs["include_entity_names"] = list(dict.fromkeys([*exact, *included]))
        return original_build(self, query, *args, **kwargs)

    def build_text_unit_context(
        self, selected_entities: list[Any], max_context_tokens: int = 8000,
        return_candidate_context: bool = False, column_delimiter: str = "|", context_name: str = "Sources",
    ):
        if return_candidate_context or not selected_entities or not self.text_units:
            return original_units(
                self, selected_entities, max_context_tokens, return_candidate_context, column_delimiter, context_name,
            )
        query = getattr(self, "_f69_query", "")
        unique = list({entity.id: entity for entity in selected_entities}.values())
        entities = [(entity.title, list(entity.text_unit_ids or [])) for entity in unique]
        texts = {
            unit_id: str(self.text_units[unit_id].text or "")
            for _, unit_ids in entities for unit_id in unit_ids if unit_id in self.text_units
        }
        similarity: dict[str, float] = {}
        if similarity_function is not None and query:
            try:
                similarity = similarity_function(self, query)
            except Exception as exc:  # ranking must never block an answer
                report["similarity_error"] = str(exc)
        ordered, ranking = rank_text_units(
            entities, texts, question_identifiers(query), similarity,
            getattr(self, "_f69_hub_threshold", None) or HUB_MIN_TEXT_UNITS, getattr(self, "_f69_exact", ()),
        )
        report.update(ranking)
        units = [self.text_units[unit_id] for unit_id in ordered]
        context_text, context_data = format_text_units(
            text_units=units, tokenizer=self.tokenizer, max_context_tokens=max_context_tokens,
            shuffle_data=False, context_name=context_name, column_delimiter=column_delimiter,
        )
        return str(context_text), context_data

    build_context._local_ranking_patch = original_build  # type: ignore[attr-defined]
    build_text_unit_context._local_ranking_patch = original_units  # type: ignore[attr-defined]
    LocalSearchMixedContext.build_context = build_context  # type: ignore[method-assign]
    LocalSearchMixedContext._build_text_unit_context = build_text_unit_context  # type: ignore[method-assign]


def reset_local_ranking() -> None:
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext

    for name in ("build_context", "_build_text_unit_context"):
        current = getattr(LocalSearchMixedContext, name)
        original = getattr(current, "_local_ranking_patch", None)
        if original is not None:
            setattr(LocalSearchMixedContext, name, original)


def text_unit_similarity(root_dir: Any, k: int = SIMILARITY_CANDIDATES) -> SimilarityFunction:
    """Similarity of text units to the query from the index's text-unit vectors (Basic's store)."""

    def similarity(context_builder: Any, query: str) -> dict[str, float]:
        from pathlib import Path

        from graphrag.config.embeddings import text_unit_text_embedding
        from graphrag.config.load_config import load_config
        from graphrag.utils.api import get_embedding_store

        config = load_config(root_dir=Path(root_dir))
        store = get_embedding_store(config=config.vector_store, embedding_name=text_unit_text_embedding)
        embedder = context_builder.text_embedder
        results = store.similarity_search_by_text(
            text=query, text_embedder=lambda value: embedder.embedding(input=[value]).first_embedding, k=k,
        )
        return {str(result.document.id): float(result.score) for result in results}

    return similarity
