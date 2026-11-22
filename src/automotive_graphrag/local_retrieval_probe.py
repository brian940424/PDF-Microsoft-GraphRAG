"""Measure which gold pages Local search can reach and actually puts in its context (F-69).

Builds only the Local context (one embedding call per question, plus one for text-unit
similarity with the F-69 ranking); no answer is generated and no judge runs. For each question
it reports the matched entities, the pages in context, and two ratios over all gold pages:

- ``gold_with_entity``: the page has a text unit linked to an entity (the upper bound for Local).
- ``gold_in_context``: the page is in the Local context.

Usage::

    python -m automotive_graphrag.local_retrieval_probe projects/<id> <question_set.json> \\
        [--mode baseline|ranked] [--output result.json]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .connections import ConnectionSettings
from .local_ranking import install_local_ranking, reset_local_ranking, text_unit_similarity


def page_index(output: Path) -> tuple[dict[str, list[tuple[str, int]]], set[tuple[str, int]]]:
    """Text unit id → (document, page) pairs, and the pages that have a unit linked to an entity."""
    documents = pd.read_parquet(output / "documents.parquet", columns=["id", "raw_data"])
    pages_by_document: dict[str, list[tuple[str, int]]] = {}
    for row in documents.to_dict(orient="records"):
        raw = row.get("raw_data") or {}
        pages = raw.get("pages")
        numbers = [int(page) for page in pages] if pages is not None and len(pages) else [raw.get("page")]
        pages_by_document[str(row["id"])] = [
            (str(raw.get("document_id", "")), int(page)) for page in numbers if page is not None
        ]
    units = pd.read_parquet(output / "text_units.parquet", columns=["id", "document_id", "entity_ids"])
    unit_pages: dict[str, list[tuple[str, int]]] = {}
    with_entity: set[tuple[str, int]] = set()
    for row in units.to_dict(orient="records"):
        pages = pages_by_document.get(str(row["document_id"]), [])
        unit_pages[str(row["id"])] = pages
        if row.get("entity_ids") is not None and len(row["entity_ids"]):
            with_entity.update(pages)
    return unit_pages, with_entity


def gold_pages(question: dict[str, Any]) -> set[tuple[str, int]]:
    return {
        (str(item.get("document_id", "")), int(page))
        for item in question.get("gold_evidence") or [] for page in item.get("pages") or []
    }


def local_context_builder(root: Path):
    """The Local search engine GraphRAG's CLI would build, without running a search."""
    from graphrag.config.embeddings import entity_description_embedding
    from graphrag.config.load_config import load_config
    from graphrag.query.factory import get_local_search_engine
    from graphrag.query.indexer_adapters import (
        read_indexer_covariates, read_indexer_entities, read_indexer_relationships,
        read_indexer_reports, read_indexer_text_units,
    )
    from graphrag.utils.api import get_embedding_store

    config = load_config(root_dir=root)
    output = root / "output"
    tables = {
        name: pd.read_parquet(output / f"{name}.parquet")
        for name in ("communities", "community_reports", "text_units", "relationships", "entities")
    }
    covariates_path = output / "covariates.parquet"
    covariates = read_indexer_covariates(pd.read_parquet(covariates_path)) if covariates_path.is_file() else []
    level = 2
    engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(tables["community_reports"], tables["communities"], level),
        text_units=read_indexer_text_units(tables["text_units"]),
        entities=read_indexer_entities(tables["entities"], tables["communities"], level),
        relationships=read_indexer_relationships(tables["relationships"]),
        covariates={"claims": covariates},
        description_embedding_store=get_embedding_store(
            config=config.vector_store, embedding_name=entity_description_embedding,
        ),
        response_type="Single Paragraph",
    )
    short_ids = {str(row["human_readable_id"]): str(row["id"]) for row in tables["text_units"].to_dict(orient="records")}
    return engine, short_ids


def retrieval_hints(
    project: Path, questions: list[dict[str, Any]], path: Path, model: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Manual-language query per question (Basic's rewrite), from ``path``; missing ones are asked once and saved."""
    hints: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    missing = [question for question in questions if str(question.get("question_id")) not in hints]
    if missing:
        from .query_translation import document_language_sample
        from .strategy_routing import StrategyRouter

        connections = ConnectionSettings(project.parent)
        model = model or connections.get_chat_model()
        _script, excerpt = document_language_sample(project / "graphrag")
        router = StrategyRouter()
        for question in missing:
            query, _calls = router.rewrite_query(
                question["question"], excerpt, model, connections.get_api_base_url(),
                connections.get_api_key(project.name),
            )
            hints[str(question.get("question_id"))] = {"query": query}
        path.write_text(json.dumps(hints, ensure_ascii=False, indent=2), encoding="utf-8")
    return hints


def probe(
    project: Path, questions: list[dict[str, Any]], mode: str,
    hints: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = project / "graphrag"
    unit_pages, with_entity = page_index(root / "output")
    engine, short_ids = local_context_builder(root)
    results: list[dict[str, Any]] = []
    page_counts: collections.Counter[tuple[str, int]] = collections.Counter()
    for question in questions:
        report: dict[str, Any] = {}
        if mode == "ranked":
            hint = (hints or {}).get(str(question.get("question_id")), {})
            install_local_ranking(report, text_unit_similarity(root), hint.get("query"))
        try:
            built = engine.context_builder.build_context(query=question["question"], **engine.context_builder_params)
            if asyncio.iscoroutine(built):
                built = asyncio.run(built)
        finally:
            reset_local_ranking()
        records = built.context_records
        entities = records.get("entities")
        sources = records.get("sources")
        unit_ids = [short_ids.get(str(value), str(value)) for value in (sources["id"] if sources is not None and len(sources) else [])]
        pages = list(dict.fromkeys(page for unit in unit_ids for page in unit_pages.get(unit, [])))
        page_counts.update(set(pages))
        gold = gold_pages(question)
        results.append({
            "question_id": question.get("question_id"),
            "entities": [str(value) for value in (entities["entity"] if entities is not None and len(entities) else [])],
            "context_pages": [f"{document} p.{page}" for document, page in pages],
            "gold_pages": sorted(f"{document} p.{page}" for document, page in gold),
            "gold_with_entity": len(gold & with_entity),
            "gold_in_context": len(gold & set(pages)),
            "gold_total": len(gold),
            "ranking": report,
        })
    total = sum(item["gold_total"] for item in results) or 1
    return {
        "mode": mode,
        "questions": len(results),
        "gold_pages": total,
        "gold_with_entity_ratio": round(sum(item["gold_with_entity"] for item in results) / total, 3),
        "gold_in_context_ratio": round(sum(item["gold_in_context"] for item in results) / total, 3),
        "questions_without_gold_in_context": [item["question_id"] for item in results if item["gold_in_context"] == 0],
        "most_repeated_pages": [
            {"page": f"{document} p.{page}", "questions": count} for (document, page), count in page_counts.most_common(10)
        ],
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", type=Path, help="project directory, for example projects/SM3E-1L33U3")
    parser.add_argument("question_set", type=Path, help="question set JSON with gold_evidence pages")
    parser.add_argument("--mode", choices=("baseline", "ranked"), default="ranked")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--hints", type=Path,
        help="JSON of the manual-language query per question; missing entries are asked from the chat model and saved",
    )
    parser.add_argument("--model", help="chat model for --hints (default: the connection's chat model)")
    args = parser.parse_args(argv)
    # Resolve paths first: loading the GraphRAG config can change the working directory.
    project = args.project.resolve()
    output = args.output.resolve() if args.output else None
    hints_path = args.hints.resolve() if args.hints else None
    # settings.yaml reads the API key and models from the environment the app normally sets.
    ConnectionSettings(project.parent).apply_to_environment(project.name)
    data = json.loads(args.question_set.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    hints = retrieval_hints(project, questions, hints_path, args.model) if hints_path else None
    summary = probe(project, questions, args.mode, hints)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
    brief = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
