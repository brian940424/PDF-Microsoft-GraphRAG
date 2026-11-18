"""Reasoning-model-compatible execution for Microsoft GraphRAG DRIFT queries."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any


_DRIFT_PATCH_LOCK = threading.RLock()


class SamplingParameterFilter:
    """Delegate a completion client while omitting unsupported sampling arguments."""

    def __init__(self, completion: Any) -> None:
        self._completion = completion

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completion, name)

    def completion(self, *args: Any, **kwargs: Any) -> Any:
        return self._completion.completion(*args, **self._without_sampling(kwargs))

    async def completion_async(self, *args: Any, **kwargs: Any) -> Any:
        return await self._completion.completion_async(*args, **self._without_sampling(kwargs))

    @staticmethod
    def _without_sampling(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in kwargs.items() if key not in {"temperature", "top_p"}}


class LocalContextCollector:
    """Keep the context of every Local Search action that one DRIFT query runs.

    GraphRAG's DRIFT API stores only the latest ``on_context`` payload, so the
    sources of earlier follow-up questions are lost although the reduced answer
    still cites them. This collector records all payloads for merging.
    """

    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        # Other query callbacks (map/reduce/token events) are irrelevant here.
        if name.startswith("on_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def on_context(self, context: Any) -> None:
        if isinstance(context, dict):
            self.contexts.append(dict(context))

    def merged(self) -> dict[str, Any]:
        return merge_local_contexts(self.contexts)


def merge_local_contexts(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Union per-action context tables, de-duplicated by record id in first-seen order."""
    import pandas as pd

    merged: dict[str, Any] = {}
    for context in contexts:
        for key, value in context.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(merged[key], pd.DataFrame) and isinstance(value, pd.DataFrame):
                merged[key] = pd.concat([merged[key], value], ignore_index=True)
    for key, value in merged.items():
        if not isinstance(value, pd.DataFrame) or "id" not in value.columns or value.empty:
            continue
        frame = value.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
        if "in_context" in value.columns:
            # A record used by any action counts as in context.
            used_ids = set(value.loc[value["in_context"].map(lambda item: item is not False), "id"])
            frame["in_context"] = frame["id"].isin(used_ids)
        merged[key] = frame
    return merged


def _filter_engine_model(engine: Any, config: Any) -> Any:
    """Wrap every model reference used by one DRIFT engine when it selects a reasoning model."""
    model_settings = config.get_completion_model_config(config.drift_search.completion_model_id)
    if "reasoning_effort" not in model_settings.call_args:
        return engine

    compatible_model = SamplingParameterFilter(engine.model)
    engine.model = compatible_model
    engine.context_builder.model = compatible_model
    engine.primer.chat_model = compatible_model
    engine.local_search.model = compatible_model
    return engine


def run_drift_search(
    *,
    root_dir: str | Path,
    query: str,
    community_level: int = 2,
    response_type: str = "Single Paragraph",
) -> tuple[str, dict[str, Any]]:
    """Run the official GraphRAG DRIFT entry point with a scoped reasoning-model adapter.

    GraphRAG 3.2 builds its DRIFT model arguments in the search implementation,
    so this scoped factory wrapper is needed to remove unsupported sampling
    arguments without modifying the installed GraphRAG package or other queries.
    """
    from graphrag.cli import query as cli_query
    from graphrag.api import query as query_api

    def execute() -> tuple[str, dict[str, Any]]:
        original_factory = query_api.get_drift_search_engine
        collector = LocalContextCollector()

        def compatible_factory(*args: Any, **kwargs: Any) -> Any:
            engine = original_factory(*args, **kwargs)
            config = kwargs.get("config", args[0] if args else None)
            # DRIFT shares this list with its inner LocalSearch, so the collector
            # sees every follow-up action's context records.
            callbacks = getattr(engine, "callbacks", None)
            if isinstance(callbacks, list):
                callbacks.append(collector)
            return _filter_engine_model(engine, config)

        query_api.get_drift_search_engine = compatible_factory
        try:
            result = cli_query.run_drift_search(
                data_dir=None,
                root_dir=Path(root_dir),
                community_level=community_level,
                response_type=response_type,
                streaming=False,
                query=query,
                verbose=False,
            )
            answer, context = result
            if collector.contexts:
                return str(answer), collector.merged()
            return str(answer), context if isinstance(context, dict) else {}
        finally:
            query_api.get_drift_search_engine = original_factory

    # The GraphRAG API resolves its factory through a module global. Serialize
    # DRIFT calls while that reference is temporarily wrapped to prevent races.
    with _DRIFT_PATCH_LOCK:
        try:
            original_cwd = os.getcwd()
        except OSError:
            # Recover cleanly if an earlier GraphRAG call left the app in a
            # temporary directory that has since been removed.
            original_cwd = str(Path(__file__).resolve().parents[2])
        try:
            return execute()
        finally:
            try:
                os.chdir(original_cwd)
            except FileNotFoundError:
                os.chdir(Path(__file__).resolve().parents[2])
