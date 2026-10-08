"""GPT-6-compatible execution for Microsoft GraphRAG DRIFT queries."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .connections import GPT6_LUNA_MODEL


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


def _filter_engine_model(engine: Any, config: Any) -> Any:
    """Wrap every model reference used by one DRIFT engine when it selects GPT-6 Luna."""
    model_settings = config.get_completion_model_config(config.drift_search.completion_model_id)
    if model_settings.model != GPT6_LUNA_MODEL:
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
    response_type: str = "Multiple Paragraphs",
) -> tuple[str, dict[str, Any]]:
    """Run the official GraphRAG DRIFT entry point with a scoped Luna adapter.

    GraphRAG 3.2 builds its DRIFT model arguments in the search implementation,
    so this scoped factory wrapper is needed to remove unsupported sampling
    arguments without modifying the installed GraphRAG package or other queries.
    """
    from graphrag.cli import query as cli_query
    from graphrag.api import query as query_api

    def execute() -> tuple[str, dict[str, Any]]:
        original_factory = query_api.get_drift_search_engine

        def compatible_factory(*args: Any, **kwargs: Any) -> Any:
            engine = original_factory(*args, **kwargs)
            config = kwargs.get("config", args[0] if args else None)
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
            return str(answer), context if isinstance(context, dict) else {}
        finally:
            query_api.get_drift_search_engine = original_factory

    # The GraphRAG API resolves its factory through a module global. Serialize
    # DRIFT calls while that reference is temporarily wrapped to prevent races.
    with _DRIFT_PATCH_LOCK:
        return execute()
