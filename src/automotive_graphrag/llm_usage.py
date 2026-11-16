"""Record LLM calls and token usage for one query, and summarize them for comparison."""

from __future__ import annotations

import functools
import time
from typing import Any, Iterable, Mapping

# Stages of one query's LLM calls, as stored in `execution_metadata["llm_calls"]`.
ROUTING_STAGE = "routing"
SEARCH_STAGE = "search"
ANSWER_CHECK_STAGE = "answer_check"
QUERY_REWRITE_STAGE = "query_rewrite"
AGENTIC_STAGE = "agentic"


def call_record(
    stage: str,
    method: str | None,
    model: str | None,
    usage: Mapping[str, Any] | None,
    duration_seconds: float,
    error: str | None = None,
) -> dict[str, Any]:
    """One LLM call; token counts are None when the API did not report usage."""
    prompt_tokens, completion_tokens = usage_tokens(usage)
    return {
        "stage": stage,
        "method": method,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "duration_seconds": round(duration_seconds, 3),
        "error": error,
    }


def usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    """Prompt and completion tokens from an OpenAI-style `usage` dict or object."""
    if usage is None:
        return None, None
    if not isinstance(usage, Mapping):
        usage = {
            key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens")
        }
    def number(key: str) -> int | None:
        value = usage.get(key)
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return number("prompt_tokens"), number("completion_tokens")


def tag_calls(calls: Iterable[Mapping[str, Any]], stage: str, method: str | None) -> list[dict[str, Any]]:
    """Calls reported by a search worker, labelled with the stage and method that made them."""
    return [{**dict(call), "stage": stage, "method": method} for call in calls if isinstance(call, Mapping)]


def usage_metadata(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Execution-metadata fields derived from the recorded calls."""
    measured = [call for call in calls if call.get("prompt_tokens") is not None]
    if not calls:
        status = "unknown_no_llm_calls_recorded"
    elif len(measured) == len(calls):
        status = "measured"
    elif measured:
        status = "partially_measured"
    else:
        status = "unknown_usage_not_returned"
    return {
        "llm_calls": calls,
        # None rather than 0: with no records the count is unknown, not zero.
        "model_call_count": len(calls) if calls else None,
        "prompt_tokens": sum(int(call["prompt_tokens"]) for call in measured) if measured else None,
        "completion_tokens": (
            sum(int(call.get("completion_tokens") or 0) for call in measured) if measured else None
        ),
        "cost_status": status,
    }


def query_cost_summary(execution_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-query search count, LLM calls, tokens and whether the first-ranked method answered.

    Older records have no `llm_calls`; their call and token fields are None rather than zero.
    """
    metadata = execution_metadata or {}
    attempts = metadata.get("strategy_attempts")
    attempts = attempts if isinstance(attempts, list) else []
    agentic = metadata.get("search_step_count")
    if attempts:
        search_count: int | None = len(attempts)
    elif isinstance(agentic, int):
        search_count = agentic
    elif metadata.get("requested_method") in {"local", "global", "drift", "basic"}:
        search_count = 1
    else:
        search_count = None
    has_calls = isinstance(metadata.get("llm_calls"), list)
    prompt_tokens = metadata.get("prompt_tokens") if has_calls else None
    completion_tokens = metadata.get("completion_tokens") if has_calls else None
    return {
        "search_count": search_count,
        "llm_call_count": metadata.get("model_call_count") if has_calls else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (
            int(prompt_tokens) + int(completion_tokens or 0) if isinstance(prompt_tokens, int) else None
        ),
        "first_method": str(attempts[0].get("method", "")) if attempts else None,
        "first_choice_found": attempts[0].get("status") == "found" if attempts else None,
    }


def aggregate_cost(summaries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average search count, LLM calls and tokens, plus the first-choice hit rate per first method."""
    items = list(summaries)

    def mean(key: str) -> float | None:
        values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
        return round(sum(values) / len(values), 2) if values else None

    routed = [item for item in items if item.get("first_choice_found") is not None]
    by_method: dict[str, dict[str, int]] = {}
    for item in routed:
        bucket = by_method.setdefault(str(item.get("first_method") or ""), {"first": 0, "found": 0})
        bucket["first"] += 1
        bucket["found"] += int(bool(item["first_choice_found"]))
    return {
        "query_count": len(items),
        "average_search_count": mean("search_count"),
        "average_llm_call_count": mean("llm_call_count"),
        "average_prompt_tokens": mean("prompt_tokens"),
        "average_completion_tokens": mean("completion_tokens"),
        "average_total_tokens": mean("total_tokens"),
        "measured_query_count": sum(1 for item in items if isinstance(item.get("total_tokens"), int)),
        "first_choice_hit_rate": (
            round(sum(bool(item["first_choice_found"]) for item in routed) / len(routed), 4) if routed else None
        ),
        "first_choice_by_method": by_method,
    }


_recorder_target: list[dict[str, Any]] = []


def install_litellm_recorder(calls: list[dict[str, Any]]) -> None:
    """Record every LiteLLM chat completion made in this process (used by the query worker).

    Streaming calls ask for usage in the final chunk; the stream is passed through unchanged.
    LiteLLM is patched once per process; a later call only redirects the records to `calls`.
    """
    import litellm

    global _recorder_target
    _recorder_target = calls
    if getattr(litellm.acompletion, "_usage_recorder", False):
        return
    original_async = litellm.acompletion
    original_sync = litellm.completion

    def record(kwargs: dict[str, Any], usage: Any, started: float, error: str | None = None) -> None:
        _recorder_target.append(
            call_record(SEARCH_STAGE, None, kwargs.get("model"), _as_usage(usage), time.monotonic() - started, error)
        )

    def prepare(kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("stream") and "stream_options" not in kwargs:
            kwargs = {**kwargs, "stream_options": {"include_usage": True}}
        return kwargs

    @functools.wraps(original_async)
    async def acompletion(*args: Any, **kwargs: Any) -> Any:
        kwargs = prepare(kwargs)
        started = time.monotonic()
        try:
            response = await original_async(*args, **kwargs)
        except Exception as exc:
            record(kwargs, None, started, str(exc) or type(exc).__name__)
            raise
        if not kwargs.get("stream"):
            record(kwargs, getattr(response, "usage", None), started)
            return response

        async def stream() -> Any:
            usage = None
            try:
                async for chunk in response:
                    usage = getattr(chunk, "usage", None) or usage
                    yield chunk
            finally:
                record(kwargs, usage, started)
        return stream()

    @functools.wraps(original_sync)
    def completion(*args: Any, **kwargs: Any) -> Any:
        kwargs = prepare(kwargs)
        started = time.monotonic()
        try:
            response = original_sync(*args, **kwargs)
        except Exception as exc:
            record(kwargs, None, started, str(exc) or type(exc).__name__)
            raise
        if not kwargs.get("stream"):
            record(kwargs, getattr(response, "usage", None), started)
            return response

        def stream() -> Any:
            usage = None
            try:
                for chunk in response:
                    usage = getattr(chunk, "usage", None) or usage
                    yield chunk
            finally:
                record(kwargs, usage, started)
        return stream()

    acompletion._usage_recorder = True  # type: ignore[attr-defined]
    litellm.acompletion = acompletion
    litellm.completion = completion


def _as_usage(usage: Any) -> dict[str, Any] | None:
    prompt_tokens, completion_tokens = usage_tokens(usage)
    if prompt_tokens is None and completion_tokens is None:
        return None
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

