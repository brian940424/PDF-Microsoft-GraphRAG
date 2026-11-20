"""Run an isolated Microsoft GraphRAG query for retrieval experiments."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from .answer_extraction import install_two_stage_basic
from .cross_references import capture_basic_retrieval, follow_references
from .llm_usage import install_litellm_recorder
from .variants import select_basic_context
from .querying import QueryService


def use_retrieval_query_for_basic(retrieval_query: str) -> None:
    """Make Basic pick text units with `retrieval_query` while the answer prompt keeps the question."""
    from graphrag.query.structured_search.basic_search.basic_context import BasicSearchContext

    original = BasicSearchContext.build_context
    if getattr(original, "_retrieval_query_patch", False):
        original = original.__wrapped__

    def build_context(self, query: str, *args: Any, **kwargs: Any):
        return original(self, retrieval_query if query else query, *args, **kwargs)

    build_context._retrieval_query_patch = True  # type: ignore[attr-defined]
    build_context.__wrapped__ = original  # type: ignore[attr-defined]
    BasicSearchContext.build_context = build_context  # type: ignore[method-assign]


def execute_graph_query(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one CLI query and return JSON-safe answer and context data."""
    method = str(payload.get("method", ""))
    root_dir = Path(str(payload["root_dir"]))
    question = str(payload["question"])
    response_type = str(payload.get("response_type") or "Single Paragraph")
    community_level = int(payload.get("community_level", 2))
    dynamic_community_selection = bool(payload.get("dynamic_community_selection", False))
    if method not in {"local", "global", "drift", "basic"}:
        raise ValueError(f"不支援的 GraphRAG 檢索策略：{method}")

    from graphrag.cli import query as cli_query

    llm_calls: list[dict[str, Any]] = []
    variant_report: dict[str, Any] = {}
    extraction_report: dict[str, Any] = {}
    cross_reference_report: dict[str, Any] = {}
    install_litellm_recorder(llm_calls)
    common = {
        "data_dir": None,
        "root_dir": root_dir,
        "response_type": response_type,
        "streaming": False,
        "query": question,
        "verbose": False,
    }
    if method == "local":
        answer, context = cli_query.run_local_search(community_level=community_level, **common)
    elif method == "global":
        answer, context = cli_query.run_global_search(
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            **common,
        )
    elif method == "drift":
        from .drift_compat import run_drift_search

        answer, context = run_drift_search(
            root_dir=root_dir,
            query=question,
            community_level=community_level,
            response_type=response_type,
        )
    else:
        retrieval_query = str(payload.get("retrieval_query") or "").strip()
        if retrieval_query:
            use_retrieval_query_for_basic(retrieval_query)

        retrieval: dict[str, Any] = {}
        capture_basic_retrieval(retrieval)

        def select_variants(query: str, result: Any) -> Any:
            try:
                result, reference_report = follow_references(result, retrieval, root_dir)
            except Exception as exc:  # following references must never block an answer
                reference_report = {"error": str(exc)}
            if reference_report:
                cross_reference_report.update(reference_report)
            try:
                # A rewritten retrieval query carries variant names of a question in another language.
                selected, report = select_basic_context(result, f"{query}\n{retrieval_query}", root_dir)
            except Exception as exc:  # variant selection must never block an answer
                variant_report["error"] = str(exc)
                return result
            variant_report.update(report)
            return selected

        install_two_stage_basic(select_variants, extraction_report)
        answer, context = cli_query.run_basic_search(
            data_dir=None,
            root_dir=root_dir,
            response_type=response_type,
            streaming=False,
            query=question,
            verbose=False,
        )
    serialized = QueryService._serialize_context(context)
    if variant_report and isinstance(serialized, dict):
        serialized["variant_selection"] = variant_report
    if cross_reference_report and isinstance(serialized, dict):
        serialized["cross_references"] = cross_reference_report
    if extraction_report and isinstance(serialized, dict):
        serialized["answer_extraction"] = extraction_report
    return {
        "answer": str(answer),
        "context": serialized,
        "llm_calls": llm_calls,
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        # GraphRAG writes its answer to stdout; keep the worker protocol to one
        # JSON object so the parent can reliably parse the response.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = execute_graph_query(payload)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
