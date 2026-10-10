"""Run an isolated Microsoft GraphRAG query for retrieval experiments."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from .connections import GPT6_LUNA_MODEL
from .querying import QueryService


def execute_graph_query(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one CLI query and return JSON-safe answer and context data."""
    method = str(payload.get("method", ""))
    root_dir = Path(str(payload["root_dir"]))
    question = str(payload["question"])
    model = str(payload.get("model", ""))
    response_type = str(payload.get("response_type") or "Single Paragraph")
    if method not in {"local", "global", "drift", "basic"}:
        raise ValueError(f"不支援的 GraphRAG 檢索策略：{method}")

    from graphrag.cli import query as cli_query

    common = {
        "data_dir": None,
        "root_dir": root_dir,
        "response_type": response_type,
        "streaming": False,
        "query": question,
        "verbose": False,
    }
    if method == "local":
        answer, context = cli_query.run_local_search(community_level=2, **common)
    elif method == "global":
        answer, context = cli_query.run_global_search(
            community_level=2,
            dynamic_community_selection=False,
            **common,
        )
    elif method == "drift" and model == GPT6_LUNA_MODEL:
        from .drift_compat import run_drift_search

        answer, context = run_drift_search(
            root_dir=root_dir,
            query=question,
            community_level=2,
            response_type=response_type,
        )
    elif method == "drift":
        answer, context = cli_query.run_drift_search(community_level=2, **common)
    else:
        answer, context = cli_query.run_basic_search(
            data_dir=None,
            root_dir=root_dir,
            response_type=response_type,
            streaming=False,
            query=question,
            verbose=False,
        )
    return {
        "answer": str(answer),
        "context": QueryService._serialize_context(context),
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
