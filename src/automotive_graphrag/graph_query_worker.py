"""Run an isolated Microsoft GraphRAG query for retrieval experiments."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from .answer_extraction import install_two_stage_basic, reset_two_stage_basic
from .cross_references import capture_basic_retrieval, follow_references, reset_capture_basic_retrieval
from .global_prefilter import DEFAULT_GLOBAL_REPORT_TOP_N, prefilter_reports
from .global_rewrite import rewrite_global_answer
from .global_reports import community_manual_pages, keep_cited_reports, label_reports
from .global_sources import ground_global_answer
from .llm_usage import install_litellm_recorder
from .local_ranking import install_local_ranking, reset_local_ranking, text_unit_similarity
from .variants import select_basic_context
from .querying import QueryService

_IN_PROCESS_QUERY_LOCK = threading.RLock()


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


def reset_retrieval_query_for_basic() -> None:
    """Remove the per-query Basic retrieval rewrite before another in-process query runs."""
    from graphrag.query.structured_search.basic_search.basic_context import BasicSearchContext

    current = BasicSearchContext.build_context
    if getattr(current, "_retrieval_query_patch", False):
        BasicSearchContext.build_context = current.__wrapped__  # type: ignore[method-assign]


def execute_graph_query(
    payload: dict[str, Any], preloaded_dataframes: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Execute one CLI query and return JSON-safe answer and context data."""
    # Search adapters patch GraphRAG module-level loaders and callbacks. Keep those
    # temporary hooks scoped to one request while allowing the loaded tables to persist.
    with _IN_PROCESS_QUERY_LOCK, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            original_cwd = Path.cwd()
        except OSError:
            original_cwd = Path(__file__).resolve().parents[2]
            os.chdir(original_cwd)
        try:
            return _execute_graph_query(payload, preloaded_dataframes)
        finally:
            try:
                os.chdir(original_cwd)
            except FileNotFoundError:
                os.chdir(Path(__file__).resolve().parents[2])


def _execute_graph_query(
    payload: dict[str, Any], preloaded_dataframes: dict[str, object] | None,
) -> dict[str, Any]:
    method = str(payload.get("method", ""))
    root_dir = Path(str(payload["root_dir"]))
    question = str(payload["question"])
    response_type = str(payload.get("response_type") or "Single Paragraph")
    community_level = int(payload.get("community_level", 2))
    dynamic_community_selection = bool(payload.get("dynamic_community_selection", False))
    if method not in {"local", "global", "drift", "basic"}:
        raise ValueError(f"不支援的 GraphRAG 檢索策略：{method}")

    from graphrag.cli import query as cli_query

    from . import llm_usage

    previous_recorder_target = llm_usage._recorder_target
    llm_calls: list[dict[str, Any]] = []
    variant_report: dict[str, Any] = {}
    extraction_report: dict[str, Any] = {}
    cross_reference_report: dict[str, Any] = {}
    local_ranking_report: dict[str, Any] = {}
    install_litellm_recorder(llm_calls)
    original_resolver = cli_query._resolve_output_files
    if preloaded_dataframes is not None:
        def resolve_cached_output_files(config, output_list, optional_list=None):
            missing = [name for name in output_list if name not in preloaded_dataframes]
            if missing:
                raise FileNotFoundError(f"GraphRAG 索引缺少必要資料表：{', '.join(missing)}")
            result = {name: preloaded_dataframes[name] for name in output_list}
            for name in optional_list or []:
                result[name] = preloaded_dataframes.get(name)
            return result

        cli_query._resolve_output_files = resolve_cached_output_files
    manual_pages: dict[str, dict[str, tuple[int, ...]]] | None = None
    global_communities: Any = None
    global_sources: list[dict[str, Any]] = []
    grounding_report: dict[str, Any] = {}
    prefilter_report: dict[str, Any] = {}
    rewrite_report: dict[str, Any] = {}
    if method == "global":
        active_resolver = cli_query._resolve_output_files
        report_top_n = int(payload.get("global_report_top_n", DEFAULT_GLOBAL_REPORT_TOP_N))

        def resolve_labelled_reports(config, output_list, optional_list=None):
            # Name the manuals in each report so Global can answer "which manuals" (F-70).
            nonlocal manual_pages, global_communities
            result = dict(active_resolver(config, output_list, optional_list))
            reports, communities = result.get("community_reports"), result.get("communities")
            global_communities = communities
            if reports is not None and not dynamic_community_selection:
                # Map only the reports closest to the question (P-05); dynamic selection walks
                # the report hierarchy itself and needs every report.
                try:
                    reports, report = prefilter_reports(
                        reports, question, root_dir, report_top_n, community_level,
                    )
                except Exception as exc:  # the prefilter must never block an answer
                    report = {"applied": False, "error": str(exc)}
                prefilter_report.update(report)
                result["community_reports"] = reports
            if reports is not None and communities is not None:
                try:
                    output_dir = root_dir / "output"
                    manual_pages = community_manual_pages(output_dir, communities)
                    result["community_reports"] = label_reports(reports, communities, output_dir)
                except (OSError, KeyError, ValueError):  # an index without source metadata answers unlabelled
                    manual_pages = None
            return result

        cli_query._resolve_output_files = resolve_labelled_reports
    common = {
        "data_dir": None,
        "root_dir": root_dir,
        "response_type": response_type,
        "streaming": False,
        "query": question,
        "verbose": False,
    }
    try:
        if method == "local":
            install_local_ranking(
                local_ranking_report, text_unit_similarity(root_dir),
                str(payload.get("retrieval_query") or "").strip() or None,
            )
            answer, context = cli_query.run_local_search(community_level=community_level, **common)
        elif method == "global":
            answer, context = cli_query.run_global_search(
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                **common,
            )
            if global_communities is not None:
                # Point each cited statement at the closest passages of its reports (F-70).
                try:
                    answer, global_sources, grounding_report = ground_global_answer(
                        str(answer), global_communities, root_dir,
                        preloaded_dataframes.get("text_units") if preloaded_dataframes is not None else None,
                    )
                except Exception as exc:  # grounding must never block an answer
                    grounding_report = {"error": str(exc)}
                if bool(payload.get("global_source_rewrite", True)) and "error" not in grounding_report:
                    # Rewrite the summary-based answer from the cited passages (F-70 step 4).
                    try:
                        rewritten, rewrite_sources, rewrite_report = rewrite_global_answer(
                            question, str(answer), [str(item["id"]) for item in global_sources],
                            global_communities, root_dir, response_type,
                            preloaded_dataframes.get("text_units") if preloaded_dataframes is not None else None,
                        )
                    except Exception as exc:  # the rewrite must never block an answer
                        rewrite_report = {"applied": False, "error": str(exc)}
                    else:
                        if rewrite_report.get("applied"):
                            seen = {str(item["id"]) for item in rewrite_sources}
                            answer = rewritten
                            global_sources = [
                                *rewrite_sources, *(item for item in global_sources if str(item["id"]) not in seen),
                            ]
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
            preloaded_text_units = preloaded_dataframes.get("text_units") if preloaded_dataframes is not None else None

            def select_variants(query: str, result: Any) -> Any:
                try:
                    result, reference_report = follow_references(result, retrieval, root_dir, preloaded_text_units)
                except Exception as exc:  # following references must never block an answer
                    reference_report = {"error": str(exc)}
                if reference_report:
                    cross_reference_report.update(reference_report)
                try:
                    # A rewritten retrieval query carries variant names of a question in another language.
                    selected, report = select_basic_context(
                        result, f"{query}\n{retrieval_query}", root_dir, preloaded_text_units,
                    )
                except Exception as exc:  # variant selection must never block an answer
                    variant_report["error"] = str(exc)
                    return result
                variant_report.update(report)
                return selected

            install_two_stage_basic(select_variants, extraction_report)  # qualifier revision off: SM3E Q2 rerun 2026-11-20 scored lower (F-56)
            answer, context = cli_query.run_basic_search(
                data_dir=None,
                root_dir=root_dir,
                response_type=response_type,
                streaming=False,
                query=question,
                verbose=False,
            )
    finally:
        cli_query._resolve_output_files = original_resolver
        reset_retrieval_query_for_basic()
        reset_capture_basic_retrieval()
        reset_two_stage_basic()
        reset_local_ranking()
        llm_usage._recorder_target = previous_recorder_target
    serialized = QueryService._serialize_context(context)
    if method == "global" and isinstance(serialized, dict):
        serialized = keep_cited_reports(serialized, str(answer), manual_pages)
        if global_sources:
            serialized["sources"] = global_sources
        if grounding_report:
            serialized["source_grounding"] = grounding_report
        if prefilter_report:
            serialized["report_prefilter"] = prefilter_report
        if rewrite_report:
            serialized["source_rewrite"] = rewrite_report
    if variant_report and isinstance(serialized, dict):
        serialized["variant_selection"] = variant_report
    if cross_reference_report and isinstance(serialized, dict):
        serialized["cross_references"] = cross_reference_report
    if local_ranking_report and isinstance(serialized, dict):
        serialized["local_ranking"] = local_ranking_report
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
