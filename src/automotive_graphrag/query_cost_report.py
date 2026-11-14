"""Search-cost statistics of a query log (P-02), for comparing runs before and after a change.

Usage: python -m automotive_graphrag.query_cost_report projects/<id>/runs/queries.jsonl --source automatic_qa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .llm_usage import aggregate_cost, query_cost_summary



def summarize_query_log(path: str, source: str | None = None, since: str | None = None,
                        method: str | None = "auto") -> dict[str, Any]:
    """Aggregate a project's `runs/queries.jsonl`, optionally filtered by source, start time and method."""
    summaries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if source and record.get("source") != source:
            continue
        if since and str(record.get("started_at", "")) < since:
            continue
        if method and (record.get("requested_method") or record.get("method")) != method:
            continue
        summaries.append(query_cost_summary(record.get("execution_metadata")))
    return aggregate_cost(summaries)


def main(argv: list[str] | None = None) -> int:
    """Print search-cost statistics of a query log, for comparing runs before and after a change."""
    parser = argparse.ArgumentParser(description="統計 queries.jsonl 的搜尋次數、LLM 呼叫、Token 與路由首選命中率")
    parser.add_argument("path", help="專案的 runs/queries.jsonl")
    parser.add_argument("--source", help="只統計此來源（例如自動問答測試）")
    parser.add_argument("--since", help="只統計此 ISO 時間之後開始的查詢，例如 2026-11-14")
    parser.add_argument("--method", default="auto", help="要求的查詢方法，預設 auto；all 表示不篩選")
    args = parser.parse_args(argv)
    method = None if args.method == "all" else args.method
    print(json.dumps(summarize_query_log(args.path, args.source, args.since, method), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
