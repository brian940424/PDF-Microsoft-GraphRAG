"""Conservative semantic query-cache policy and OpenAI-compatible embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from .connections import ConnectionSettings, llm_request_headers
from .query_cache import QueryResultCache, normalize_query

SCHEMA_VERSION = 1
EMBEDDING_VERSION = "semantic-query-embedding-v1"
_NUMBER_UNIT = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?\s*(?:n\s*[·.]?\s*m|nm|kgf\s*/?\s*cm2|kpa|mpa|bar|psi|°?c|°?f|v|a|l|ml|mm|cm|km|s|sec|min|h|rpm|%|頁|年|號)?", re.I)
_IDENTIFIER = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9]+(?:-[A-Z0-9]+)*\b", re.I)
_NEGATION = re.compile(r"不可以|不可|不能|不要|無法|沒有|未|勿|非|禁止|不得|可以|能否|是否|啟用|停用|接上|拔除|disconnect|connect|enable|disable|not|never|no\b|cannot|shouldn't|should not", re.I)
_ACTIONS = re.compile(r"檢查|測量|更換|拆卸|安裝|重設|校正|充電|啟動|排除|維修|清潔|檢修|replace|remove|install|inspect|check|measure|reset|calibrate|charge|start|troubleshoot", re.I)
_CONTEXT = re.compile(r"上述|剛才|前面|那個|接下來|然後呢|我的車|這個|那要|之後呢|上述步驟|what about then|after that|my car|the previous", re.I)
_MULTI = re.compile(r"比較|差異|不同|哪一個|哪個|或者|以及|並且|同時|compare|difference|which one|versus|\bvs\b|\bor\b|\band\b", re.I)
_CONDITION = re.compile(r"如果|若|當|只在|僅在|冷車|熱車|怠速時|行駛時|發動時|停止時|故障時|\bif\b|\bwhen\b|\bwhile\b|\bunless\b|\bduring\b|\bonly when\b", re.I)


def project_mode_path(project_path: Path) -> Path:
    return project_path / "query_cache_settings.json"


def read_project_mode(project_path: Path) -> str:
    try:
        value = json.loads(project_mode_path(project_path).read_text(encoding="utf-8"))
        mode = value.get("semantic_cache_mode")
        return mode if mode in {"off", "shadow", "serving"} else "off"
    except (OSError, json.JSONDecodeError, AttributeError):
        return "off"


def write_project_mode(project_path: Path, mode: str) -> None:
    if mode not in {"off", "shadow", "serving"}:
        raise ValueError("invalid semantic cache mode")
    project_path.mkdir(parents=True, exist_ok=True)
    path = project_mode_path(project_path)
    descriptor, temporary = tempfile.mkstemp(prefix=".query-cache-settings-", dir=project_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            output.write(json.dumps({"schema_version": SCHEMA_VERSION, "semantic_cache_mode": mode}, ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def semantic_config() -> dict[str, Any] | None:
    if os.environ.get("GRAPHRAG_ENABLE_SEMANTIC_QUERY_CACHE", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        top_k = int(os.environ.get("GRAPHRAG_SEMANTIC_QUERY_CACHE_TOP_K", "5"))
        if not 1 <= top_k <= 20:
            return None
    except (TypeError, ValueError):
        return None
    try:
        threshold = float(os.environ["GRAPHRAG_SEMANTIC_QUERY_CACHE_THRESHOLD"])
        margin = float(os.environ["GRAPHRAG_SEMANTIC_QUERY_CACHE_MIN_MARGIN"])
        calibrated = 0 < threshold <= 1 and 0 <= margin <= 1
    except (KeyError, TypeError, ValueError):
        threshold, margin, calibrated = None, None, False
    return {"threshold": threshold if calibrated else None, "margin": margin if calibrated else None,
            "top_k": top_k, "calibrated": calibrated}


def scope_hash(scope: dict[str, Any]) -> str:
    body = {"schema": SCHEMA_VERSION, "embedding_version": EMBEDDING_VERSION, **scope}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def hard_guard(source: str, candidate: str) -> tuple[bool, str | None]:
    """Reject high-risk differences before considering semantic similarity."""
    left, right = normalize_query(source), normalize_query(candidate)
    for pattern, reason in ((_CONTEXT, "context_dependent"), (_MULTI, "multi_part_or_comparison")):
        if pattern.search(left) or pattern.search(right):
            return False, reason
    if sorted(_IDENTIFIER.findall(left)) != sorted(_IDENTIFIER.findall(right)):
        return False, "identifier_mismatch"
    if sorted(x.lower().replace(" ", "") for x in _NUMBER_UNIT.findall(left)) != sorted(x.lower().replace(" ", "") for x in _NUMBER_UNIT.findall(right)):
        return False, "numeric_or_unit_mismatch"
    if sorted(x.lower() for x in _NEGATION.findall(left)) != sorted(x.lower() for x in _NEGATION.findall(right)):
        return False, "polarity_mismatch"
    if sorted(x.lower() for x in _ACTIONS.findall(left)) != sorted(x.lower() for x in _ACTIONS.findall(right)):
        return False, "action_mismatch"
    left_conditions = [match.start() for match in _CONDITION.finditer(left)]
    right_conditions = [match.start() for match in _CONDITION.finditer(right)]
    if bool(left_conditions) != bool(right_conditions):
        return False, "condition_mismatch"
    if left_conditions and left[min(left_conditions):] != right[min(right_conditions):]:
        return False, "condition_mismatch"
    return True, None


Embedding = Callable[[str, str, str], tuple[list[float], float]]


def proxy_embedding(text: str, model: str, project_id: str, connections: ConnectionSettings) -> tuple[list[float], float]:
    """Request one embedding from the configured OpenAI-compatible proxy."""
    started = time.monotonic()
    base = connections.get_api_base_url().rstrip("/")
    key = connections.get_api_key(project_id)
    payload = json.dumps({"model": model, "input": text}).encode("utf-8")
    request = urllib.request.Request(f"{base}/embeddings", data=payload, headers=llm_request_headers(key), method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    vector = body["data"][0]["embedding"]
    if not isinstance(vector, list) or not vector or any(not math.isfinite(float(v)) for v in vector):
        raise ValueError("proxy returned invalid embedding")
    return [float(v) for v in vector], (time.monotonic() - started) * 1000


def candidate(cache: QueryResultCache, query: str, scope: dict[str, Any], model: str,
              embed: Embedding, top_k: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    stats: dict[str, Any] = {"semantic_embedding_calls": 0, "semantic_embedding_model": model,
                              "semantic_embedding_latency_ms": None, "semantic_cache_scope_hash": scope_hash(scope)}
    embedding_started = time.monotonic()
    try:
        stats["semantic_embedding_calls"] = 1
        vector, latency = embed(query, model, str(scope["project_id"]))
        stats["semantic_embedding_latency_ms"] = round(latency, 2)
        rows = cache.semantic_candidates(stats["semantic_cache_scope_hash"], vector, top_k)
        stats["_query_vector"] = vector
    except Exception as exc:
        stats["semantic_embedding_latency_ms"] = round((time.monotonic() - embedding_started) * 1000, 2)
        stats["semantic_cache_reject_reason"] = f"embedding_or_cache_error:{type(exc).__name__}"
        return None, stats
    if not rows:
        stats["semantic_cache_reject_reason"] = "no_candidate"
        return None, stats
    for row in rows:
        allowed, reason = hard_guard(query, row["query"])
        row["guard_pass"] = allowed
        row["guard_reason"] = reason
    valid = [row for row in rows if row["guard_pass"]]
    if not valid:
        stats["semantic_cache_reject_reason"] = rows[0]["guard_reason"] or "hard_guard"
        return None, stats
    best = valid[0]
    second = valid[1]["score"] if len(valid) > 1 else -1.0
    stats.update({"semantic_cache_score": round(best["score"], 6), "semantic_cache_candidate_id": best["item_id"]})
    config = semantic_config()
    if config is None or not config["calibrated"]:
        stats["semantic_cache_reject_reason"] = "uncalibrated_threshold"
        return None, stats
    stats["semantic_cache_threshold"] = config["threshold"]
    if best["score"] < config["threshold"]:
        stats["semantic_cache_reject_reason"] = "below_threshold"
        return None, stats
    if best["score"] - second < config["margin"]:
        stats["semantic_cache_reject_reason"] = "insufficient_margin"
        return None, stats
    stats["semantic_cache_reject_reason"] = None
    return best, stats

