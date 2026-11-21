"""Project-isolated SQLite cache for completed GraphRAG query results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import math
import struct
from pathlib import Path
from typing import Any


DEFAULT_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_MAX_BYTES = 500 * 1024 * 1024


def normalize_query(query: str) -> str:
    """Normalize compatible whitespace, width, and Latin case differences."""
    normalized = unicodedata.normalize("NFKC", query)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.lower()


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def index_signature(project_path: Path) -> str:
    """Use successful index metadata, falling back to the indexed Parquet file stamps."""
    metadata = project_path / "graphrag" / "last_index_run.json"
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        if str(value.get("status", "")).upper() in {"INDEXED", "COMPLETED", "SUCCESS"}:
            for key in ("run_id", "completed_at"):
                if value.get(key):
                    return f"{key}:{value[key]}"
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    output = project_path / "graphrag" / "output"
    files = sorted(output.glob("*.parquet"))
    stamps = []
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        stamps.append((path.name, stat.st_mtime_ns, stat.st_size))
    return "files:" + hashlib.sha256(json.dumps(stamps, separators=(",", ":")).encode()).hexdigest()


class QueryResultCache:
    """A WAL-backed, multi-process-safe cache stored inside one project directory."""

    def __init__(self, project_path: Path, ttl_seconds: int | None = None,
                 max_bytes: int | None = None) -> None:
        self.path = project_path / "cache" / "query_cache.db"
        self.ttl_seconds = ttl_seconds if ttl_seconds is not None else _positive_env_int(
            "GRAPHRAG_QUERY_CACHE_TTL_SECONDS", DEFAULT_TTL_SECONDS
        )
        self.max_bytes = max_bytes if max_bytes is not None else _positive_env_int(
            "GRAPHRAG_QUERY_CACHE_MAX_BYTES", DEFAULT_MAX_BYTES
        )
        self._lock = threading.RLock()

    @staticmethod
    def make_key(parameters: dict[str, Any]) -> str:
        payload = json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("CREATE TABLE IF NOT EXISTS query_cache (cache_key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL, size INTEGER NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS semantic_query_cache (item_id TEXT PRIMARY KEY, scope_hash TEXT NOT NULL, query TEXT NOT NULL, vector BLOB NOT NULL, dimension INTEGER NOT NULL, value TEXT NOT NULL, source_key TEXT NOT NULL, created REAL NOT NULL, last_used REAL NOT NULL, size INTEGER NOT NULL, schema_version INTEGER NOT NULL)")
        connection.execute("CREATE INDEX IF NOT EXISTS semantic_scope_created ON semantic_query_cache(scope_hash, created DESC)")
        return connection

    @staticmethod
    def _vector_bytes(vector: list[float] | tuple[float, ...]) -> bytes:
        return struct.pack(f"<{len(vector)}f", *vector)

    @staticmethod
    def _vector_values(payload: bytes, dimension: int) -> tuple[float, ...]:
        if dimension <= 0 or len(payload) != dimension * 4:
            raise ValueError("invalid semantic vector")
        return struct.unpack(f"<{dimension}f", payload)

    def put_semantic(self, item_id: str, scope_hash: str, query: str, vector: list[float] | tuple[float, ...], value: dict[str, Any], source_key: str) -> None:
        if not vector or any(not math.isfinite(float(component)) for component in vector):
            raise ValueError("invalid semantic vector")
        vector_bytes = self._vector_bytes(vector)
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        now = time.time()
        size = len(vector_bytes) + len(payload.encode("utf-8")) + len(query.encode("utf-8"))
        with self._lock, self._connect() as connection:
            connection.execute("INSERT OR REPLACE INTO semantic_query_cache(item_id,scope_hash,query,vector,dimension,value,source_key,created,last_used,size,schema_version) VALUES(?,?,?,?,?,?,?,?,?,?,1)",
                               (item_id, scope_hash, query, vector_bytes, len(vector), payload, source_key, now, now, size))
            self._evict(connection)

    def semantic_candidates(self, scope_hash: str, vector: list[float] | tuple[float, ...], top_k: int) -> list[dict[str, Any]]:
        if not vector or top_k <= 0:
            return []
        now = time.time()
        candidates = []
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM semantic_query_cache WHERE created < ?", (now - self.ttl_seconds,))
            rows = connection.execute("SELECT item_id,query,vector,dimension,value,source_key,created FROM semantic_query_cache WHERE scope_hash=? AND schema_version=1 ORDER BY created DESC LIMIT 1000", (scope_hash,)).fetchall()
            for item_id, query, raw, dimension, payload, source_key, created in rows:
                if time.time() - created > self.ttl_seconds or dimension != len(vector):
                    continue
                try:
                    other = self._vector_values(raw, dimension)
                    value = json.loads(payload)
                    if not isinstance(value, dict):
                        continue
                    denominator = math.sqrt(sum(x*x for x in vector) * sum(x*x for x in other))
                    if denominator == 0:
                        continue
                    score = sum(float(x)*y for x,y in zip(vector, other)) / denominator
                except (ValueError, TypeError, json.JSONDecodeError, struct.error):
                    continue
                candidates.append({"item_id": item_id, "query": query, "value": value, "source_key": source_key, "score": score})
            candidates.sort(key=lambda item: item["score"], reverse=True)
            result = candidates[:top_k]
            for candidate in result:
                connection.execute("UPDATE semantic_query_cache SET last_used=? WHERE item_id=?", (now, candidate["item_id"]))
            return result

    def _evict(self, connection: sqlite3.Connection) -> None:
        exact = connection.execute("SELECT cache_key,created,size FROM query_cache").fetchall()
        semantic = connection.execute("SELECT item_id,last_used,size FROM semantic_query_cache").fetchall()
        rows = [(created, "exact", key, size) for key, created, size in exact]
        rows += [(last_used, "semantic", key, size) for key, last_used, size in semantic]
        total = sum(row[3] for row in rows)
        for _created, kind, key, size in sorted(rows):
            if total <= self.max_bytes:
                break
            table, column = ("query_cache", "cache_key") if kind == "exact" else ("semantic_query_cache", "item_id")
            connection.execute(f"DELETE FROM {table} WHERE {column}=?", (key,))
            total -= size

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT value, created FROM query_cache WHERE cache_key=?", (key,)).fetchone()
            if row is None:
                return None
            if time.time() - row[1] > self.ttl_seconds:
                connection.execute("DELETE FROM query_cache WHERE cache_key=?", (key,))
                return None
            value = json.loads(row[0])
            if not isinstance(value, dict):
                raise ValueError("query cache value is not an object")
            return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO query_cache(cache_key,value,created,size) VALUES(?,?,?,?) "
                "ON CONFLICT(cache_key) DO UPDATE SET value=excluded.value,created=excluded.created,size=excluded.size",
                (key, payload, time.time(), len(payload.encode("utf-8"))),
            )
            self._evict(connection)

    def clear(self) -> int:
        with self._lock:
            try:
                with self._connect() as connection:
                    count = connection.execute("SELECT (SELECT COUNT(*) FROM query_cache)+(SELECT COUNT(*) FROM semantic_query_cache)").fetchone()[0]
                    connection.execute("DELETE FROM query_cache")
                    connection.execute("DELETE FROM semantic_query_cache")
                    return int(count)
            except sqlite3.DatabaseError:
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{self.path}{suffix}").unlink(missing_ok=True)
                return -1

    def count(self) -> int:
        with self._lock:
            try:
                with self._connect() as connection:
                    return int(connection.execute("SELECT (SELECT COUNT(*) FROM query_cache)+(SELECT COUNT(*) FROM semantic_query_cache)").fetchone()[0])
            except sqlite3.DatabaseError:
                return -1

    def semantic_count(self) -> int:
        with self._lock:
            try:
                with self._connect() as connection:
                    return int(connection.execute("SELECT COUNT(*) FROM semantic_query_cache").fetchone()[0])
            except sqlite3.DatabaseError:
                return -1
