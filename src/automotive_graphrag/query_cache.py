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
        return connection

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
            total = connection.execute("SELECT COALESCE(SUM(size),0) FROM query_cache").fetchone()[0]
            while total > self.max_bytes:
                oldest = connection.execute("SELECT cache_key,size FROM query_cache ORDER BY created LIMIT 1").fetchone()
                if oldest is None:
                    break
                connection.execute("DELETE FROM query_cache WHERE cache_key=?", (oldest[0],))
                total -= oldest[1]

    def clear(self) -> int:
        with self._lock:
            try:
                with self._connect() as connection:
                    count = connection.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
                    connection.execute("DELETE FROM query_cache")
                    return int(count)
            except sqlite3.DatabaseError:
                for suffix in ("", "-wal", "-shm"):
                    Path(f"{self.path}{suffix}").unlink(missing_ok=True)
                return -1

    def count(self) -> int:
        with self._lock:
            try:
                with self._connect() as connection:
                    return int(connection.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0])
            except sqlite3.DatabaseError:
                return -1
