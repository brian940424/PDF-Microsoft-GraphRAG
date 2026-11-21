"""Thread-safe in-process LRU cache for memory-mapped GraphRAG index tables."""

from __future__ import annotations

import gc
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .query_cache import index_signature


@dataclass(slots=True)
class LoadedGraphIndex:
    project_id: str
    signature: str
    dataframes: dict[str, object]
    arrow_tables: dict[str, object]
    load_seconds: float


class GraphIndexManager:
    """Retain typed Arrow-backed DataFrames by project and successful index signature."""

    _instance: "GraphIndexManager | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, max_loaded_projects: int | None = None,
                 loader: Callable[[Path], tuple[dict[str, object], dict[str, object]]] | None = None) -> None:
        try:
            configured_limit = int(os.environ.get("GRAPHRAG_MAX_LOADED_PROJECTS", "2"))
        except ValueError:
            configured_limit = 2
        self.max_loaded_projects = max(1, max_loaded_projects if max_loaded_projects is not None else configured_limit)
        self.loader = loader or self._load_index
        self._entries: OrderedDict[tuple[str, str], LoadedGraphIndex] = OrderedDict()
        self._project_locks: dict[str, threading.Lock] = {}
        self._lock = threading.RLock()

    @classmethod
    def instance(cls) -> "GraphIndexManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get(self, project_id: str, project_path: Path) -> tuple[dict[str, object], bool, float]:
        """Return frames, whether all were already resident, and time spent loading them."""
        signature = index_signature(project_path)
        key = (project_id, signature)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry.dataframes, True, 0.0
            project_lock = self._project_locks.setdefault(project_id, threading.Lock())
        with project_lock:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    return entry.dataframes, True, 0.0
                self._remove_project_versions(project_id)
            started = time.monotonic()
            dataframes, arrow_tables = self.loader(project_path)
            elapsed = time.monotonic() - started
            loaded = LoadedGraphIndex(project_id, signature, dataframes, arrow_tables, elapsed)
            with self._lock:
                self._entries[key] = loaded
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_loaded_projects:
                    _, evicted = self._entries.popitem(last=False)
                    del evicted
                    gc.collect()
            return dataframes, False, elapsed

    def clear(self, project_id: str | None = None) -> int:
        """Drop one or every resident index and collect released DataFrame references."""
        with self._lock:
            if project_id is None:
                count = len(self._entries)
                self._entries.clear()
            else:
                count = self._remove_project_versions(project_id)
            if count:
                gc.collect()
            return count

    def loaded_projects(self) -> list[str]:
        with self._lock:
            return list(dict.fromkeys(project_id for project_id, _ in self._entries))

    def _remove_project_versions(self, project_id: str) -> int:
        keys = [key for key in self._entries if key[0] == project_id]
        for key in keys:
            del self._entries[key]
        if keys:
            gc.collect()
        return len(keys)

    @staticmethod
    def _load_index(project_path: Path) -> tuple[dict[str, object], dict[str, object]]:
        import pyarrow.parquet as parquet
        from graphrag.data_model.dfs import (
            communities_typed,
            community_reports_typed,
            covariates_typed,
            entities_typed,
            relationships_typed,
            text_units_typed,
        )

        transforms = {
            "entities": entities_typed,
            "relationships": relationships_typed,
            "communities": communities_typed,
            "community_reports": community_reports_typed,
            "text_units": text_units_typed,
            "covariates": covariates_typed,
        }
        output = project_path / "graphrag" / "output"
        dataframes: dict[str, object] = {}
        arrow_tables: dict[str, object] = {}
        for name, transform in transforms.items():
            path = output / f"{name}.parquet"
            if not path.exists() and name == "covariates":
                dataframes[name] = None
                continue
            table = parquet.read_table(path, memory_map=True)
            arrow_tables[name] = table
            dataframes[name] = transform(table.to_pandas())
        return dataframes, arrow_tables
