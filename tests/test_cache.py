import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.graph_index_cache import GraphIndexManager
from automotive_graphrag.query_cache import QueryResultCache, index_signature, normalize_query


class QueryResultCacheTests(unittest.TestCase):
    def test_normalizes_width_whitespace_and_latin_case(self):
        self.assertEqual(normalize_query("  Ｐ０３００   Misfire　Test "), "p0300 misfire test")

    def test_isolates_projects_and_clears_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            first = QueryResultCache(Path(directory) / "project-a", ttl_seconds=60)
            second = QueryResultCache(Path(directory) / "project-b")
            key_a = first.make_key({"project_id": "a", "strategy": "local"})
            key_b = second.make_key({"project_id": "b", "strategy": "local"})
            first.put(key_a, {"answer": "A"})
            self.assertEqual(first.get(key_a), {"answer": "A"})
            self.assertIsNone(second.get(key_b))
            self.assertEqual(first.clear(), 1)

    def test_expires_entries_and_enforces_byte_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = QueryResultCache(Path(directory) / "project", ttl_seconds=60, max_bytes=100)
            cache.put("old", {"answer": "x" * 80})
            cache.put("new", {"answer": "y" * 80})
            self.assertIsNone(cache.get("old"))
            self.assertEqual(cache.get("new"), {"answer": "y" * 80})
            cache.ttl_seconds = -1
            self.assertIsNone(cache.get("new"))

    def test_key_changes_when_strategy_model_parameters_or_signature_change(self):
        base = {
            "project_id": "A", "index_signature": "one", "strategy": "local",
            "chat_model": "m1", "strategy_params": {"level": 2}, "normalized_query": "question",
        }
        key = QueryResultCache.make_key(base)
        for name, value in (
            ("index_signature", "two"), ("strategy", "basic"), ("chat_model", "m2"),
            ("strategy_params", {"level": 3}),
        ):
            changed = dict(base)
            changed[name] = value
            self.assertNotEqual(QueryResultCache.make_key(changed), key)

    def test_signature_tracks_successful_run_and_parquet_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            output = project / "graphrag" / "output"
            output.mkdir(parents=True)
            parquet = output / "entities.parquet"
            parquet.write_bytes(b"first")
            fallback = index_signature(project)
            (project / "graphrag" / "last_index_run.json").write_text(
                json.dumps({"status": "INDEXED", "completed_at": "2026-11-20T00:00:00Z"}), encoding="utf-8",
            )
            self.assertEqual(index_signature(project), "completed_at:2026-11-20T00:00:00Z")
            (project / "graphrag" / "last_index_run.json").unlink()
            parquet.write_bytes(b"second version")
            self.assertNotEqual(index_signature(project), fallback)


    def test_semantic_candidates_are_scoped_and_clear_with_exact_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = QueryResultCache(Path(directory) / "project", ttl_seconds=60)
            cache.put_semantic("candidate-a", "scope-a", "inspect wiper fuse", [1.0, 0.0], {"answer": "A"}, "exact-a")
            self.assertEqual(cache.semantic_candidates("scope-a", [0.9, 0.1], 5)[0]["value"], {"answer": "A"})
            self.assertEqual(cache.semantic_candidates("scope-b", [1.0, 0.0], 5), [])
            self.assertEqual(cache.clear(), 1)
            self.assertEqual(cache.semantic_candidates("scope-a", [1.0, 0.0], 5), [])

    def test_existing_exact_only_database_gets_semantic_table_without_losing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            database = project / "cache" / "query_cache.db"
            database.parent.mkdir(parents=True)
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE query_cache (cache_key TEXT PRIMARY KEY, value TEXT NOT NULL, created REAL NOT NULL, size INTEGER NOT NULL)")
                connection.execute("INSERT INTO query_cache VALUES('legacy','{\"answer\":\"old\"}',?,15)", (time.time(),))
            cache = QueryResultCache(project)
            self.assertEqual(cache.get("legacy"), {"answer": "old"})
            cache.put_semantic("new", "scope", "query", [1.0], {"answer": "new"}, "legacy")
            self.assertEqual(cache.count(), 2)


class GraphIndexManagerTests(unittest.TestCase):
    def make_project(self, root: Path, project_id: str) -> Path:
        project = root / project_id
        output = project / "graphrag" / "output"
        output.mkdir(parents=True)
        (output / "entities.parquet").write_bytes(b"fake parquet marker")
        return project

    def test_first_load_hit_lru_eviction_and_explicit_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projects = {name: self.make_project(root, name) for name in ("A", "B")}
            calls = []

            def loader(path):
                calls.append(path.name)
                return {"entities": object()}, {"entities": object()}

            manager = GraphIndexManager(max_loaded_projects=1, loader=loader)
            _, first_hit, load_seconds = manager.get("A", projects["A"])
            self.assertEqual(load_seconds, manager._entries[("A", index_signature(projects["A"]))].load_seconds)
            _, second_hit, second_load = manager.get("A", projects["A"])
            manager.get("B", projects["B"])

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(second_load, 0.0)
            self.assertEqual(calls, ["A", "B"])
            self.assertEqual(manager.loaded_projects(), ["B"])
            self.assertEqual(manager.clear("B"), 1)
            self.assertEqual(manager.loaded_projects(), [])

    def test_parallel_first_access_loads_project_once_and_signature_change_reloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = self.make_project(root, "A")
            calls = []
            calls_lock = threading.Lock()

            def loader(path):
                with calls_lock:
                    calls.append(path)
                time.sleep(0.02)
                return {"entities": object()}, {"entities": object()}

            manager = GraphIndexManager(max_loaded_projects=2, loader=loader)
            with ThreadPoolExecutor(max_workers=5) as pool:
                results = list(pool.map(lambda _: manager.get("A", project), range(5)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(sum(not hit for _, hit, _ in results), 1)

            (project / "graphrag" / "last_index_run.json").write_text(
                json.dumps({"status": "INDEXED", "completed_at": "2026-11-20T00:00:00Z"}), encoding="utf-8",
            )
            manager.get("A", project)
            self.assertEqual(len(calls), 2)

    def test_default_loader_uses_pyarrow_memory_map_and_typed_graph_rag_frames(self):
        import pandas as pd
        from graphrag.data_model.schemas import (
            COMMUNITY_CHILDREN, COMMUNITY_ID, COMMUNITY_LEVEL, FINDINGS, N_TOKENS,
            PERIOD, RATING, SIZE,
        )

        source_frames = {
            "entities": pd.DataFrame({"short_id": [1]}),
            "relationships": pd.DataFrame({"short_id": [1]}),
            "communities": pd.DataFrame({
                COMMUNITY_ID: [1], COMMUNITY_LEVEL: [0], COMMUNITY_CHILDREN: [[]], PERIOD: ["p"], SIZE: [1],
            }),
            "community_reports": pd.DataFrame({
                COMMUNITY_ID: [1], COMMUNITY_LEVEL: [0], COMMUNITY_CHILDREN: [[]],
                FINDINGS: [[]], RATING: [1.0], SIZE: [1],
            }),
            "text_units": pd.DataFrame({N_TOKENS: [4]}),
            "covariates": pd.DataFrame({"short_id": [1]}),
        }

        class ArrowTable:
            def __init__(self, name):
                self.name = name

            def to_pandas(self):
                return source_frames[self.name].copy()

        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            output = project / "graphrag" / "output"
            output.mkdir(parents=True)
            for name in source_frames:
                (output / f"{name}.parquet").write_bytes(b"parquet")
            with patch(
                "pyarrow.parquet.read_table", side_effect=lambda source, **kwargs: ArrowTable(Path(source).stem),
            ) as read_table:
                frames, tables = GraphIndexManager._load_index(project)

        self.assertEqual(set(frames), set(source_frames))
        self.assertEqual(set(tables), set(source_frames))
        self.assertTrue(all(call.kwargs == {"memory_map": True} for call in read_table.call_args_list))


if __name__ == "__main__":
    unittest.main()
