import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.graph_query_worker import execute_graph_query


class GraphQueryWorkerTests(unittest.TestCase):
    def test_routes_strategies_to_graph_rag_cli_functions(self):
        from graphrag.cli import query as cli_query

        calls = {}

        def capture(name):
            def run(**kwargs):
                calls[name] = kwargs
                return "answer", {"sources": []}

            return run

        with (
            patch.object(cli_query, "run_local_search", side_effect=capture("local")),
            patch.object(cli_query, "run_global_search", side_effect=capture("global")),
            patch.object(cli_query, "run_drift_search", side_effect=capture("drift")),
            patch.object(cli_query, "run_basic_search", side_effect=capture("basic")),
        ):
            for method in ("local", "global", "drift", "basic"):
                result = execute_graph_query({
                    "root_dir": "/tmp/project",
                    "question": "問題",
                    "method": method,
                    "model": "gpt-4.1-mini",
                })
                self.assertEqual(result["answer"], "answer")
                self.assertEqual(result["context"], {"sources": []})

        self.assertEqual(set(calls), {"local", "global", "drift", "basic"})
        self.assertEqual(calls["local"]["community_level"], 2)
        self.assertFalse(calls["global"]["dynamic_community_selection"])
        self.assertEqual(calls["drift"]["community_level"], 2)
        self.assertEqual(calls["basic"]["root_dir"], Path("/tmp/project"))

    def test_luna_drift_uses_compatibility_adapter(self):
        with patch(
            "automotive_graphrag.drift_compat.run_drift_search",
            return_value=("luna answer", {"sources": []}),
        ) as drift:
            result = execute_graph_query({
                "root_dir": "/tmp/project",
                "question": "問題",
                "method": "drift",
                "model": "gpt-6-luna",
            })

        self.assertEqual(result["answer"], "luna answer")
        drift.assert_called_once_with(
            root_dir=Path("/tmp/project"),
            query="問題",
            community_level=2,
            response_type="Multiple Paragraphs",
        )


if __name__ == "__main__":
    unittest.main()
