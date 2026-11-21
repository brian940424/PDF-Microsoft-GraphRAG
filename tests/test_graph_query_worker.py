import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.graph_query_worker import execute_graph_query


class GraphQueryWorkerTests(unittest.TestCase):
    def test_preloaded_frames_bypass_graph_rag_disk_table_reader_for_every_strategy(self):
        from graphrag.cli import query as cli_query
        import pandas as pd

        required = {
            "local": ["communities", "community_reports", "text_units", "relationships", "entities"],
            "global": ["entities", "communities", "community_reports"],
            "drift": ["communities", "community_reports", "text_units", "relationships", "entities"],
            "basic": ["text_units"],
        }
        frames = {name: pd.DataFrame({"id": [name]}) for names in required.values() for name in names}
        resolved = []

        def search(name):
            def run(**kwargs):
                values = cli_query._resolve_output_files(config=None, output_list=required[name])
                resolved.append((name, values))
                return "answer", {"sources": []}
            return run

        original_resolver = cli_query._resolve_output_files
        with (
            patch.object(cli_query, "run_local_search", side_effect=search("local")),
            patch.object(cli_query, "run_global_search", side_effect=search("global")),
            patch.object(cli_query, "run_basic_search", side_effect=search("basic")),
            patch("automotive_graphrag.drift_compat.run_drift_search", side_effect=lambda **kwargs: search("drift")()),
        ):
            for method in ("local", "global", "drift", "basic"):
                result = execute_graph_query({
                    "root_dir": "/tmp/project", "question": "question", "method": method,
                }, frames)
                self.assertEqual(result["answer"], "answer")

        self.assertEqual(cli_query._resolve_output_files, original_resolver)
        self.assertEqual({name for name, _ in resolved}, set(required))
        for name, values in resolved:
            self.assertEqual(set(values), set(required[name]))
            self.assertTrue(all(values[item] is frames[item] for item in required[name]))

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
                    "response_type": "List of 3-7 Points",
                })
                self.assertEqual(result["answer"], "answer")
                self.assertEqual(result["context"], {"sources": []})

        self.assertEqual(set(calls), {"local", "global", "drift", "basic"})
        self.assertEqual(calls["local"]["community_level"], 2)
        self.assertFalse(calls["global"]["dynamic_community_selection"])
        self.assertEqual(calls["drift"]["community_level"], 2)
        self.assertEqual(calls["basic"]["root_dir"], Path("/tmp/project"))
        self.assertTrue(all(call["response_type"] == "List of 3-7 Points" for call in calls.values()))

    def test_basic_embeds_retrieval_query_but_keeps_question_for_the_answer(self):
        from graphrag.cli import query as cli_query
        from graphrag.query.structured_search.basic_search.basic_context import BasicSearchContext

        seen = []

        def build_context(self, query, *args, **kwargs):
            seen.append(query)
            return "context"

        def run_basic(**kwargs):
            BasicSearchContext.build_context(object(), kwargs["query"])
            return "answer", {"sources": []}

        with (
            patch.object(BasicSearchContext, "build_context", build_context),
            patch.object(cli_query, "run_basic_search", side_effect=run_basic) as basic,
        ):
            for _ in range(2):
                execute_graph_query({
                    "root_dir": "/tmp/project",
                    "question": "煞車排氣順序為何？",
                    "method": "basic",
                    "retrieval_query": "Brake bleeding order",
                })

        self.assertEqual(basic.call_args.kwargs["query"], "煞車排氣順序為何？")
        self.assertEqual(seen, ["Brake bleeding order", "Brake bleeding order"])

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
                "response_type": "Single Sentence",
            })

        self.assertEqual(result["answer"], "luna answer")
        drift.assert_called_once_with(
            root_dir=Path("/tmp/project"),
            query="問題",
            community_level=2,
            response_type="Single Sentence",
        )


if __name__ == "__main__":
    unittest.main()
