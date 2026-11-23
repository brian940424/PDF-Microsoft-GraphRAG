import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from automotive_graphrag.global_prefilter import prefilter_reports, prefilter_signature, select_reports
from automotive_graphrag.graph_query_worker import execute_graph_query


def _reports():
    return pd.DataFrame({
        "id": ["r-brake", "r-audio", "r-seat", "r-deep", "r-novec"],
        "community": [1, 2, 3, 4, 5],
        "level": [0, 1, 2, 3, 2],
        "full_content": ["# Brakes", "# Audio", "# Seats", "# Deep brakes", "# No vector"],
        "summary": ["b", "a", "s", "d", "n"],
    })


VECTORS = {
    "r-brake": np.array([1.0, 0.0]),
    "r-audio": np.array([0.0, 1.0]),
    "r-seat": np.array([0.6, 0.8]),
    "r-deep": np.array([1.0, 0.0]),
}


class SelectReportsTests(unittest.TestCase):
    def test_keeps_most_similar_reports_within_the_level_and_ranks_missing_vectors_last(self):
        selected, stats = select_reports(_reports(), [1.0, 0.0], VECTORS, top_n=2, max_level=2)

        self.assertEqual(list(selected["id"]), ["r-brake", "r-seat"])  # r-deep is above level 2
        self.assertEqual(stats["candidates"], 4)
        self.assertEqual(stats["kept"], 2)
        self.assertEqual(stats["missing_vectors"], 1)
        self.assertAlmostEqual(stats["min_kept_score"], 0.6)

        selected, _ = select_reports(_reports(), [1.0, 0.0], VECTORS, top_n=4, max_level=2)
        self.assertIn("r-novec", set(selected["id"]))

    def test_without_a_level_every_report_is_a_candidate(self):
        selected, stats = select_reports(_reports(), [1.0, 0.0], VECTORS, top_n=2, max_level=None)

        self.assertEqual(set(selected["id"]), {"r-brake", "r-deep"})
        self.assertEqual(stats["candidates"], 5)


class PrefilterReportsTests(unittest.TestCase):
    def _never_embed(self, texts):
        raise AssertionError("no embedding call expected")

    def test_disabled_or_few_reports_or_no_vectors_leave_reports_unchanged_without_embedding(self):
        reports = _reports()
        for top_n, vectors, reason in ((0, VECTORS, "disabled"), (4, VECTORS, "few reports"), (2, {}, "no report vectors")):
            result, report = prefilter_reports(
                reports, "q", "/tmp/none", top_n, 2, embed=self._never_embed, report_vectors=vectors,
            )
            self.assertIs(result, reports)
            self.assertFalse(report["applied"])
            self.assertEqual(report["reason"], reason)

    def test_reports_without_levels_are_not_filtered(self):
        reports = pd.DataFrame({"community": [1, 2]})
        result, report = prefilter_reports(reports, "q", "/tmp/none", 1, 2, embed=self._never_embed)
        self.assertIs(result, reports)
        self.assertFalse(report["applied"])

    def test_applies_with_one_embedding_call_for_the_question(self):
        calls = []

        def embed(texts):
            calls.append(texts)
            return [[0.0, 1.0]]

        result, report = prefilter_reports(_reports(), "音響", "/tmp/none", 1, 2, embed=embed, report_vectors=VECTORS)

        self.assertEqual(calls, [["音響"]])
        self.assertEqual(list(result["id"]), ["r-audio"])
        self.assertTrue(report["applied"])
        self.assertEqual(report["top_n"], 1)

    def test_signature_names_version_and_size(self):
        self.assertEqual(prefilter_signature(), "v1-top200")
        self.assertEqual(prefilter_signature(100), "v1-top100")


class WorkerPrefilterTests(unittest.TestCase):
    def _run(self, payload):
        from graphrag.cli import query as cli_query

        frames = {
            "entities": pd.DataFrame({"id": ["e"]}),
            "communities": pd.DataFrame({"community": [1, 2, 3, 4, 5], "text_unit_ids": [[]] * 5}),
            "community_reports": _reports(),
        }
        seen = {}

        def run(**kwargs):
            values = cli_query._resolve_output_files(config=None, output_list=list(frames))
            seen["ids"] = list(values["community_reports"]["id"])
            return "answer", {"reports": pd.DataFrame([{"id": "1", "title": "Brakes"}])}

        with (
            patch.object(cli_query, "run_global_search", side_effect=run),
            patch("automotive_graphrag.global_prefilter.config_embedder", return_value=lambda texts: [[1.0, 0.0]]),
            patch("automotive_graphrag.global_prefilter.text_unit_vectors", return_value=VECTORS),
        ):
            result = execute_graph_query({"root_dir": "/tmp/project", "question": "brakes", "method": "global", **payload}, frames)
        self.assertEqual(len(frames["community_reports"]), 5)  # the cached frame is not modified
        return seen["ids"], result["context"]

    def test_global_maps_only_the_reports_closest_to_the_question(self):
        ids, context = self._run({"global_report_top_n": 2})

        self.assertEqual(ids, ["r-brake", "r-seat"])
        self.assertTrue(context["report_prefilter"]["applied"])
        self.assertEqual(context["report_prefilter"]["candidates"], 4)

    def test_default_top_n_skips_small_indexes_and_dynamic_selection_skips_the_filter(self):
        ids, context = self._run({})
        self.assertEqual(len(ids), 5)
        self.assertEqual(context["report_prefilter"]["reason"], "few reports")

        ids, context = self._run({"global_report_top_n": 2, "dynamic_community_selection": True})
        self.assertEqual(len(ids), 5)
        self.assertNotIn("report_prefilter", context)

    def test_a_failing_prefilter_still_answers_with_every_report(self):
        from graphrag.cli import query as cli_query

        frames = {
            "entities": pd.DataFrame({"id": ["e"]}),
            "communities": pd.DataFrame({"community": [1], "text_unit_ids": [[]]}),
            "community_reports": _reports(),
        }

        def run(**kwargs):
            values = cli_query._resolve_output_files(config=None, output_list=list(frames))
            return f"{len(values['community_reports'])} reports", {}

        with (
            patch.object(cli_query, "run_global_search", side_effect=run),
            patch("automotive_graphrag.global_prefilter.text_unit_vectors", side_effect=OSError("broken")),
        ):
            result = execute_graph_query(
                {"root_dir": "/tmp/project", "question": "q", "method": "global", "global_report_top_n": 1}, frames,
            )

        self.assertEqual(result["answer"], "5 reports")
        self.assertEqual(result["context"]["report_prefilter"], {"applied": False, "error": "broken"})


if __name__ == "__main__":
    unittest.main()
