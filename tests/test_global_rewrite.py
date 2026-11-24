import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from automotive_graphrag.global_rewrite import (
    closest_units, named_manuals, rewrite_global_answer, rewrite_prompt, select_units,
)

TEXT_UNITS = pd.DataFrame({
    "id": ["u-a", "u-b", "u-c", "u-d"],
    "human_readable_id": [11, 12, 13, 14],
    "text": [
        "BRC > ABS Before Replace ECU saves the vehicle specification.",
        "BRC > Type ID from FAST, last five digits.",
        "BCS > BCM After Replace ECU writes the configuration.",
        "AV > Audio unit.",
    ],
})
COMMUNITIES = pd.DataFrame({"community": [7, 8], "text_unit_ids": [["u-a", "u-b", "u-c"], ["u-d"]]})
VECTORS = {
    "u-a": np.array([1.0, 0.0]), "u-b": np.array([0.8, 0.6]),
    "u-c": np.array([0.6, 0.8]), "u-d": np.array([0.0, 1.0]),
}
DRAFT = "Both save the configuration first [Data: Reports (7); Sources (11)]."


def words(text):
    return len(text.split())


class SelectUnitsTests(unittest.TestCase):
    def test_grounded_units_first_then_closest_to_the_question_within_the_budget(self):
        text = dict(zip(TEXT_UNITS["id"], TEXT_UNITS["text"]))
        short = dict(zip(TEXT_UNITS["id"], TEXT_UNITS["human_readable_id"].astype(str)))

        units = select_units(["u-c"], ["u-a", "u-b", "u-c", "u-d"], np.array([0.0, 1.0]), VECTORS, words, text, short)
        self.assertEqual(units, ["u-c", "u-d", "u-b", "u-a"])

        units = select_units(["u-c"], ["u-a", "u-b", "u-c", "u-d"], np.array([1.0, 0.0]), VECTORS, words, text, short, max_tokens=20)
        self.assertEqual(units, ["u-c", "u-a"])


class ClosestUnitsTests(unittest.TestCase):
    def test_returns_the_whole_index_units_closest_to_the_question(self):
        self.assertEqual(closest_units(np.array([0.0, 1.0]), VECTORS, 2), ["u-d", "u-c"])
        self.assertEqual(closest_units(np.array([0.0, 1.0]), VECTORS, 0), [])
        self.assertEqual(closest_units(np.array([0.0, 1.0]), {}, 3), [])


class NamedManualsTests(unittest.TestCase):
    def test_matches_whole_manual_codes_only(self):
        manuals = {"BR", "BRC", "BRM", "AV", "BCS"}
        self.assertEqual(named_manuals("Across the BR and BRC manuals, which parts are non-reusable?", manuals), {"BR", "BRC"})
        self.assertEqual(named_manuals("車身修理手冊（BRM）中，後期車型怎麼判定？", manuals), {"BRM"})
        self.assertEqual(named_manuals("Compare ABS and BCM replacement.", manuals), set())


MANUALS = {"u-a": "BRC", "u-b": "BRC", "u-c": "BCS", "u-d": "AV"}


class RewriteTests(unittest.TestCase):
    def _rewrite(
        self, answer=DRAFT, grounded=("11",), reply="In BRC, use Before Replace ECU [Data: Sources (11, 12)].",
        question="Compare ABS and BCM replacement.",
    ):
        calls = []

        def complete(messages):
            calls.append(messages)
            return reply

        result = rewrite_global_answer(
            question, answer, list(grounded), COMMUNITIES, "/tmp/none", "Single Paragraph",
            TEXT_UNITS, embed=lambda texts: [[1.0, 0.0]], complete=complete, token_count=words, unit_vectors=VECTORS,
            manual_by_unit=MANUALS,
        )
        return result, calls

    def test_rewrites_from_grounded_cited_community_and_searched_passages(self):
        with patch("automotive_graphrag.global_rewrite.GLOBAL_SEARCH_UNITS", 0):
            (_, sources, report), _ = self._rewrite()
        self.assertEqual([item["id"] for item in sources], ["11", "12", "13"])  # community 8 is not cited
        self.assertEqual(report["searched_units"], 0)

        (answer, sources, report), calls = self._rewrite()

        self.assertEqual(answer, "In BRC, use Before Replace ECU [Data: Sources (11, 12)].")
        # u-d is in no cited community but comes from the whole-index search
        self.assertEqual([item["id"] for item in sources], ["11", "12", "13", "14"])
        self.assertTrue(report["applied"])
        self.assertEqual((report["grounded_units"], report["added_units"], report["searched_units"]), (1, 3, 1))
        self.assertEqual(report["searched_manuals"], [])

        system, user = calls[0]
        self.assertEqual(user, {"role": "user", "content": "Compare ABS and BCM replacement."})
        self.assertIn("12|BRC > Type ID from FAST, last five digits.", system["content"])
        self.assertIn("-----Draft answer-----\n" + DRAFT, system["content"])
        self.assertIn("Single Paragraph", system["content"])

    def test_question_naming_manuals_searches_only_those_manuals(self):
        (_, sources, report), _ = self._rewrite(question="In the BRC and BCS manuals, compare the replacements.")

        self.assertEqual([item["id"] for item in sources], ["11", "12", "13"])  # AV's u-d is not searched
        self.assertEqual((report["searched_units"], report["searched_manuals"]), (0, ["BCS", "BRC"]))

    def test_keeps_the_answer_without_cited_passages_or_with_an_empty_reply(self):
        (answer, sources, report), calls = self._rewrite(answer="No citation.", grounded=())
        self.assertEqual((answer, sources, report["applied"]), ("No citation.", [], False))
        self.assertEqual(calls, [])

        (answer, sources, report), _ = self._rewrite(reply="  ")
        self.assertEqual((answer, sources, report["reason"]), (DRAFT, [], "empty rewrite"))

    def test_prompt_has_basic_format_answer_rules_and_rewrite_rules(self):
        prompt = rewrite_prompt("List of 3-7 Points", "DATA")

        self.assertIn("---Answer requirements---", prompt)
        self.assertIn("---Rewriting the draft answer---", prompt)
        self.assertLess(prompt.index("---Rewriting the draft answer---"), prompt.rindex("---Target response length and format---"))
        self.assertIn("目前取得的資料不足以回答此部分", prompt)  # not "the provided data does not state it"
        self.assertNotIn("say only that the provided data does not state it", prompt)
        self.assertIn("do not add remarks about what is unconfirmed, incomplete", prompt)
        self.assertIn("Keep only the points the question asks for.", prompt)
        self.assertIn("keep what its passages say inside the steps", prompt)
        self.assertIn("Do not present work that is the same for both as a difference.", prompt)
        self.assertIn("List of 3-7 Points", prompt)
        self.assertIn("DATA", prompt)


class WorkerRewriteTests(unittest.TestCase):
    def _run(self, payload, rewrite):
        from graphrag.cli import query as cli_query

        from automotive_graphrag.graph_query_worker import execute_graph_query

        frames = {
            "entities": pd.DataFrame({"id": ["e"]}),
            "communities": COMMUNITIES,
            "community_reports": pd.DataFrame({"community": [7, 8], "full_content": ["# A", "# B"], "summary": ["a", "b"]}),
            "text_units": TEXT_UNITS,
        }

        def run(**kwargs):
            cli_query._resolve_output_files(config=None, output_list=["entities", "communities", "community_reports"])
            return "Draft [Data: Reports (7)].", {}

        grounded = ("Draft [Data: Reports (7); Sources (11)].", [{"id": "11", "text": "a", "in_context": True}], {"grounded_citations": 1})
        with (
            patch.object(cli_query, "run_global_search", side_effect=run),
            patch("automotive_graphrag.graph_query_worker.ground_global_answer", return_value=grounded),
            patch("automotive_graphrag.graph_query_worker.rewrite_global_answer", side_effect=rewrite) as mocked,
        ):
            result = execute_graph_query({"root_dir": "/tmp/project", "question": "q", "method": "global", **payload}, frames)
        return result, mocked

    def test_rewritten_answer_replaces_the_draft_and_keeps_every_source(self):
        rewritten = ("Final [Data: Sources (12)].", [{"id": "12", "text": "b", "in_context": True}], {"applied": True})
        result, mocked = self._run({}, lambda *args, **kwargs: rewritten)

        self.assertEqual(result["answer"], "Final [Data: Sources (12)].")
        self.assertEqual([item["id"] for item in result["context"]["sources"]], ["12", "11"])
        self.assertEqual(result["context"]["source_rewrite"], {"applied": True})
        self.assertEqual(mocked.call_args.args[:3], ("q", "Draft [Data: Reports (7); Sources (11)].", ["11"]))

    def test_failure_or_switch_off_keeps_the_grounded_answer(self):
        def fail(*args, **kwargs):
            raise RuntimeError("model down")

        result, _ = self._run({}, fail)
        self.assertEqual(result["answer"], "Draft [Data: Reports (7); Sources (11)].")
        self.assertEqual(result["context"]["source_rewrite"], {"applied": False, "error": "model down"})

        result, mocked = self._run({"global_source_rewrite": False}, fail)
        self.assertEqual(result["answer"], "Draft [Data: Reports (7); Sources (11)].")
        self.assertNotIn("source_rewrite", result["context"])
        mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
