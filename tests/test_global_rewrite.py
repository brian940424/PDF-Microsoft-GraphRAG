import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from automotive_graphrag.global_rewrite import rewrite_global_answer, rewrite_prompt, select_units

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


class RewriteTests(unittest.TestCase):
    def _rewrite(self, answer=DRAFT, grounded=("11",), reply="In BRC, use Before Replace ECU [Data: Sources (11, 12)]."):
        calls = []

        def complete(messages):
            calls.append(messages)
            return reply

        result = rewrite_global_answer(
            "Compare ABS and BCM replacement.", answer, list(grounded), COMMUNITIES, "/tmp/none", "Single Paragraph",
            TEXT_UNITS, embed=lambda texts: [[1.0, 0.0]], complete=complete, token_count=words, unit_vectors=VECTORS,
        )
        return result, calls

    def test_rewrites_from_grounded_and_cited_community_passages(self):
        (answer, sources, report), calls = self._rewrite()

        self.assertEqual(answer, "In BRC, use Before Replace ECU [Data: Sources (11, 12)].")
        self.assertEqual([item["id"] for item in sources], ["11", "12", "13"])  # community 8 is not cited
        self.assertTrue(report["applied"])
        self.assertEqual((report["grounded_units"], report["added_units"]), (1, 2))
        system, user = calls[0]
        self.assertEqual(user, {"role": "user", "content": "Compare ABS and BCM replacement."})
        self.assertIn("12|BRC > Type ID from FAST, last five digits.", system["content"])
        self.assertIn("-----Draft answer-----\n" + DRAFT, system["content"])
        self.assertIn("Single Paragraph", system["content"])

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
