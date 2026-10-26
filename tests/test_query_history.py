import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
import json

import gradio as gr

from automotive_graphrag.app import QUERY_HISTORY_LIMIT, create_app
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectStore
from automotive_graphrag.querying import QueryResult


def query_record(query_id: str, question: str, source: str | None, **changes) -> QueryResult:
    record = QueryResult(
        query_id=query_id,
        project_id="L33-SM3E",
        question=question,
        method="local",
        status="COMPLETED",
        answer=f"{question} 的回答",
        error=None,
        started_at="2026-10-25T01:00:00+00:00",
        completed_at="2026-10-25T01:00:05+00:00",
        duration_seconds=5.0,
        source=source,
    )
    return replace(record, **changes)


class QueryHistoryUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "projects"
        store = ProjectStore(self.root)
        store.create(project_id="L33-SM3E", display_name="L33", vehicle_name="L33", manual_version="SM3E")
        self.history_path = store.path_for("L33-SM3E") / "runs" / "queries.jsonl"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        app = create_app(self.root)
        self.functions = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_history(self, records: list[QueryResult]) -> None:
        self.history_path.write_text(
            "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
        )

    def test_history_lists_manual_and_legacy_records_newest_first(self) -> None:
        self.write_history(
            [
                query_record("q1", "舊紀錄", None),
                query_record("q2", "自動問答", "automatic_qa"),
                query_record("q3", "手動問題", "manual", method="global", requested_method="auto"),
                query_record("q4", "前台", "portal"),
            ]
        )

        rows, ids = self.functions["load_query_history"]("L33-SM3E")

        self.assertEqual(ids, ["q3", "q1"])
        self.assertEqual(rows[0][1:], ["手動問題", "Global（自動）", "COMPLETED", 5.0])
        self.assertEqual(self.functions["load_query_history"](None), ([], []))

    def test_history_is_limited_and_truncates_long_questions(self) -> None:
        records = [query_record(f"q{index}", f"問題{index}", "manual") for index in range(QUERY_HISTORY_LIMIT + 5)]
        records.append(query_record("long", "長" * 80, "manual"))
        self.write_history(records)

        rows, ids = self.functions["load_query_history"]("L33-SM3E")

        self.assertEqual(len(ids), QUERY_HISTORY_LIMIT)
        self.assertEqual(ids[0], "long")
        self.assertEqual(rows[0][1], "長" * 60 + "…")

    def test_selecting_row_restores_question_answer_and_evidence(self) -> None:
        evidence = Evidence(
            evidence_id="E1", rank=1, context_id="7", text_unit_id="tu", chunk_id="c1", section_id="WW",
            section_name="Wiper", document_id="WW.pdf", page=25, block_id="b01", text="來源全文",
        )
        self.write_history(
            [query_record("q1", "雨刷不動？", "manual", requested_method="auto", response_type="List of 3-7 Points",
                          evidence=(evidence,))]
        )
        _, ids = self.functions["load_query_history"]("L33-SM3E")
        event = gr.SelectData(None, {"index": [0, 1], "value": "雨刷不動？"})

        outputs = self.functions["select_query_history"]("L33-SM3E", ids, event)

        question, method, response_type, answer, summary, rows, selector, detail, _, values, group = outputs
        self.assertEqual(question, "雨刷不動？")
        self.assertEqual(method["value"], "auto")
        self.assertEqual(response_type["value"], "List of 3-7 Points")
        self.assertEqual(answer, "雨刷不動？ 的回答")
        self.assertIn("✅", summary)
        self.assertEqual(rows[0][1], "E1")
        self.assertEqual(selector.value, "E1")
        self.assertIn("來源全文", detail)
        self.assertEqual(values[0]["evidence_id"], "E1")
        self.assertTrue(group.visible)

    def test_selecting_missing_row_raises(self) -> None:
        event = gr.SelectData(None, {"index": [0, 0], "value": ""})
        with self.assertRaises(gr.Error):
            self.functions["select_query_history"]("L33-SM3E", [], event)


if __name__ == "__main__":
    unittest.main()
