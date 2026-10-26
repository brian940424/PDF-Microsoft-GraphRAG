from __future__ import annotations

import unittest

import pandas as pd

from automotive_graphrag.agentic_search import AgenticSearch
from automotive_graphrag.querying import QueryExecution
from automotive_graphrag.querying import QueryService


class AgenticSearchTests(unittest.TestCase):
    def test_evidence_gap_triggers_one_bounded_follow_up_and_grounded_answer(self) -> None:
        model_responses = iter([
            {"method": "local", "query": "雨刷故障檢查", "purpose": "取得檢查程序"},
            {"sufficient": False, "gap": "缺少保險絲規格", "method": "basic", "query": "雨刷保險絲規格"},
            {"answer": "先檢查保險絲規格 [E1]，再執行馬達檢查 [E2]。"},
        ])
        model_prompts: list[str] = []
        searches: list[tuple[str, str]] = []

        def chat(_base: str, _key: str, _model: str, prompt: str) -> object:
            model_prompts.append(prompt)
            return next(model_responses)

        def search(method: str, query: str) -> QueryExecution:
            searches.append((method, query))
            return QueryExecution(0, f"結果 {len(searches)}", "", {"sources": [{"id": str(len(searches)), "text": f"來源 {len(searches)}"}]})

        decision = AgenticSearch(chat).run("雨刷不會動如何檢修？", "model", "https://api", "key", search)

        self.assertEqual(searches, [("local", "雨刷故障檢查"), ("basic", "雨刷保險絲規格")])
        self.assertEqual(decision.answer, "先檢查保險絲規格 [E1]，再執行馬達檢查 [E2]。")
        self.assertEqual(len(decision.steps), 2)
        self.assertEqual(decision.returncode, 0)
        self.assertEqual(
            [row["citation_id"] for step in decision.context["agentic_steps"] for row in step["evidence"]],
            ["E1", "E2"],
        )
        self.assertIn("Retrieved results", model_prompts[1])
        self.assertTrue(model_prompts[2].startswith("Answer the user's question"))

    def test_sufficient_evidence_stops_without_follow_up(self) -> None:
        model_responses = iter([
            {"method": "basic", "query": "雨刷保險絲規格", "purpose": "找到規格"},
            {"sufficient": True, "gap": "", "method": "", "query": ""},
            {"answer": "規格為 15 A [E1]。"},
        ])
        searches: list[str] = []
        agent = AgenticSearch(lambda *_: next(model_responses))

        result = agent.run(
            "雨刷保險絲規格？", "model", "https://api", "key",
            lambda method, _query: searches.append(method) or QueryExecution(0, "規格 15 A", "", {"sources": [{"id": "1", "text": "15 A"}]}),
        )

        self.assertEqual(searches, ["basic"])
        self.assertEqual(result.method, "basic")
        self.assertEqual(result.answer, "規格為 15 A [E1]。")

    def test_failed_optional_follow_up_keeps_successful_first_search(self) -> None:
        model_responses = iter([
            {"method": "local", "query": "主查詢", "purpose": "取得依據"},
            {"sufficient": False, "gap": "需找數值", "method": "basic", "query": "數值"},
            {"answer": "依第一輪來源可確認的部分 [E1]。"},
        ])
        calls = 0

        def search(method: str, query: str) -> QueryExecution:
            nonlocal calls
            calls += 1
            if calls == 1:
                return QueryExecution(0, "初次回答", "", {"sources": [{"id": "tu1", "text": "可用來源"}]})
            return QueryExecution(2, "", "補搜錯誤", {})

        result = AgenticSearch(lambda *_: next(model_responses)).run("q", "model", "https://api", "key", search)

        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.model_call_count, 3)
        self.assertIsNone(result.estimated_cost_usd)
        self.assertEqual(result.execution_metadata["prompt_version"], "agentic-v1")
        self.assertEqual(result.execution_metadata["search_step_count"], 2)
        self.assertGreaterEqual(result.context["agentic_steps"][0]["duration_seconds"], 0)
        self.assertEqual(result.context["agentic_steps"][0]["evidence"][0]["source_type"], "text_unit")
        self.assertIsNone(result.error)
        self.assertIn("補充搜尋失敗", result.stop_reason)
        self.assertEqual(result.answer, "依第一輪來源可確認的部分 [E1]。")

    def test_search_receives_remaining_total_time_budget(self) -> None:
        model_responses = iter([
            {"method": "local", "query": "q", "purpose": "p"},
            {"sufficient": True, "gap": "", "method": "", "query": ""},
            {"answer": "A [E1]"},
        ])
        budgets: list[float] = []

        def search(_method: str, _query: str, timeout_seconds: float) -> QueryExecution:
            budgets.append(timeout_seconds)
            return QueryExecution(0, "A", "", {"sources": [{"id": "tu1", "text": "evidence"}]})

        result = AgenticSearch(lambda *_: next(model_responses)).run("q", "model", "https://api", "key", search)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(budgets), 1)
        self.assertGreater(budgets[0], 0)
        self.assertLessEqual(budgets[0], 300)

    def test_evidence_extractor_reads_graph_rag_32_serialized_context_records(self) -> None:
        graph_context = QueryService._serialize_context({
            "sources": pd.DataFrame([
                {"id": "text-unit-1", "text": "檢查保險絲", "in_context": True},
                {"id": "candidate-only", "text": "未進入查詢脈絡", "in_context": False},
            ]),
            "reports": pd.DataFrame([
                {"id": "report-2", "title": "電氣系統", "summary": "系統級摘要"},
            ]),
        })

        evidence = AgenticSearch._extract_evidence(graph_context)

        self.assertEqual([item.source_id for item in evidence], ["text-unit-1", "report-2"])
        self.assertEqual([item.source_type for item in evidence], ["text_unit", "community_report"])
        self.assertEqual([item.text for item in evidence], ["檢查保險絲", "系統級摘要"])

    def test_invalid_planner_method_falls_back_to_local(self) -> None:
        calls = 0

        def search(method: str, query: str) -> QueryExecution:
            nonlocal calls
            calls += 1
            self.assertEqual((method, query), ("local", "原始問題"))
            return QueryExecution(0, "Local 回答", "", {})

        result = AgenticSearch(lambda *_: {"method": "shell", "query": "unsafe"}).run(
            "原始問題", "model", "https://api", "key", search,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(result.method, "local")
        self.assertEqual(result.answer, "Local 回答")
        self.assertIn("回退 Local", result.stop_reason)

    def test_missing_api_key_skips_planner_and_uses_local(self) -> None:
        result = AgenticSearch(lambda *_: self.fail("planner should not be called")).run(
            "原始問題", "model", "https://api", None,
            lambda method, query: QueryExecution(0, f"{method}:{query}", "", {}),
        )
        self.assertEqual(result.answer, "local:原始問題")
        self.assertEqual(result.method, "local")

    def test_model_error_after_search_uses_graph_rag_answer(self) -> None:
        responses = iter([{"method": "local", "query": "q", "purpose": "p"}])
        result = AgenticSearch(lambda *_: next(responses)).run(
            "q", "model", "https://api", "key",
            lambda *_: QueryExecution(0, "GraphRAG answer", "", {}),
        )
        self.assertEqual(result.answer, "GraphRAG answer")
        self.assertIn("改用 GraphRAG 原始回答", result.stop_reason)


if __name__ == "__main__":
    unittest.main()
