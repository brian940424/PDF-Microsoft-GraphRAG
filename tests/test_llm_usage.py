import asyncio
import unittest

import litellm

from automotive_graphrag.llm_usage import (
    aggregate_cost,
    call_record,
    install_litellm_recorder,
    query_cost_summary,
    usage_metadata,
)
from automotive_graphrag.strategy_routing import StrategyRouter


class UsageMetadataTests(unittest.TestCase):
    def test_metadata_totals_measured_calls_and_reports_status(self) -> None:
        calls = [
            call_record("routing", None, "m", {"prompt_tokens": 100, "completion_tokens": 20}, 0.5),
            call_record("search", "local", "m", {"prompt_tokens": 3000, "completion_tokens": 200}, 4.0),
        ]
        metadata = usage_metadata(calls)
        self.assertEqual(metadata["model_call_count"], 2)
        self.assertEqual(metadata["prompt_tokens"], 3100)
        self.assertEqual(metadata["completion_tokens"], 220)
        self.assertEqual(metadata["cost_status"], "measured")

        calls.append(call_record("answer_check", "local", "m", None, 0.3))
        self.assertEqual(usage_metadata(calls)["cost_status"], "partially_measured")
        self.assertIsNone(usage_metadata([])["model_call_count"])

    def test_cost_summary_and_aggregate_tolerate_records_without_llm_calls(self) -> None:
        new = {
            "requested_method": "auto",
            "strategy_attempts": [
                {"method": "local", "status": "not_found"},
                {"method": "basic", "status": "found"},
            ],
            **usage_metadata([
                call_record("routing", None, "m", {"prompt_tokens": 10, "completion_tokens": 5}, 0.1),
            ]),
        }
        old = {"requested_method": "auto", "strategy_attempts": [{"method": "basic", "status": "found"}]}
        new_summary = query_cost_summary(new)
        self.assertEqual(new_summary["search_count"], 2)
        self.assertEqual(new_summary["total_tokens"], 15)
        self.assertFalse(new_summary["first_choice_found"])
        old_summary = query_cost_summary(old)
        self.assertIsNone(old_summary["llm_call_count"])
        self.assertTrue(old_summary["first_choice_found"])
        self.assertEqual(query_cost_summary({"requested_method": "basic"})["search_count"], 1)

        aggregate = aggregate_cost([new_summary, old_summary])
        self.assertEqual(aggregate["average_search_count"], 1.5)
        self.assertEqual(aggregate["average_total_tokens"], 15)
        self.assertEqual(aggregate["measured_query_count"], 1)
        self.assertEqual(aggregate["first_choice_hit_rate"], 0.5)
        self.assertEqual(
            aggregate["first_choice_by_method"], {"local": {"first": 1, "found": 0}, "basic": {"first": 1, "found": 1}},
        )


class QueryLogReportTests(unittest.TestCase):
    def test_report_filters_by_source_and_method(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from automotive_graphrag.query_cost_report import summarize_query_log

        records = [
            {"source": "automatic_qa", "requested_method": "auto", "started_at": "2026-11-14T01:00:00",
             "execution_metadata": {"strategy_attempts": [{"method": "basic", "status": "found"}]}},
            {"source": "automatic_qa", "requested_method": "auto", "started_at": "2026-11-13T01:00:00",
             "execution_metadata": {"strategy_attempts": [{"method": "local", "status": "not_found"},
                                                          {"method": "basic", "status": "found"}]}},
            {"source": None, "requested_method": "basic", "started_at": "2026-11-14T02:00:00",
             "execution_metadata": {"requested_method": "basic"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.jsonl"
            path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            report = summarize_query_log(str(path), source="automatic_qa")
            self.assertEqual(report["query_count"], 2)
            self.assertEqual(report["average_search_count"], 1.5)
            self.assertEqual(report["first_choice_hit_rate"], 0.5)
            self.assertEqual(summarize_query_log(str(path), since="2026-11-14")["query_count"], 1)
            self.assertEqual(summarize_query_log(str(path), method=None)["query_count"], 3)


class RouterUsageTests(unittest.TestCase):
    def test_route_and_check_record_usage_from_chat(self) -> None:
        def chat(_base: str, _key: str, _model: str, prompt: str, usage_sink: dict | None = None) -> object:
            usage_sink.update({"prompt_tokens": 50, "completion_tokens": 7})
            if "Rank all four methods" in prompt:
                return {"methods": ["basic", "local", "global", "drift"], "reason": "原因"}
            return {"found": True, "reason": "ok"}

        router = StrategyRouter(chat)
        decision = router.route("Q", "m", "https://example", "key")
        self.assertEqual(decision.llm_calls[0]["stage"], "routing")
        self.assertEqual(decision.llm_calls[0]["prompt_tokens"], 50)
        check = router.check_answer("Q", "A", "m", "https://example", "key")
        self.assertEqual(check.llm_calls[0]["stage"], "answer_check")
        self.assertEqual(check.llm_calls[0]["completion_tokens"], 7)

    def test_failed_chat_is_recorded_without_tokens(self) -> None:
        from automotive_graphrag.projects import ProjectError

        def chat(*_: object) -> object:
            raise ProjectError("HTTP 500")

        decision = StrategyRouter(chat).route("Q", "m", "https://example", "key")
        self.assertEqual(decision.method, "basic")
        self.assertEqual(len(decision.llm_calls), 1)
        self.assertEqual(decision.llm_calls[0]["error"], "HTTP 500")
        self.assertIsNone(decision.llm_calls[0]["prompt_tokens"])


class LiteLLMRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = (litellm.acompletion, litellm.completion)

    def tearDown(self) -> None:
        litellm.acompletion, litellm.completion = self.original

    def test_records_plain_and_streaming_completions(self) -> None:
        calls: list[dict] = []
        install_litellm_recorder(calls)
        messages = [{"role": "user", "content": "hello"}]

        async def run() -> str:
            await litellm.acompletion(model="gpt-4o-mini", messages=messages, mock_response="plain answer")
            stream = await litellm.acompletion(
                model="gpt-4o-mini", messages=messages, mock_response="streamed answer", stream=True,
            )
            text = ""
            async for chunk in stream:
                text += chunk.choices[0].delta.content or ""
            return text

        self.assertEqual(asyncio.run(run()), "streamed answer")
        litellm.completion(model="gpt-4o-mini", messages=messages, mock_response="sync answer")
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["stage"] == "search" for call in calls))
        self.assertIsNotNone(calls[0]["prompt_tokens"])
        self.assertIsNotNone(calls[1]["completion_tokens"])
        self.assertIsNotNone(calls[2]["completion_tokens"])

    def test_failed_completion_is_recorded_and_raised(self) -> None:
        calls: list[dict] = []
        install_litellm_recorder(calls)
        with self.assertRaises(Exception):
            litellm.completion(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}],
                mock_response=Exception("boom"),
            )
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0]["error"])


if __name__ == "__main__":
    unittest.main()
