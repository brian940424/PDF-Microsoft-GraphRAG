import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.projects import ProjectError
from automotive_graphrag.strategy_routing import (
    ANSWER_CHECK_PROMPT,
    AnswerCheck,
    LOCAL_MIN_SOURCES,
    NOT_FOUND_ANSWER,
    SearchOutcome,
    StrategyDecision,
    StrategyRouter,
    context_source_count,
    search_with_fallback,
)


class StrategyRouterTests(unittest.TestCase):
    def test_route_returns_model_selected_method_and_reason(self) -> None:
        calls: list[tuple[str, str, str, str]] = []

        def chat(api_base_url: str, api_key: str, model: str, prompt: str) -> object:
            calls.append((api_base_url, api_key, model, prompt))
            return {"method": "Global", "reason": "問題要求整體主題歸納"}

        decision = StrategyRouter(chat).route("這批手冊涵蓋哪些系統？", "gpt-4.1-mini", "https://api", "key")

        self.assertEqual(decision.method, "global")
        self.assertEqual(decision.reason, "問題要求整體主題歸納")
        api_base_url, api_key, model, prompt = calls[0]
        self.assertEqual((api_base_url, api_key, model), ("https://api", "key", "gpt-4.1-mini"))
        self.assertIn("這批手冊涵蓋哪些系統？", prompt)
        self.assertTrue(prompt.startswith("You choose the Microsoft GraphRAG search method"))

    def test_invalid_method_falls_back_to_local(self) -> None:
        decision = StrategyRouter(lambda *_: {"method": "hybrid", "reason": "x"}).route("q", "m", "u", "k")

        self.assertEqual(decision.method, "local")
        self.assertIn("自動判斷失敗", decision.reason)
        self.assertIn("hybrid", decision.reason)

    def test_api_error_falls_back_to_local(self) -> None:
        def chat(*_: object) -> object:
            raise ProjectError("策略判斷 API 回應 HTTP 500")

        decision = StrategyRouter(chat).route("q", "m", "u", "k")

        self.assertEqual(decision.method, "local")
        self.assertIn("HTTP 500", decision.reason)

    def test_missing_api_key_falls_back_without_calling_model(self) -> None:
        def chat(*_: object) -> object:
            raise AssertionError("must not call the model without an API key")

        decision = StrategyRouter(chat).route("q", "m", "u", None)

        self.assertEqual(decision.method, "local")
        self.assertIn("API Key", decision.reason)

    def test_missing_reason_is_marked(self) -> None:
        decision = StrategyRouter(lambda *_: {"method": "basic"}).route("q", "m", "u", "k")

        self.assertEqual(decision.method, "basic")
        self.assertEqual(decision.reason, "模型未提供原因")


class StrategyRankingTests(unittest.TestCase):
    def test_ranking_is_kept_and_prompt_asks_for_it(self) -> None:
        prompts: list[str] = []

        def chat(_base: str, _key: str, _model: str, prompt: str) -> object:
            prompts.append(prompt)
            return {"methods": ["Drift", "global", "basic", "local"], "reason": "多跳"}

        decision = StrategyRouter(chat).route("q", "m", "u", "k")

        self.assertEqual(decision.attempt_order, ("drift", "global", "basic", "local"))
        self.assertIn('"methods"', prompts[0])

    def test_partial_ranking_is_completed_with_default_order_without_duplicates(self) -> None:
        response = {"methods": ["global", "global", "basic"], "reason": "x"}

        decision = StrategyRouter(lambda *_: response).route("q", "m", "u", "k")

        self.assertEqual(decision.attempt_order, ("global", "basic", "local", "drift"))

    def test_unsupported_method_in_ranking_falls_back_to_default_order(self) -> None:
        response = {"methods": ["basic", "hybrid"], "reason": "x"}

        decision = StrategyRouter(lambda *_: response).route("q", "m", "u", "k")

        self.assertIn("hybrid", decision.reason)
        self.assertEqual(decision.attempt_order, ("local", "basic", "global", "drift"))


class AnswerCheckTests(unittest.TestCase):
    def test_model_verdict_is_returned_and_prompt_is_english(self) -> None:
        prompts: list[str] = []

        def chat(_base: str, _key: str, _model: str, prompt: str) -> object:
            prompts.append(prompt)
            return {"found": False, "reason": "回答表示資料不足"}

        check = StrategyRouter(chat).check_answer("問題", "沒有資料", "m", "u", "k")

        self.assertEqual(check, AnswerCheck(False, "回答表示資料不足"))
        self.assertTrue(prompts[0].startswith("You check whether"))
        self.assertIn("問題", prompts[0])
        self.assertIn("沒有資料", prompts[0])

    def test_unusable_check_accepts_the_answer(self) -> None:
        def broken(*_: object) -> object:
            raise ProjectError("HTTP 500")

        for router, key in (
            (StrategyRouter(broken), "k"),
            (StrategyRouter(lambda *_: {"found": "no"}), "k"),
            (StrategyRouter(broken), None),
        ):
            with self.subTest(key=key):
                self.assertTrue(router.check_answer("q", "a", "m", "u", key).found)


class SearchWithFallbackTests(unittest.TestCase):
    def test_first_answer_keeps_router_reason_and_runs_nothing_else(self) -> None:
        tried: list[str] = []

        def search(method: str) -> SearchOutcome:
            tried.append(method)
            return SearchOutcome("答案")

        result = search_with_fallback(
            StrategyDecision("basic", "原因"), search, lambda _: AnswerCheck(True, "ok"),
        )

        self.assertEqual(tried, ["basic"])
        self.assertEqual((result.method, result.reason, result.found), ("basic", "原因", True))

    def test_blank_answer_counts_as_failed_without_checking(self) -> None:
        def check(_: str) -> AnswerCheck:
            raise AssertionError("blank answers must not reach the checker")

        result = search_with_fallback(
            StrategyDecision("local", "原因"), lambda _: SearchOutcome("  "), check,
        )

        self.assertEqual(len(result.attempts), 4)
        self.assertEqual({item["status"] for item in result.attempts}, {"failed"})
        self.assertEqual(result.outcome.error, "GraphRAG 未回傳回答")

    def test_all_not_found_returns_clear_answer_and_first_answered_method(self) -> None:
        def search(method: str) -> SearchOutcome:
            return SearchOutcome("", "壞了") if method == "local" else SearchOutcome("沒有")

        result = search_with_fallback(
            StrategyDecision("local", "原因"), search, lambda _: AnswerCheck(False, "沒有答案"),
        )

        self.assertFalse(result.found)
        self.assertEqual((result.method, result.outcome.answer), ("basic", NOT_FOUND_ANSWER))
        self.assertIsNone(result.outcome.error)

    def test_local_with_too_few_sources_falls_back_without_checking(self) -> None:
        checked: list[str] = []

        def search(method: str) -> SearchOutcome:
            if method == "local":
                return SearchOutcome("來源未說明，只提到相簿群組", source_count=LOCAL_MIN_SOURCES - 1)
            return SearchOutcome("聯絡人群組步驟", source_count=2)

        def check(answer: str) -> AnswerCheck:
            checked.append(answer)
            return AnswerCheck(True, "ok")

        result = search_with_fallback(StrategyDecision("local", "原因"), search, check)

        self.assertEqual((result.method, result.found), ("basic", True))
        self.assertEqual(checked, ["聯絡人群組步驟"])
        self.assertEqual(result.attempts[0]["status"], "not_found")
        self.assertIn(f"只取得 {LOCAL_MIN_SOURCES - 1} 個原文段落", result.attempts[0]["detail"])
        self.assertIn("Local 未找到答案，改用 Basic 找到答案", result.reason)

    def test_local_with_enough_or_unknown_sources_is_checked_normally(self) -> None:
        for count in (LOCAL_MIN_SOURCES, None):
            with self.subTest(count=count):
                tried: list[str] = []

                def search(method: str) -> SearchOutcome:
                    tried.append(method)
                    return SearchOutcome("答案", source_count=count)

                result = search_with_fallback(
                    StrategyDecision("local", "原因"), search, lambda _: AnswerCheck(True, "ok"),
                )
                self.assertEqual((tried, result.found), (["local"], True))

    def test_matching_section_title_accepts_answer_without_overstrict_check(self) -> None:
        checked: list[str] = []
        result = search_with_fallback(
            StrategyDecision("basic", "原因"),
            lambda method: SearchOutcome("回答", payload={}, section_titles=("關閉螢幕時辨識指紋",)),
            lambda answer: checked.append(answer) or AnswerCheck(False, "未找到"),
            question="螢幕關閉時辨識指紋如何設定？",
        )
        self.assertEqual((result.method, result.found), ("basic", True))
        self.assertEqual(checked, [])

    def test_generic_s_pen_title_does_not_stop_fallback_for_specific_questions(self) -> None:
        questions = (
            "使用 S Pen 的懸浮翻譯時，要如何切換逐字或整句翻譯？",
            "螢幕錄影時，如何在畫面上顯示 S Pen 游標？",
            "如何用 S Pen 按鈕遠端拍照？拍照時要如何變更拍攝模式？",
        )
        for question in questions:
            with self.subTest(question=question):
                tried: list[str] = []

                def search(method: str) -> SearchOutcome:
                    tried.append(method)
                    if method == "local":
                        return SearchOutcome(
                            "提供的資料未說明這項操作。", source_count=8,
                            section_titles=("使用 S Pen",),
                        )
                    return SearchOutcome(
                        "找到具體操作步驟。", source_count=3,
                        section_titles=("使用 S Pen 拍照",),
                    )

                result = search_with_fallback(
                    StrategyDecision("local", "specific manual lookup"), search,
                    lambda answer: AnswerCheck("未說明" not in answer, "answer check"),
                    question,
                )

                self.assertEqual(tried[:2], ["local", "basic"])
                self.assertEqual((result.method, result.found), ("basic", True))

    def test_unsupported_drift_does_not_override_rejected_basic_answer(self) -> None:
        tried: list[str] = []
        def search(method: str) -> SearchOutcome:
            tried.append(method)
            if method == "local":
                return SearchOutcome("Local answer")
            if method == "basic":
                return SearchOutcome("Basic 正確答案", payload={"sources": [{"id": "1"}]})
            return SearchOutcome("手冊未提供此設定，請查閱使用手冊。", payload={"sources": [{"id": "2"}]})
        result = search_with_fallback(
            StrategyDecision("local", "原因", ("local", "basic", "drift")), search,
            lambda _: AnswerCheck(False, "答案檢查未通過"), question="如何設定？",
        )
        self.assertEqual(tried, ["local", "basic", "drift", "global"])
        self.assertFalse(result.found)
        self.assertEqual(result.outcome.answer, NOT_FOUND_ANSWER)
        self.assertEqual(
            [item["status"] for item in result.attempts],
            ["not_found", "not_found", "not_found", "not_found"],
        )


class AnswerCheckPromptTests(unittest.TestCase):
    def test_prompt_treats_unstated_or_other_feature_answers_as_not_found(self) -> None:
        self.assertIn("even when the answer also contains other related or unrelated information", ANSWER_CHECK_PROMPT)
        self.assertIn("提供的資料未說明", ANSWER_CHECK_PROMPT)
        self.assertIn("different feature, app, setting or procedure", ANSWER_CHECK_PROMPT)
        self.assertIn("minor detail", ANSWER_CHECK_PROMPT)


class ContextSourceCountTests(unittest.TestCase):
    def test_counts_distinct_in_context_sources_including_nested(self) -> None:
        context = {
            "sources": [{"id": "1"}, {"id": "1"}, {"id": "2", "in_context": False}],
            "drift": [{"sources": [{"id": "3"}]}],
        }
        self.assertEqual(context_source_count(context), 2)
        self.assertEqual(context_source_count({"sources": []}), 0)

    def test_context_without_sources_table_counts_zero(self) -> None:
        self.assertEqual(context_source_count({}), 0)
        self.assertEqual(context_source_count({"entities": [{"id": "1"}]}), 0)

    def test_unknown_without_context(self) -> None:
        for context in (None, "raw"):
            with self.subTest(context=context):
                self.assertIsNone(context_source_count(context))


class QueryTabAutoStrategyTests(unittest.TestCase):
    def test_query_tab_defaults_to_auto_and_automatic_qa_also_offers_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(Path(directory) / "projects")

        method_dropdowns = [
            component.get_config()
            for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "查詢方法"
        ]
        self.assertEqual(len(method_dropdowns), 1)
        self.assertEqual(method_dropdowns[0]["value"], "auto")
        self.assertIn("auto", [value for _label, value in method_dropdowns[0]["choices"]])
        self.assertIn("agentic", [value for _label, value in method_dropdowns[0]["choices"]])
        reasoning_dropdowns = [
            component.get_config()
            for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") in {
                "推理深度", "建圖推理深度", "評測推理深度",
            }
        ]
        self.assertGreaterEqual(len(reasoning_dropdowns), 7)

        handlers = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        _model, method_update = handlers["update_query_method_compatibility"]("gpt-4o-mini", "auto")
        self.assertEqual(method_update["value"], "auto")
        _model, other_update = handlers["update_autoqa_model_method_compatibility"]("gpt-4o-mini", "local")
        self.assertEqual(other_update["value"], "local")
        self.assertIn("agentic", [value for _label, value in other_update["choices"]])


if __name__ == "__main__":
    unittest.main()
