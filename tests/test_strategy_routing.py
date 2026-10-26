import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.projects import ProjectError
from automotive_graphrag.strategy_routing import StrategyRouter


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
        reasoning_dropdowns = [
            component.get_config()
            for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "推理深度"
        ]
        self.assertGreaterEqual(len(reasoning_dropdowns), 7)

        handlers = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        _model, method_update = handlers["update_query_method_compatibility"]("gpt-4o-mini", "auto")
        self.assertEqual(method_update["value"], "auto")
        _model, other_update = handlers["update_model_method_compatibility"]("gpt-4o-mini", "local")
        self.assertEqual(other_update["value"], "local")
        self.assertIn("auto", [value for _label, value in other_update["choices"]])


if __name__ == "__main__":
    unittest.main()
