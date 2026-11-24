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
    PARTIAL_ANSWER_NOTE,
    SearchOutcome,
    StrategyDecision,
    StrategyRouter,
    context_source_count,
    evidence_page_records,
    question_identifiers,
    search_with_fallback,
)
from automotive_graphrag.answer_prompts import INSUFFICIENT_DATA_PHRASE


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

    def test_invalid_method_falls_back_to_basic(self) -> None:
        decision = StrategyRouter(lambda *_: {"method": "hybrid", "reason": "x"}).route("q", "m", "u", "k")

        self.assertEqual(decision.method, "basic")
        self.assertIn("自動判斷失敗", decision.reason)
        self.assertIn("hybrid", decision.reason)

    def test_api_error_falls_back_to_basic(self) -> None:
        def chat(*_: object) -> object:
            raise ProjectError("策略判斷 API 回應 HTTP 500")

        decision = StrategyRouter(chat).route("q", "m", "u", "k")

        self.assertEqual(decision.method, "basic")
        self.assertIn("HTTP 500", decision.reason)

    def test_missing_api_key_falls_back_without_calling_model(self) -> None:
        def chat(*_: object) -> object:
            raise AssertionError("must not call the model without an API key")

        decision = StrategyRouter(chat).route("q", "m", "u", None)

        self.assertEqual(decision.method, "basic")
        self.assertIn("API Key", decision.reason)

    def test_comparison_question_tries_basic_first_and_keeps_ranking_as_fallback(self) -> None:
        response = {"methods": ["drift", "local", "basic", "global"], "reason": "需結合不同章節"}

        for question in (
            "Compare the additional service required when replacing the BCM with the AV control unit.",
            "QR25DE 與 VQ35DE 在加速踏板處理上有什麼不同？",
        ):
            with self.subTest(question=question):
                decision = StrategyRouter(lambda *_: response).route(question, "m", "u", "k")

                self.assertEqual(decision.method, "basic")
                self.assertEqual(decision.attempt_order, ("basic", "drift", "local", "global"))
                self.assertIn("需結合不同章節", decision.reason)
                self.assertIn("比較題固定先用 Basic", decision.reason)
                self.assertIn("DRIFT", decision.reason)

    def test_non_comparison_question_keeps_model_ranking(self) -> None:
        response = {"methods": ["drift", "local", "basic", "global"], "reason": "多步推理"}

        decision = StrategyRouter(lambda *_: response).route("How is the BCM replaced?", "m", "u", "k")

        self.assertEqual(decision.method, "drift")
        self.assertEqual(decision.reason, "多步推理")

    def test_comparison_question_already_basic_first_is_unchanged(self) -> None:
        response = {"methods": ["basic", "drift", "local", "global"], "reason": "查原文"}

        decision = StrategyRouter(lambda *_: response).route("What is the difference between A and B?", "m", "u", "k")

        self.assertEqual(decision.reason, "查原文")
        self.assertEqual(decision.attempt_order, ("basic", "drift", "local", "global"))

    def test_missing_reason_is_marked(self) -> None:
        decision = StrategyRouter(lambda *_: {"method": "basic"}).route("q", "m", "u", "k")

        self.assertEqual(decision.method, "basic")
        self.assertEqual(decision.reason, "模型未提供原因")

    def test_rewrite_query_returns_manual_language_query_and_records_call(self):
        prompts = []

        def chat(api_base_url, api_key, model, prompt, reasoning_effort=None):
            prompts.append(prompt)
            return {"query": "  Brake bleeding procedure:\n first caliper  "}

        query, calls = StrategyRouter(chat).rewrite_query(
            "煞車系統排氣時先接哪個卡鉗？", "Connect a vinyl tube", "m", "u", "k",
        )

        self.assertEqual(query, "Brake bleeding procedure: first caliper")
        self.assertIn("煞車系統排氣時先接哪個卡鉗？", prompts[0])
        self.assertIn("Connect a vinyl tube", prompts[0])
        self.assertEqual([call["stage"] for call in calls], ["query_rewrite"])

    def test_rewrite_query_returns_none_without_key_or_usable_reply(self):
        self.assertEqual(StrategyRouter(lambda *args: {"query": "x"}).rewrite_query("q", "e", "m", "u", None), (None, ()))
        query, _ = StrategyRouter(lambda *args: {"text": "x"}).rewrite_query("q", "e", "m", "u", "k")
        self.assertIsNone(query)


class StrategyRankingTests(unittest.TestCase):
    def test_ranking_is_kept_and_prompt_asks_for_it(self) -> None:
        prompts: list[str] = []

        def chat(_base: str, _key: str, _model: str, prompt: str) -> object:
            prompts.append(prompt)
            return {"methods": ["Drift", "global", "basic", "local"], "reason": "多跳"}

        decision = StrategyRouter(chat).route("q", "m", "u", "k")

        self.assertEqual(decision.attempt_order, ("drift", "global", "basic", "local"))
        self.assertIn('"methods"', prompts[0])
        self.assertIn("Do not rank drift first for a single setting", prompts[0])
        self.assertIn("Reserve drift for questions that must combine information from distinct sections", prompts[0])

    def test_partial_ranking_is_completed_with_default_order_without_duplicates(self) -> None:
        response = {"methods": ["global", "global", "basic"], "reason": "x"}

        decision = StrategyRouter(lambda *_: response).route("q", "m", "u", "k")

        self.assertEqual(decision.attempt_order, ("global", "basic", "local", "drift"))

    def test_unsupported_method_in_ranking_falls_back_to_default_order(self) -> None:
        response = {"methods": ["basic", "hybrid"], "reason": "x"}

        decision = StrategyRouter(lambda *_: response).route("q", "m", "u", "k")

        self.assertIn("hybrid", decision.reason)
        self.assertEqual(decision.attempt_order, ("basic", "local", "global", "drift"))


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

    def test_partial_verdict_is_parsed_only_when_not_found(self) -> None:
        responses = {
            "partial": {"found": False, "partial": True, "answered_parts": 1, "missing": "失效保護與解除條件", "reason": "只答出偵測條件"},
            "found_wins": {"found": True, "partial": True, "answered_parts": 3, "missing": "x", "reason": "ok"},
            "bad_parts": {"found": False, "partial": True, "answered_parts": True, "reason": "r"},
            "not_partial": {"found": False, "partial": False, "answered_parts": 2, "missing": "x", "reason": "r"},
        }
        expected = {
            "partial": (False, True, 1, "失效保護與解除條件"),
            "found_wins": (True, False, 0, ""),
            "bad_parts": (False, True, 1, ""),
            "not_partial": (False, False, 0, ""),
        }
        for name, response in responses.items():
            with self.subTest(name=name):
                check = StrategyRouter(lambda *_, response=response: response).check_answer("q", "a", "m", "u", "k")
                self.assertEqual((check.found, check.partial, check.answered_parts, check.missing), expected[name])

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
            return SearchOutcome("答案 [Data: Sources (1)]")

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

    def test_attempts_record_answer_and_evidence_pages(self) -> None:
        pages = ({"document": "BCS.pdf", "page": 70},)

        def search(method: str) -> SearchOutcome:
            return SearchOutcome(f"{method} 答案 [Data: Sources (1)]", evidence_pages=pages)

        result = search_with_fallback(
            StrategyDecision("basic", "原因"), search, lambda _: AnswerCheck(False, "沒有"),
        )

        self.assertEqual(result.outcome.answer, NOT_FOUND_ANSWER)
        self.assertEqual(
            [(item["answer"], item["evidence"]) for item in result.attempts],
            [(f"{method} 答案 [Data: Sources (1)]", [{"document": "BCS.pdf", "page": 70}])
             for method in ("basic", "local", "global", "drift")],
        )

    def test_best_partial_answer_replaces_not_found_reply(self) -> None:
        verdicts = {
            "basic": AnswerCheck(False, "只答一項", partial=True, answered_parts=1, missing="失效保護、解除條件"),
            "local": AnswerCheck(False, "答兩項", partial=True, answered_parts=2, missing="解除條件"),
            "global": AnswerCheck(False, "也答兩項", partial=True, answered_parts=2, missing="x"),
            "drift": AnswerCheck(False, "沒有"),
        }
        tried: list[str] = []

        def search(method: str) -> SearchOutcome:
            tried.append(method)
            return SearchOutcome(f"{method} 偵測條件 [Data: Sources (1)]")

        result = search_with_fallback(StrategyDecision("basic", "原因"), search, lambda _: verdicts[tried[-1]])

        self.assertEqual((result.found, result.partial, result.method), (False, True, "local"))
        self.assertEqual(
            result.outcome.answer,
            "local 偵測條件 [Data: Sources (1)]\n\n" + PARTIAL_ANSWER_NOTE.format(missing="解除條件"),
        )
        self.assertEqual(
            [item["status"] for item in result.attempts], ["partial", "partial", "partial", "not_found"],
        )
        self.assertEqual(result.attempts[1]["answered_parts"], 2)
        self.assertEqual(result.attempts[1]["missing"], "解除條件")
        self.assertIn("採用 Local 的部分回答", result.reason)
        self.assertIn("Basic 只找到部分答案", result.reason)

    def test_found_answer_still_wins_over_earlier_partial(self) -> None:
        tried: list[str] = []

        def check(_: str) -> AnswerCheck:
            if tried[-1] == "basic":
                return AnswerCheck(False, "部分", partial=True, answered_parts=1)
            return AnswerCheck(True, "ok")

        def search(method: str) -> SearchOutcome:
            tried.append(method)
            return SearchOutcome(f"{method} [Data: Sources (1)]", source_count=10)

        result = search_with_fallback(StrategyDecision("basic", "原因"), search, check)

        self.assertEqual((result.found, result.partial, result.method), (True, False, "local"))
        self.assertEqual(result.outcome.answer, "local [Data: Sources (1)]")

    def test_unsupported_answer_is_never_kept_as_partial(self) -> None:
        result = search_with_fallback(
            StrategyDecision("basic", "原因"),
            lambda method: SearchOutcome("", "壞了") if method == "global" else SearchOutcome("沒有引用的回答"),
            lambda _: AnswerCheck(False, "部分", partial=True, answered_parts=1),
        )

        self.assertFalse(result.partial)
        self.assertEqual(result.outcome.answer, NOT_FOUND_ANSWER)

    def test_evidence_page_records_dedupe_document_pages(self) -> None:
        class Item:
            def __init__(self, document_id: str, page: int | None, pages: tuple[int, ...] = ()) -> None:
                self.document_id, self.page, self.pages = document_id, page, pages

        records = evidence_page_records([Item("A.pdf", 3), Item("A.pdf", 3), Item("B.pdf", 4, (4, 5))])

        self.assertEqual(records, (
            {"document": "A.pdf", "page": 3}, {"document": "B.pdf", "page": 4}, {"document": "B.pdf", "page": 5},
        ))

    def test_local_with_too_few_sources_falls_back_without_checking(self) -> None:
        checked: list[str] = []

        def search(method: str) -> SearchOutcome:
            if method == "local":
                return SearchOutcome("來源未說明，只提到相簿群組", source_count=LOCAL_MIN_SOURCES - 1)
            return SearchOutcome("聯絡人群組步驟 [Data: Sources (1)]", source_count=2)

        def check(answer: str) -> AnswerCheck:
            checked.append(answer)
            return AnswerCheck(True, "ok")

        result = search_with_fallback(StrategyDecision("local", "原因"), search, check)

        self.assertEqual((result.method, result.found), ("basic", True))
        self.assertEqual(checked, ["聯絡人群組步驟 [Data: Sources (1)]"])
        self.assertEqual(result.attempts[0]["status"], "not_found")
        self.assertIn(f"只取得 {LOCAL_MIN_SOURCES - 1} 個原文段落", result.attempts[0]["detail"])
        self.assertIn("Local 未找到答案，改用 Basic 找到答案", result.reason)

    def test_local_with_enough_or_unknown_sources_is_checked_normally(self) -> None:
        for count in (LOCAL_MIN_SOURCES, None):
            with self.subTest(count=count):
                tried: list[str] = []

                def search(method: str) -> SearchOutcome:
                    tried.append(method)
                    return SearchOutcome("答案 [Data: Sources (1)]", source_count=count)

                result = search_with_fallback(
                    StrategyDecision("local", "原因"), search, lambda _: AnswerCheck(True, "ok"),
                )
                self.assertEqual((tried, result.found), (["local"], True))

    def test_question_identifiers_keep_codes_and_skip_page_references(self) -> None:
        self.assertEqual(
            question_identifiers("C1154 與C1155 的差異？接頭E24端子2、SP130、12V、VDC、參考 AV-97"),
            ("C1154", "C1155", "E24", "SP130"),
        )

    def test_local_context_without_named_codes_falls_back_without_checking(self) -> None:
        # SM3E v4 Q07: Local matched C1154 but its passages were the C1155 steps (F-69).
        checked: list[str] = []
        local_context = {"sources": [{"id": str(index), "text": "C1155 BR FLUID LEVEL LOW steps"} for index in range(8)]}

        def search(method: str) -> SearchOutcome:
            if method == "local":
                return SearchOutcome("端子檢查 [Data: Sources (1)]", payload=local_context, source_count=8)
            return SearchOutcome("C1154 步驟 [Data: Sources (2)]", source_count=3)

        def check(answer: str) -> AnswerCheck:
            checked.append(answer)
            return AnswerCheck(True, "ok")

        result = search_with_fallback(
            StrategyDecision("local", "原因"), search, check, named_terms=question_identifiers("C1154 怎麼診斷？"),
        )

        self.assertEqual((result.method, result.found), ("basic", True))
        self.assertEqual(checked, ["C1154 步驟 [Data: Sources (2)]"])
        self.assertEqual(result.attempts[0]["status"], "not_found")
        self.assertIn("沒有題目點名的 C1154", result.attempts[0]["detail"])

    def test_local_context_with_most_named_codes_is_checked(self) -> None:
        context = {"sources": [{"id": "1", "text": "C1154 PNP SWITCH and connector E24"}]}
        result = search_with_fallback(
            StrategyDecision("local", "原因"),
            lambda _: SearchOutcome("答案 [Data: Sources (1)]", payload=context, source_count=None),
            lambda _: AnswerCheck(True, "ok"),
            named_terms=("C1154", "E24", "M24"),
        )
        self.assertEqual((result.method, result.found), ("local", True))

    def test_answer_with_insufficient_data_phrase_is_partial_and_fallback_continues(self) -> None:
        local_answer = f"拆下飾板 [Data: Sources (1)]。安裝扭力：{INSUFFICIENT_DATA_PHRASE}。"

        def search(method: str) -> SearchOutcome:
            if method == "local":
                return SearchOutcome(local_answer)
            return SearchOutcome("無關 [Data: Sources (2)]")

        def check(answer: str) -> AnswerCheck:
            return AnswerCheck(answer == local_answer, "檢查")

        result = search_with_fallback(StrategyDecision("local", "原因", ("local", "basic")), search, check)

        self.assertEqual(result.attempts[0]["status"], "partial")
        self.assertEqual(result.attempts[1]["method"], "basic")
        self.assertEqual((result.method, result.found, result.partial), ("local", False, True))
        self.assertIn("目前取得的資料未涵蓋的部分", result.outcome.answer)

    def test_matching_section_title_accepts_answer_without_overstrict_check(self) -> None:
        checked: list[str] = []
        result = search_with_fallback(
            StrategyDecision("basic", "原因"),
            lambda method: SearchOutcome("回答 [Data: Sources (1)]", payload={}, section_titles=("關閉螢幕時辨識指紋",)),
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
                        "找到具體操作步驟。 [Data: Sources (1)]", source_count=3,
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

    def test_answers_not_traceable_to_passages_fall_back_without_checking(self) -> None:
        context = {"sources": [{"id": "5"}]}
        cases = {
            "no citation": ("GPS 需要四顆衛星。", "沒有引用任何原文段落"),
            "general knowledge": (
                "GPS 需要四顆衛星。 [Data: General Knowledge (standard GPS principles)]", "一般知識",
            ),
            "reports only": ("GPS 需要四顆衛星。 [Data: Reports (3)]", "沒有引用任何原文段落"),
            "id not in context": ("GPS 需要四顆衛星。 [Data: Sources (99)]", "未出現在證據中"),
            "short form id not in context": ("GPS 需要四顆衛星。 [Sources: 99]", "未出現在證據中"),
        }
        for name, (answer, detail) in cases.items():
            with self.subTest(name):
                checked: list[str] = []

                def search(method: str) -> SearchOutcome:
                    if method == "drift":
                        return SearchOutcome(answer, payload=context)
                    return SearchOutcome("", "壞了")

                result = search_with_fallback(
                    StrategyDecision("drift", "原因", ("drift",)), search,
                    lambda text: checked.append(text) or AnswerCheck(True, "ok"),
                )

                self.assertFalse(result.found)
                self.assertEqual(checked, [])
                self.assertEqual(result.attempts[0]["status"], "not_found")
                self.assertIn(detail, result.attempts[0]["detail"])

    def test_short_form_sources_citation_counts_as_passage_citation(self) -> None:
        tried: list[str] = []

        def search(method: str) -> SearchOutcome:
            tried.append(method)
            return SearchOutcome("Check the ECM first [Sources: 857].", source_count=9)

        result = search_with_fallback(
            StrategyDecision("local", "原因", ("local", "basic")), search, lambda answer: AnswerCheck(True, "ok"),
        )

        self.assertEqual(tried, ["local"])
        self.assertTrue(result.found)

    def test_local_and_basic_answers_also_need_passage_citations(self) -> None:
        tried: list[str] = []

        def search(method: str) -> SearchOutcome:
            tried.append(method)
            if method == "local":
                return SearchOutcome("開啟設定後點選 Wi-Fi。", source_count=9)
            return SearchOutcome("開啟設定後點選連線。 [Data: Sources (2, +more)]")

        result = search_with_fallback(
            StrategyDecision("local", "原因", ("local", "basic")), search,
            lambda _: AnswerCheck(True, "ok"),
        )

        self.assertEqual(tried, ["local", "basic"])
        self.assertEqual((result.method, result.found), ("basic", True))
        self.assertIn("Local 回答沒有引用任何原文段落", result.attempts[0]["detail"])

    def test_cited_drift_and_uncited_global_answers_are_checked(self) -> None:
        for method, answer in (
            ("drift", "答案 [Data: Sources (5, +more)]"),
            ("drift", "答案 [E2]"),
            ("global", "總覽 [Data: Reports (1)]"),
        ):
            with self.subTest(answer):
                result = search_with_fallback(
                    StrategyDecision(method, "原因", (method,)),
                    lambda _: SearchOutcome(answer, payload={"sources": [{"id": "5"}]}),
                    lambda _: AnswerCheck(True, "ok"),
                )
                self.assertEqual((result.method, result.found), (method, True))


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
        # Automatic QA defaults to auto: the dropdown starts on the "auto" value, not its label.
        _model, default_update = handlers["update_autoqa_model_method_compatibility"]("gpt-4o-mini", None)
        self.assertEqual(default_update["value"], "auto")
        autoqa_method = next(
            component.get_config() for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "檢索模式"
        )
        self.assertEqual(autoqa_method["value"], "auto")


if __name__ == "__main__":
    unittest.main()
