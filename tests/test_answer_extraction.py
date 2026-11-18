import asyncio
import unittest
from types import SimpleNamespace

from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.structured_search.basic_search.search import BasicSearch

from automotive_graphrag.answer_extraction import (
    EXTRACTION_HEADING,
    EXTRACTION_PROMPT,
    context_with_extraction,
    extract_source_sentences,
    install_two_stage_basic,
    needs_two_stage,
)
from automotive_graphrag.answer_prompts import ANSWER_REQUIREMENTS


class FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()


class FakeModel:
    """Returns `extraction` for the non-streaming call and streams `answer` for the answer call."""

    def __init__(self, extraction: str = "", answer: str = "answer", fail_extraction: bool = False) -> None:
        self.extraction = extraction
        self.answer = answer
        self.fail_extraction = fail_extraction
        self.calls: list[dict] = []

    async def completion_async(self, messages, stream=False, **params):
        self.calls.append({"messages": messages, "stream": stream, "params": params})
        if not stream:
            if self.fail_extraction:
                raise RuntimeError("proxy down")
            return SimpleNamespace(content=self.extraction)

        async def chunks():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=self.answer))])
        return chunks()


class FakeContextBuilder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def build_context(self, query, conversation_history=None, **kwargs):
        self.queries.append(query)
        return ContextBuilderResult(context_chunks="id|text\n486|BCM page\n502|keyfob page", context_records={})


class ContextWithExtractionTests(unittest.TestCase):
    def test_appends_block_to_text_and_list_contexts(self) -> None:
        self.assertEqual(
            context_with_extraction("ctx", " [502] re-register keyfobs \n"),
            f"ctx\n\n{EXTRACTION_HEADING}\n[502] re-register keyfobs\n",
        )
        self.assertEqual(context_with_extraction(["a"], "[1] x"), ["a", f"\n\n{EXTRACTION_HEADING}\n[1] x\n"])

    def test_empty_or_none_extraction_keeps_context(self) -> None:
        self.assertEqual(context_with_extraction("ctx", ""), "ctx")
        self.assertEqual(context_with_extraction("ctx", " none "), "ctx")


class NeedsTwoStageTests(unittest.TestCase):
    def test_comparison_questions_use_two_stages(self) -> None:
        for question in (
            "Compare the additional service required when replacing the BCM with the AV control unit.",
            "How does the rear rotor runout limit differ from the front rotor's?",
            "How is the hose aligned on the front caliper versus the rear caliper?",
            "和 B2190 的解除方式有什麼不同？",
            "兩者的可能原因有何差別？",
        ):
            self.assertTrue(needs_two_stage(question), question)

    def test_single_object_questions_skip_extraction(self) -> None:
        for question in (
            "What cautions must be observed when removing the accelerator pedal assembly?",
            "安裝真空軟管到煞車助力器時有哪些注意事項？軟管至少要插入多深？",
            "前碟煞來令片的新品標準厚度與磨耗極限厚度分別是多少？",
            "前碟盤的最大偏擺量與厚度差限制值各是多少？",
            "基本音響要進入「硬體/軟體版本與喇叭聲道檢查」模式，操作步驟為何？",
        ):
            self.assertFalse(needs_two_stage(question), question)


class ExtractionCallTests(unittest.TestCase):
    def test_extraction_sends_context_and_question_without_streaming(self) -> None:
        model = FakeModel(extraction="[502] line")

        result = asyncio.run(extract_source_sentences(model, "Q?", ["c1", "c2"], {"reasoning_effort": "medium"}))

        self.assertEqual(result, "[502] line")
        call = model.calls[0]
        self.assertFalse(call["stream"])
        self.assertEqual(call["params"], {"reasoning_effort": "medium"})
        self.assertIn("c1\nc2", call["messages"][0]["content"])
        self.assertEqual(call["messages"][1], {"role": "user", "content": "Q?"})

    def test_failed_extraction_returns_empty_string(self) -> None:
        self.assertEqual(asyncio.run(extract_source_sentences(FakeModel(fail_extraction=True), "Q?", "c", {})), "")

    def test_prompts_are_english_and_ask_for_verbatim_whole_sentences(self) -> None:
        for phrase in ("copy word for word", "Copy whole sentences and steps", "Check all of them", "output only NONE"):
            self.assertIn(phrase, EXTRACTION_PROMPT)
        self.assertIn("{context_data}", EXTRACTION_PROMPT)
        self.assertIn(EXTRACTION_HEADING, ANSWER_REQUIREMENTS)


class TwoStageBasicSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = BasicSearch.stream_search
        self.report: dict = {}
        install_two_stage_basic()
        install_two_stage_basic(report=self.report)  # idempotent

    def tearDown(self) -> None:
        BasicSearch.stream_search = self.original

    def _search(self, model: FakeModel, question: str = "Compare BCM and AV") -> tuple[str, FakeContextBuilder]:
        builder = FakeContextBuilder()
        search = BasicSearch(
            model=model, context_builder=builder, tokenizer=FakeTokenizer(),
            system_prompt="DATA:\n{context_data}\nFORMAT:{response_type}", model_params={"reasoning_effort": "low"},
        )
        async def collect() -> str:
            return "".join([token async for token in search.stream_search(question)])

        result = asyncio.run(collect())
        self.assertNotIn("build_context", builder.__dict__)
        return result, builder

    def test_answer_call_sees_full_context_plus_extracted_sentences(self) -> None:
        model = FakeModel(extraction="[502] If new BCM does not come with keyfobs, re-register all keyfobs.", answer="final")

        result, builder = self._search(model)

        self.assertEqual(result, "final")
        self.assertEqual(builder.queries, ["Compare BCM and AV"])  # context built once
        self.assertEqual([call["stream"] for call in model.calls], [False, True])
        answer_prompt = model.calls[1]["messages"][0]["content"]
        self.assertIn("486|BCM page", answer_prompt)
        self.assertIn(f"{EXTRACTION_HEADING}\n[502] If new BCM", answer_prompt)
        self.assertEqual(self.report, {"two_stage": True})

    def test_non_comparison_question_answers_in_one_stage(self) -> None:
        model = FakeModel(extraction="[502] should not be used", answer="single")

        result, builder = self._search(model, "What cautions apply when removing the BCM?")

        self.assertEqual(result, "single")
        self.assertEqual(builder.queries, ["What cautions apply when removing the BCM?"])
        self.assertEqual([call["stream"] for call in model.calls], [True])
        self.assertIn("486|BCM page", model.calls[0]["messages"][0]["content"])
        self.assertNotIn(EXTRACTION_HEADING, model.calls[0]["messages"][0]["content"])
        self.assertEqual(self.report, {"two_stage": False})

    def test_failed_extraction_falls_back_to_single_stage_context(self) -> None:
        model = FakeModel(fail_extraction=True, answer="still answered")

        result, _ = self._search(model)

        self.assertEqual(result, "still answered")
        self.assertNotIn(EXTRACTION_HEADING, model.calls[1]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
