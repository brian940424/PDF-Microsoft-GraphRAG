import asyncio
import unittest
from types import SimpleNamespace

import pandas as pd
from graphrag.query.context_builder.builders import ContextBuilderResult
from graphrag.query.structured_search.basic_search.search import BasicSearch

from automotive_graphrag.answer_extraction import install_two_stage_basic
from automotive_graphrag.answer_revision import (
    REVISION_PROMPT,
    check_revision,
    cited_source_ids,
    cited_source_texts,
    revise_answer,
)

SOURCES = pd.DataFrame({
    "id": ["649", "650", "254"],
    "text": [
        "Bleed the rear right caliper first. With the reservoir kept at least half full, bleed the others in order.",
        "Inspect the hose for damage.",
        "The position is compared with map data and shown as the vehicle mark.",
    ],
})
RECORDS = {"sources": SOURCES}
DRAFT = "Connect the vinyl tube to the rear right caliper first; bleed the others in order. [Data: Sources (649)]"
REVISED = (
    "Connect the vinyl tube to the rear right caliper first; with the reservoir kept at least half full, "
    "bleed the others in order. [Data: Sources (649)]"
)


class FakeModel:
    """Returns `revision` for non-streaming calls and streams `answer` for the answer call."""

    def __init__(self, revision: str = "", answer: str = DRAFT, fail: bool = False) -> None:
        self.revision = revision
        self.answer = answer
        self.fail = fail
        self.calls: list[dict] = []

    async def completion_async(self, messages, stream=False, **params):
        self.calls.append({"messages": messages, "stream": stream, "params": params})
        if not stream:
            if self.fail:
                raise RuntimeError("proxy down")
            return SimpleNamespace(content=self.revision)

        async def chunks():
            for part in (self.answer[:10], self.answer[10:]):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=part))])
        return chunks()


class CitationTests(unittest.TestCase):
    def test_collects_source_ids_in_order_without_duplicates(self) -> None:
        answer = "A [Data: Sources (267, 585)]. B [Data: Sources (584, 267, +more)]. C [Data: Entities (3)]"
        self.assertEqual(cited_source_ids(answer), ["267", "585", "584"])

    def test_texts_come_only_from_cited_sources_in_context(self) -> None:
        self.assertEqual(cited_source_texts("x [Data: Sources (254, 999)]", RECORDS), [("254", SOURCES.text[2])])
        self.assertEqual(cited_source_texts("x [Data: Sources (254)]", {}), [])
        self.assertEqual(cited_source_texts("no citation", RECORDS), [])


class CheckRevisionTests(unittest.TestCase):
    def test_accepts_added_qualifier(self) -> None:
        self.assertIsNone(check_revision(DRAFT, REVISED))

    def test_rejects_changed_citations_growth_shortening_and_new_paragraphs(self) -> None:
        self.assertEqual(check_revision(DRAFT, REVISED.replace("649", "650")), "citations_changed")
        self.assertEqual(check_revision(DRAFT, DRAFT.replace("(649)", "(649, 650)")), "citations_changed")
        self.assertEqual(check_revision(DRAFT, DRAFT + " extra" * 100), "too_long")
        self.assertEqual(check_revision(DRAFT, "First rear right. [Data: Sources (649)]"), "shortened")
        self.assertEqual(check_revision(DRAFT, DRAFT + "\n\nAlso inspect. [Data: Sources (649)]"), "paragraph_added")
        self.assertEqual(check_revision(DRAFT, "  "), "empty")


class ReviseAnswerTests(unittest.TestCase):
    def test_applies_revision_and_sends_only_cited_sources(self) -> None:
        model = FakeModel(revision=REVISED)

        answer, report = asyncio.run(revise_answer(model, "Which caliper first?", DRAFT, RECORDS, {"reasoning_effort": "low"}))

        self.assertEqual(answer, REVISED)
        self.assertEqual(report["status"], "applied")
        self.assertEqual(report["draft"], DRAFT)
        self.assertEqual(report["cited_sources"], ["649"])
        call = model.calls[0]
        self.assertFalse(call["stream"])
        self.assertEqual(call["params"], {"reasoning_effort": "low"})
        prompt = call["messages"][0]["content"]
        self.assertIn("[649]\nBleed the rear right caliper", prompt)
        self.assertNotIn("Inspect the hose", prompt)
        self.assertIn("Which caliper first?", prompt)
        self.assertIn(DRAFT, prompt)

    def test_keeps_draft_when_skipped_unchanged_rejected_or_failed(self) -> None:
        not_found = "The provided data does not state this."
        answer, report = asyncio.run(revise_answer(FakeModel(), "Q", not_found, RECORDS, {}))
        self.assertEqual((answer, report["status"]), (not_found, "skipped"))

        model = FakeModel(revision=DRAFT)
        answer, report = asyncio.run(revise_answer(model, "Q", DRAFT, RECORDS, {}))
        self.assertEqual((answer, report["status"]), (DRAFT, "unchanged"))

        answer, report = asyncio.run(revise_answer(FakeModel(revision=DRAFT + "\n\nMore."), "Q", DRAFT, RECORDS, {}))
        self.assertEqual((answer, report["status"], report["reason"]), (DRAFT, "rejected", "paragraph_added"))

        answer, report = asyncio.run(revise_answer(FakeModel(fail=True), "Q", DRAFT, RECORDS, {}))
        self.assertEqual((answer, report["status"]), (DRAFT, "error"))

    def test_prompt_is_english_and_limits_the_revision(self) -> None:
        for phrase in ("put back", "Do not:", "add a sentence, step or item", "expand a reference", "add, remove or change citation identifiers", "output the draft exactly as it is"):
            self.assertIn(phrase, REVISION_PROMPT)
        for field in ("{question}", "{sources}", "{draft}"):
            self.assertIn(field, REVISION_PROMPT)


class RevisedBasicSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = BasicSearch.stream_search

    def tearDown(self) -> None:
        BasicSearch.stream_search = self.original

    def _search(self, model: FakeModel, revise: bool, report: dict) -> str:
        install_two_stage_basic(report=report, revise=revise)
        builder = SimpleNamespace(build_context=lambda **kwargs: ContextBuilderResult(context_chunks="id|text\n649|...", context_records=RECORDS))
        search = BasicSearch(
            model=model, context_builder=builder, tokenizer=SimpleNamespace(encode=str.split),
            system_prompt="DATA:\n{context_data}\nFORMAT:{response_type}",
        )

        async def collect() -> str:
            return "".join([token async for token in search.stream_search("Which caliper first?")])

        return asyncio.run(collect())

    def test_revision_replaces_the_streamed_draft(self) -> None:
        report: dict = {}
        model = FakeModel(revision=REVISED)

        self.assertEqual(self._search(model, revise=True, report=report), REVISED)
        self.assertEqual([call["stream"] for call in model.calls], [True, False])
        self.assertEqual(report["two_stage"], False)
        self.assertEqual(report["qualifier_revision"]["status"], "applied")

    def test_without_revise_the_answer_streams_unchanged(self) -> None:
        report: dict = {}
        model = FakeModel(revision=REVISED)

        self.assertEqual(self._search(model, revise=False, report=report), DRAFT)
        self.assertEqual([call["stream"] for call in model.calls], [True])
        self.assertNotIn("qualifier_revision", report)


if __name__ == "__main__":
    unittest.main()
