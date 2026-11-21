import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automotive_graphrag.automatic_qa import AutomaticQATestService
from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.question_sets import QuestionSetService
from automotive_graphrag.source_sampling import SourceSample, SourceSamplingService
from automotive_graphrag.strategy_questions import (
    COMMON_INSTRUCTIONS,
    SourceBundle,
    StrategyBundleBuilder,
    StrategyQuestionGenerator,
    allocate_quotas,
    normalize_strategy_mix,
)

PROJECT = "SM3E"
POWER_TOOLS = "PRECAUTIONS WHEN USING POWER TOOLS (AIR OR ELECTRIC) AND HAMMERS"


def sample(document: str, page: int, text: str = "", section_path: str = "") -> SourceSample:
    body = text or f"{document} page {page} body text. " * 10
    chunk = f"{PROJECT}-{document}-p{page:04d}-b01"
    return SourceSample(
        sample_id=chunk, chunk_id=chunk, project_id=PROJECT, section_id=document.removesuffix(".pdf"),
        section_name=document.removesuffix(".pdf"), document_id=document, page=page, block_id="b01",
        content_type="general", character_count=len(body), text=body,
        section_path=section_path or document.removesuffix(".pdf"),
    )


class FakeStrategyClient:
    """Answers every bundle with one question citing all of its sources."""

    def __init__(self):
        self.prompts: list[str] = []

    def __call__(self, base_url, api_key, model, prompt):
        self.prompts.append(prompt)
        bundles = json.loads(prompt.split("Bundles: ", 1)[1])
        questions = []
        for bundle in bundles:
            ids = [item["sample_id"] for item in bundle["sources"]]
            questions.append({
                "bundle_id": bundle["bundle_id"],
                "question": f"{bundle['bundle_id']} 的情境下應該怎麼處理？",
                "reference_answer": "依手冊步驟處理。",
                "question_source_sample_ids": ids[:1],
                "answer_source_sample_ids": ids,
                "entities": ["BRAKE PEDAL", "ACCELERATOR PEDAL"],
                "reasoning_chain": ["先查第一頁 → 得知要拆加速踏板", "再查參照頁 → 拔接頭後要學習"],
            })
        return {"questions": questions}


class AllocationTests(unittest.TestCase):
    def test_default_mix_matches_the_guideline_counts(self):
        self.assertEqual(allocate_quotas(40, normalize_strategy_mix(None)),
                         {"basic": 20, "local": 8, "global": 5, "drift": 7})
        self.assertEqual(allocate_quotas(10, {"basic": 50, "local": 20, "drift": 17.5, "global": 12.5}),
                         {"basic": 5, "local": 2, "global": 1, "drift": 2})
        self.assertEqual(allocate_quotas(3, {"basic": 1, "global": 0}), {"basic": 3, "local": 0, "global": 0, "drift": 0})

    def test_invalid_mix_is_rejected(self):
        with self.assertRaisesRegex(ProjectError, "至少要有一項"):
            normalize_strategy_mix({"basic": 0, "local": 0})
        with self.assertRaisesRegex(ProjectError, "不可小於 0"):
            normalize_strategy_mix({"basic": -1})
        with self.assertRaisesRegex(ProjectError, "不支援的出題策略"):
            normalize_strategy_mix({"hybrid": 10})


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary.name) / "projects")
        self.store.create(project_id=PROJECT, display_name="SM3E", vehicle_name="L33", manual_version="SM3E")
        self.labels: list[tuple[str, int]] = []

        def label_matches(project_id, document_id, page):
            self.labels.append((document_id, page))
            return (document_id, page) != ("ACC.pdf", 9)

        self.builder = StrategyBundleBuilder(self.store, label_matches=label_matches)

    def tearDown(self):
        self.temporary.cleanup()

    def test_basic_pairs_only_consecutive_pages_of_the_same_section(self):
        samples = [
            sample("BR.pdf", 14, section_path="BR > BRAKE FLUID"), sample("BR.pdf", 15, section_path="BR > BRAKE FLUID"),
            sample("BR.pdf", 16, section_path="BR > BRAKE PEDAL"), sample("BR.pdf", 18, section_path="BR > BRAKE PEDAL"),
            sample("BR.pdf", 30, text="short"),
        ]
        bundles = self.builder.build(PROJECT, "basic", samples)

        pairs = [[item.page for item in bundle.samples] for bundle in bundles if len(bundle.samples) == 2]
        singles = sorted(bundle.samples[0].page for bundle in bundles if len(bundle.samples) == 1)
        self.assertEqual(pairs, [[14, 15]])
        # Pages shorter than MIN_SOURCE_CHARACTERS are not used.
        self.assertEqual(singles, [14, 15, 16, 18])

    def test_drift_follows_labelled_references_to_far_pages_once_per_tail(self):
        samples = [
            sample("BR.pdf", 30, "Remove the accelerator pedal assembly. Refer to ACC-3. " * 4),
            sample("BR.pdf", 31, "Remove the accelerator pedal assembly. Refer to ACC-3. " * 4),
            sample("ACC.pdf", 3, "Disconnect the harness connector. Refer to ACC-4. " * 4),
            sample("ACC.pdf", 4, "Perform accelerator pedal released position learning. " * 4),
            # Adjacent page reference: not a reasoning chain.
            sample("BR.pdf", 13, "Adjust the stop lamp switch. Refer to BR-14. " * 4),
            sample("BR.pdf", 14, "Stop lamp switch clearance values listed here. " * 4),
            # The printed label of ACC.pdf page 9 does not match, so it is not followed.
            sample("BR.pdf", 50, "Refer to ACC-9 for the procedure details here. " * 4),
            sample("ACC.pdf", 9, "Unrelated page text that is long enough to use. " * 4),
        ]
        bundles = self.builder.build(PROJECT, "drift", samples)

        chains = [[(item.document_id, item.page) for item in bundle.samples] for bundle in bundles]
        self.assertEqual(len(chains), 1)
        self.assertIn(chains[0][0], {("BR.pdf", 30), ("BR.pdf", 31)})
        self.assertEqual(chains[0][1:], [("ACC.pdf", 3), ("ACC.pdf", 4)])
        self.assertIn("Reference chain:", bundles[0].hint)
        self.assertIn(("ACC.pdf", 9), self.labels)

    def test_global_uses_specific_headings_shared_by_documents(self):
        samples = [
            sample("ACC.pdf", 2, f"PRECAUTION\n{POWER_TOOLS}\nSwitch the ignition OFF and wait three minutes. " * 2),
            sample("BR.pdf", 3, f"{POWER_TOOLS}\nDisconnect the battery before using power tools here. " * 2),
            sample("BCS.pdf", 5, "REMOVAL AND INSTALLATION\nINSTALLATION\nInstall in the reverse order of removal. " * 2),
            sample("BR.pdf", 9, "REMOVAL AND INSTALLATION\nInstall in the reverse order of removal steps. " * 2),
        ]
        bundles = self.builder.build(PROJECT, "global", samples)

        self.assertEqual(len(bundles), 1)
        self.assertEqual([(item.document_id, item.page) for item in bundles[0].samples], [("ACC.pdf", 2), ("BR.pdf", 3)])
        self.assertIn(POWER_TOOLS, bundles[0].hint)

    def test_local_joins_related_entities_and_skips_hubs(self):
        samples = [sample("BR.pdf", 18), sample("ACC.pdf", 3), sample("ACC.pdf", 4), sample("BR.pdf", 2)]
        output = self.store.path_for(PROJECT) / "graphrag" / "output"
        output.mkdir(parents=True, exist_ok=True)
        unit_of = {item.chunk_id: f"unit-{index}" for index, item in enumerate(samples)}
        (output / "source_metadata.jsonl").write_text("".join(
            json.dumps({"text_unit_id": unit, "chunk_id": chunk}) + "\n" for chunk, unit in unit_of.items()
        ), encoding="utf-8")
        units = list(unit_of.values())
        pd.DataFrame([
            {"title": "BRAKE PEDAL", "text_unit_ids": [units[0]]},
            {"title": "ACCELERATOR PEDAL", "text_unit_ids": [units[1], units[2]]},
            {"title": "NISSAN", "text_unit_ids": units * 3},
        ]).to_parquet(output / "entities.parquet")
        pd.DataFrame([
            {"source": "BRAKE PEDAL", "target": "ACCELERATOR PEDAL", "description": "Remove the accelerator pedal first.",
             "weight": 9.0, "text_unit_ids": [units[0]]},
            {"source": "NISSAN", "target": "BRAKE PEDAL", "description": "Generic.", "weight": 20.0,
             "text_unit_ids": [units[3]]},
        ]).to_parquet(output / "relationships.parquet")

        bundles = self.builder.build(PROJECT, "local", samples)

        self.assertEqual(len(bundles), 1)
        self.assertEqual([(item.document_id, item.page) for item in bundles[0].samples], [("BR.pdf", 18), ("ACC.pdf", 3)])
        self.assertIn("BRAKE PEDAL, ACCELERATOR PEDAL", bundles[0].hint)
        self.assertNotIn("NISSAN", bundles[0].hint)

    def test_local_without_index_has_no_bundles(self):
        self.assertEqual(self.builder.build(PROJECT, "local", [sample("BR.pdf", 18)]), [])


class ResponseTests(unittest.TestCase):
    def bundle(self, strategy, *samples):
        return SourceBundle(f"{strategy}-1", strategy, tuple(samples))

    def parse(self, strategy, bundle, **fields):
        raw = {
            "bundle_id": bundle.bundle_id, "question": "拆下煞車助力器後還要做什麼？", "reference_answer": "要執行學習程序。",
            "question_source_sample_ids": [bundle.samples[0].sample_id],
            "answer_source_sample_ids": [item.sample_id for item in bundle.samples], **fields,
        }
        return StrategyQuestionGenerator.parse_response({"questions": [raw]}, strategy, [bundle])

    def test_strategy_rules_reject_questions_that_do_not_fit(self):
        cross_document = self.bundle("basic", sample("BR.pdf", 18), sample("ACC.pdf", 3))
        self.assertIn("同一份文件", self.parse("basic", cross_document)[2][0])

        adjacent = self.bundle("drift", sample("BR.pdf", 14), sample("BR.pdf", 15))
        self.assertIn("不相鄰", self.parse("drift", adjacent, reasoning_chain=["一", "二"])[2][0])
        far = self.bundle("drift", sample("BR.pdf", 30), sample("ACC.pdf", 4))
        self.assertIn("至少要兩步", self.parse("drift", far, reasoning_chain=["一"])[2][0])
        questions, _, errors = self.parse("drift", far, reasoning_chain=["BR p.30 → 拆加速踏板", "ACC p.4 → 學習"])
        self.assertEqual(errors, [])
        self.assertEqual((questions[0].design_strategy, questions[0].answer_scope), ("drift", "cross_document"))
        self.assertEqual(len(questions[0].reasoning_chain), 2)

        local = self.bundle("local", sample("BR.pdf", 13), sample("BR.pdf", 48))
        self.assertIn("兩個實體", self.parse("local", local, entities=["STOP LAMP SWITCH"])[2][0])
        single_page = self.parse("local", local, entities=["STOP LAMP SWITCH", "BRAKE PEDAL"],
                                 answer_source_sample_ids=[local.samples[0].sample_id])
        self.assertIn("至少兩個不同頁面", single_page[2][0])
        self.assertEqual(self.parse("local", local, entities=["STOP LAMP SWITCH", "BRAKE PEDAL"])[0][0].answer_scope,
                         "cross_page")

    def test_skips_unknown_sources_and_english_text_are_reported_per_item(self):
        bundle = self.bundle("basic", sample("BR.pdf", 49))
        questions, skipped, errors = StrategyQuestionGenerator.parse_response({"questions": [
            {"bundle_id": "basic-1", "skip": "only a table of contents"},
        ]}, "basic", [bundle])
        self.assertEqual((questions, errors), ([], []))
        self.assertEqual(skipped[0]["reason"], "only a table of contents")
        self.assertIn("sample_id", self.parse("basic", bundle, answer_source_sample_ids=["other"])[2][0])
        self.assertIn("繁體中文", self.parse("basic", bundle, question="What is the pad thickness?")[2][0])

    def test_literal_backslash_n_becomes_a_line_break(self):
        bundle = self.bundle("basic", sample("BR.pdf", 49))
        questions, _, _ = self.parse("basic", bundle, reference_answer="- 新品 11.0 mm\\n- 極限 2.0 mm")
        self.assertEqual(questions[0].reference_answer, "- 新品 11.0 mm\n- 極限 2.0 mm")

    def test_citation_markers_with_sample_ids_are_removed(self):
        first, second = sample("ACC.pdf", 2), sample("BR.pdf", 3)
        bundle = self.bundle("global", first, second)
        questions, _, _ = self.parse(
            "global", bundle,
            reference_answer=f"- 兩份手冊都有規定。【{first.sample_id}】【{second.sample_id}】\n- 先斷開電池 [{first.sample_id}]。\n- 【注意】保留",
        )
        self.assertEqual(questions[0].reference_answer, "- 兩份手冊都有規定。\n- 先斷開電池。\n- 【注意】保留")

    def test_prompts_are_english_and_carry_strategy_conditions(self):
        bundle = self.bundle("drift", sample("BR.pdf", 30), sample("ACC.pdf", 4))
        prompt = StrategyQuestionGenerator.build_prompt("drift", [bundle], ["既有題目？"])
        self.assertTrue(prompt.startswith(COMMON_INSTRUCTIONS))
        self.assertIn("Never state the intermediate conclusion", prompt)
        self.assertIn("Counter-example", prompt)
        self.assertIn('"reasoning_chain"', prompt)
        self.assertIn("既有題目？", prompt)
        self.assertIn("Never join two unrelated sub-questions", StrategyQuestionGenerator.build_prompt("basic", [bundle]))


class AutomaticStrategyGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary.name) / "projects")
        self.store.create(project_id=PROJECT, display_name="SM3E", vehicle_name="L33", manual_version="SM3E")
        self.store.update_status(PROJECT, "INDEXED")
        project = self.store.path_for(PROJECT)
        for name in ("BR.pdf", "ACC.pdf"):
            (project / "source" / name).write_bytes(b"%PDF-1.4")
        records = [
            sample("BR.pdf", 14), sample("BR.pdf", 15), sample("BR.pdf", 16),
            sample("BR.pdf", 30, "Remove the accelerator pedal assembly. Refer to ACC-3. " * 4),
            sample("ACC.pdf", 3, "Disconnect the harness connector first. Then lift it out. " * 4),
            sample("ACC.pdf", 7, "Accelerator pedal specification values listed here. " * 4),
        ]
        (project / "processed" / "input.jsonl").write_text("".join(
            json.dumps({
                "chunk_id": item.chunk_id, "project_id": PROJECT, "section_id": item.section_id,
                "section_name": item.section_name, "section_path": item.section_path,
                "document_id": item.document_id, "page": item.page, "block_id": item.block_id, "text": item.text,
            }) + "\n" for item in records
        ), encoding="utf-8")
        self.connections = ConnectionSettings(self.store.root)
        self.connections.save("https://api.openai.com/v1", "test-key")
        self.question_sets = QuestionSetService(self.store)
        self.client = FakeStrategyClient()
        generator = StrategyQuestionGenerator(
            self.store, self.connections, self.client,
            StrategyBundleBuilder(self.store, label_matches=lambda *args: True),
        )
        self.service = AutomaticQATestService(
            self.store, SourceSamplingService(self.store), object(), object(), self.question_sets,
            object(), object(), strategy_generator=generator,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_question_set_records_strategy_scope_and_design_log(self):
        model = self.connections.get_chat_model()
        question_set = self.service.generate_question_set(
            PROJECT, 2, False, model, strategy_mix={"basic": 50, "local": 25, "drift": 25, "global": 0},
        )

        strategies = [item.design_strategy for item in question_set.questions]
        # No GraphRAG index, so the Local question becomes Basic; the DRIFT chain BR p.30 → ACC p.3 is used.
        self.assertEqual(strategies, ["basic", "basic", "basic", "drift"])
        drift = question_set.questions[-1]
        self.assertEqual(drift.answer_scope, "cross_document")
        self.assertTrue(all(item.answer_scope for item in question_set.questions))
        self.assertIn("Local 只找到 0 組相關原文，1 題改出 Basic", question_set.description)
        self.assertIn("Basic 3、DRIFT 1", question_set.description)
        self.assertNotIn("Global 題", question_set.description)

        log = self.service.question_design_log(PROJECT, question_set.question_set_id)
        self.assertEqual(log["quotas"], {"basic": 3, "local": 0, "global": 0, "drift": 1})
        self.assertEqual(log["questions"][-1]["question_id"], drift.question_id)
        self.assertEqual(len(log["questions"][-1]["reasoning_chain"]), 2)
        self.assertTrue(all("Bundles: " in prompt for prompt in self.client.prompts))

        exported = QuestionSetService.to_question_set_json(question_set.questions)["questions"]
        self.assertEqual(exported[-1]["design_strategy"], "drift")

    def test_pages_used_by_one_question_are_not_reused(self):
        # BR p.15 neighbours BR p.14 and p.16, so after a question on p.14–15 the p.16 page is free only
        # for one more question; with three Basic questions wanted, the third falls short.
        result = self.service.strategy_generator.generate(
            PROJECT, SourceSamplingService(self.store).scan(PROJECT, minimum_characters=1), 6,
            {"basic": 1}, self.connections.get_chat_model(),
        )
        page_sets = [
            {(item.document_name, page) for item in question.answer_source_evidence for page in item.pages}
            for question in result.questions
        ]
        self.assertGreater(len(page_sets), 1)
        for index, pages in enumerate(page_sets):
            for other in page_sets[index + 1:]:
                self.assertFalse(any(
                    left[0] == right[0] and abs(left[1] - right[1]) <= 1 for left in pages for right in other
                ), (pages, other))
        self.assertIn("Basic 完成", "；".join(result.notes))

    def test_nothing_generated_raises_with_the_reason(self):
        def skip_all(base_url, api_key, model, prompt):
            bundles = json.loads(prompt.split("Bundles: ", 1)[1])
            return {"questions": [{"bundle_id": bundle["bundle_id"], "skip": "no"} for bundle in bundles]}

        self.service.strategy_generator.client = skip_all
        with self.assertRaisesRegex(ProjectError, "未能生成任何有效題目"):
            self.service.generate_question_set(
                PROJECT, 1, False, self.connections.get_chat_model(), strategy_mix={"basic": 1},
            )

if __name__ == "__main__":
    unittest.main()
