import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.answer_prompts import (
    ANSWER_PROMPT_DIRECTORY,
    ANSWER_REQUIREMENTS,
    GLOBAL_ANSWER_REQUIREMENTS,
    INSUFFICIENT_DATA_PHRASE,
    answer_prompts,
    install_answer_prompts,
    with_answer_requirements,
)


class AnswerPromptTests(unittest.TestCase):
    def test_requirements_are_inserted_before_the_last_format_section(self) -> None:
        prompt = "A\n---Target response length and format---\nx\n---Data tables---\n---Target response length and format---\n{response_type}"

        result = with_answer_requirements(prompt)

        self.assertEqual(result.count(ANSWER_REQUIREMENTS), 1)
        self.assertLess(result.index("---Data tables---"), result.index("---Answer requirements---"))
        self.assertTrue(result.endswith("---Target response length and format---\n{response_type}"))
        with self.assertRaises(ValueError):
            with_answer_requirements("no format section")

    def test_all_answer_prompts_keep_placeholders_and_cover_f43_weaknesses(self) -> None:
        prompts = answer_prompts()

        self.assertEqual(set(prompts), {
            "local_search_system_prompt.txt", "basic_search_system_prompt.txt",
            "drift_search_system_prompt.txt", "drift_search_reduce_prompt.txt",
            "global_search_map_system_prompt.txt", "global_search_reduce_system_prompt.txt",
        })
        for name, text in prompts.items():
            if name.startswith("global_search"):
                continue
            with self.subTest(name=name):
                self.assertIn("{context_data}", text)
                self.assertIn("{response_type}", text)
                self.assertIn("---Answer requirements---", text)
        for phrase in (
            "similarly named functions", "include every item of that list", '"To ..."',
            "Do not add anything the data does not state", "do not soften it", "different device models",
            "single section that directly answers the question", "Preserve the source order",
            "Printed page numbers inside source text are content",
            # v3 (F-43): matching conditions, required notes kept, unrelated notes dropped, bare specs
            "Match the conditions in the question", "Include every step, condition and note that is required",
            "Leave out every such item that is not part of the answer", "give only the requested values with their units",
            # v4 (F-55): every retrieved source is checked, multi-page evidence combined, comparisons sorted
            "go through the sources in the data one by one", "Never answer from only the first-ranked source",
            "what all objects share and what is specific to each object", "Cite every source whose content you use",
            # v5 (F-56): qualifiers inside a used sentence kept, separate unrelated notes still dropped
            "Keep the qualifiers inside each sentence or step you use", "keeping only the main action and the numbers is not enough",
            "state that instruction and its conditions together with the fact", "Final check before you answer",
            "qualifiers inside the same sentence or step as the asked content are kept",
            # v7 (F-60): sentence-level citations and the source of the question's situation
            "Cite sentence by sentence", "cite the source that establishes the situation in the question",
            # v8 (B-14): no general knowledge, even where the GraphRAG base prompt invites it
            "replace any earlier instruction to incorporate general knowledge", 'never write a "General Knowledge" citation',
        ):
            self.assertIn(phrase, ANSWER_REQUIREMENTS)

    def test_local_prompt_never_claims_the_manual_lacks_a_point(self) -> None:
        prompts = answer_prompts()
        local = prompts["local_search_system_prompt.txt"]

        self.assertIn(INSUFFICIENT_DATA_PHRASE, local)
        self.assertIn("never write that the manual, the data or the sources do not state", local)
        self.assertNotIn("say only that the provided data does not state it", local)
        for name in ("basic_search_system_prompt.txt", "drift_search_system_prompt.txt"):
            self.assertIn("say only that the provided data does not state it", prompts[name])
            self.assertNotIn(INSUFFICIENT_DATA_PHRASE, prompts[name])

    def test_global_prompts_name_manuals_and_drop_hedging(self) -> None:
        prompts = answer_prompts()
        map_prompt = prompts["global_search_map_system_prompt.txt"]
        reduce_prompt = prompts["global_search_reduce_system_prompt.txt"]

        for text in (map_prompt, reduce_prompt):
            self.assertEqual(text.count(GLOBAL_ANSWER_REQUIREMENTS), 1)
            self.assertIn("name the manuals from those lines", text)
            self.assertIn(INSUFFICIENT_DATA_PHRASE, text)
            self.assertIn("do not add remarks about what is unconfirmed", text)
        self.assertNotIn("just say so", reduce_prompt)
        # Both are str.format templates in GraphRAG; the inserted text must not break them.
        self.assertIn("ctx", map_prompt.format(context_data="ctx", max_length=10))
        self.assertLess(map_prompt.index("---Answer requirements---"), map_prompt.index("---Data tables---"))
        formatted = reduce_prompt.format(report_data="rep", response_type="Single Paragraph", max_length=10)
        self.assertLess(formatted.index("---Answer requirements---"), formatted.rindex("Single Paragraph"))

    def test_install_writes_files_and_points_settings_to_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"local_search": {"prompt": "prompts/local_search_system_prompt.txt", "k": 3}}

            install_answer_prompts(settings, root)
            install_answer_prompts(settings, root)

            self.assertEqual(settings["local_search"]["k"], 3)
            for section, key in (
                ("local_search", "prompt"), ("basic_search", "prompt"),
                ("drift_search", "prompt"), ("drift_search", "reduce_prompt"),
                ("global_search", "map_prompt"), ("global_search", "reduce_prompt"),
            ):
                path = Path(settings[section][key])
                self.assertTrue(path.is_absolute())
                self.assertEqual(path.parent, (root / ANSWER_PROMPT_DIRECTORY).resolve())
                self.assertIn("---Answer requirements---", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
