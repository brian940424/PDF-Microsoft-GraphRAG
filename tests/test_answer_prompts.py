import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.answer_prompts import (
    ANSWER_PROMPT_DIRECTORY,
    ANSWER_REQUIREMENTS,
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
        })
        for name, text in prompts.items():
            with self.subTest(name=name):
                self.assertIn("{context_data}", text)
                self.assertIn("{response_type}", text)
                self.assertIn("---Answer requirements---", text)
        for phrase in (
            "similarly named functions", "include every item of that list", '"To ..."',
            "Do not add anything the data does not state", "do not soften it", "different device models",
            "single section that directly answers the question", "Preserve the source order",
            "Printed page numbers inside source text are content",
            "Answer only the requested parts", "immediate first action",
            "Do not add relevant but unrequested details",
            "Treat each clause of the question as the scope boundary",
            "Ground every claim in the supplied context",
            "only when they answer a requested part",
        ):
            self.assertIn(phrase, ANSWER_REQUIREMENTS)

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
            ):
                path = Path(settings[section][key])
                self.assertTrue(path.is_absolute())
                self.assertEqual(path.parent, (root / ANSWER_PROMPT_DIRECTORY).resolve())
                self.assertIn("---Answer requirements---", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
