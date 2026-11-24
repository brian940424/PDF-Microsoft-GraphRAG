import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.graph_extraction import (
    ENTITY_TYPES,
    EXTRACT_GRAPH_PROMPT,
    GRAPH_EXTRACTION_PROMPT_PATH,
    configure_graph_extraction,
)


class GraphExtractionTests(unittest.TestCase):
    def test_prompt_keeps_graphrag_placeholders_and_format(self) -> None:
        filled = EXTRACT_GRAPH_PROMPT.format(entity_types=",".join(ENTITY_TYPES), input_text="BRC > C1155")
        self.assertIn("Entity_types: COMPONENT,SYSTEM", filled)
        self.assertTrue(filled.rstrip().endswith("Output:"))
        for marker in ('("entity"<|>', '("relationship"<|>', "**##**", "<|COMPLETE|>"):
            self.assertIn(marker, EXTRACT_GRAPH_PROMPT)

    def test_prompt_asks_for_part_names_without_action_suffixes(self) -> None:
        self.assertIn("write STEERING SWITCH, not STEERING SWITCH REPLACEMENT", EXTRACT_GRAPH_PROMPT)
        self.assertIn("Every passage that contains manual content has at least one entity", EXTRACT_GRAPH_PROMPT)

    def test_configure_writes_prompt_and_overrides_default_entity_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"extract_graph": {"completion_model_id": "m", "entity_types": ["organization", "person"]}}

            configure_graph_extraction(settings, root)
            configure_graph_extraction(settings, root)

            section = settings["extract_graph"]
            self.assertEqual(section["completion_model_id"], "m")
            self.assertEqual(section["entity_types"], list(ENTITY_TYPES))
            self.assertEqual(section["max_gleanings"], 1)
            self.assertEqual(Path(section["prompt"]), (root / GRAPH_EXTRACTION_PROMPT_PATH).resolve())
            self.assertEqual(Path(section["prompt"]).read_text(encoding="utf-8"), EXTRACT_GRAPH_PROMPT)


if __name__ == "__main__":
    unittest.main()
