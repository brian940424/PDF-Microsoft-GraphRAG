import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automotive_graphrag.query_translation import document_language_sample, needs_rewrite, text_script


class QueryTranslationTests(unittest.TestCase):
    def test_text_script_weighs_cjk_characters_against_latin_words(self) -> None:
        self.assertEqual(text_script("煞車系統排氣時，第一個要接上乙烯管的是哪一個卡鉗？"), "cjk")
        self.assertEqual(text_script("DTC B2562（LOW VOLTAGE）的偵測條件為何？"), "cjk")
        self.assertEqual(text_script("Connect a vinyl tube to rear right brake caliper bleeder valve."), "latin")
        self.assertIsNone(text_script("12.5 (0.49)"))

    def test_needs_rewrite_only_when_scripts_differ(self) -> None:
        self.assertTrue(needs_rewrite("煞車排氣順序為何？", "latin"))
        self.assertFalse(needs_rewrite("指紋如何註冊？", "cjk"))
        self.assertFalse(needs_rewrite("What is the brake fluid type?", "latin"))
        self.assertFalse(needs_rewrite("煞車排氣順序為何？", None))

    def test_document_language_sample_reads_indexed_text_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            texts = ["SECTION BR", "Connect a vinyl tube to rear right brake caliper bleeder valve. " * 5]
            pd.DataFrame({"text": texts}).to_parquet(root / "output" / "text_units.parquet")

            script, excerpt = document_language_sample(root)

            self.assertEqual(script, "latin")
            self.assertTrue(excerpt.startswith("Connect a vinyl tube"))
            self.assertLessEqual(len(excerpt), 400)
            self.assertEqual(document_language_sample(root / "missing"), (None, ""))


if __name__ == "__main__":
    unittest.main()
