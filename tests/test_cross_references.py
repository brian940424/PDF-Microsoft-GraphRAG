import tempfile
import unittest
from pathlib import Path

import numpy as np

from automotive_graphrag.cross_references import (
    page_label_matches,
    referenced_pages,
    select_referenced_units,
)


def unit(angle: float) -> np.ndarray:
    """A 2-D unit vector whose cosine with the query (1, 0) is cos(angle)."""
    return np.array([np.cos(angle), np.sin(angle)])


def with_cosine(value: float) -> np.ndarray:
    return unit(float(np.arccos(value)))


QUERY = np.array([1.0, 0.0])


class ReferencedPagesTests(unittest.TestCase):
    def test_finds_references_to_known_documents_in_order(self) -> None:
        texts = [
            'Replace BCM. Refer to BCS-79, "Removal and Installation". Refer to XYZ-3.',
            "Remove the accelerator pedal. Refer to ACC-3. Refer to BCS-79.",
        ]

        pages = referenced_pages(texts, {"BCS.pdf", "ACC.pdf"})

        self.assertEqual(pages, [("BCS.pdf", 79), ("ACC.pdf", 3)])


class SelectReferencedUnitsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.locations = {
            "top1": ("BCS.pdf", 63), "top2": ("BCS.pdf", 64),
            "p79a": ("BCS.pdf", 79), "p79b": ("BCS.pdf", 79),
            "acc3": ("ACC.pdf", 3), "br23": ("BR.pdf", 23), "far": ("AV.pdf", 284),
        }
        self.vectors = {
            "p79a": with_cosine(0.596), "p79b": with_cosine(0.40),
            "acc3": with_cosine(0.58), "br23": with_cosine(0.59), "far": with_cosine(0.50),
        }
        self.top = {"top1": with_cosine(0.70), "top2": with_cosine(0.60)}

    def select(self, texts, label_matches=lambda *_: True, **kwargs):
        return select_referenced_units(
            QUERY, self.top, texts, self.locations, self.vectors.get, label_matches, **kwargs,
        )

    def test_adds_close_referenced_page_with_only_its_close_passages(self) -> None:
        selected = self.select(["Replace BCM. Refer to BCS-79."])

        self.assertEqual([(item["document_id"], item["page"]) for item in selected], [("BCS.pdf", 79)])
        self.assertEqual(selected[0]["units"], ["p79a"])
        self.assertAlmostEqual(selected[0]["gap"], 0.004, places=3)

    def test_skips_pages_beyond_margin_already_retrieved_or_unlabeled(self) -> None:
        texts = ["Refer to AV-284. Refer to BCS-64. Refer to BCS-79."]

        self.assertEqual(self.select(texts, label_matches=lambda *_: False), [])
        selected = self.select(texts)
        self.assertEqual([item["page"] for item in selected], [79])

    def test_keeps_closest_pages_up_to_the_limit(self) -> None:
        texts = ["Refer to ACC-3. Refer to BR-23. Refer to BCS-79."]

        selected = self.select(texts)

        self.assertEqual([item["page"] for item in selected], [79, 23])
        self.assertEqual([item["page"] for item in self.select(texts, max_pages=3)], [79, 23, 3])

    def test_no_retrieved_results_selects_nothing(self) -> None:
        self.top = {}

        self.assertEqual(self.select(["Refer to BCS-79."]), [])


class PageLabelTests(unittest.TestCase):
    def test_reads_the_printed_label_from_the_footer(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "BCS.pdf"
            document = pymupdf.open()
            for number in (1, 2):
                page = document.new_page(width=600, height=800)
                page.insert_text((50, 100), "Refer to BCS-1 in the body text.")
                page.insert_text((280, 785), f"BCS-{number + 10 if number == 2 else number}")
            document.save(path)
            document.close()

            self.assertTrue(page_label_matches(path, 1, "BCS-1"))
            self.assertFalse(page_label_matches(path, 2, "BCS-2"))
            self.assertFalse(page_label_matches(path, 2, "BCS-1"))
            self.assertFalse(page_label_matches(path, 3, "BCS-3"))
            self.assertFalse(page_label_matches(Path(directory) / "missing.pdf", 1, "BCS-1"))


if __name__ == "__main__":
    unittest.main()
