import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automotive_graphrag.global_reports import (
    cited_report_ids,
    community_manual_pages,
    keep_cited_reports,
    label_reports,
    manuals_line,
)


class GlobalReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output = Path(self.temporary_directory.name)
        rows = [
            {"text_unit_id": "tu-1", "document_id": "BR.pdf", "page": 3},
            {"text_unit_id": "tu-2", "document_id": "BR.pdf", "page": 4, "pages": [4, 5]},
            {"text_unit_id": "tu-3", "document_id": "BRC.pdf", "page": 56},
            {"text_unit_id": "tu-4", "document_id": "BRC.pdf", "page": 57},
        ]
        (self.output / "source_metadata.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
        self.communities = pd.DataFrame([
            {"community": 1, "text_unit_ids": ["tu-1", "tu-2", "tu-3"]},
            {"community": 3, "text_unit_ids": ["tu-1", "tu-3", "tu-4"]},
            {"community": 2, "text_unit_ids": ["missing"]},
        ])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_communities_map_to_manual_pages(self) -> None:
        pages = community_manual_pages(self.output, self.communities)

        self.assertEqual(pages["1"], {"BR.pdf": (3, 4, 5), "BRC.pdf": (56,)})
        self.assertEqual(pages["2"], {})
        # The manual holding most of the community's text units comes first.
        self.assertEqual(list(pages["3"]), ["BRC.pdf", "BR.pdf"])
        self.assertEqual(manuals_line(pages["1"]), "Manuals: BR (pages 3–5); BRC (pages 56)")
        self.assertEqual(manuals_line({}), "")

    def test_many_page_ranges_collapse_to_the_span(self) -> None:
        line = manuals_line({"AV.pdf": (1, 3, 5, 7, 9, 11, 13)})

        self.assertEqual(line, "Manuals: AV (pages 1–13)")

    def test_labels_copy_reports_without_touching_the_cached_frame(self) -> None:
        reports = pd.DataFrame([
            {"community": 1, "full_content": "# Brakes", "summary": "Brake fluid"},
            {"community": 2, "full_content": "# Other", "summary": None},
        ])

        labelled = label_reports(reports, self.communities, self.output)

        self.assertEqual(labelled["full_content"][0], "Manuals: BR (pages 3–5); BRC (pages 56)\n\n# Brakes")
        self.assertTrue(labelled["summary"][0].startswith("Manuals: BR"))
        self.assertEqual(labelled["full_content"][1], "# Other")
        self.assertEqual(reports["full_content"][0], "# Brakes")

    def test_only_cited_reports_are_kept_with_their_pages(self) -> None:
        context = {"reports": [{"id": str(index), "title": f"R{index}"} for index in range(5)]}
        answer = "BR covers it [Data: Reports (3, 1, +more)]. Also [Data: Reports (1, 9)]."
        pages = {"1": {"BR.pdf": (3, 4)}, "3": {}}

        self.assertEqual(cited_report_ids(answer), ["3", "1", "9"])
        result = keep_cited_reports(context, answer, pages)

        self.assertEqual([row["id"] for row in result["reports"]], ["3", "1"])
        self.assertEqual(result["reports"][1]["manual_pages"], {"BR.pdf": [3, 4]})
        self.assertEqual(result["reports_in_context_count"], 5)


if __name__ == "__main__":
    unittest.main()
