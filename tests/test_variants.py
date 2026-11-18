import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import pymupdf
from graphrag.query.context_builder.builders import ContextBuilderResult

from automotive_graphrag.evidence import section_path_with_variant
from automotive_graphrag.variants import (
    CACHE_FILE,
    VariantSource,
    assign_variants,
    earliest_equivalent_pages,
    header_tags,
    project_variants,
    question_variants,
    select_basic_context,
    select_sources,
    with_variant_prefix,
)

AV_VARIANTS = ["BASE AUDIO", "DISPLAY AUDIO WITH BOSE", "NAVIGATION WITHOUT BOSE", "NAVIGATION WITH BOSE"]
PRECAUTION = (
    "Precaution for power tools. When working near the airbag sensor unit with the ignition ON, "
    "do not use air or electric power tools or hammers. Turn the ignition OFF and disconnect the battery first."
)
OTHER = "Speaker channel check. Press and hold the preset button while turning the volume knob to select the channel."


def write_pdf(path: Path, pages: list[tuple[str, str, str]]) -> None:
    """pages: (chapter, variant, body); header lines at the top, body lower on the page."""
    document = pymupdf.open()
    for chapter, variant, body in pages:
        page = document.new_page(width=612, height=792)
        header = "\n".join(line for line in (f"< {chapter} >" if chapter else "", f"[{variant}]" if variant else "") if line)
        if header:
            page.insert_text((52, 30), header, fontsize=9)
        page.insert_textbox(pymupdf.Rect(52, 120, 560, 700), body, fontsize=10)
    document.save(path)
    document.close()


class HeaderTagTests(unittest.TestCase):
    def test_reads_variant_and_chapter_lines_only(self) -> None:
        self.assertEqual(header_tags("< PRECAUTION >\n[NAVIGATION WITH BOSE]\nPRECAUTIONS"), ("NAVIGATION WITH BOSE", "PRECAUTION"))
        self.assertEqual(header_tags("PRECAUTIONS\n[Data: Sources (1)]"), ("", ""))

    def test_untagged_page_inherits_only_within_the_same_chapter(self) -> None:
        headers = [("BASE AUDIO", "PRECAUTION"), ("", "PRECAUTION"), ("", "SYSTEM"), ("", "")]
        self.assertEqual(assign_variants(headers), ["BASE AUDIO", "BASE AUDIO", "", ""])


class QuestionVariantTests(unittest.TestCase):
    def test_full_name_wins_and_with_is_not_without(self) -> None:
        self.assertEqual(question_variants("AV control unit (navigation with BOSE)", AV_VARIANTS), {"NAVIGATION WITH BOSE"})
        self.assertEqual(question_variants("Navigation without BOSE unit", AV_VARIANTS), {"NAVIGATION WITHOUT BOSE"})

    def test_family_name_and_alias(self) -> None:
        both = {"NAVIGATION WITH BOSE", "NAVIGATION WITHOUT BOSE"}
        self.assertEqual(question_variants("How does the navigation system work?", AV_VARIANTS), both)
        self.assertEqual(question_variants("導航系統如何定位？", AV_VARIANTS), both)
        self.assertEqual(question_variants("How do I bleed the brakes?", AV_VARIANTS), set())


class SelectSourcesTests(unittest.TestCase):
    catalog = {
        "AV.pdf": {8: "BASE AUDIO", 119: "DISPLAY AUDIO WITH BOSE", 204: "NAVIGATION WITHOUT BOSE", 298: "NAVIGATION WITH BOSE"},
        "BCS.pdf": {63: "BCM", 79: "BCM"},
    }

    def source(self, document: str, page: int, text: str = OTHER) -> VariantSource:
        return VariantSource(f"{document}-{page}", document, page, self.catalog[document].get(page, ""), text)

    def test_named_variant_drops_other_versions_but_not_single_variant_documents(self) -> None:
        sources = [self.source("AV.pdf", 204), self.source("AV.pdf", 298, PRECAUTION), self.source("BCS.pdf", 79)]

        kept, report = select_sources(sources, "Compare the BCM with the AV control unit (navigation with BOSE)", self.catalog)

        self.assertEqual([item.key for item in kept], ["AV.pdf-298", "BCS.pdf-79"])
        self.assertEqual(report["named_variants"], {"AV.pdf": ["NAVIGATION WITH BOSE"]})
        self.assertEqual(report["dropped"][0]["page"], 204)

    def test_named_variant_without_retrieved_match_keeps_everything(self) -> None:
        sources = [self.source("AV.pdf", 8), self.source("AV.pdf", 204, PRECAUTION)]

        kept, report = select_sources(sources, "navigation with BOSE screen", self.catalog)

        self.assertEqual(len(kept), 2)
        self.assertEqual(report["named_variants"], {})

    def test_repeated_passage_keeps_the_first_chapter_only(self) -> None:
        sources = [
            self.source("AV.pdf", 204, PRECAUTION + " Refer to AV-204. INFOID:0000000008659102"),
            self.source("AV.pdf", 8, PRECAUTION + " Refer to AV-8. INFOID:0000000008571550"),
            self.source("AV.pdf", 119, OTHER),
        ]

        kept, report = select_sources(sources, "What must be done before using power tools?", self.catalog)

        self.assertEqual([item.page for item in kept], [8, 119])
        self.assertIn("same passage as AV.pdf p.8", report["dropped"][0]["reason"])

    def test_compare_text_is_preferred_over_chunk_text(self) -> None:
        first = VariantSource("a", "AV.pdf", 8, "BASE AUDIO", "figure described one way", PRECAUTION)
        later = VariantSource("b", "AV.pdf", 204, "NAVIGATION WITHOUT BOSE", "figure described another way", PRECAUTION)

        kept, _ = select_sources([later, first], "power tools", self.catalog)

        self.assertEqual([item.key for item in kept], ["a"])


class EarliestEquivalentTests(unittest.TestCase):
    catalog = {"AV.pdf": {8: "BASE AUDIO", 9: "BASE AUDIO", 57: "DISPLAY AUDIO", 205: "NAVIGATION WITH BOSE", 206: "NAVIGATION WITH BOSE"}}
    bodies = {"AV.pdf": {8: OTHER, 9: PRECAUTION, 57: PRECAUTION, 205: PRECAUTION, 206: OTHER + " navigation only"}}

    def test_first_matching_page_of_an_earlier_variant_wins(self) -> None:
        sources = [
            VariantSource("a", "AV.pdf", 205, "NAVIGATION WITH BOSE", "chunk"),
            VariantSource("b", "AV.pdf", 206, "NAVIGATION WITH BOSE", "chunk"),
            VariantSource("c", "AV.pdf", 9, "BASE AUDIO", "chunk"),
        ]

        self.assertEqual(earliest_equivalent_pages(sources, self.catalog, self.bodies), {"a": 9})

    def test_named_variants_limit_the_candidates(self) -> None:
        sources = [VariantSource("a", "AV.pdf", 205, "NAVIGATION WITH BOSE", "chunk")]

        self.assertEqual(earliest_equivalent_pages(sources, self.catalog, self.bodies, {"AV.pdf": ["DISPLAY AUDIO"]}), {"a": 57})
        self.assertEqual(earliest_equivalent_pages(sources, self.catalog, self.bodies, {"AV.pdf": ["NAVIGATION WITH BOSE"]}), {})


class DisplayTests(unittest.TestCase):
    def test_prefix_and_section_path(self) -> None:
        self.assertEqual(with_variant_prefix("Text", "BCM"), "[BCM] Text")
        self.assertEqual(with_variant_prefix("[BCM] Text", "BCM"), "[BCM] Text")
        self.assertEqual(with_variant_prefix("Text", ""), "Text")
        self.assertEqual(section_path_with_variant("AV > Removal", "BASE AUDIO"), "AV > [BASE AUDIO] > Removal")
        self.assertEqual(section_path_with_variant("AV", "BASE AUDIO"), "AV > [BASE AUDIO]")
        self.assertEqual(section_path_with_variant("BR", ""), "BR")


class ProjectIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.project = Path(self.directory.name) / "project"
        (self.project / "source").mkdir(parents=True)
        self.output = self.project / "graphrag" / "output"
        self.output.mkdir(parents=True)
        write_pdf(self.project / "source" / "AV.pdf", [
            ("", "", "Contents"),
            ("PRECAUTION", "BASE AUDIO", PRECAUTION),
            ("PRECAUTION", "", OTHER),
            ("PRECAUTION", "NAVIGATION WITH BOSE", PRECAUTION),
            ("SYSTEM", "NAVIGATION WITH BOSE", OTHER + " Navigation map matching."),
        ])
        write_pdf(self.project / "source" / "BR.pdf", [("", "", "Bleeding brake system")])
        units = [("u1", "1", "AV.pdf", 2), ("u2", "2", "AV.pdf", 4), ("u3", "3", "AV.pdf", 5), ("u4", "4", "BR.pdf", 1)]
        pd.DataFrame({"id": [u[0] for u in units], "human_readable_id": [u[1] for u in units]}).to_parquet(
            self.output / "text_units.parquet"
        )
        (self.output / "source_metadata.jsonl").write_text("".join(
            json.dumps({"text_unit_id": u[0], "document_id": u[2], "page": u[3]}) + "\n" for u in units
        ), encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_catalog_reads_headers_inherits_within_chapter_and_is_cached(self) -> None:
        catalog = project_variants(self.project, self.output)

        self.assertEqual(catalog["AV.pdf"], {2: "BASE AUDIO", 3: "BASE AUDIO", 4: "NAVIGATION WITH BOSE", 5: "NAVIGATION WITH BOSE"})
        self.assertEqual(catalog["BR.pdf"], {})
        self.assertTrue((self.output / CACHE_FILE).is_file())
        self.assertEqual(project_variants(self.project, self.output), catalog)

    def test_basic_context_drops_repeated_chapter_and_tags_sources(self) -> None:
        frame = pd.DataFrame({"id": ["2", "1", "3", "4"], "text": ["nav precaution", "base precaution", "nav system", "brakes"]})
        result = ContextBuilderResult(context_chunks="old", context_records={"sources": frame})

        selected, report = select_basic_context(result, "What precautions apply to power tools?", self.project / "graphrag")

        kept = selected.context_records["sources"]
        self.assertEqual(list(kept["id"]), ["1", "3", "4"])
        self.assertEqual(list(kept["text"]), ["[BASE AUDIO] base precaution", "[NAVIGATION WITH BOSE] nav system", "brakes"])
        self.assertIn("[BASE AUDIO] base precaution", selected.context_chunks)
        self.assertEqual([item["page"] for item in report["dropped"]], [4])

    def test_later_chapter_is_replaced_by_the_first_chapter_page(self) -> None:
        pd.DataFrame({"id": ["u1", "u2", "u3", "u4"], "human_readable_id": ["1", "2", "3", "4"],
                      "text": ["base precaution", "nav precaution", "nav system", "brakes"]}).to_parquet(
            self.output / "text_units.parquet"
        )
        frame = pd.DataFrame({"id": ["2", "4"], "text": ["nav precaution", "brakes"]})
        result = ContextBuilderResult(context_chunks="old", context_records={"sources": frame})

        selected, report = select_basic_context(result, "power tool precautions", self.project / "graphrag")

        kept = selected.context_records["sources"]
        self.assertEqual(list(kept["id"]), ["1", "4"])
        self.assertEqual(list(kept["text"]), ["[BASE AUDIO] base precaution", "brakes"])
        self.assertEqual(report["replaced"][0]["replaced_by_page"], 2)

        named, named_report = select_basic_context(result, "navigation with BOSE precautions", self.project / "graphrag")
        self.assertEqual(list(named.context_records["sources"]["id"]), ["2", "4"])
        self.assertEqual(named_report["replaced"], [])

    def test_basic_context_unchanged_without_multi_variant_documents(self) -> None:
        (self.project / "source" / "AV.pdf").unlink()
        result = ContextBuilderResult(context_chunks="old", context_records={"sources": pd.DataFrame({"id": ["4"], "text": ["x"]})})

        selected, report = select_basic_context(result, "q", self.project / "graphrag")

        self.assertIs(selected, result)
        self.assertEqual(report, {})


if __name__ == "__main__":
    unittest.main()
