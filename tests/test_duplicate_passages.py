import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pymupdf

from automotive_graphrag.duplicate_passages import (
    ADDED_BY,
    CACHE_FILE,
    PageEquivalence,
    PassageGroup,
    expansion_locations,
    group_passages,
    project_passage_groups,
    used_groups,
    with_duplicate_citations,
)
from automotive_graphrag.evaluation.source_scoring import cited_evidence, source_score
from automotive_graphrag.evidence import Evidence

WARNING = (
    "WARNING: When working near the Airbag Diagnosis Sensor Unit with the ignition ON, do not use air or "
    "electric power tools or strike near the sensor with a hammer. When using power tools or hammers, always "
    "switch the ignition OFF, disconnect the battery and wait at least three minutes before performing any service."
)
WARNING_VARIANT = WARNING.replace("three", "3") + " Precaution for Procedure without Cowl Top Cover"
CAUTION = (
    "CAUTION: After disconnecting the CONSULT vehicle interface from the data link connector, the ignition must "
    "be cycled OFF and then ON again, otherwise the vehicle systems may not operate normally after the work."
)
UNIQUE = (
    "Solder the repaired parts and wrap them with tape. Frays of the twisted line must be within 110 mm. "
    "Do not perform bypass wire connections for the repair parts of the communication system harness."
)


def gold(document: str, pages: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(document_id=document, pages=pages)


def evidence(rank: int, document: str, page: int, context_id: str, **changes) -> Evidence:
    values = dict(
        evidence_id=f"E{rank}", rank=rank, context_id=context_id, text_unit_id=f"t{context_id}", chunk_id="c",
        section_id="s", section_name="s", document_id=document, page=page, block_id="b01", text="text",
    )
    values.update(changes)
    return Evidence(**values)


class GroupPassagesTests(unittest.TestCase):
    def test_exact_and_near_repeats_group_but_unique_and_short_blocks_do_not(self) -> None:
        groups = group_passages([
            ("ACC.pdf", 2, WARNING), ("BR.pdf", 3, WARNING_VARIANT), ("BCS.pdf", 3, WARNING),
            ("BCS.pdf", 14, CAUTION), ("BCS.pdf", 22, CAUTION),
            ("AV.pdf", 8, UNIQUE), ("AV.pdf", 9, "Short heading"), ("AV.pdf", 10, "Short heading"),
        ])

        self.assertEqual([group.locations for group in groups], [
            (("ACC.pdf", 2), ("BCS.pdf", 3), ("BR.pdf", 3)),
            (("BCS.pdf", 14), ("BCS.pdf", 22)),
        ])

    def test_passage_inside_a_much_larger_block_is_not_a_repeat(self) -> None:
        large = WARNING + " " + UNIQUE + " " + CAUTION
        self.assertEqual(group_passages([("A.pdf", 1, WARNING), ("B.pdf", 1, large)]), [])

    def test_same_page_twice_is_not_a_repeat(self) -> None:
        self.assertEqual(group_passages([("A.pdf", 1, WARNING), ("A.pdf", 1, WARNING)]), [])


class UsedGroupsTests(unittest.TestCase):
    groups = [
        PassageGroup((("ACC.pdf", 2), ("AV.pdf", 8), ("BR.pdf", 3)), WARNING),
        PassageGroup((("AV.pdf", 8), ("AV.pdf", 56)), CAUTION),
    ]

    def test_only_passages_an_answer_sentence_is_built_on(self) -> None:
        answer = "Switch the ignition OFF, disconnect the battery and wait at least 3 minutes before using power tools. [Data: Sources (12)]"

        self.assertEqual(used_groups(answer, {("AV.pdf", 8)}, self.groups), [self.groups[0]])

    def test_uncited_page_or_unrelated_or_other_language_answer_adds_nothing(self) -> None:
        self.assertEqual(used_groups("Switch the ignition OFF and disconnect the battery before power tools.", {("BR.pdf", 9)}, self.groups), [])
        self.assertEqual(used_groups("Keep frays of the twisted line within 110 mm and solder the joint.", {("AV.pdf", 8)}, self.groups), [])
        self.assertEqual(used_groups("先將點火開關轉至 OFF，拆下電瓶並等待三分鐘。", {("AV.pdf", 8)}, self.groups), [])


class ExpansionLocationTests(unittest.TestCase):
    variants = {"AV.pdf": {8: "BASE AUDIO", 9: "BASE AUDIO", 56: "DISPLAY AUDIO", 204: "NAVIGATION", 205: "NAVIGATION"}}

    def test_all_documents_but_only_the_first_variant_chapter(self) -> None:
        group = PassageGroup((("ACC.pdf", 2), ("AV.pdf", 9), ("AV.pdf", 56), ("AV.pdf", 204), ("BR.pdf", 3), ("BR.pdf", 7)), WARNING)

        self.assertEqual(expansion_locations([group], {("AV.pdf", 204)}, self.variants), [
            ("ACC.pdf", 2), ("AV.pdf", 9), ("BR.pdf", 3), ("BR.pdf", 7),
        ])

    def test_named_variant_keeps_its_chapter(self) -> None:
        group = PassageGroup((("AV.pdf", 9), ("AV.pdf", 204), ("AV.pdf", 205)), WARNING)

        self.assertEqual(
            expansion_locations([group], {("AV.pdf", 204)}, self.variants, {"AV.pdf": ["NAVIGATION"]}),
            [("AV.pdf", 205)],
        )


class SourceScoringTests(unittest.TestCase):
    def test_code_added_evidence_counts_as_cited(self) -> None:
        items = [evidence(1, "ACC.pdf", 2, "1"), evidence(2, "BR.pdf", 3, "5", added_by=ADDED_BY), evidence(3, "BR.pdf", 9, "6")]

        self.assertEqual([item.page for item in cited_evidence("Done. [Data: Sources (1)]", items)], [2, 3])

    def test_equivalent_pages_hit_gold(self) -> None:
        items = [evidence(1, "AV.pdf", 209, "214")]
        same = lambda document, first, second: {first, second} == {209, 303}

        self.assertEqual(source_score("x [Data: Sources (214)]", items, [gold("AV.pdf", (303,))], 0.0), 0.0)
        self.assertEqual(source_score("x [Data: Sources (214)]", items, [gold("AV.pdf", (303,))], 0.0, same), 1.0)


class ProjectIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.project = root / "P1"
        (self.project / "source").mkdir(parents=True)
        self.output = self.project / "graphrag" / "output"
        self.output.mkdir(parents=True)
        for name, bodies in {"ACC.pdf": [WARNING], "BR.pdf": [UNIQUE, CAUTION, WARNING_VARIANT], "AV.pdf": [UNIQUE + " Navigation.", UNIQUE + " Navigation."]}.items():
            document = pymupdf.open()
            for body in bodies:
                page = document.new_page(width=612, height=792)
                page.insert_textbox(pymupdf.Rect(52, 120, 560, 700), body, fontsize=10)
            document.save(self.project / "source" / name)
            document.close()
        units = [("t1", "1", "ACC.pdf", 1), ("t2", "2", "BR.pdf", 2), ("t3", "3", "BR.pdf", 3), ("t4", "4", "AV.pdf", 1)]
        pd.DataFrame({"id": [u[0] for u in units], "human_readable_id": [u[1] for u in units]}).to_parquet(self.output / "text_units.parquet")
        (self.output / "source_metadata.jsonl").write_text("".join(json.dumps({
            "text_unit_id": u[0], "chunk_id": f"c{u[1]}", "project_id": "P1", "section_id": "s", "section_name": "S",
            "document_id": u[2], "page": u[3], "block_id": "b01", "text": WARNING if u[0] in {"t1", "t3"} else "other",
            "section_path": u[2].removesuffix(".pdf"),
        }) + "\n" for u in units), encoding="utf-8")
        self.projects = SimpleNamespace(path_for=lambda project_id: root / project_id)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_groups_are_cached(self) -> None:
        groups = project_passage_groups(self.project, self.output)

        self.assertEqual([group.locations for group in groups], [
            (("ACC.pdf", 1), ("BR.pdf", 3)),
            (("AV.pdf", 1), ("AV.pdf", 2), ("BR.pdf", 1)),
        ])
        self.assertTrue((self.output / CACHE_FILE).is_file())
        self.assertEqual(project_passage_groups(self.project, self.output), groups)

    def test_cited_repeat_adds_the_other_page_with_its_text_unit(self) -> None:
        from automotive_graphrag.source_metadata import SourceMetadataService

        answer = "Switch the ignition OFF, disconnect the battery and wait at least 3 minutes before using power tools. [Data: Sources (1)]"
        original = (evidence(1, "ACC.pdf", 1, "1"),)
        projects = SimpleNamespace(path_for=self.projects.path_for)
        result, added = with_duplicate_citations(projects, "P1", answer, original, {})

        self.assertEqual(added, [{"document_id": "BR.pdf", "page": 3, "retrieved": False}])
        self.assertEqual((result[1].document_id, result[1].page, result[1].context_id, result[1].added_by), ("BR.pdf", 3, "3", ADDED_BY))
        self.assertEqual(source_score(answer, result, [gold("ACC.pdf", (1,)), gold("BR.pdf", (3,))]), 1.0)

    def test_retrieved_but_uncited_repeat_is_marked(self) -> None:
        answer = "Switch the ignition OFF, disconnect the battery and wait at least 3 minutes before using power tools. [Data: Sources (1)]"
        original = (evidence(1, "ACC.pdf", 1, "1"), evidence(2, "BR.pdf", 3, "3"))

        result, added = with_duplicate_citations(self.projects, "P1", answer, original, {})

        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].added_by, ADDED_BY)
        self.assertEqual(added, [{"document_id": "BR.pdf", "page": 3, "retrieved": True}])

    def test_page_equivalence_uses_the_pdf_text_layer(self) -> None:
        same = PageEquivalence(self.project)

        self.assertTrue(same("AV.pdf", 1, 2))
        self.assertFalse(same("BR.pdf", 1, 2))
        self.assertFalse(same("MISSING.pdf", 1, 2))


if __name__ == "__main__":
    unittest.main()
