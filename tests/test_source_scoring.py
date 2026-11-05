import unittest

from automotive_graphrag.evaluation.source_scoring import (
    cited_evidence,
    cited_source_ids,
    combined_score,
    source_score,
    validate_source_weight,
)
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.question_sets import GoldEvidence


def evidence(context_id: str, page: int | None, document_id: str = "A.pdf", rank: int = 1,
             source_type: str = "text_unit") -> Evidence:
    return Evidence(
        evidence_id=f"E{rank}", rank=rank, context_id=context_id, text_unit_id=f"tu-{context_id}",
        chunk_id="", section_id="", section_name="", document_id=document_id, page=page,
        block_id="", text="", source_type=source_type,
    )


def gold(pages: tuple[int, ...], document_id: str = "A.pdf") -> GoldEvidence:
    return GoldEvidence(document_id=document_id, pages=pages, chunk_ids=())


class CitationParsingTests(unittest.TestCase):
    def test_parses_graphrag_sources_and_agentic_ids(self) -> None:
        answer = (
            "步驟一 [Data: Sources (12, 34, +more); Entities (5)]。"
            "步驟二 [Data: Reports (7); Sources (56)]。另見 [E3]。"
        )
        sources, evidence_ids = cited_source_ids(answer)
        self.assertEqual(sources, {"12", "34", "56"})
        self.assertEqual(evidence_ids, {"E3"})

    def test_entity_and_report_ids_are_not_sources(self) -> None:
        sources, _ = cited_source_ids("[Data: Entities (1, 2); Reports (3)]")
        self.assertEqual(sources, set())

    def test_cited_evidence_keeps_only_cited_text_units(self) -> None:
        retrieved = [evidence("1", 10, rank=1), evidence("2", 11, rank=2), evidence("3", None, rank=3)]
        cited = cited_evidence("[Data: Sources (2)] 與 [E3]", retrieved)
        self.assertEqual([item.context_id for item in cited], ["2", "3"])


class SourceScoreTests(unittest.TestCase):
    def test_not_applicable_without_gold_or_text_units(self) -> None:
        self.assertIsNone(source_score("[Data: Sources (1)]", [evidence("1", 10)], []))
        report = evidence("9", None, "GraphRAG Community Report", source_type="community_report")
        self.assertIsNone(source_score("[Data: Reports (9)]", [report], [gold((10,))]))

    def test_exact_and_adjacent_page_hits(self) -> None:
        retrieved = [evidence("1", 10)]
        self.assertEqual(source_score("[Data: Sources (1)]", retrieved, [gold((10,))]), 1.0)
        self.assertEqual(source_score("[Data: Sources (1)]", retrieved, [gold((11,))]), 1.0)
        self.assertEqual(source_score("[Data: Sources (1)]", retrieved, [gold((12,))]), 0.5)

    def test_page_recall_over_multiple_gold_pages(self) -> None:
        retrieved = [evidence("1", 10), evidence("2", 40, rank=2)]
        self.assertAlmostEqual(
            source_score("[Data: Sources (1)]", retrieved, [gold((10, 20, 30))]), 0.3333
        )

    def test_wrong_document_or_no_citation_scores_zero(self) -> None:
        retrieved = [evidence("1", 10, "B.pdf"), evidence("2", 10, rank=2)]
        self.assertEqual(source_score("[Data: Sources (1)]", retrieved, [gold((10,))]), 0.0)
        self.assertEqual(source_score("沒有引用", retrieved, [gold((10,))]), 0.0)
        # A cited id missing from the retrieved context cannot be resolved.
        self.assertEqual(source_score("[Data: Sources (99)]", retrieved, [gold((10,))]), 0.0)

    def test_document_only_gold(self) -> None:
        retrieved = [evidence("1", 10)]
        self.assertEqual(source_score("[Data: Sources (1)]", retrieved, [gold(())]), 1.0)
        self.assertEqual(source_score("[Data: Sources (1)]", retrieved, [gold((), "B.pdf")]), 0.0)


class CombinedScoreTests(unittest.TestCase):
    def test_weighted_mix_and_fallback(self) -> None:
        self.assertEqual(combined_score(1.0, 0.0), 0.9)
        self.assertEqual(combined_score(0.0, 1.0), 0.1)
        self.assertEqual(combined_score(0.5, 1.0, 0.2), 0.6)
        self.assertEqual(combined_score(0.75, None), 0.75)
        self.assertIsNone(combined_score(None, 1.0))

    def test_weight_validation(self) -> None:
        self.assertEqual(validate_source_weight("0.2"), 0.2)
        for bad in (-0.1, 1.5, "x"):
            with self.assertRaises(ValueError):
                validate_source_weight(bad)


if __name__ == "__main__":
    unittest.main()
