import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automotive_graphrag.evidence import EvidenceService
from automotive_graphrag.projects import ProjectStore


class EvidenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        self.output = self.store.path_for("L33-SM3E") / "graphrag" / "output"
        self.output.mkdir()
        pd.DataFrame(
            [
                {"id": "full-tu-1", "human_readable_id": 7},
                {"id": "full-tu-2", "human_readable_id": 8},
            ]
        ).to_parquet(self.output / "text_units.parquet")
        rows = [
            {
                "text_unit_id": "full-tu-1",
                "chunk_id": "L33-SM3E-WW-p0025-b01",
                "project_id": "L33-SM3E",
                "section_id": "WW",
                "section_name": "Wiper & Washer",
                "document_id": "WW.pdf",
                "page": 25,
                "block_id": "b01",
                "text": "檢查雨刷保險絲",
            },
            {
                "text_unit_id": "full-tu-2",
                "chunk_id": "L33-SM3E-PG-p0042-b01",
                "project_id": "L33-SM3E",
                "section_id": "PG",
                "section_name": "Power Supply",
                "document_id": "PG.pdf",
                "page": 42,
                "block_id": "b01",
                "text": "檢查電源供應",
            },
        ]
        (self.output / "source_metadata.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_context_sources_resolve_to_ranked_pdf_evidence(self) -> None:
        context = {
            "sources": [
                {"id": "8", "text": "檢查電源供應", "in_context": True},
                {"id": "7", "text": "檢查雨刷保險絲", "in_context": True},
            ]
        }

        evidence = EvidenceService(self.store).from_context("L33-SM3E", context)

        self.assertEqual([item.evidence_id for item in evidence], ["E1", "E2"])
        self.assertEqual((evidence[0].document_id, evidence[0].page), ("PG.pdf", 42))
        self.assertEqual(evidence[1].chunk_id, "L33-SM3E-WW-p0025-b01")
        self.assertIsNone(evidence[0].score)

    def test_candidate_source_not_in_context_is_excluded(self) -> None:
        context = {
            "sources": [
                {"id": "7", "in_context": False},
                {"id": "8", "in_context": True},
            ]
        }

        evidence = EvidenceService(self.store).from_context("L33-SM3E", context)

        self.assertEqual([item.text_unit_id for item in evidence], ["full-tu-2"])


if __name__ == "__main__":
    unittest.main()
