import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.source_metadata import SourceMetadataService


class SourceMetadataServiceTests(unittest.TestCase):
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
        self.service = SourceMetadataService(self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_parquets(self, raw_data: dict[str, object] | None = None) -> None:
        source = raw_data or {
            "chunk_id": "L33-SM3E-WW-p0025-b01",
            "project_id": "L33-SM3E",
            "section_id": "WW",
            "section_name": "WW",
            "document_id": "WW.pdf",
            "page": 25,
            "block_id": "b01",
        }
        pd.DataFrame([{"id": "page-document-id", "raw_data": source}]).to_parquet(
            self.output / "documents.parquet"
        )
        pd.DataFrame(
            [
                {
                    "id": "graph-text-unit-id",
                    "document_id": "page-document-id",
                    "text": "雨刷保險絲檢查內容",
                }
            ]
        ).to_parquet(self.output / "text_units.parquet")

    def test_build_maps_text_unit_to_original_pdf_metadata(self) -> None:
        self.write_parquets()

        mappings = self.service.build("L33-SM3E")

        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0].text_unit_id, "graph-text-unit-id")
        self.assertEqual(mappings[0].chunk_id, "L33-SM3E-WW-p0025-b01")
        self.assertEqual((mappings[0].document_id, mappings[0].page), ("WW.pdf", 25))
        self.assertEqual(self.service.load("L33-SM3E"), mappings)
        saved = json.loads((self.output / "source_metadata.jsonl").read_text().strip())
        self.assertEqual(saved["section_id"], "WW")

    def test_build_rejects_text_unit_without_source_metadata(self) -> None:
        self.write_parquets(raw_data={"document_id": "WW.pdf"})

        with self.assertRaisesRegex(ProjectError, "無法回連"):
            self.service.build("L33-SM3E")

        self.assertFalse((self.output / "source_metadata.jsonl").exists())

    def test_build_requires_graphrag_source_tables(self) -> None:
        with self.assertRaisesRegex(ProjectError, "缺少 documents"):
            self.service.build("L33-SM3E")


if __name__ == "__main__":
    unittest.main()
