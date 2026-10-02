import csv
import json
import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.source_sampling import SourceSamplingService


class SourceSamplingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        self.processed = self.store.path_for("L33-SM3E") / "processed" / "input.jsonl"
        records = [
            self.record("WW", "WW.pdf", 1, "CONTENTS\nDIAGNOSIS ........ 2\nREMOVAL ........ 8\nSPECIFICATIONS ........ 10"),
            self.record("WW", "WW.pdf", 2, "Diagnostic inspection and check the wiper motor circuit carefully."),
            self.record("WW", "WW.pdf", 3, "Removal and installation procedure. Follow each step and caution."),
            self.record("PG", "PG.pdf", 10, "Service data specification and tightening torque tolerance values."),
            self.record("PG", "PG.pdf", 11, "General system overview with enough explanatory original manual text."),
            self.record("PG", "PG.pdf", 12, "short"),
        ]
        self.processed.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        self.service = SourceSamplingService(self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def record(section: str, document: str, page: int, text: str) -> dict[str, object]:
        chunk_id = f"L33-SM3E-{section}-p{page:04d}-b01"
        return {
            "id": chunk_id.removesuffix("-b01"),
            "chunk_id": chunk_id,
            "project_id": "L33-SM3E",
            "section_id": section,
            "section_name": section,
            "document_id": document,
            "page": page,
            "block_id": "b01",
            "text": text,
        }

    def test_scan_filters_toc_short_text_section_page_and_content_type(self) -> None:
        diagnostic = self.service.scan(
            "L33-SM3E",
            section_ids=["WW"],
            page_from=2,
            page_to=3,
            content_type="diagnostic",
            minimum_characters=20,
        )

        self.assertEqual([(item.document_id, item.page) for item in diagnostic], [("WW.pdf", 2)])
        all_candidates = self.service.scan("L33-SM3E", minimum_characters=20)
        self.assertEqual([item.page for item in all_candidates], [10, 11, 2, 3])
        self.assertNotIn(1, [item.page for item in all_candidates])
        self.assertNotIn(12, [item.page for item in all_candidates])

    def test_available_sections_are_derived_from_processed_source(self) -> None:
        self.assertEqual(self.service.available_sections("L33-SM3E"), [("PG", "PG"), ("WW", "WW")])

    def test_sampling_is_reproducible_and_persisted(self) -> None:
        first = self.service.sample("L33-SM3E", count=2, minimum_characters=20, seed=17)
        second = self.service.sample("L33-SM3E", count=2, minimum_characters=20, seed=17)

        self.assertEqual(
            [item.chunk_id for item in first.samples],
            [item.chunk_id for item in second.samples],
        )
        self.assertEqual(self.service.get("L33-SM3E", first.sample_batch_id), first)

    def test_sample_rejects_request_larger_than_candidates(self) -> None:
        with self.assertRaisesRegex(ProjectError, "只有 1 筆"):
            self.service.sample(
                "L33-SM3E",
                count=2,
                content_type="specification",
                minimum_characters=20,
            )

    def test_export_writes_source_metadata_and_text(self) -> None:
        batch = self.service.sample("L33-SM3E", count=1, minimum_characters=20)

        json_path, csv_path = self.service.export("L33-SM3E", batch.sample_batch_id)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["requested_count"], 1)
        self.assertIn("chunk_id", payload["samples"][0])
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertTrue(rows[0]["text"])

    def test_old_processed_records_require_reprocessing(self) -> None:
        self.processed.write_text('{"document_id":"WW.pdf","page":1,"text":"content"}\n')

        with self.assertRaisesRegex(ProjectError, "重新前處理"):
            self.service.scan("L33-SM3E")


if __name__ == "__main__":
    unittest.main()
