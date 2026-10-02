import json
import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.ground_truth import GroundTruthService
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.question_sets import QuestionSetService


class GroundTruthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = ProjectStore(self.root / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        source = self.root / "questions.json"
        source.write_text(
            json.dumps(
                {
                    "name": "Gold Evidence 測試",
                    "questions": [{"question_id": "Q001", "question": "雨刷不會動"}],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.question_sets = QuestionSetService(self.store)
        self.question_set = self.question_sets.import_file("L33-SM3E", source)
        output = self.store.path_for("L33-SM3E") / "graphrag" / "output"
        output.mkdir()
        metadata = [
            {
                "text_unit_id": "tu-1",
                "chunk_id": "L33-SM3E-WW-p0025-b01",
                "project_id": "L33-SM3E",
                "section_id": "WW",
                "section_name": "Wiper & Washer",
                "document_id": "WW.pdf",
                "page": 25,
                "block_id": "b01",
                "text": "雨刷保險絲",
            },
            {
                "text_unit_id": "tu-2",
                "chunk_id": "L33-SM3E-WW-p0026-b01",
                "project_id": "L33-SM3E",
                "section_id": "WW",
                "section_name": "Wiper & Washer",
                "document_id": "WW.pdf",
                "page": 26,
                "block_id": "b01",
                "text": "雨刷馬達",
            },
        ]
        (output / "source_metadata.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
            encoding="utf-8",
        )
        self.service = GroundTruthService(self.store, self.question_sets)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_save_valid_gold_evidence_to_question(self) -> None:
        updated = self.service.save(
            "L33-SM3E",
            self.question_set.question_set_id,
            "Q001",
            [
                {
                    "document_id": "WW.pdf",
                    "pages": [25, 26],
                    "chunk_ids": ["L33-SM3E-WW-p0025-b01", "L33-SM3E-WW-p0026-b01"],
                }
            ],
        )

        gold = updated.questions[0].gold_evidence[0]
        self.assertEqual(gold.document_id, "WW.pdf")
        self.assertEqual(gold.pages, (25, 26))
        reloaded = self.question_sets.get("L33-SM3E", self.question_set.question_set_id)
        self.assertEqual(reloaded.questions[0].gold_evidence, updated.questions[0].gold_evidence)

    def test_save_rejects_unknown_pdf_page_and_chunk(self) -> None:
        with self.assertRaisesRegex(ProjectError, "找不到 PDF"):
            self.service.save(
                "L33-SM3E",
                self.question_set.question_set_id,
                "Q001",
                [{"document_id": "PG.pdf", "pages": [1], "chunk_ids": []}],
            )
        with self.assertRaisesRegex(ProjectError, "找不到頁碼.*99"):
            self.service.save(
                "L33-SM3E",
                self.question_set.question_set_id,
                "Q001",
                [{"document_id": "WW.pdf", "pages": [99], "chunk_ids": []}],
            )
        with self.assertRaisesRegex(ProjectError, "找不到 Chunk"):
            self.service.save(
                "L33-SM3E",
                self.question_set.question_set_id,
                "Q001",
                [{"document_id": "WW.pdf", "pages": [], "chunk_ids": ["missing"]}],
            )

    def test_question_set_import_reports_invalid_gold_evidence_position(self) -> None:
        source = self.root / "invalid.json"
        source.write_text(
            json.dumps(
                {
                    "name": "Invalid",
                    "questions": [
                        {
                            "question_id": "Q001",
                            "question": "問題",
                            "gold_evidence": [{"document_id": "WW.pdf", "pages": [0], "chunk_ids": []}],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ProjectError, r"questions\[0\].gold_evidence\[0\].pages"):
            self.question_sets.import_file("L33-SM3E", source)


if __name__ == "__main__":
    unittest.main()
