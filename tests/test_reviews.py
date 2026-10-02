import csv
import json
import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.question_sets import QuestionSetService
from automotive_graphrag.reviews import ReviewService


class ReviewServiceTests(unittest.TestCase):
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
                    "name": "人工評測集",
                    "questions": [
                        {"question_id": "Q001", "question": "問題一"},
                        {"question_id": "Q002", "question": "問題二"},
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.question_sets = QuestionSetService(self.store)
        self.question_set = self.question_sets.import_file("L33-SM3E", source)
        value = json.loads(
            (
                self.store.path_for("L33-SM3E")
                / "question_sets"
                / f"{self.question_set.question_set_id}.json"
            ).read_text()
        )
        value["questions"][0].update(status="COMPLETED", answer="回答一", duration_seconds=0.5)
        question_set_path = (
            self.store.path_for("L33-SM3E") / "question_sets" / f"{self.question_set.question_set_id}.json"
        )
        question_set_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        self.reviews = ReviewService(self.store, self.question_sets)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_save_review_and_update_existing_label(self) -> None:
        first = self.reviews.save(
            "L33-SM3E", self.question_set.question_set_id, "Q001", "partially_correct", "缺少步驟"
        )
        updated = self.reviews.save(
            "L33-SM3E", self.question_set.question_set_id, "Q001", "correct", "重新確認"
        )

        self.assertEqual(first.human_label, "partially_correct")
        self.assertEqual(updated.human_label, "correct")
        self.assertEqual(updated.reviewer_note, "重新確認")
        records = self.reviews.records("L33-SM3E", self.question_set.question_set_id)
        self.assertEqual(records[0].answer, "回答一")
        self.assertEqual(records[0].human_label, "correct")
        self.assertIsNotNone(records[0].reviewed_at)

    def test_progress_counts_reviewed_questions(self) -> None:
        self.reviews.save("L33-SM3E", self.question_set.question_set_id, "Q001", "incorrect")

        progress = self.reviews.progress("L33-SM3E", self.question_set.question_set_id)

        self.assertEqual((progress.reviewed, progress.total), (1, 2))

    def test_invalid_label_and_unknown_question_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProjectError, "有效"):
            self.reviews.save("L33-SM3E", self.question_set.question_set_id, "Q001", "maybe")
        with self.assertRaisesRegex(ProjectError, "找不到題號"):
            self.reviews.save("L33-SM3E", self.question_set.question_set_id, "Q999", "correct")

    def test_export_contains_answers_and_human_reviews(self) -> None:
        self.reviews.save("L33-SM3E", self.question_set.question_set_id, "Q001", "correct", "完整")

        json_path, csv_path = self.reviews.export("L33-SM3E", self.question_set.question_set_id)

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["reviews"][0]["answer"], "回答一")
        self.assertEqual(payload["reviews"][0]["human_label"], "correct")
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["reviewer_note"], "完整")
        self.assertEqual(rows[1]["human_label"], "")


if __name__ == "__main__":
    unittest.main()
