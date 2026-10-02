import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.downloads import stage_downloads
from automotive_graphrag.projects import ProjectError


class DownloadStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_stages_exports_under_system_temp_with_original_names(self) -> None:
        json_source = self.root / "question_set.json"
        csv_source = self.root / "report.csv"
        json_source.write_text('{"name":"test"}', encoding="utf-8")
        csv_source.write_text("id,status\n1,ok\n", encoding="utf-8")

        staged_json, staged_csv = stage_downloads((json_source, csv_source))

        system_temp = Path(tempfile.gettempdir()).resolve()
        self.assertTrue(staged_json.is_relative_to(system_temp))
        self.assertTrue(staged_csv.is_relative_to(system_temp))
        self.assertEqual(staged_json.name, json_source.name)
        self.assertEqual(staged_csv.name, csv_source.name)
        self.assertEqual(staged_json.read_text(encoding="utf-8"), json_source.read_text(encoding="utf-8"))

    def test_rejects_missing_export(self) -> None:
        with self.assertRaisesRegex(ProjectError, "找不到可下載"):
            stage_downloads((self.root / "missing.json",))


if __name__ == "__main__":
    unittest.main()
