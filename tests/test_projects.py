import json
import tempfile
import unittest
from pathlib import Path

from automotive_graphrag.projects import PROJECT_DIRECTORIES, ProjectError, ProjectStore


class ProjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def create_project(self, project_id: str = "L33-SM3E", display_name: str = "L33 / SM3E"):
        return self.store.create(
            project_id=project_id,
            display_name=display_name,
            vehicle_name="L33",
            manual_version="SM3E",
            description="測試專案",
        )

    def test_create_project_builds_complete_isolated_workspace(self) -> None:
        project = self.create_project()

        project_path = self.store.root / project.project_id
        self.assertEqual(project.status, "EMPTY")
        self.assertTrue(all((project_path / name).is_dir() for name in PROJECT_DIRECTORIES))
        metadata = json.loads((project_path / "project.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["description"], "測試專案")

    def test_create_project_rejects_invalid_project_ids(self) -> None:
        for project_id in ("", "with space", "../escape", "中文", "a/b"):
            with self.subTest(project_id=project_id), self.assertRaises(ProjectError):
                self.create_project(project_id=project_id)
        self.assertTrue(not self.store.root.exists() or list(self.store.root.iterdir()) == [])

    def test_create_project_rejects_duplicate_id_without_changing_existing_data(self) -> None:
        first = self.create_project(project_id="T30-SM5E", display_name="First")

        with self.assertRaisesRegex(ProjectError, "已存在"):
            self.create_project(project_id=first.project_id, display_name="Replacement")

        self.assertEqual(self.store.get(first.project_id).display_name, "First")

    def test_list_rows_include_pdf_count_status_and_updated_time(self) -> None:
        second = self.create_project(project_id="T30-SM5E", display_name="T30 / SM5E")
        first = self.create_project()
        (self.store.root / first.project_id / "source" / "WW.PDF").write_bytes(b"pdf")
        (self.store.root / first.project_id / "source" / "notes.txt").write_text("not a PDF")

        rows = self.store.table_rows()

        self.assertEqual([row[0] for row in rows], ["L33-SM3E", "T30-SM5E"])
        self.assertEqual(rows[0][2:4], [1, "EMPTY"])
        self.assertEqual(rows[0][4], first.updated_at)
        self.assertEqual(rows[1][4], second.updated_at)

    def test_required_display_fields_are_validated(self) -> None:
        with self.assertRaisesRegex(ProjectError, "顯示名稱為必填"):
            self.create_project(display_name="  ")


if __name__ == "__main__":
    unittest.main()
