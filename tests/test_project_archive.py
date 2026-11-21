import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from automotive_graphrag.project_archive import (
    ARCHIVE_FORMAT,
    ARCHIVE_VERSION,
    MANIFEST_NAME,
    export_project,
    import_project,
)
from automotive_graphrag.projects import ProjectError, ProjectStore


class ProjectArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        base = Path(self.temporary_directory.name)
        self.source = ProjectStore(base / "source-projects")
        self.target = ProjectStore(base / "target-projects")
        self.project = self.source.create(
            project_id="L33-SM3E", display_name="L33 / SM3E", vehicle_name="L33", manual_version="SM3E",
        )
        self.source.update_status("L33-SM3E", "INDEXED")
        root = self.source.root / "L33-SM3E"
        (root / "source" / "manual.pdf").write_bytes(b"%PDF-1.4")
        (root / "graphrag" / "output").mkdir()
        (root / "graphrag" / "output" / "entities.parquet").write_bytes(b"graph")
        (root / "graphrag" / "cache").mkdir()
        (root / "graphrag" / "cache" / "llm.json").write_text("{}")
        (root / "cache").mkdir(exist_ok=True)
        (root / "cache" / "query_cache.db").write_text("cache")
        (root / "graphrag" / ".env").write_text("GRAPHRAG_API_KEY=secret")
        self.base = base

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_round_trip_keeps_graph_and_metadata_but_not_secrets_or_cache(self) -> None:
        archive = export_project(self.source, "L33-SM3E")

        project = import_project(self.target, archive)

        self.assertEqual(project.project_id, "L33-SM3E")
        self.assertEqual(project.status, "INDEXED")
        root = self.target.root / "L33-SM3E"
        self.assertEqual((root / "graphrag" / "output" / "entities.parquet").read_bytes(), b"graph")
        self.assertEqual((root / "source" / "manual.pdf").read_bytes(), b"%PDF-1.4")
        self.assertFalse((root / "graphrag" / ".env").exists())
        self.assertFalse((root / "graphrag" / "cache").exists())
        self.assertFalse((root / "cache" / "query_cache.db").exists())
        self.assertTrue((root / "exports").is_dir())
        self.assertEqual([p.name for p in self.target.root.iterdir()], ["L33-SM3E"])

    def test_cache_is_exported_when_requested(self) -> None:
        archive = export_project(self.source, "L33-SM3E", include_cache=True)

        import_project(self.target, archive)

        self.assertTrue((self.target.root / "L33-SM3E" / "graphrag" / "cache" / "llm.json").is_file())
        self.assertTrue((self.target.root / "L33-SM3E" / "cache" / "query_cache.db").is_file())

    def test_import_refuses_existing_project_id(self) -> None:
        archive = export_project(self.source, "L33-SM3E")

        with self.assertRaisesRegex(ProjectError, "已存在"):
            import_project(self.source, archive)

    def test_export_refuses_busy_project(self) -> None:
        self.source.update_status("L33-SM3E", "INDEXING")

        with self.assertRaisesRegex(ProjectError, "建圖"):
            export_project(self.source, "L33-SM3E")

    def write_archive(self, members: dict[str, str], manifest: dict | None = None) -> Path:
        path = self.base / "crafted.zip"
        with zipfile.ZipFile(path, "w") as archive:
            if manifest is not None:
                archive.writestr(MANIFEST_NAME, json.dumps(manifest))
            for name, content in members.items():
                archive.writestr(name, content)
        return path

    def test_import_rejects_archive_without_manifest(self) -> None:
        path = self.write_archive({"L33-SM3E/project.json": "{}"})

        with self.assertRaisesRegex(ProjectError, "不是本系統"):
            import_project(self.target, path)

    def test_import_rejects_path_traversal(self) -> None:
        manifest = {"format": ARCHIVE_FORMAT, "version": ARCHIVE_VERSION, "project_id": "L33-SM3E"}
        metadata = (self.source.root / "L33-SM3E" / "project.json").read_text()
        path = self.write_archive(
            {"L33-SM3E/project.json": metadata, "L33-SM3E/../evil.txt": "x"}, manifest
        )

        with self.assertRaisesRegex(ProjectError, "不安全"):
            import_project(self.target, path)
        self.assertFalse((self.target.root / "evil.txt").exists())
        self.assertFalse((self.target.root / "L33-SM3E").exists())

    def test_import_rejects_non_zip(self) -> None:
        path = self.base / "not.zip"
        path.write_text("hello")

        with self.assertRaisesRegex(ProjectError, "無法讀取"):
            import_project(self.target, path)


if __name__ == "__main__":
    unittest.main()
