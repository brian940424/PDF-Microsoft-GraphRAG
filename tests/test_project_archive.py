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

    def test_import_conflict_message_suggests_new_project_id(self) -> None:
        archive = export_project(self.source, "L33-SM3E")

        with self.assertRaisesRegex(ProjectError, "新的 Project ID"):
            import_project(self.source, archive)

    def test_import_under_new_project_id_keeps_original_project(self) -> None:
        root = self.source.root / "L33-SM3E"
        (root / "processed" / "input.jsonl").write_text(
            '{"id":"L33-SM3E-ACC-p0001-b01","project_id":"L33-SM3E","text":"x"}\n'
            '{"id":"L33-SM3E-ACC-p0002-b01","project_id":"L33-SM3E","text":"y"}\n'
        )
        (root / "question_sets").mkdir(exist_ok=True)
        (root / "question_sets" / "set.json").write_text(
            '{\n  "project_id": "L33-SM3E",\n  "note": "project_id is L33-SM3E"\n}'
        )
        archive = export_project(self.source, "L33-SM3E")

        project = import_project(self.source, archive, project_id=" L33-SM3E-copy ")

        self.assertEqual(project.project_id, "L33-SM3E-copy")
        self.assertEqual(project.display_name, "L33 / SM3E")
        self.assertEqual(project.status, "INDEXED")
        self.assertEqual({p.project_id for p in self.source.list()}, {"L33-SM3E", "L33-SM3E-copy"})
        copy = self.source.root / "L33-SM3E-copy"
        self.assertEqual((copy / "graphrag" / "output" / "entities.parquet").read_bytes(), b"graph")
        # Only project_id fields change; chunk IDs and free text keep the original ID.
        self.assertEqual(
            (copy / "processed" / "input.jsonl").read_text(),
            '{"id":"L33-SM3E-ACC-p0001-b01","project_id":"L33-SM3E-copy","text":"x"}\n'
            '{"id":"L33-SM3E-ACC-p0002-b01","project_id":"L33-SM3E-copy","text":"y"}\n',
        )
        self.assertEqual(
            (copy / "question_sets" / "set.json").read_text(),
            '{\n  "project_id": "L33-SM3E-copy",\n  "note": "project_id is L33-SM3E"\n}',
        )
        self.assertEqual(json.loads((root / "project.json").read_text())["project_id"], "L33-SM3E")
        self.assertIn('"project_id":"L33-SM3E"', (root / "processed" / "input.jsonl").read_text())
        self.assertEqual([p.name for p in copy.rglob("*.rewrite")], [])

    def test_import_with_same_project_id_as_archive_behaves_like_default(self) -> None:
        archive = export_project(self.source, "L33-SM3E")

        project = import_project(self.target, archive, project_id="L33-SM3E")

        self.assertEqual(project.project_id, "L33-SM3E")

    def test_import_refuses_new_project_id_that_already_exists(self) -> None:
        self.source.create(project_id="Other", display_name="Other", vehicle_name="X", manual_version="Y")
        archive = export_project(self.source, "L33-SM3E")

        with self.assertRaisesRegex(ProjectError, "請換一個新的 Project ID"):
            import_project(self.source, archive, project_id="Other")

    def test_import_rejects_invalid_new_project_id(self) -> None:
        archive = export_project(self.source, "L33-SM3E")

        with self.assertRaisesRegex(ProjectError, "只能包含"):
            import_project(self.target, archive, project_id="../evil")
        self.assertFalse(self.target.root.exists() and any(self.target.root.iterdir()))

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
