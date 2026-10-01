import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from automotive_graphrag.indexing import IndexingService
from automotive_graphrag.projects import ProjectError, ProjectStore


class FakeGraphRag:
    def __init__(self, fail_index: bool = False) -> None:
        self.commands: list[list[str]] = []
        self.fail_index = fail_index

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        root = Path(command[command.index("--root") + 1])
        if "init" in command:
            (root / "prompts").mkdir(exist_ok=True)
            (root / "settings.yaml").write_text("input: {}\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "initialized\n", "")
        if self.fail_index:
            (root / "output").mkdir(exist_ok=True)
            (root / "output" / "partial.txt").write_text("partial")
            return subprocess.CompletedProcess(command, 2, "", "index failed\n")
        (root / "output").mkdir(exist_ok=True)
        (root / "output" / "entities.parquet").write_bytes(b"index")
        return subprocess.CompletedProcess(command, 0, "index complete\n", "")


class IndexingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        project_path = self.store.path_for("L33-SM3E")
        (project_path / "processed" / "input.jsonl").write_text('{"id":"p1","title":"Page","text":"content"}\n')
        self.store.update_status("L33-SM3E", "READY")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_initialize_uses_cli_and_configures_jsonl_input(self) -> None:
        runner = FakeGraphRag()
        service = IndexingService(self.store, runner)

        settings_path = service.initialize("L33-SM3E")

        self.assertIn("init", runner.commands[0])
        self.assertEqual(runner.commands[0][runner.commands[0].index("--model") + 1], "gpt-4.1")
        self.assertEqual(
            runner.commands[0][runner.commands[0].index("--embedding") + 1],
            "text-embedding-3-large",
        )
        settings = yaml.safe_load(settings_path.read_text())
        self.assertEqual(settings["input"]["type"], "jsonl")
        self.assertEqual(settings["input"]["file_pattern"], r".*\.jsonl$$")
        self.assertEqual(settings["input"]["storage"]["base_dir"], "input")
        graph_input = settings_path.parent / "input" / "input.jsonl"
        self.assertIn('"id":"p1"', graph_input.read_text())

    def test_successful_build_sets_indexed_and_saves_result(self) -> None:
        runner = FakeGraphRag()
        service = IndexingService(self.store, runner)

        result = service.build("L33-SM3E")

        self.assertEqual(result.status, "INDEXED")
        self.assertEqual(self.store.get("L33-SM3E").status, "INDEXED")
        self.assertTrue((self.store.path_for("L33-SM3E") / "graphrag" / "output" / "entities.parquet").is_file())
        self.assertEqual(service.last_result("L33-SM3E"), result)
        self.assertFalse((self.store.path_for("L33-SM3E") / "graphrag" / ".indexing.lock").exists())

    def test_failed_rebuild_restores_last_successful_output(self) -> None:
        graph_root = self.store.path_for("L33-SM3E") / "graphrag"
        (graph_root / "settings.yaml").write_text("input: {}\n")
        output = graph_root / "output"
        output.mkdir()
        (output / "previous.parquet").write_bytes(b"previous")
        self.store.update_status("L33-SM3E", "STALE")
        service = IndexingService(self.store, FakeGraphRag(fail_index=True))

        result = service.build("L33-SM3E")

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.last_message, "index failed")
        self.assertEqual(self.store.get("L33-SM3E").status, "FAILED")
        self.assertEqual((output / "previous.parquet").read_bytes(), b"previous")
        self.assertFalse((output / "partial.txt").exists())

    def test_existing_lock_prevents_duplicate_index_job(self) -> None:
        lock = self.store.path_for("L33-SM3E") / "graphrag" / ".indexing.lock"
        lock.write_text("busy")

        with self.assertRaisesRegex(ProjectError, "正在執行"):
            IndexingService(self.store, FakeGraphRag()).build("L33-SM3E")

        self.assertEqual(self.store.get("L33-SM3E").status, "READY")

    def test_build_rejects_project_that_is_not_ready(self) -> None:
        self.store.update_status("L33-SM3E", "UPLOADED")

        with self.assertRaisesRegex(ProjectError, "不允許建圖"):
            IndexingService(self.store, FakeGraphRag()).build("L33-SM3E")


if __name__ == "__main__":
    unittest.main()
