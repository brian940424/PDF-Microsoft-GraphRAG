import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from automotive_graphrag.connections import ConnectionSettings, ConnectionTestResult
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
            (root / "settings.yaml").write_text(
                "input: {}\n"
                "completion_models:\n  default_completion_model: {}\n"
                "embedding_models:\n  default_embedding_model: {}\n",
                encoding="utf-8",
            )
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
        self.connections = ConnectionSettings(
            self.store.root,
            lambda base_url, key: ConnectionTestResult(True, "ok"),
        )
        self.connections.save("https://api.openai.com/v1", "test-key")
        self.metadata_projects: list[str] = []

    def build_metadata(self, project_id: str) -> None:
        self.metadata_projects.append(project_id)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_initialize_uses_cli_and_configures_jsonl_input(self) -> None:
        runner = FakeGraphRag()
        service = IndexingService(
            self.store,
            runner,
            connection_settings=self.connections,
            metadata_builder=self.build_metadata,
        )

        settings_path = service.initialize("L33-SM3E")

        self.assertIn("init", runner.commands[0])
        self.assertEqual(runner.commands[0][runner.commands[0].index("--model") + 1], "gpt-4o-mini")
        self.assertEqual(
            runner.commands[0][runner.commands[0].index("--embedding") + 1],
            "text-embedding-3-small",
        )
        settings = yaml.safe_load(settings_path.read_text())
        self.assertEqual(settings["input"]["type"], "jsonl")
        self.assertEqual(settings["input"]["file_pattern"], r".*\.jsonl$$")
        self.assertEqual(settings["input"]["storage"]["base_dir"], "input")
        self.assertEqual(
            settings["completion_models"]["default_completion_model"]["api_base"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(settings["completion_models"]["default_completion_model"]["model"], "gpt-4o-mini")
        self.assertEqual(
            settings["embedding_models"]["default_embedding_model"]["model"],
            "text-embedding-3-small",
        )
        graph_input = settings_path.parent / "input" / "input.jsonl"
        self.assertIn('"id":"p1"', graph_input.read_text())

    def test_successful_build_sets_indexed_and_saves_result(self) -> None:
        runner = FakeGraphRag()
        service = IndexingService(
            self.store,
            runner,
            connection_settings=self.connections,
            metadata_builder=self.build_metadata,
        )

        result = service.build("L33-SM3E")

        self.assertEqual(result.status, "INDEXED")
        self.assertEqual(self.store.get("L33-SM3E").status, "INDEXED")
        self.assertTrue((self.store.path_for("L33-SM3E") / "graphrag" / "output" / "entities.parquet").is_file())
        self.assertEqual(service.last_result("L33-SM3E"), result)
        self.assertEqual(self.metadata_projects, ["L33-SM3E"])
        self.assertFalse((self.store.path_for("L33-SM3E") / "graphrag" / ".indexing.lock").exists())

    def test_build_uses_selected_gpt6_luna_model(self) -> None:
        runner = FakeGraphRag()
        service = IndexingService(
            self.store,
            runner,
            connection_settings=self.connections,
            metadata_builder=self.build_metadata,
        )

        result = service.build("L33-SM3E", chat_model="gpt-6-luna")

        self.assertEqual(result.status, "INDEXED")
        settings = yaml.safe_load(
            (self.store.path_for("L33-SM3E") / "graphrag" / "settings.yaml").read_text()
        )
        completion_model = settings["completion_models"]["default_completion_model"]
        self.assertEqual(completion_model["model"], "gpt-6-luna")
        self.assertEqual(completion_model["call_args"]["reasoning_effort"], "medium")
        self.assertNotIn("temperature", completion_model["call_args"])

    def test_indexed_project_can_be_rebuilt_with_selected_model(self) -> None:
        self.store.update_status("L33-SM3E", "INDEXED")
        runner = FakeGraphRag()
        service = IndexingService(
            self.store,
            runner,
            connection_settings=self.connections,
            metadata_builder=self.build_metadata,
        )

        result = service.build("L33-SM3E", chat_model="gpt-6-luna")

        self.assertEqual(result.status, "INDEXED")
        self.assertEqual(self.store.get("L33-SM3E").status, "INDEXED")
        self.assertFalse(
            (self.store.path_for("L33-SM3E") / "graphrag" / ".last-successful-output").exists()
        )

    def test_failed_rebuild_restores_last_successful_output(self) -> None:
        graph_root = self.store.path_for("L33-SM3E") / "graphrag"
        (graph_root / "settings.yaml").write_text("input: {}\n")
        output = graph_root / "output"
        output.mkdir()
        (output / "previous.parquet").write_bytes(b"previous")
        self.store.update_status("L33-SM3E", "INDEXED")
        service = IndexingService(
            self.store,
            FakeGraphRag(fail_index=True),
            connection_settings=self.connections,
            metadata_builder=self.build_metadata,
        )

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
            IndexingService(
                self.store,
                FakeGraphRag(),
                connection_settings=self.connections,
                metadata_builder=self.build_metadata,
            ).build("L33-SM3E")

        self.assertEqual(self.store.get("L33-SM3E").status, "READY")

    def test_build_rejects_project_that_is_not_ready(self) -> None:
        self.store.update_status("L33-SM3E", "UPLOADED")

        with self.assertRaisesRegex(ProjectError, "不允許建圖"):
            IndexingService(
                self.store,
                FakeGraphRag(),
                connection_settings=self.connections,
                metadata_builder=self.build_metadata,
            ).build("L33-SM3E")

    def test_build_rejects_unsupported_model(self) -> None:
        with self.assertRaisesRegex(ProjectError, "不支援的建圖模型"):
            IndexingService(
                self.store,
                FakeGraphRag(),
                connection_settings=self.connections,
                metadata_builder=self.build_metadata,
            ).build("L33-SM3E", chat_model="unsupported-model")


if __name__ == "__main__":
    unittest.main()
