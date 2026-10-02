import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryService


class FakeQueryRunner:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self.return_code:
            return subprocess.CompletedProcess(command, self.return_code, "", "query failed\n")
        return subprocess.CompletedProcess(command, 0, "先檢查保險絲，再檢查馬達。\n", "")


class QueryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        self.graph_root = self.store.path_for("L33-SM3E") / "graphrag"
        (self.graph_root / "output").mkdir()
        (self.graph_root / "settings.yaml").write_text(
            "completion_models:\n  default_completion_model: {}\n"
            "embedding_models:\n  default_embedding_model: {}\n",
            encoding="utf-8",
        )
        self.store.update_status("L33-SM3E", "INDEXED")
        self.connections = ConnectionSettings(self.store.root)
        self.connections.save("https://api.openai.com/v1", "test-key")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_local_query_uses_selected_project_and_saves_history(self) -> None:
        runner = FakeQueryRunner()
        service = QueryService(self.store, runner, self.connections)

        result = service.ask("L33-SM3E", " 雨刷不會動，如何檢修？ ")

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.answer, "先檢查保險絲，再檢查馬達。")
        self.assertEqual(result.method, "local")
        command = runner.commands[0]
        self.assertEqual(command[command.index("--root") + 1], str(self.graph_root))
        self.assertEqual(command[command.index("--method") + 1], "local")
        self.assertEqual(command[-1], "雨刷不會動，如何檢修？")
        self.assertEqual(service.history("L33-SM3E"), [result])
        settings = yaml.safe_load((self.graph_root / "settings.yaml").read_text())
        self.assertEqual(
            settings["completion_models"]["default_completion_model"]["api_base"],
            "https://api.openai.com/v1",
        )

    def test_failed_query_is_saved_with_error(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(return_code=2), self.connections)

        result = service.ask("L33-SM3E", "測試問題")

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.error, "query failed")
        self.assertEqual(service.history("L33-SM3E")[0], result)

    def test_query_rejects_project_without_completed_index(self) -> None:
        self.store.update_status("L33-SM3E", "READY")
        service = QueryService(self.store, FakeQueryRunner(), self.connections)

        with self.assertRaisesRegex(ProjectError, "尚未完成建圖"):
            service.ask("L33-SM3E", "測試問題")

    def test_query_rejects_blank_question_and_unknown_method(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(), self.connections)
        with self.assertRaisesRegex(ProjectError, "問題為必填"):
            service.ask("L33-SM3E", "  ")
        with self.assertRaisesRegex(ProjectError, "不支援"):
            service.ask("L33-SM3E", "測試問題", "unknown")


if __name__ == "__main__":
    unittest.main()
