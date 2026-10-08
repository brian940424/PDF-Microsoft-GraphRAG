import subprocess
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from unittest.mock import patch

from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.querying import QueryExecution, QueryService


class FakeQueryRunner:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if self.return_code:
            return subprocess.CompletedProcess(command, self.return_code, "", "query failed\n")
        return subprocess.CompletedProcess(command, 0, "先檢查保險絲，再檢查馬達。\n", "")


class ContextQueryRunner:
    def __call__(self, command: list[str]) -> QueryExecution:
        return QueryExecution(0, "有來源的回答", "", {"sources": [{"id": "7", "text": "來源"}]})


class FakeEvidenceService:
    def from_context(self, project_id: str, context: dict[str, object]) -> list[Evidence]:
        return [
            Evidence(
                evidence_id="E1",
                rank=1,
                context_id="7",
                text_unit_id="full-tu-1",
                chunk_id="L33-SM3E-WW-p0025-b01",
                section_id="WW",
                section_name="Wiper & Washer",
                document_id="WW.pdf",
                page=25,
                block_id="b01",
                text="來源",
            )
        ]


class FailingEvidenceService:
    def from_context(self, project_id: str, context: dict[str, object]) -> list[Evidence]:
        raise ProjectError(f"找不到專案：{project_id}")


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
        self.assertEqual(settings["completion_models"]["default_completion_model"]["model"], "gpt-4o-mini")
        self.assertEqual(
            settings["embedding_models"]["default_embedding_model"]["model"],
            "text-embedding-3-small",
        )

    def test_failed_query_is_saved_with_error(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(return_code=2), self.connections)

        result = service.ask("L33-SM3E", "測試問題")

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.error, "query failed")
        self.assertEqual(service.history("L33-SM3E")[0], result)

    def test_query_can_override_chat_model(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(), self.connections)

        service.ask("L33-SM3E", "測試模型", chat_model="gpt-4.1-mini")

        settings = yaml.safe_load((self.graph_root / "settings.yaml").read_text())
        self.assertEqual(settings["completion_models"]["default_completion_model"]["model"], "gpt-4.1-mini")

    def test_gpt6_query_sets_medium_reasoning_compatibility_args(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(), self.connections)

        service.ask("L33-SM3E", "測試 GPT-6", chat_model="gpt-6-luna")

        settings = yaml.safe_load((self.graph_root / "settings.yaml").read_text())
        model = settings["completion_models"]["default_completion_model"]
        self.assertEqual(model["model"], "gpt-6-luna")
        self.assertEqual(model["call_args"]["reasoning_effort"], "medium")
        self.assertNotIn("temperature", model["call_args"])
        self.assertNotIn("top_p", model["call_args"])

    def test_gpt6_drift_uses_compatible_entrypoint(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(), self.connections)

        with patch(
            "automotive_graphrag.querying.run_compatible_drift_search",
            return_value=("Luna DRIFT 回答", {"sources": []}),
        ) as drift_search:
            result = service.ask("L33-SM3E", "測試 GPT-6", method="drift", chat_model="gpt-6-luna")

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.answer, "Luna DRIFT 回答")
        drift_search.assert_called_once_with(root_dir=self.graph_root, query="測試 GPT-6")

    def test_query_rejects_unknown_chat_model(self) -> None:
        service = QueryService(self.store, FakeQueryRunner(), self.connections)

        with self.assertRaisesRegex(ProjectError, "不支援的 Chat 模型"):
            service.ask("L33-SM3E", "測試模型", chat_model="not-a-model")

    def test_query_saves_context_and_resolved_evidence(self) -> None:
        service = QueryService(
            self.store,
            ContextQueryRunner(),
            self.connections,
            evidence_service=FakeEvidenceService(),
        )

        result = service.ask("L33-SM3E", "測試來源")

        self.assertEqual(result.evidence[0].evidence_id, "E1")
        self.assertEqual(result.context["sources"][0]["id"], "7")
        self.assertEqual(service.history("L33-SM3E")[0], result)

    def test_context_summary_is_bounded_for_large_query_context(self) -> None:
        context = {"sources": [{"text": "x" * 10_000} for _ in range(100)], "metadata": {"a": 1}}

        summary = QueryService.context_summary(context)

        self.assertEqual(summary["sections"]["sources"]["items"], 100)
        self.assertNotIn("x" * 100, json.dumps(summary))
        self.assertLess(len(json.dumps(summary)), 1000)

    def test_answer_survives_evidence_resolution_project_error(self) -> None:
        service = QueryService(
            self.store,
            ContextQueryRunner(),
            self.connections,
            evidence_service=FailingEvidenceService(),
        )

        result = service.ask("L33-SM3E", "測試來源錯誤")

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.answer, "有來源的回答")
        self.assertEqual(result.evidence, ())
        self.assertIn("找不到專案", result.context["_evidence_warning"])
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
