import subprocess
import json
import tempfile
import unittest
from pathlib import Path

import yaml
import pandas as pd
from unittest.mock import patch

from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.evidence import Evidence, EvidenceService
from automotive_graphrag.querying import QueryExecution, QueryService
from automotive_graphrag.source_metadata import SourceMetadata


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

        result = service.ask("L33-SM3E", " 雨刷不會動，如何檢修？ ", response_type="List of 3-7 Points")

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.answer, "先檢查保險絲，再檢查馬達。")
        self.assertEqual(result.method, "local")
        command = runner.commands[0]
        self.assertEqual(command[command.index("--root") + 1], str(self.graph_root))
        self.assertEqual(command[command.index("--method") + 1], "local")
        self.assertEqual(command[command.index("--response-type") + 1], "List of 3-7 Points")
        self.assertEqual(result.response_type, "List of 3-7 Points")
        self.assertEqual(command[-1], "雨刷不會動，如何檢修？")
        self.assertEqual(service.history("L33-SM3E"), [result])
        settings = yaml.safe_load((self.graph_root / "settings.yaml").read_text())
        self.assertEqual(
            settings["completion_models"]["default_completion_model"]["api_base"],
            "https://api.openai.com/v1",
        )
        self.assertEqual(settings["completion_models"]["default_completion_model"]["model"], "gpt-6-luna")
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
        service = QueryService(self.store, connection_settings=self.connections)

        def run_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            payload = json.loads(str(kwargs["input"]))
            self.assertEqual(payload["method"], "drift")
            self.assertEqual(payload["model"], "gpt-6-luna")
            return subprocess.CompletedProcess(
                command, 0, json.dumps({"answer": "Luna DRIFT 回答", "context": {"sources": []}}), ""
            )

        with patch("automotive_graphrag.querying.subprocess.run", side_effect=run_worker):
            result = service.ask("L33-SM3E", "測試 GPT-6", method="drift", chat_model="gpt-6-luna")

        self.assertEqual(result.status, "COMPLETED")
        self.assertEqual(result.answer, "Luna DRIFT 回答")
        self.assertEqual(result.context["sources"], [])

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

    def test_all_retrieval_methods_save_returned_evidence(self) -> None:
        service = QueryService(
            self.store,
            connection_settings=self.connections,
            evidence_service=FakeEvidenceService(),
        )
        calls: list[dict[str, object]] = []

        def run_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            payload = json.loads(str(kwargs["input"]))
            calls.append(payload)
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({
                    "answer": f"{payload['method']} answer",
                    "context": {"sources": [{"id": "7", "text": "來源"}]},
                }),
                "",
            )

        with patch("automotive_graphrag.querying.subprocess.run", side_effect=run_worker):
            results = [
                service.ask("L33-SM3E", "測試證據", method=method)
                for method in ("local", "global", "drift", "basic")
            ]

        self.assertEqual([call["method"] for call in calls], ["local", "global", "drift", "basic"])
        self.assertTrue(all(result.evidence and result.evidence[0].page == 25 for result in results))
        self.assertEqual([len(item.evidence) for item in service.history("L33-SM3E")], [1, 1, 1, 1])

    def test_global_community_reports_are_saved_as_report_level_evidence(self) -> None:
        evidence = EvidenceService(self.store).from_context(
            "L33-SM3E",
            {
                "reports": [{
                    "id": "report-1",
                    "community": "community-3",
                    "title": "冷卻系統",
                    "summary": "檢查冷卻液液位及管路狀況。",
                }]
            },
        )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].document_id, "GraphRAG Community Report")
        self.assertIsNone(evidence[0].page)
        self.assertEqual(evidence[0].text, "檢查冷卻液液位及管路狀況。")

    def test_drift_nested_context_is_serialized_and_mapped_to_source_metadata(self) -> None:
        nested_context = {
            "查詢 A": {"sources": pd.DataFrame([{"id": "7", "text": "來源 A"}])},
            "查詢 B": {"sources": pd.DataFrame([{"id": "8", "text": "來源 B"}])},
        }
        serialized = QueryService._serialize_context(nested_context)
        self.assertEqual(serialized["查詢 A"]["sources"][0]["id"], "7")
        self.assertEqual(serialized["查詢 B"]["sources"][0]["text"], "來源 B")

        output = self.graph_root / "output"
        (output / "text_units.parquet").touch()
        metadata = [
            SourceMetadata("full-7", "chunk-7", "L33-SM3E", "WW", "雨刷", "WW.pdf", 25, "b01", "原文 A"),
            SourceMetadata("full-8", "chunk-8", "L33-SM3E", "PG", "電源", "PG.pdf", 42, "b02", "原文 B"),
        ]

        class MetadataService:
            def load(self, project_id):
                self_project_id = project_id
                return metadata if self_project_id == "L33-SM3E" else []

        with patch(
            "automotive_graphrag.evidence.pd.read_parquet",
            return_value=pd.DataFrame([
                {"id": "full-7", "human_readable_id": "7"},
                {"id": "full-8", "human_readable_id": "8"},
            ]),
        ):
            evidence = EvidenceService(self.store, MetadataService()).from_context("L33-SM3E", serialized)

        self.assertEqual([(item.document_id, item.page, item.chunk_id) for item in evidence], [
            ("WW.pdf", 25, "chunk-7"),
            ("PG.pdf", 42, "chunk-8"),
        ])

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
