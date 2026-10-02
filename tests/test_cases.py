import csv
import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.cases import CaseService
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.portal import UserPortalService, create_portal_app


class StoredQueries:
    def __init__(self, results: list[QueryResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, str, str]] = []

    def history(self, project_id: str) -> list[QueryResult]:
        return [item for item in self.results if item.project_id == project_id]

    def ask(self, project_id: str, question: str, method: str = "local") -> QueryResult:
        self.calls.append((project_id, question, method))
        return replace(self.results[0], project_id=project_id, question=question, method=method)


class CaseServiceTests(unittest.TestCase):
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
        now = datetime.now(timezone.utc).isoformat()
        source = Evidence(
            evidence_id="E1",
            rank=1,
            context_id="1",
            text_unit_id="tu-1",
            chunk_id="chunk-1",
            section_id="WW",
            section_name="Wiper",
            document_id="WW.pdf",
            page=25,
            block_id="b01",
            text="檢查雨刷保險絲",
        )
        self.query = QueryResult(
            query_id="query-1",
            project_id="L33-SM3E",
            question="雨刷不會動",
            method="local",
            status="COMPLETED",
            answer="先檢查保險絲。",
            error=None,
            started_at=now,
            completed_at=now,
            duration_seconds=0.1,
            evidence=(source,),
        )
        self.service = CaseService(self.store, StoredQueries([self.query]))  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_save_uses_persisted_query_and_preserves_evidence(self) -> None:
        record = self.service.save_query("L33-SM3E", "query-1", "後續追蹤")

        self.assertEqual(record.question, "雨刷不會動")
        self.assertEqual(record.evidence[0].document_id, "WW.pdf")
        self.assertEqual(self.service.list("L33-SM3E"), [record])

    def test_rejects_unknown_or_duplicate_query(self) -> None:
        with self.assertRaisesRegex(ProjectError, "找不到"):
            self.service.save_query("L33-SM3E", "invented")
        self.service.save_query("L33-SM3E", "query-1")
        with self.assertRaisesRegex(ProjectError, "已加入"):
            self.service.save_query("L33-SM3E", "query-1")

    def test_export_writes_json_and_csv(self) -> None:
        self.service.save_query("L33-SM3E", "query-1", "追蹤")

        json_path, csv_path = self.service.export("L33-SM3E")

        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))[0]["query_id"], "query-1")
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["note"], "追蹤")

    def test_portal_binds_query_to_enabled_indexed_project(self) -> None:
        self.store.update_status("L33-SM3E", "INDEXED")
        queries = StoredQueries([self.query])
        portal = UserPortalService(self.store, queries)  # type: ignore[arg-type]

        result = portal.ask("L33-SM3E", "新的維修問題")

        self.assertEqual(result.project_id, "L33-SM3E")
        self.assertEqual(queries.calls, [("L33-SM3E", "新的維修問題", "local")])
        self.store.set_enabled("L33-SM3E", False)
        with self.assertRaisesRegex(ProjectError, "未啟用"):
            portal.ask("L33-SM3E", "不可執行")
        self.assertEqual(len(queries.calls), 1)

    def test_user_and_admin_apps_expose_separate_controls(self) -> None:
        portal_app = create_portal_app(self.root / "portal-projects")
        admin_app = create_app(self.root / "admin-projects")

        portal_labels = {
            component.get_config().get("label")
            for component in portal_app.blocks.values()
            if hasattr(component, "get_config")
        }
        admin_labels = {
            component.get_config().get("label")
            for component in admin_app.blocks.values()
            if hasattr(component, "get_config")
        }
        self.assertIn("車型／專案", portal_labels)
        self.assertIn("維修問題", portal_labels)
        self.assertIn("回答採用的 Evidence", portal_labels)
        self.assertIn("案例 JSON", portal_labels)
        self.assertNotIn("API Base URL", portal_labels)
        self.assertIn("API Base URL", admin_labels)
        self.assertNotIn("車型／專案", admin_labels)


if __name__ == "__main__":
    unittest.main()
