import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.cases import CaseService
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult


class StoredQueries:
    def __init__(self, results: list[QueryResult]) -> None:
        self.results = results

    def history(self, project_id: str) -> list[QueryResult]:
        return [item for item in self.results if item.project_id == project_id]


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

    def test_app_builds_with_unified_query_and_case_controls(self) -> None:
        app = create_app(self.root / "ui-projects")

        labels = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }
        self.assertIn("車型／專案", labels)
        self.assertIn("維修問題", labels)
        self.assertIn("回答採用的 Evidence", labels)
        self.assertIn("案例 JSON", labels)


if __name__ == "__main__":
    unittest.main()
