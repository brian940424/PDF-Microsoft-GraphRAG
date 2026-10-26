import html
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automotive_graphrag.dashboard3d import build_dashboard_data, dashboard_iframe, render_dashboard_html
from automotive_graphrag.projects import ProjectStore


class Dashboard3DTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.project = self.store.create(
            project_id="demo", display_name="Demo", vehicle_name="V", manual_version="1", description=""
        )
        self.project_path = self.store.path_for("demo")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_index(self) -> None:
        output = self.project_path / "graphrag" / "output"
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "id": ["e1", "e2", "e3"],
            "title": ["ENGINE", "BRAKE", "LONELY </script>"],
            "type": ["EVENT", "ORGANIZATION", "EVENT"],
            "description": ["x" * 1000, "brake", None],
            "frequency": [3, 2, 1],
            "degree": [1, 1, 0],
        }).to_parquet(output / "entities.parquet")
        pd.DataFrame({
            "source": ["ENGINE", "ENGINE"],
            "target": ["BRAKE", "MISSING"],
            "description": ["linked", "dangling"],
            "weight": [2.0, 1.0],
        }).to_parquet(output / "relationships.parquet")
        pd.DataFrame({
            "community": [0, 1],
            "level": [0, 1],
            "title": ["Community 0", "Community 1"],
            "entity_ids": [["e1", "e2"], ["e1"]],
        }).to_parquet(output / "communities.parquet")
        pd.DataFrame({
            "community": [0],
            "level": [0],
            "title": ["Drivetrain"],
            "summary": ["summary"],
            "rank": [7.5],
        }).to_parquet(output / "community_reports.parquet")
        pd.DataFrame({"id": ["d1", "d2"]}).to_parquet(output / "documents.parquet")
        (output / "stats.json").write_text(json.dumps({"total_runtime": 12.34}), encoding="utf-8")
        queries = [
            {"question": "q1", "method": "local", "status": "COMPLETED", "completed_at": "2026-10-23T21:40:07"},
            {"question": "q2", "method": "global", "status": "FAILED", "started_at": "2026-10-23T21:41:00"},
        ]
        (self.project_path / "runs" / "queries.jsonl").write_text(
            "\n".join(json.dumps(query) for query in queries) + "\nnot json\n", encoding="utf-8"
        )

    def test_returns_none_before_indexing(self) -> None:
        self.assertIsNone(build_dashboard_data(self.project, self.project_path))
        self.assertIn("尚未建圖", dashboard_iframe(self.project, self.project_path))
        self.assertIn("開啟專案", dashboard_iframe(None, None))

    def test_builds_regions_nodes_links_and_events(self) -> None:
        self.write_index()

        data = build_dashboard_data(self.project, self.project_path)

        self.assertEqual(data["regions"], [{"c": 0, "t": "Drivetrain", "rank": 7.5, "s": "summary", "n": 2}])
        self.assertEqual([node["r"] for node in data["nodes"]], [0, 0, -1])
        self.assertLessEqual(len(data["nodes"][0]["d"]), 320)
        self.assertEqual(data["nodes"][2]["d"], "")
        self.assertEqual(data["links"], [[0, 1, 2.0, "linked"]])
        self.assertEqual(data["stats"]["documents"], 2)
        self.assertEqual(data["stats"]["index_runtime_s"], 12.3)
        self.assertEqual((data["stats"]["queries"], data["stats"]["queries_completed"]), (2, 1))
        self.assertEqual([event["q"] for event in data["events"]], ["q2", "q1"])
        self.assertEqual(data["types"][0], {"name": "EVENT", "count": 2})

    def test_alerts_list_failed_queries_isolated_entities_and_frontier(self) -> None:
        self.write_index()

        alerts = build_dashboard_data(self.project, self.project_path)["alerts"]

        self.assertEqual([alert["tone"] for alert in alerts], ["bad", "warn", "info"])
        self.assertEqual(alerts[0], {"tone": "bad", "title": "問答未完成：FAILED", "body": "q2", "sel": {"kind": "query", "k": 0}})
        self.assertEqual(alerts[1]["sel"], {"kind": "node", "i": 2})
        self.assertEqual(alerts[2]["sel"], {"kind": "region", "r": -1})

    def test_rendered_page_embeds_data_safely_in_iframe(self) -> None:
        self.write_index()
        data = build_dashboard_data(self.project, self.project_path)

        page = render_dashboard_html(data)
        self.assertNotIn("LONELY </script>", page)
        self.assertNotIn("__DASHBOARD_DATA__", page)
        self.assertNotIn("__THREE_VERSION__", page)
        self.assertNotIn("__WORLD_JS__", page)
        # 城鎮、車流與物流倉庫的共用模擬（由 progress-dashboard/lib/world.ts 產生）已內嵌
        self.assertIn("function createTraffic(", page)
        self.assertIn("function layoutTown(", page)
        self.assertIn("three@0.160.0", page)
        embedded = page.split('id="data">', 1)[1].split("</script>", 1)[0]
        self.assertEqual(json.loads(embedded), data)

        frame = dashboard_iframe(self.project, self.project_path)
        self.assertTrue(frame.startswith("<iframe"))
        srcdoc = frame.split('srcdoc="', 1)[1].split('" style=', 1)[0]
        self.assertEqual(html.unescape(srcdoc), page)

    def test_render_loop_stops_when_hidden_or_paused_and_idle(self) -> None:
        self.write_index()
        page = render_dashboard_html(build_dashboard_data(self.project, self.project_path))

        # 看不見時不排下一幀
        self.assertIn("new IntersectionObserver(", page)
        self.assertIn('document.addEventListener("visibilitychange", wake)', page)
        self.assertIn("if (frameRequested || !onScreen || document.hidden) return;", page)
        # 暫停且閒置時停止重畫；互動與鏡頭阻尼會喚醒
        self.assertIn("if (!paused || goal || performance.now() < awakeUntil) scheduleFrame();", page)
        self.assertIn("if (viewMoved) awakeUntil = performance.now() + IDLE_GRACE_MS;", page)
        self.assertNotIn('controls.addEventListener("change", wake)', page)
        self.assertNotIn("updateTag();\n  requestAnimationFrame(tick);", page)


if __name__ == "__main__":
    unittest.main()
