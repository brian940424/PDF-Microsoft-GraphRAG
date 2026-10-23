import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.app import create_app
from automotive_graphrag.connections import API_KEY_ENVIRONMENT_VARIABLE, ConnectionSettings
from automotive_graphrag.projects import ProjectStore

LOCKED_TABS = ("1 文件與建圖", "2 問答測試", "3 自動問答測試", "4 檢索實驗")


class WorkspaceLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "projects"
        self.environment = patch.dict(os.environ, {API_KEY_ENVIRONMENT_VARIABLE: ""})
        self.environment.start()
        ProjectStore(self.root).create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        self.app = create_app(self.root)
        self.workspace_tab_states = next(
            fn.fn for fn in self.app.fns.values()
            if getattr(fn.fn, "__name__", "") == "workspace_tab_states"
        )

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

    def test_tabs_start_locked(self) -> None:
        tab_states = {
            component.get_config().get("label"): component.get_config().get("interactive")
            for component in self.app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") in LOCKED_TABS
        }
        self.assertEqual(tab_states, dict.fromkeys(LOCKED_TABS, False))

    def test_lock_reason_names_missing_project_and_api_key(self) -> None:
        banner, *tabs = self.workspace_tab_states(None)
        self.assertIn("尚未開啟專案", banner)
        self.assertIn("API Key 尚未設定", banner)
        self.assertTrue(all(tab.interactive is False for tab in tabs))

        banner, *tabs = self.workspace_tab_states("L33-SM3E")
        self.assertNotIn("尚未開啟專案", banner)
        self.assertIn("API Key 尚未設定", banner)
        self.assertTrue(all(tab.interactive is False for tab in tabs))

        banner, *tabs = self.workspace_tab_states("deleted-project")
        self.assertIn("目前開啟的專案已不存在", banner)

    def test_tabs_unlock_when_project_open_and_api_key_set(self) -> None:
        ConnectionSettings(self.root).save("https://api.openai.com/v1", "sk-test-1234")

        banner, *tabs = self.workspace_tab_states(None)
        self.assertIn("尚未開啟專案", banner)
        self.assertNotIn("API Key", banner)

        banner, *tabs = self.workspace_tab_states("L33-SM3E")
        self.assertIn("目前開啟專案", banner)
        self.assertTrue(all(tab.interactive is True for tab in tabs))


if __name__ == "__main__":
    unittest.main()
