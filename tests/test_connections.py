import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.connections import ConnectionSettings, ConnectionTestResult
from automotive_graphrag.projects import ProjectError


class ConnectionSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name) / "projects"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_saved_key_is_shared_and_applied_to_environment(self) -> None:
        first = ConnectionSettings(self.project_root)
        first.save_api_key("sk-shared-1234")

        second = ConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(second.apply_to_environment(), "sk-shared-1234")
            self.assertEqual(os.environ["GRAPHRAG_API_KEY"], "sk-shared-1234")

        self.assertEqual(second.masked_api_key(), "已設定（••••1234）")
        self.assertEqual(stat.S_IMODE(second.path.stat().st_mode), 0o600)

    def test_test_uses_unsaved_key_without_persisting_it(self) -> None:
        tested: list[str] = []
        settings = ConnectionSettings(
            self.project_root,
            lambda base_url, key: tested.append(f"{base_url}|{key}") or ConnectionTestResult(True, "連線成功"),
        )

        result = settings.test("https://gateway.example/v1/", "candidate-key")

        self.assertTrue(result.success)
        self.assertEqual(tested, ["https://gateway.example/v1|candidate-key"])
        self.assertFalse(settings.path.exists())

    def test_missing_key_blocks_graphrag_connection(self) -> None:
        settings = ConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ProjectError, "尚未設定"):
            settings.apply_to_environment()

    def test_blank_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProjectError, "必填"):
            ConnectionSettings(self.project_root).save_api_key("  ")

    def test_base_url_is_persisted_and_applied(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save("https://gateway.example/v1/", "shared-key")

        with patch.dict(os.environ, {}, clear=True):
            settings.apply_to_environment()
            self.assertEqual(os.environ["GRAPHRAG_API_BASE"], "https://gateway.example/v1")
        self.assertEqual(settings.get_api_base_url(), "https://gateway.example/v1")

    def test_invalid_base_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ProjectError, "HTTP"):
            ConnectionSettings(self.project_root).save("gateway.example", "shared-key")


if __name__ == "__main__":
    unittest.main()
