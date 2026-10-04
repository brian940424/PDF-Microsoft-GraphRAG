import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.connections import (
    ALLOWED_CHAT_MODELS,
    ConnectionSettings,
    ConnectionTestResult,
    chat_completion_api_params,
    configure_completion_model,
)
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

    def test_project_dotenv_key_is_read_and_applied(self) -> None:
        graph_root = self.project_root / "L33-SM3E" / "graphrag"
        graph_root.mkdir(parents=True)
        (graph_root / ".env").write_text("GRAPHRAG_API_KEY=project-dotenv-key\n", encoding="utf-8")
        settings = ConnectionSettings(self.project_root)

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.get_api_key(), "project-dotenv-key")
            self.assertEqual(settings.get_api_key("L33-SM3E"), "project-dotenv-key")
            self.assertEqual(settings.apply_to_environment("L33-SM3E"), "project-dotenv-key")
            self.assertEqual(os.environ["GRAPHRAG_API_KEY"], "project-dotenv-key")

    def test_saved_key_wins_over_project_dotenv_after_restart(self) -> None:
        graph_root = self.project_root / "L33-SM3E" / "graphrag"
        graph_root.mkdir(parents=True)
        (graph_root / ".env").write_text("GRAPHRAG_API_KEY=old-dotenv-key\n", encoding="utf-8")
        settings = ConnectionSettings(self.project_root)
        settings.save("https://api.openai.com/v1", "new-saved-key")

        # Simulate a fresh app process, where save()'s environment update is gone.
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.get_api_key("L33-SM3E"), "new-saved-key")

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

    def test_low_cost_models_are_defaults(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save("https://api.openai.com/v1", "shared-key")

        self.assertEqual(settings.get_chat_model(), "gpt-4o-mini")
        self.assertEqual(settings.get_embedding_model(), "text-embedding-3-small")

    def test_models_are_restricted_to_allowed_options(self) -> None:
        settings = ConnectionSettings(self.project_root)
        with self.assertRaisesRegex(ProjectError, "允許清單"):
            settings.save(
                "https://api.openai.com/v1",
                "shared-key",
                "gpt-4.1",
                "text-embedding-3-small",
            )

    def test_gpt6_luna_uses_reasoning_compatible_chat_completion_parameters(self) -> None:
        self.assertIn("gpt-6-luna", ALLOWED_CHAT_MODELS)
        self.assertEqual(
            chat_completion_api_params("gpt-6-luna", 4000),
            {"max_completion_tokens": 4000, "reasoning_effort": "medium"},
        )
        self.assertEqual(
            chat_completion_api_params("gpt-4o-mini", 4000),
            {"temperature": 0, "max_tokens": 4000},
        )

    def test_gpt6_luna_graph_rag_settings_drop_incompatible_sampling_options(self) -> None:
        model_config = {
            "model": "gpt-4o-mini",
            "call_args": {"temperature": 0, "top_p": 1, "max_tokens": 1200},
        }

        configure_completion_model(model_config, "gpt-6-luna")

        self.assertEqual(model_config["model"], "gpt-6-luna")
        self.assertEqual(
            model_config["call_args"],
            {"max_completion_tokens": 1200, "reasoning_effort": "medium"},
        )

        configure_completion_model(model_config, "gpt-4o-mini")
        self.assertEqual(model_config["call_args"], {"max_tokens": 1200})


if __name__ == "__main__":
    unittest.main()
