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
    DEFAULT_LITELLM_API_BASE_URL,
    JudgeConnectionSettings,
    JudgeConnectionTestResult,
    chat_completion_api_params,
    configure_completion_model,
    judge_completion_api_params,
    judge_model_label,
    format_judge_models,
    parse_judge_models,
    parse_litellm_model_list,
    parse_litellm_reasoning_models,
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

    def test_luna_is_default_chat_model(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save("https://api.openai.com/v1", "shared-key")

        self.assertEqual(settings.get_chat_model(), "gpt-6-luna")
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
            chat_completion_api_params("gpt-6-luna", 4000, "high"),
            {"max_completion_tokens": 4000, "reasoning_effort": "high"},
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



class JudgeConnectionSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name) / "projects"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_unconfigured_litellm_offers_no_models(self) -> None:
        settings = JudgeConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(settings.is_configured())
            self.assertEqual(settings.available_models(), ())
            self.assertEqual(settings.get_api_base_url(), DEFAULT_LITELLM_API_BASE_URL)
            with self.assertRaisesRegex(ProjectError, "清單"):
                settings.endpoint_for("litellm:claude-judge")

    def test_saved_litellm_settings_are_separate_from_graphrag_connection(self) -> None:
        ConnectionSettings(self.project_root).save("https://api.openai.com/v1", "sk-openai")
        settings = JudgeConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True):
            settings.save(
                "http://localhost:4000/", "sk-litellm-9876",
                ["claude-judge", "gemini-flash", "claude-judge", "litellm:gpt-4o-mini"],
                ["claude-judge", "unknown"],
            )
            reloaded = JudgeConnectionSettings(self.project_root)
            self.assertEqual(
                reloaded.available_models(),
                ("litellm:claude-judge", "litellm:gemini-flash", "litellm:gpt-4o-mini"),
            )
            self.assertEqual(reloaded.get_reasoning_models(), ("claude-judge",))
            endpoint = reloaded.endpoint_for("litellm:claude-judge")
            self.assertEqual(endpoint.api_base_url, "http://localhost:4000")
            self.assertEqual(endpoint.api_key, "sk-litellm-9876")
            self.assertTrue(endpoint.supports_reasoning)
            self.assertFalse(reloaded.endpoint_for("litellm:gemini-flash").supports_reasoning)
            self.assertTrue(reloaded.supports_reasoning("litellm:claude-judge"))
            self.assertFalse(reloaded.supports_reasoning("claude-judge"))
            self.assertEqual(reloaded.masked_api_key(), "已設定（••••9876）")
            self.assertNotIn("GRAPHRAG_API_KEY", os.environ)
            with self.assertRaisesRegex(ProjectError, "清單"):
                reloaded.endpoint_for("litellm:unlisted")
            with self.assertRaisesRegex(ProjectError, "不支援"):
                reloaded.endpoint_for("gpt-4o-mini")
        self.assertEqual(stat.S_IMODE(settings.path.stat().st_mode), 0o600)
        self.assertEqual(ConnectionSettings(self.project_root).get_api_key(), "sk-openai")

    def test_blank_key_on_save_keeps_existing_key_and_key_is_optional(self) -> None:
        settings = JudgeConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True):
            settings.save(DEFAULT_LITELLM_API_BASE_URL, "sk-first", ["a"])
            settings.save(DEFAULT_LITELLM_API_BASE_URL, "", "b")
            self.assertEqual(settings.get_api_key(), "sk-first")
            self.assertEqual(settings.get_models(), ("b",))
            other = JudgeConnectionSettings(Path(self.temporary_directory.name) / "other")
            other.save(DEFAULT_LITELLM_API_BASE_URL, "", ["open-proxy"])
            self.assertEqual(other.endpoint_for("litellm:open-proxy").api_key, "")

    def test_environment_key_is_used_when_no_saved_key(self) -> None:
        settings = JudgeConnectionSettings(self.project_root)
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-env"}, clear=True):
            settings.save(DEFAULT_LITELLM_API_BASE_URL, "", ["claude-judge"])
            self.assertEqual(settings.endpoint_for("litellm:claude-judge").api_key, "sk-env")
            self.assertNotIn("api_key", settings.path.read_text(encoding="utf-8"))

    def test_save_rejects_invalid_values(self) -> None:
        settings = JudgeConnectionSettings(self.project_root)
        with self.assertRaisesRegex(ProjectError, "至少"):
            settings.save(DEFAULT_LITELLM_API_BASE_URL, "k", " , ")
        with self.assertRaisesRegex(ProjectError, "空白"):
            settings.save(DEFAULT_LITELLM_API_BASE_URL, "k", ["bad model"])
        with self.assertRaisesRegex(ProjectError, "HTTP"):
            settings.save("ftp://proxy", "k", ["a"])
        self.assertFalse(settings.path.exists())

    def test_connection_test_uses_unsaved_values(self) -> None:
        tested: list[tuple[str, str]] = []
        settings = JudgeConnectionSettings(
            self.project_root,
            tester=lambda url, key: tested.append((url, key)) or JudgeConnectionTestResult(True, "ok", ("a",)),
        )
        result = settings.test("http://proxy.test:4000/", "sk-unsaved")
        self.assertTrue(result.success)
        self.assertEqual(result.models, ("a",))
        self.assertEqual(tested, [("http://proxy.test:4000", "sk-unsaved")])
        self.assertFalse(settings.path.exists())

    def test_litellm_test_lists_aliases_and_reasoning_support(self) -> None:
        responses = {
            "http://proxy:4000/v1/models": {"data": [{"id": "gemini-flash"}, {"id": "claude-judge"}]},
            "http://proxy:4000/model/info": {"data": [
                {"model_name": "claude-judge", "model_info": {"supports_reasoning": True}},
                {"model_name": "gemini-flash", "model_info": {"supports_reasoning": None}},
                {"model_name": "not-listed", "model_info": {"supports_reasoning": True}},
            ]},
        }
        requested: list[tuple[str, str | None]] = []

        def fake_get_json(url: str, api_key: str, access_headers=None) -> object:
            requested.append((url, api_key))
            return responses[url]

        with patch.object(JudgeConnectionSettings, "_get_json", staticmethod(fake_get_json)):
            result = JudgeConnectionSettings(self.project_root).test("http://proxy:4000", "sk-x")
        self.assertTrue(result.success)
        self.assertEqual(result.models, ("claude-judge", "gemini-flash"))
        self.assertEqual(result.reasoning_models, ("claude-judge",))
        self.assertEqual([url for url, _key in requested], list(responses))
        self.assertEqual(parse_litellm_model_list({"data": [{"id": ""}, "x"]}), ())
        self.assertEqual(parse_litellm_reasoning_models(None), ())

    def test_cloudflare_access_credentials_are_environment_headers(self) -> None:
        settings = JudgeConnectionSettings(self.project_root)
        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "client-id.example",
            "CF_ACCESS_CLIENT_SECRET": "client-secret",
        }, clear=True):
            settings.save(DEFAULT_LITELLM_API_BASE_URL, "", ["claude"])
            self.assertEqual(settings.endpoint_for("litellm:claude").extra_headers, {
                "CF-Access-Client-Id": "client-id.example",
                "CF-Access-Client-Secret": "client-secret",
            })
            self.assertNotIn("client-secret", settings.path.read_text(encoding="utf-8"))

    def test_litellm_test_and_judge_use_app_env_with_external_projects_root(self) -> None:
        app_root = Path(self.temporary_directory.name) / "app"
        app_root.mkdir()
        (app_root / ".env").write_text(
            "CF_ACCESS_CLIENT_ID=app-client-id\nCF_ACCESS_CLIENT_SECRET=app-client-secret\n",
            encoding="utf-8",
        )
        settings = JudgeConnectionSettings(Path(self.temporary_directory.name) / "volume" / "projects")
        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "", "CF_ACCESS_CLIENT_SECRET": "",
        }, clear=True), patch("pathlib.Path.cwd", return_value=app_root):
            connectivity_headers = settings._configured_access_headers()
            judge_headers = settings.get_access_headers()
        self.assertEqual(judge_headers, connectivity_headers)
        self.assertEqual(judge_headers, {
            "CF-Access-Client-Id": "app-client-id",
            "CF-Access-Client-Secret": "app-client-secret",
        })

    def test_litellm_connectivity_requests_include_cloudflare_headers(self) -> None:
        captured: list[dict[str, str]] = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return b'{"data": []}'

        def fake_urlopen(request, timeout):
            captured.append(dict(request.header_items()))
            return Response()

        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "id", "CF_ACCESS_CLIENT_SECRET": "secret",
        }, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            result = JudgeConnectionSettings._test_litellm("http://proxy", "lite-key")
        self.assertTrue(result.success)
        self.assertEqual(len(captured), 2)
        for headers in captured:
            self.assertEqual(headers["Cf-access-client-id"], "id")
            self.assertEqual(headers["Cf-access-client-secret"], "secret")
            self.assertEqual(headers["Authorization"], "Bearer lite-key")

    def test_litellm_judge_params_only_send_reasoning_when_supported(self) -> None:
        self.assertEqual(judge_completion_api_params("litellm:claude", 4000, "high"), {"reasoning_effort": "high"})
        self.assertEqual(
            judge_completion_api_params("litellm:gemini", 4000, None), {"temperature": 0, "max_tokens": 4000}
        )
        self.assertEqual(
            judge_completion_api_params("gpt-4o-mini", 4000, "high"), chat_completion_api_params("gpt-4o-mini", 4000)
        )
        with self.assertRaises(ProjectError):
            judge_completion_api_params("litellm:claude", 4000, "none")
        self.assertEqual(judge_model_label("litellm:claude-judge"), "LiteLLM／claude-judge")
        self.assertEqual(judge_model_label("gpt-4.1-mini"), "gpt-4.1-mini")

    def test_judge_panel_ids_round_trip_through_saved_string(self) -> None:
        self.assertEqual(parse_judge_models("gpt-4o-mini, litellm:claude , gpt-4o-mini"), ("gpt-4o-mini", "litellm:claude"))
        self.assertEqual(parse_judge_models(["gpt-4o-mini", "", "litellm:claude"]), ("gpt-4o-mini", "litellm:claude"))
        self.assertEqual(parse_judge_models("—"), ())
        self.assertEqual(format_judge_models(["gpt-4o-mini"]), "gpt-4o-mini")
        self.assertEqual(judge_model_label("gpt-4o-mini, litellm:claude"), "gpt-4o-mini、LiteLLM／claude")
        self.assertEqual(judge_model_label(["litellm:claude"]), "LiteLLM／claude")


if __name__ == "__main__":
    unittest.main()
