import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.connections import (
    ConnectionSettings,
    ConnectionTestResult,
    DEFAULT_API_BASE_URL,
    DEFAULT_CHAT_MODELS,
    chat_completion_api_params,
    configure_completion_model,
    configure_embedding_model,
    extract_json_object,
    format_judge_models,
    graphrag_request_headers,
    judge_model_label,
    llm_request_headers,
    parse_judge_models,
    parse_litellm_embedding_models,
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

    def test_models_outside_the_saved_list_are_rejected_at_read_time(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.path.parent.mkdir(parents=True)
        settings.path.write_text(
            '{"api_key": "k", "chat_model": "removed", "models": ["gpt-4o-mini"]}', encoding="utf-8"
        )
        with self.assertRaisesRegex(ProjectError, "可用清單"):
            settings.get_chat_model()

    def test_chat_parameters_follow_reasoning_support_not_model_name(self) -> None:
        self.assertEqual(chat_completion_api_params(4000, "high"), {"reasoning_effort": "high"})
        self.assertEqual(chat_completion_api_params(4000, None), {"temperature": 0, "max_tokens": 4000})
        with self.assertRaises(ProjectError):
            chat_completion_api_params(4000, "none")

    def test_reasoning_models_keep_the_parameter_through_graphrag_client(self) -> None:
        model_config = {
            "model": "gpt-4o-mini",
            "call_args": {"temperature": 0, "top_p": 1, "max_tokens": 1200},
        }
        with patch("automotive_graphrag.connections._cloudflare_access_headers", return_value={}):
            configure_completion_model(model_config, "gemini-flash", "medium")
            self.assertEqual(model_config["model"], "gemini-flash")
            self.assertEqual(model_config["call_args"], {
                "max_completion_tokens": 1200,
                "reasoning_effort": "medium",
                "allowed_openai_params": ["reasoning_effort"],
                "extra_headers": {"User-Agent": "automotive-graphrag/0.1"},
            })

            configure_completion_model(model_config, "gpt-4o-mini", None)
            self.assertEqual(model_config["call_args"], {
                "max_tokens": 1200, "extra_headers": {"User-Agent": "automotive-graphrag/0.1"},
            })

    def test_graphrag_settings_reference_cloudflare_credentials_by_variable(self) -> None:
        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "id-value", "CF_ACCESS_CLIENT_SECRET": "secret-value",
        }, clear=True):
            expected = {
                "User-Agent": "automotive-graphrag/0.1",
                "CF-Access-Client-Id": "${CF_ACCESS_CLIENT_ID}",
                "CF-Access-Client-Secret": "${CF_ACCESS_CLIENT_SECRET}",
            }
            self.assertEqual(graphrag_request_headers(), expected)
            embedding = {"model": "old"}
            configure_embedding_model(embedding, "new-embedding")
            self.assertEqual(embedding, {"model": "new-embedding", "call_args": {"extra_headers": expected}})

    def test_apply_to_environment_exports_cloudflare_values_from_app_dotenv(self) -> None:
        app_root = Path(self.temporary_directory.name) / "app"
        app_root.mkdir()
        (app_root / ".env").write_text(
            "CF_ACCESS_CLIENT_ID=app-id\nCF_ACCESS_CLIENT_SECRET=app-secret\n", encoding="utf-8"
        )
        settings = ConnectionSettings(self.project_root)
        settings.save(DEFAULT_API_BASE_URL, "k")
        with patch.dict(os.environ, {}, clear=True), patch("pathlib.Path.cwd", return_value=app_root):
            settings.apply_to_environment()
            self.assertEqual(os.environ["CF_ACCESS_CLIENT_ID"], "app-id")
            self.assertEqual(os.environ["CF_ACCESS_CLIENT_SECRET"], "app-secret")
        self.assertNotIn("app-secret", settings.path.read_text(encoding="utf-8"))

    def test_direct_requests_send_identity_and_cloudflare_headers(self) -> None:
        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "id", "CF_ACCESS_CLIENT_SECRET": "secret",
        }, clear=True):
            self.assertEqual(llm_request_headers("k"), {
                "Content-Type": "application/json",
                "User-Agent": "automotive-graphrag/0.1",
                "Authorization": "Bearer k",
                "CF-Access-Client-Id": "id",
                "CF-Access-Client-Secret": "secret",
            })

    def test_base_url_is_normalized_to_the_openai_compatible_root(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save("http://localhost:4000/", "k")
        self.assertEqual(settings.get_api_base_url(), "http://localhost:4000/v1")

    def test_saved_model_lists_drive_choices_and_reasoning(self) -> None:
        settings = ConnectionSettings(self.project_root)
        self.assertEqual(settings.available_chat_models(), DEFAULT_CHAT_MODELS)
        settings.save(
            DEFAULT_API_BASE_URL, "sk-litellm-9876", "claude", "embed-a",
            ["claude", "gemini", "claude"], ["claude", "unknown"], ["embed-a"],
        )
        reloaded = ConnectionSettings(self.project_root)
        self.assertEqual(reloaded.available_chat_models(), ("claude", "gemini"))
        self.assertEqual(reloaded.available_embedding_models(), ("embed-a",))
        self.assertEqual(reloaded.get_reasoning_models(), ("claude",))
        self.assertEqual(reloaded.reasoning_effort_for("claude", "high"), "high")
        self.assertIsNone(reloaded.reasoning_effort_for("gemini", "high"))
        self.assertEqual(reloaded.masked_api_key(), "已設定（••••9876）")
        self.assertEqual(stat.S_IMODE(reloaded.path.stat().st_mode), 0o600)

    def test_default_models_are_always_kept_selectable(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save(DEFAULT_API_BASE_URL, "k", "claude", "embed-a", ["gemini"], [], [])
        self.assertEqual(settings.available_chat_models(), ("gemini", "claude"))
        self.assertEqual(settings.available_embedding_models(), ("embed-a",))
        self.assertEqual(settings.get_chat_model(), "claude")

    def test_save_without_lists_keeps_saved_lists_and_blank_key_keeps_key(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save(DEFAULT_API_BASE_URL, "sk-first", "claude", "embed-a", ["claude"], ["claude"], ["embed-a"])
        settings.save("http://other:4000", "", "claude", "embed-a")
        self.assertEqual(settings.get_api_key(), "sk-first")
        self.assertEqual(settings.available_chat_models(), ("claude",))
        self.assertTrue(settings.supports_reasoning("claude"))

    def test_saved_key_wins_over_environment_after_restart(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save(DEFAULT_API_BASE_URL, "sk-saved")
        with patch.dict(os.environ, {"GRAPHRAG_API_KEY": "old", "LITELLM_API_KEY": "other"}, clear=True):
            self.assertEqual(settings.get_api_key(), "sk-saved")

    def test_environment_key_is_used_when_no_saved_key(self) -> None:
        settings = ConnectionSettings(self.project_root)
        with patch.dict(os.environ, {"LITELLM_API_KEY": "sk-env", "GRAPHRAG_API_KEY": "old"}, clear=True):
            self.assertEqual(settings.get_api_key(), "sk-env")

    def test_environment_key_wins_over_project_dotenv(self) -> None:
        graph_root = self.project_root / "L33-SM3E" / "graphrag"
        graph_root.mkdir(parents=True)
        (graph_root / ".env").write_text("GRAPHRAG_API_KEY=dotenv-key\n", encoding="utf-8")
        settings = ConnectionSettings(self.project_root)
        with patch.dict(os.environ, {"GRAPHRAG_API_KEY": "env-key"}, clear=True):
            self.assertEqual(settings.get_api_key("L33-SM3E"), "env-key")

    def test_saved_connection_wins_over_environment(self) -> None:
        settings = ConnectionSettings(self.project_root)
        settings.save("http://saved:4000/v1", "sk-saved", "gpt-4o-mini", "text-embedding-3-large")
        environment = {
            "GRAPHRAG_API_BASE": "http://env:4000/v1",
            "GRAPHRAG_CHAT_MODEL": "gpt-4.1-mini",
            "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-small",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(settings.get_api_base_url(), "http://saved:4000/v1")
            self.assertEqual(settings.get_chat_model(), "gpt-4o-mini")
            self.assertEqual(settings.get_embedding_model(), "text-embedding-3-large")

    def test_environment_connection_is_used_when_nothing_is_saved(self) -> None:
        settings = ConnectionSettings(self.project_root)
        environment = {
            "GRAPHRAG_API_BASE": "http://env:4000/v1",
            "GRAPHRAG_CHAT_MODEL": "gpt-4.1-mini",
            "GRAPHRAG_EMBEDDING_MODEL": "text-embedding-3-large",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(settings.get_api_base_url(), "http://env:4000/v1")
            self.assertEqual(settings.get_chat_model(), "gpt-4.1-mini")
            self.assertEqual(settings.get_embedding_model(), "text-embedding-3-large")

    def test_local_dotenv_only_supplies_the_key(self) -> None:
        self.project_root.mkdir(parents=True)
        (self.project_root.parent / ".env").write_text(
            "GRAPHRAG_API_KEY=dotenv-key\n"
            "GRAPHRAG_API_BASE=http://dotenv:4000/v1\n"
            "GRAPHRAG_CHAT_MODEL=gpt-4.1-mini\n"
            "GRAPHRAG_EMBEDDING_MODEL=text-embedding-3-large\n",
            encoding="utf-8",
        )
        settings = ConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.get_api_key(), "dotenv-key")
            self.assertEqual(settings.get_api_base_url(), DEFAULT_API_BASE_URL)
            self.assertEqual(settings.get_chat_model(), "gpt-6-luna")
            self.assertEqual(settings.get_embedding_model(), "text-embedding-3-small")

    def test_save_rejects_invalid_values(self) -> None:
        settings = ConnectionSettings(self.project_root)
        with self.assertRaisesRegex(ProjectError, "空白"):
            settings.save(DEFAULT_API_BASE_URL, "k", "gpt-6-luna", "e", ["bad model"])
        with self.assertRaisesRegex(ProjectError, "HTTP"):
            settings.save("ftp://proxy", "k")
        self.assertFalse(settings.path.exists())

    def test_former_connections_merge_into_the_proxy_on_first_read(self) -> None:
        self.project_root.mkdir(parents=True)
        (self.project_root / ".connection.json").write_text(json.dumps({
            "api_base_url": "http://127.0.0.1:8899/v1", "api_key": "sk-old-openai",
            "chat_model": "gpt-6-luna", "embedding_model": "text-embedding-3-small",
        }), encoding="utf-8")
        (self.project_root / ".judge_connection.json").write_text(json.dumps({
            "provider": "litellm", "api_base_url": "https://llm.example", "api_key": "sk-proxy",
            "models": ["gemini-flash", "gpt-6-luna"], "reasoning_models": ["gemini-flash"],
        }), encoding="utf-8")
        settings = ConnectionSettings(self.project_root)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.get_api_base_url(), "https://llm.example/v1")
            self.assertEqual(settings.get_api_key(), "sk-proxy")
            self.assertEqual(settings.get_chat_model(), "gpt-6-luna")
            self.assertEqual(settings.get_embedding_model(), "text-embedding-3-small")
            self.assertEqual(
                settings.available_chat_models(), ("gemini-flash", "gpt-6-luna", "gpt-4o-mini", "gpt-4.1-mini")
            )
            self.assertEqual(settings.get_reasoning_models(), ("gemini-flash", "gpt-6-luna"))
            settings.save("https://new.example", "", "gpt-4o-mini", "text-embedding-3-small")
            self.assertEqual(settings.get_api_base_url(), "https://new.example/v1")
            self.assertEqual(settings.get_api_key(), "sk-proxy")
            self.assertEqual(settings.get_chat_model(), "gpt-4o-mini")

    def test_connection_test_uses_unsaved_values(self) -> None:
        tested: list[tuple[str, str]] = []
        settings = ConnectionSettings(
            self.project_root,
            tester=lambda url, key: tested.append((url, key)) or ConnectionTestResult(True, "ok", ("a",)),
        )
        result = settings.test("http://proxy.test:4000/", "sk-unsaved")
        self.assertTrue(result.success)
        self.assertEqual(result.models, ("a",))
        self.assertEqual(tested, [("http://proxy.test:4000/v1", "sk-unsaved")])
        self.assertFalse(settings.path.exists())

    def test_litellm_test_separates_chat_embedding_and_reasoning_models(self) -> None:
        responses = {
            "http://proxy:4000/v1/models": {"data": [
                {"id": "gemini-flash"}, {"id": "claude-judge"}, {"id": "my-vectors"}, {"id": "text-embedding-3-small"},
            ]},
            "http://proxy:4000/model/info": {"data": [
                {"model_name": "claude-judge", "model_info": {"supports_reasoning": True, "mode": "chat"}},
                {"model_name": "gemini-flash", "model_info": {"supports_reasoning": None, "mode": "chat"}},
                {"model_name": "my-vectors", "model_info": {"mode": "embedding"}},
                {"model_name": "not-listed", "model_info": {"supports_reasoning": True}},
            ]},
        }
        requested: list[tuple[str, str | None]] = []

        def fake_get_json(url: str, api_key: str, access_headers=None) -> object:
            requested.append((url, api_key))
            return responses[url]

        with patch.object(ConnectionSettings, "_get_json", staticmethod(fake_get_json)):
            result = ConnectionSettings(self.project_root).test("http://proxy:4000", "sk-x")
        self.assertTrue(result.success)
        self.assertEqual(result.models, ("claude-judge", "gemini-flash"))
        self.assertEqual(result.reasoning_models, ("claude-judge",))
        self.assertEqual(result.embedding_models, ("my-vectors", "text-embedding-3-small"))
        self.assertEqual([url for url, _key in requested], list(responses))
        self.assertEqual(parse_litellm_model_list({"data": [{"id": ""}, "x"]}), ())
        self.assertEqual(parse_litellm_reasoning_models(None), ())
        self.assertEqual(parse_litellm_embedding_models(None, ("embed-x", "chat")), ("embed-x",))

    def test_access_credentials_use_app_env_with_external_projects_root(self) -> None:
        app_root = Path(self.temporary_directory.name) / "app"
        app_root.mkdir()
        (app_root / ".env").write_text(
            "CF_ACCESS_CLIENT_ID=app-client-id\nCF_ACCESS_CLIENT_SECRET=app-client-secret\n",
            encoding="utf-8",
        )
        settings = ConnectionSettings(Path(self.temporary_directory.name) / "volume" / "projects")
        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "", "CF_ACCESS_CLIENT_SECRET": "",
        }, clear=True), patch("pathlib.Path.cwd", return_value=app_root):
            headers = settings.get_access_headers()
        self.assertEqual(headers, {
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
            result = ConnectionSettings._test_litellm("http://proxy/v1", "lite-key")
        self.assertTrue(result.success)
        self.assertEqual(len(captured), 2)
        for headers in captured:
            self.assertEqual(headers["Cf-access-client-id"], "id")
            self.assertEqual(headers["Cf-access-client-secret"], "secret")
            self.assertEqual(headers["Authorization"], "Bearer lite-key")

    def test_judge_ids_drop_the_former_prefix_and_show_plain_names(self) -> None:
        self.assertEqual(parse_judge_models("gpt-4o-mini, litellm:claude , litellm:gpt-4o-mini"), ("gpt-4o-mini", "claude"))
        self.assertEqual(parse_judge_models(["gpt-4o-mini", "", "litellm:claude"]), ("gpt-4o-mini", "claude"))
        self.assertEqual(parse_judge_models("—"), ())
        self.assertEqual(format_judge_models(["litellm:claude"]), "claude")
        self.assertEqual(judge_model_label("gpt-4o-mini, litellm:claude"), "gpt-4o-mini、claude")
        self.assertEqual(judge_model_label(["litellm:claude"]), "claude")

    def test_json_replies_are_recovered_from_fenced_text(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertIsNone(extract_json_object("no json"))


if __name__ == "__main__":
    unittest.main()
