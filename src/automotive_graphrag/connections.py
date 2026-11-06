"""Shared GraphRAG API connection settings."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

from .projects import ProjectError


API_KEY_ENVIRONMENT_VARIABLE = "GRAPHRAG_API_KEY"
API_BASE_ENVIRONMENT_VARIABLE = "GRAPHRAG_API_BASE"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
GPT6_LUNA_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORT_OPTIONS = ("low", "medium", "high")
ALLOWED_CHAT_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", GPT6_LUNA_MODEL)
ALLOWED_EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = GPT6_LUNA_MODEL
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def chat_completion_api_params(
    model: str, max_tokens: int, reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> dict[str, object]:
    """Return parameters compatible with the selected model's Chat Completions API."""
    if model == GPT6_LUNA_MODEL:
        # GPT-6 reasoning requests use max_completion_tokens and reject sampling
        # parameters such as temperature/top_p when reasoning is enabled.
        return {
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
        }
    return {"temperature": 0, "max_tokens": max_tokens}


def configure_completion_model(
    model_config: dict[str, object], model: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> None:
    """Set a GraphRAG model and its model-specific Chat Completions arguments."""
    if reasoning_effort not in REASONING_EFFORT_OPTIONS:
        raise ProjectError("不支援的 Luna 推理深度")
    model_config["model"] = model
    raw_call_args = model_config.get("call_args")
    call_args = dict(raw_call_args) if isinstance(raw_call_args, dict) else {}
    if model != GPT6_LUNA_MODEL:
        # The same project settings file is reused when users switch models.
        # Remove GPT-6-only arguments so they cannot leak into a GPT-4 request.
        call_args.pop("reasoning_effort", None)
        completion_limit = call_args.pop("max_completion_tokens", None)
        if completion_limit is not None:
            call_args.setdefault("max_tokens", completion_limit)
        model_config["call_args"] = call_args
        return
    for key in ("temperature", "top_p", "top_logprobs", "logprobs"):
        call_args.pop(key, None)
    legacy_max_tokens = call_args.pop("max_tokens", None)
    if legacy_max_tokens is not None:
        call_args.setdefault("max_completion_tokens", legacy_max_tokens)
    call_args["reasoning_effort"] = reasoning_effort
    model_config["call_args"] = call_args


@dataclass(frozen=True, slots=True)
class ConnectionTestResult:
    success: bool
    message: str


ConnectionTester = Callable[[str, str], ConnectionTestResult]


class ConnectionSettings:
    """Persist one API endpoint and key shared by every project in a project root."""

    def __init__(self, project_root: str | Path, tester: ConnectionTester | None = None) -> None:
        self.path = Path(project_root) / ".connection.json"
        self.tester = tester or self._test_openai

    def has_api_key(self, project_id: str | None = None) -> bool:
        return self.get_api_key(project_id) is not None

    def masked_api_key(self, project_id: str | None = None) -> str:
        key = self.get_api_key(project_id)
        if not key:
            return "尚未設定"
        return f"已設定（••••{key[-4:]}）"

    def get_api_key(self, project_id: str | None = None) -> str | None:
        key = os.environ.get(API_KEY_ENVIRONMENT_VARIABLE)
        if key and not self._is_placeholder_key(key):
            return key
        value = self._read()
        if value is not None:
            key = value.get("api_key")
            if not isinstance(key, str) or not key:
                raise ProjectError("共用連線設定缺少 API Key")
            return key
        return self._dotenv_api_key(project_id)

    def get_api_base_url(self) -> str:
        value = self._read()
        if value is None:
            return os.environ.get(API_BASE_ENVIRONMENT_VARIABLE, DEFAULT_API_BASE_URL).rstrip("/")
        return self._validate_api_base_url(value.get("api_base_url", DEFAULT_API_BASE_URL))

    def get_chat_model(self) -> str:
        value = self._read()
        model = (
            value.get("chat_model", DEFAULT_CHAT_MODEL)
            if value is not None
            else os.environ.get("GRAPHRAG_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        )
        return self._validate_model(model, ALLOWED_CHAT_MODELS, "Chat")

    def get_embedding_model(self) -> str:
        value = self._read()
        model = (
            value.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            if value is not None
            else os.environ.get("GRAPHRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        return self._validate_model(model, ALLOWED_EMBEDDING_MODELS, "Embedding")

    def _read(self) -> dict[str, str] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            raise ProjectError("共用連線設定無法讀取") from exc
        if not isinstance(value, dict):
            raise ProjectError("共用連線設定格式錯誤")
        return value

    def save_api_key(self, api_key: str) -> None:
        self.save(
            self.get_api_base_url(),
            api_key,
            self.get_chat_model(),
            self.get_embedding_model(),
        )

    def save(
        self,
        api_base_url: str,
        api_key: str,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        base_url = self._validate_api_base_url(api_base_url)
        selected_chat_model = self._validate_model(chat_model, ALLOWED_CHAT_MODELS, "Chat")
        selected_embedding_model = self._validate_model(
            embedding_model,
            ALLOWED_EMBEDDING_MODELS,
            "Embedding",
        )
        key = api_key.strip()
        if not key:
            raise ProjectError("API Key 為必填")
        if "\n" in key or "\r" in key:
            raise ProjectError("API Key 格式不正確")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(dir=self.path.parent, prefix=".connection-", suffix=".json")
        try:
            os.fchmod(handle, 0o600)
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                json.dump(
                    {
                        "api_base_url": base_url,
                        "api_key": key,
                        "chat_model": selected_chat_model,
                        "embedding_model": selected_embedding_model,
                    },
                    temporary,
                )
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        os.environ[API_KEY_ENVIRONMENT_VARIABLE] = key
        os.environ[API_BASE_ENVIRONMENT_VARIABLE] = base_url
        os.environ["GRAPHRAG_CHAT_MODEL"] = selected_chat_model
        os.environ["GRAPHRAG_EMBEDDING_MODEL"] = selected_embedding_model

    def apply_to_environment(self, project_id: str | None = None) -> str:
        key = self.get_api_key(project_id)
        if not key:
            raise ProjectError("尚未設定共用 GRAPHRAG_API_KEY，請先到連線設定頁設定")
        os.environ[API_KEY_ENVIRONMENT_VARIABLE] = key
        os.environ[API_BASE_ENVIRONMENT_VARIABLE] = self.get_api_base_url()
        os.environ["GRAPHRAG_CHAT_MODEL"] = self.get_chat_model()
        os.environ["GRAPHRAG_EMBEDDING_MODEL"] = self.get_embedding_model()
        return key

    def _dotenv_api_key(self, project_id: str | None) -> str | None:
        paths = [self.path.parent.parent / ".env", self.path.parent / ".env"]
        if project_id:
            paths.append(self.path.parent / project_id / "graphrag" / ".env")
        else:
            paths.extend(sorted(self.path.parent.glob("*/graphrag/.env")))
        for path in paths:
            if not path.is_file():
                continue
            key = dotenv_values(path).get(API_KEY_ENVIRONMENT_VARIABLE)
            if isinstance(key, str) and key.strip() and not self._is_placeholder_key(key):
                return key.strip()
        return None

    @staticmethod
    def _is_placeholder_key(key: str) -> bool:
        return key.strip().lower() in {
            "your-api-key",
            "your_api_key",
            "insert_key_here",
            "changeme",
            "<your-api-key>",
        }

    def test(self, api_base_url: str | None = None, api_key: str | None = None) -> ConnectionTestResult:
        base_url = self._validate_api_base_url(api_base_url) if api_base_url else self.get_api_base_url()
        key = api_key.strip() if api_key else self.get_api_key()
        if not key:
            return ConnectionTestResult(False, "尚未設定 API Key")
        return self.tester(base_url, key)

    @staticmethod
    def _validate_api_base_url(api_base_url: object) -> str:
        if not isinstance(api_base_url, str) or not api_base_url.strip():
            raise ProjectError("API Base URL 為必填")
        value = api_base_url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProjectError("API Base URL 必須是有效的 HTTP(S) 網址")
        return value

    @staticmethod
    def _validate_model(model: object, allowed: tuple[str, ...], label: str) -> str:
        if not isinstance(model, str) or model not in allowed:
            raise ProjectError(f"{label} 模型不在允許清單：{', '.join(allowed)}")
        return model

    @staticmethod
    def _test_openai(api_base_url: str, api_key: str) -> ConnectionTestResult:
        request = urllib.request.Request(
            f"{api_base_url}/models",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": "automotive-graphrag/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                if 200 <= response.status < 300:
                    return ConnectionTestResult(True, "OpenAI API 連線成功")
                return ConnectionTestResult(False, f"OpenAI API 回應 HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return ConnectionTestResult(False, "API Key 驗證失敗")
            return ConnectionTestResult(False, f"OpenAI API 回應 HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError):
            return ConnectionTestResult(False, "無法連線至 OpenAI API")



LITELLM_PROVIDER = "litellm"
LITELLM_MODEL_PREFIX = f"{LITELLM_PROVIDER}:"
LITELLM_API_KEY_ENVIRONMENT_VARIABLE = "LITELLM_API_KEY"
CF_ACCESS_CLIENT_ID_ENVIRONMENT_VARIABLE = "CF_ACCESS_CLIENT_ID"
CF_ACCESS_CLIENT_SECRET_ENVIRONMENT_VARIABLE = "CF_ACCESS_CLIENT_SECRET"


def _cloudflare_access_headers() -> dict[str, str]:
    file_values: dict[str, str] = {}
    env_files = (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env")
    for env_file in dict.fromkeys(path.resolve() for path in env_files):
        if env_file.is_file():
            for key, value in dotenv_values(env_file).items():
                if key not in file_values and isinstance(value, str) and value.strip():
                    file_values[key] = value
    headers: dict[str, str] = {}
    for variable, header in (
        (CF_ACCESS_CLIENT_ID_ENVIRONMENT_VARIABLE, "CF-Access-Client-Id"),
        (CF_ACCESS_CLIENT_SECRET_ENVIRONMENT_VARIABLE, "CF-Access-Client-Secret"),
    ):
        value = os.environ.get(variable)
        if value is None:
            value = file_values.get(variable)
        if isinstance(value, str) and value.strip():
            headers[header] = value.strip()
    return headers
DEFAULT_LITELLM_API_BASE_URL = "http://localhost:4000"


def is_litellm_model(model: object) -> bool:
    return isinstance(model, str) and model.startswith(LITELLM_MODEL_PREFIX) and len(model) > len(LITELLM_MODEL_PREFIX)


def litellm_model_id(alias: str) -> str:
    """Saved judge-model ID for a LiteLLM proxy alias; the prefix keeps it apart from OpenAI names."""
    return f"{LITELLM_MODEL_PREFIX}{alias}"


def litellm_alias(model: str) -> str:
    return model.removeprefix(LITELLM_MODEL_PREFIX)


MAX_JUDGE_MODELS = 5


def parse_judge_models(value: object) -> tuple[str, ...]:
    """Normalize one judge model, a list, or the comma-joined form saved in results."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else [value]
    result: list[str] = []
    for item in items:
        model = str(item).strip()
        if model and model != "—" and model not in result:
            result.append(model)
    return tuple(result)


def format_judge_models(models: object) -> str:
    """Saved form of a judge panel; a single judge keeps its plain model ID."""
    return ", ".join(parse_judge_models(models))


class JudgeModelPreferences:
    """Persist the automatic-QA judge selection across app refreshes."""

    def __init__(self, project_root: str | Path) -> None:
        self.path = Path(project_root) / ".judge_model_preferences.json"

    def get_autoqa_models(self) -> tuple[str, ...]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ()
        except (json.JSONDecodeError, OSError) as exc:
            raise ProjectError("評測模型偏好設定無法讀取") from exc
        if not isinstance(value, dict):
            raise ProjectError("評測模型偏好設定格式錯誤")
        return parse_judge_models(value.get("automatic_qa", []))

    def save_autoqa_models(self, models: object) -> tuple[str, ...]:
        selected = parse_judge_models(models)
        if len(selected) > MAX_JUDGE_MODELS:
            raise ProjectError(f"評測模型最多選擇 {MAX_JUDGE_MODELS} 個")
        payload = {"automatic_qa": list(selected)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=".judge-model-preferences-", suffix=".json"
        )
        try:
            os.fchmod(handle, 0o600)
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                json.dump(payload, temporary)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return selected


def judge_model_label(model: object) -> str:
    models = parse_judge_models(model)
    if len(models) > 1:
        return "、".join(judge_model_label(item) for item in models)
    if models:
        model = models[0]
    if is_litellm_model(model):
        return f"LiteLLM／{litellm_alias(str(model))}"
    return str(model) if model is not None else ""


def judge_completion_api_params(
    model: str, max_tokens: int, reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
) -> dict[str, object]:
    """Chat Completions parameters for a judge model on OpenAI or the LiteLLM proxy.

    ``reasoning_effort`` is ``None`` when the LiteLLM model does not support reasoning,
    so the parameter is never sent and the proxy cannot reject the request for it.
    """
    if is_litellm_model(model):
        if reasoning_effort is None:
            return {"temperature": 0, "max_tokens": max_tokens}
        if reasoning_effort not in REASONING_EFFORT_OPTIONS:
            raise ProjectError("不支援的評測推理深度")
        # Thinking tokens share the output budget on several providers, so no token
        # cap is sent to avoid truncating the JSON verdict.
        return {"reasoning_effort": reasoning_effort}
    return chat_completion_api_params(model, max_tokens, reasoning_effort or DEFAULT_REASONING_EFFORT)


@dataclass(frozen=True, slots=True)
class JudgeEndpoint:
    api_base_url: str
    api_key: str
    supports_reasoning: bool = False
    extra_headers: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class JudgeConnectionTestResult:
    success: bool
    message: str
    models: tuple[str, ...] = ()
    reasoning_models: tuple[str, ...] = ()


JudgeConnectionTester = Callable[[str, str], JudgeConnectionTestResult]


class JudgeConnectionSettings:
    """Persist the LiteLLM proxy used for judge models, separate from the GraphRAG connection.

    Provider keys stay in the LiteLLM proxy configuration; only the proxy URL and its
    master or virtual key are stored here.
    """

    def __init__(self, project_root: str | Path, tester: JudgeConnectionTester | None = None) -> None:
        self.path = Path(project_root) / ".judge_connection.json"
        self.tester = tester or self._test_litellm

    def get_api_base_url(self) -> str:
        value = self._read()
        if value is None or not value.get("api_base_url"):
            return DEFAULT_LITELLM_API_BASE_URL
        return ConnectionSettings._validate_api_base_url(value["api_base_url"])

    def get_api_key(self) -> str | None:
        value = self._read()
        if value is not None:
            key = value.get("api_key")
            if isinstance(key, str) and key.strip():
                return key.strip()
        key = os.environ.get(LITELLM_API_KEY_ENVIRONMENT_VARIABLE, "").strip()
        if key and not ConnectionSettings._is_placeholder_key(key):
            return key
        return None

    def get_access_headers(self) -> dict[str, str]:
        """Return optional Cloudflare Access headers from the same app env as connection tests."""
        return _cloudflare_access_headers()

    def masked_api_key(self) -> str:
        key = self.get_api_key()
        return f"已設定（••••{key[-4:]}）" if key else "尚未設定"

    def get_models(self) -> tuple[str, ...]:
        """Saved LiteLLM aliases (without the ``litellm:`` prefix)."""
        value = self._read()
        if value is None or "models" not in value:
            return ()
        return self.parse_models(value["models"])

    def get_reasoning_models(self) -> tuple[str, ...]:
        value = self._read()
        if value is None or "reasoning_models" not in value:
            return ()
        models = set(self.get_models())
        return tuple(model for model in self.parse_models(value["reasoning_models"]) if model in models)

    def is_configured(self) -> bool:
        try:
            return bool(self.get_models())
        except ProjectError:
            return False

    def available_models(self) -> tuple[str, ...]:
        """Judge-model IDs (``litellm:<alias>``) selectable in the UI; empty until saved."""
        try:
            return tuple(litellm_model_id(model) for model in self.get_models())
        except ProjectError:
            return ()

    def supports_reasoning(self, model: object) -> bool:
        if not is_litellm_model(model):
            return False
        try:
            return litellm_alias(str(model)) in self.get_reasoning_models()
        except ProjectError:
            return False

    def endpoint_for(self, model: str) -> JudgeEndpoint:
        if not is_litellm_model(model):
            raise ProjectError(f"不支援的評判模型：{model}")
        alias = litellm_alias(model)
        if alias not in self.get_models():
            raise ProjectError(f"評測模型不在評測模型設定的清單：{alias}")
        return JudgeEndpoint(
            self.get_api_base_url(), self.get_api_key() or "", alias in self.get_reasoning_models(),
            self.get_access_headers(),
        )

    def save(
        self, api_base_url: str, api_key: str, models: object, reasoning_models: object = (),
    ) -> None:
        base_url = ConnectionSettings._validate_api_base_url(api_base_url or DEFAULT_LITELLM_API_BASE_URL)
        key = (api_key or "").strip()
        if not key:
            value = self._read() or {}
            saved = value.get("api_key")
            key = saved.strip() if isinstance(saved, str) else ""
        if "\n" in key or "\r" in key:
            raise ProjectError("API Key 格式不正確")
        selected = self.parse_models(models)
        if not selected:
            raise ProjectError("請至少選擇一個評測模型")
        reasoning = [model for model in self.parse_models(reasoning_models) if model in selected]
        payload: dict[str, object] = {
            "provider": LITELLM_PROVIDER, "api_base_url": base_url,
            "models": list(selected), "reasoning_models": reasoning,
        }
        if key:
            payload["api_key"] = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(dir=self.path.parent, prefix=".judge-connection-", suffix=".json")
        try:
            os.fchmod(handle, 0o600)
            with os.fdopen(handle, "w", encoding="utf-8") as temporary:
                json.dump(payload, temporary)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def test(self, api_base_url: str | None = None, api_key: str | None = None) -> JudgeConnectionTestResult:
        base_url = (
            ConnectionSettings._validate_api_base_url(api_base_url) if api_base_url else self.get_api_base_url()
        )
        key = api_key.strip() if api_key and api_key.strip() else (self.get_api_key() or "")
        return self.tester(base_url, key)

    @staticmethod
    def parse_models(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items = value.replace("\n", ",").split(",")
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise ProjectError("評測模型清單格式錯誤")
        result: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ProjectError("評測模型清單格式錯誤")
            model = litellm_alias(item.strip())
            if any(character.isspace() for character in model):
                raise ProjectError(f"評測模型別名不可包含空白：{model}")
            if model and model not in result:
                result.append(model)
        return tuple(result)

    def _read(self) -> dict[str, object] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as exc:
            raise ProjectError("評測模型設定無法讀取") from exc
        if not isinstance(value, dict):
            raise ProjectError("評測模型設定格式錯誤")
        return value

    @staticmethod
    def _get_json(url: str, api_key: str, access_headers: dict[str, str] | None = None) -> object:
        headers = {"User-Agent": "automotive-graphrag/0.1"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(access_headers or {})
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    @classmethod
    def _test_litellm(cls, api_base_url: str, api_key: str) -> JudgeConnectionTestResult:
        models_url = f"{api_base_url}/models" if api_base_url.endswith("/v1") else f"{api_base_url}/v1/models"
        access_headers = cls._configured_access_headers()
        try:
            body = cls._get_json(models_url, api_key, access_headers)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return JudgeConnectionTestResult(False, f"LiteLLM Key 驗證失敗（HTTP {exc.code}）")
            return JudgeConnectionTestResult(False, f"LiteLLM proxy 回應 HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return JudgeConnectionTestResult(False, "無法連線至 LiteLLM proxy")
        models = parse_litellm_model_list(body)
        try:
            info = cls._get_json(f"{api_base_url}/model/info", api_key, access_headers)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            info = None
        reasoning = parse_litellm_reasoning_models(info)
        listed = f"；可用模型別名：{', '.join(models)}" if models else "；proxy 未回傳任何模型別名"
        return JudgeConnectionTestResult(
            True, f"LiteLLM proxy 連線成功{listed}", models, tuple(model for model in reasoning if model in models)
        )

    @staticmethod
    def _configured_access_headers() -> dict[str, str]:
        return _cloudflare_access_headers()


def parse_litellm_model_list(body: object) -> tuple[str, ...]:
    """Model aliases from an OpenAI-style ``/v1/models`` response."""
    items = body.get("data") if isinstance(body, dict) else None
    return tuple(sorted({
        str(item["id"]) for item in items or []
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    }))


def parse_litellm_reasoning_models(body: object) -> tuple[str, ...]:
    """Aliases whose LiteLLM ``/model/info`` entry reports ``model_info.supports_reasoning``."""
    items = body.get("data") if isinstance(body, dict) else None
    result: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict) or not isinstance(item.get("model_name"), str):
            continue
        info = item.get("model_info")
        if isinstance(info, dict) and info.get("supports_reasoning") is True:
            result.add(item["model_name"])
    return tuple(sorted(result))
