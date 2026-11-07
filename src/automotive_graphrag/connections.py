"""Shared LiteLLM proxy connection settings.

Every model call (GraphRAG indexing and queries, question generation, Vision and LLM
judges) goes through one OpenAI-compatible LiteLLM proxy. Provider keys stay in the
proxy configuration; only the proxy URL and its master or virtual key are stored here.
"""

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
LITELLM_API_KEY_ENVIRONMENT_VARIABLE = "LITELLM_API_KEY"
CF_ACCESS_CLIENT_ID_ENVIRONMENT_VARIABLE = "CF_ACCESS_CLIENT_ID"
CF_ACCESS_CLIENT_SECRET_ENVIRONMENT_VARIABLE = "CF_ACCESS_CLIENT_SECRET"
CF_ACCESS_HEADERS = (
    (CF_ACCESS_CLIENT_ID_ENVIRONMENT_VARIABLE, "CF-Access-Client-Id"),
    (CF_ACCESS_CLIENT_SECRET_ENVIRONMENT_VARIABLE, "CF-Access-Client-Secret"),
)
# Cloudflare Browser Integrity Check blocks the default urllib and SDK identities.
CLIENT_USER_AGENT = "automotive-graphrag/0.1"
DEFAULT_API_BASE_URL = "http://localhost:4000/v1"
GPT6_LUNA_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORT_OPTIONS = ("low", "medium", "high")
# Offered until a connection test has saved the proxy's own model lists.
DEFAULT_CHAT_MODELS = ("gpt-4o-mini", "gpt-4.1-mini", GPT6_LUNA_MODEL)
DEFAULT_REASONING_MODELS = (GPT6_LUNA_MODEL,)
DEFAULT_EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = GPT6_LUNA_MODEL
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Judge IDs saved before the proxy became the only connection carried this prefix.
LEGACY_MODEL_PREFIX = "litellm:"
MAX_JUDGE_MODELS = 5


def _cloudflare_access_headers() -> dict[str, str]:
    file_values: dict[str, str] = {}
    env_files = (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env")
    for env_file in dict.fromkeys(path.resolve() for path in env_files):
        if env_file.is_file():
            for key, value in dotenv_values(env_file).items():
                if key not in file_values and isinstance(value, str) and value.strip():
                    file_values[key] = value
    headers: dict[str, str] = {}
    for variable, header in CF_ACCESS_HEADERS:
        value = os.environ.get(variable)
        if not isinstance(value, str) or not value.strip():
            value = file_values.get(variable)
        if isinstance(value, str) and value.strip():
            headers[header] = value.strip()
    return headers


def llm_request_headers(api_key: str | None) -> dict[str, str]:
    """HTTP headers for a direct proxy request, including optional Cloudflare Access."""
    headers = {"Content-Type": "application/json", "User-Agent": CLIENT_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(_cloudflare_access_headers())
    return headers


def graphrag_request_headers() -> dict[str, str]:
    """``extra_headers`` for settings.yaml; credentials stay as ``${VAR}`` placeholders.

    GraphRAG expands the placeholders from the environment when it loads the file, so
    the secrets are never written to disk. ``apply_to_environment`` exports them.
    """
    configured = _cloudflare_access_headers()
    headers = {"User-Agent": CLIENT_USER_AGENT}
    for variable, header in CF_ACCESS_HEADERS:
        if header in configured:
            headers[header] = f"${{{variable}}}"
    return headers


def chat_completion_api_params(max_tokens: int, reasoning_effort: str | None) -> dict[str, object]:
    """Chat Completions parameters; ``reasoning_effort`` is None for non-reasoning models."""
    if reasoning_effort is None:
        return {"temperature": 0, "max_tokens": max_tokens}
    if reasoning_effort not in REASONING_EFFORT_OPTIONS:
        raise ProjectError("不支援的推理深度")
    # Reasoning models reject sampling parameters, and thinking tokens share the output
    # budget on several providers, so no cap is sent to avoid truncating the JSON reply.
    return {"reasoning_effort": reasoning_effort}


def extract_json_object(text: str) -> dict[str, object] | None:
    """Parse a JSON object, falling back to the first object embedded in plain text.

    Models without JSON mode may wrap the verdict in prose or Markdown code fences.
    """
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def configure_completion_model(
    model_config: dict[str, object], model: str, reasoning_effort: str | None = None,
) -> None:
    """Set a GraphRAG model and its call arguments; a None effort means no reasoning support."""
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_OPTIONS:
        raise ProjectError("不支援的推理深度")
    model_config["model"] = model
    raw_call_args = model_config.get("call_args")
    call_args = dict(raw_call_args) if isinstance(raw_call_args, dict) else {}
    call_args["extra_headers"] = graphrag_request_headers()
    if reasoning_effort is None:
        # The same project settings file is reused when users switch models.
        # Remove reasoning-only arguments so they cannot leak into another model's request.
        call_args.pop("reasoning_effort", None)
        call_args.pop("allowed_openai_params", None)
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
    # GraphRAG's LiteLLM client drops parameters it does not recognize for a proxy alias.
    call_args["allowed_openai_params"] = ["reasoning_effort"]
    model_config["call_args"] = call_args


def configure_embedding_model(model_config: dict[str, object], model: str | None = None) -> None:
    """Set a GraphRAG embedding model and the proxy request headers."""
    if model is not None:
        model_config["model"] = model
    raw_call_args = model_config.get("call_args")
    call_args = dict(raw_call_args) if isinstance(raw_call_args, dict) else {}
    call_args["extra_headers"] = graphrag_request_headers()
    model_config["call_args"] = call_args


@dataclass(frozen=True, slots=True)
class ConnectionTestResult:
    success: bool
    message: str
    models: tuple[str, ...] = ()
    reasoning_models: tuple[str, ...] = ()
    embedding_models: tuple[str, ...] = ()


ConnectionTester = Callable[[str, str], ConnectionTestResult]


class ConnectionSettings:
    """Persist the LiteLLM proxy shared by every project in a project root."""

    def __init__(self, project_root: str | Path, tester: ConnectionTester | None = None) -> None:
        self.path = Path(project_root) / ".connection.json"
        self.legacy_judge_path = Path(project_root) / ".judge_connection.json"
        self.tester = tester or self._test_litellm

    def has_api_key(self, project_id: str | None = None) -> bool:
        return self.get_api_key(project_id) is not None

    def masked_api_key(self, project_id: str | None = None) -> str:
        key = self.get_api_key(project_id)
        if not key:
            return "尚未設定"
        return f"已設定（••••{key[-4:]}）"

    def get_api_key(self, project_id: str | None = None) -> str | None:
        value = self._read()
        saved = value.get("api_key") if value is not None else None
        if isinstance(saved, str) and saved.strip():
            return saved.strip()
        for variable in (LITELLM_API_KEY_ENVIRONMENT_VARIABLE, API_KEY_ENVIRONMENT_VARIABLE):
            key = os.environ.get(variable, "").strip()
            if key and not self._is_placeholder_key(key):
                return key
        return self._dotenv_api_key(project_id)

    def get_api_base_url(self) -> str:
        value = self._read()
        if value is None or not value.get("api_base_url"):
            return self._validate_api_base_url(os.environ.get(API_BASE_ENVIRONMENT_VARIABLE, DEFAULT_API_BASE_URL))
        return self._validate_api_base_url(value["api_base_url"])

    def get_access_headers(self) -> dict[str, str]:
        """Optional Cloudflare Access headers sent with every proxy request."""
        return _cloudflare_access_headers()

    def available_chat_models(self) -> tuple[str, ...]:
        """Chat models selectable in the UI; the defaults apply until a list is saved."""
        value = self._read()
        models = self.parse_models(value.get("models")) if value is not None else ()
        return models or DEFAULT_CHAT_MODELS

    def available_embedding_models(self) -> tuple[str, ...]:
        value = self._read()
        models = self.parse_models(value.get("embedding_models")) if value is not None else ()
        return models or DEFAULT_EMBEDDING_MODELS

    def get_reasoning_models(self) -> tuple[str, ...]:
        value = self._read()
        if value is None or "reasoning_models" not in value:
            return DEFAULT_REASONING_MODELS
        return self.parse_models(value["reasoning_models"])

    def supports_reasoning(self, model: object) -> bool:
        return model in self.get_reasoning_models()

    def reasoning_effort_for(self, model: object, effort: str = DEFAULT_REASONING_EFFORT) -> str | None:
        """The effort to send to ``model``; None when it must not receive the parameter."""
        return effort if self.supports_reasoning(model) else None

    def get_chat_model(self) -> str:
        value = self._read()
        model = (
            value.get("chat_model", DEFAULT_CHAT_MODEL)
            if value is not None
            else os.environ.get("GRAPHRAG_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        )
        return self._validate_model(model, self.available_chat_models(), "Chat")

    def get_embedding_model(self) -> str:
        value = self._read()
        model = (
            value.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            if value is not None
            else os.environ.get("GRAPHRAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )
        return self._validate_model(model, self.available_embedding_models(), "Embedding")

    def _read(self) -> dict[str, object] | None:
        """Saved settings; the former GraphRAG and judge connections merge on first read."""
        value = self._read_file(self.path)
        if value is not None and "models" in value:
            return value
        legacy = self._read_file(self.legacy_judge_path)
        if legacy is None:
            return value
        merged = {key: item for key, item in (value or {}).items() if key not in {"api_base_url", "api_key"}}
        merged["api_base_url"] = legacy.get("api_base_url") or DEFAULT_API_BASE_URL
        if legacy.get("api_key"):
            merged["api_key"] = legacy["api_key"]
        chat_model = merged.get("chat_model")
        merged["models"] = list(dict.fromkeys([
            *self.parse_models(legacy.get("models")), *DEFAULT_CHAT_MODELS,
            *([chat_model] if isinstance(chat_model, str) else []),
        ]))
        merged["reasoning_models"] = list(dict.fromkeys([
            *self.parse_models(legacy.get("reasoning_models")), *DEFAULT_REASONING_MODELS,
        ]))
        return merged

    @staticmethod
    def _read_file(path: Path) -> dict[str, object] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
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
        models: object = None,
        reasoning_models: object = None,
        embedding_models: object = None,
    ) -> None:
        """Save the connection; a model list left as None keeps the saved list."""
        base_url = self._validate_api_base_url(api_base_url)
        selected_chat = self.available_chat_models() if models is None else self.parse_models(models)
        selected_embedding = (
            self.available_embedding_models() if embedding_models is None else self.parse_models(embedding_models)
        )
        # The default models are always selectable, so saving can never orphan them.
        selected_chat = tuple(dict.fromkeys([*selected_chat, *self.parse_models([chat_model])]))
        selected_embedding = tuple(dict.fromkeys([*selected_embedding, *self.parse_models([embedding_model])]))
        if not chat_model or not embedding_model:
            raise ProjectError("Chat 與 Embedding 模型為必填")
        reasoning = self.get_reasoning_models() if reasoning_models is None else self.parse_models(reasoning_models)
        key = (api_key or "").strip() or self._saved_api_key()
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
                        "chat_model": chat_model,
                        "embedding_model": embedding_model,
                        "models": list(selected_chat),
                        "reasoning_models": [model for model in reasoning if model in selected_chat],
                        "embedding_models": list(selected_embedding),
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
        os.environ["GRAPHRAG_CHAT_MODEL"] = chat_model
        os.environ["GRAPHRAG_EMBEDDING_MODEL"] = embedding_model

    def _saved_api_key(self) -> str:
        value = self._read() or {}
        saved = value.get("api_key")
        return saved.strip() if isinstance(saved, str) else ""

    def apply_to_environment(self, project_id: str | None = None) -> str:
        key = self.get_api_key(project_id)
        if not key:
            raise ProjectError("尚未設定共用 GRAPHRAG_API_KEY，請先到連線設定頁設定")
        os.environ[API_KEY_ENVIRONMENT_VARIABLE] = key
        os.environ[API_BASE_ENVIRONMENT_VARIABLE] = self.get_api_base_url()
        os.environ["GRAPHRAG_CHAT_MODEL"] = self.get_chat_model()
        os.environ["GRAPHRAG_EMBEDDING_MODEL"] = self.get_embedding_model()
        # settings.yaml references these as ${VAR}; they may only exist in the app's .env.
        access_headers = _cloudflare_access_headers()
        for variable, header in CF_ACCESS_HEADERS:
            if header in access_headers:
                os.environ[variable] = access_headers[header]
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
        key = api_key.strip() if api_key and api_key.strip() else self.get_api_key()
        if not key:
            return ConnectionTestResult(False, "尚未設定 API Key")
        return self.tester(base_url, key)

    @staticmethod
    def _validate_api_base_url(api_base_url: object) -> str:
        """Normalize to the OpenAI-compatible root, which every client appends paths to."""
        if not isinstance(api_base_url, str) or not api_base_url.strip():
            raise ProjectError("API Base URL 為必填")
        value = api_base_url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ProjectError("API Base URL 必須是有效的 HTTP(S) 網址")
        return value if value.endswith("/v1") else f"{value}/v1"

    @staticmethod
    def _validate_model(model: object, allowed: tuple[str, ...], label: str) -> str:
        if not isinstance(model, str) or model not in allowed:
            raise ProjectError(f"{label} 模型不在可用清單：{', '.join(allowed)}")
        return model

    @staticmethod
    def parse_models(value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            items = value.replace("\n", ",").split(",")
        elif isinstance(value, (list, tuple)):
            items = list(value)
        else:
            raise ProjectError("模型清單格式錯誤")
        result: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ProjectError("模型清單格式錯誤")
            model = item.strip().removeprefix(LEGACY_MODEL_PREFIX)
            if any(character.isspace() for character in model):
                raise ProjectError(f"模型名稱不可包含空白：{model}")
            if model and model not in result:
                result.append(model)
        return tuple(result)

    @staticmethod
    def _get_json(url: str, api_key: str, access_headers: dict[str, str] | None = None) -> object:
        headers = {"User-Agent": CLIENT_USER_AGENT}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(access_headers or {})
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    @classmethod
    def _test_litellm(cls, api_base_url: str, api_key: str) -> ConnectionTestResult:
        access_headers = _cloudflare_access_headers()
        try:
            body = cls._get_json(f"{api_base_url}/models", api_key, access_headers)
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return ConnectionTestResult(False, f"LiteLLM Key 驗證失敗（HTTP {exc.code}）")
            return ConnectionTestResult(False, f"LiteLLM proxy 回應 HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return ConnectionTestResult(False, "無法連線至 LiteLLM proxy")
        listed = parse_litellm_model_list(body)
        try:
            info = cls._get_json(f"{api_base_url.removesuffix('/v1')}/model/info", api_key, access_headers)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            info = None
        embedding = parse_litellm_embedding_models(info, listed)
        models = tuple(model for model in listed if model not in embedding)
        reasoning = tuple(model for model in parse_litellm_reasoning_models(info) if model in models)
        message = f"連線成功：{len(models)} 個 Chat 模型、{len(embedding)} 個 Embedding 模型"
        return ConnectionTestResult(True, message, models, reasoning, embedding)


def parse_litellm_model_list(body: object) -> tuple[str, ...]:
    """Model names from an OpenAI-style ``/v1/models`` response."""
    items = body.get("data") if isinstance(body, dict) else None
    return tuple(sorted({
        str(item["id"]) for item in items or []
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip()
    }))


def parse_litellm_reasoning_models(body: object) -> tuple[str, ...]:
    """Names whose LiteLLM ``/model/info`` entry reports ``model_info.supports_reasoning``."""
    items = body.get("data") if isinstance(body, dict) else None
    result: set[str] = set()
    for item in items or []:
        if not isinstance(item, dict) or not isinstance(item.get("model_name"), str):
            continue
        info = item.get("model_info")
        if isinstance(info, dict) and info.get("supports_reasoning") is True:
            result.add(item["model_name"])
    return tuple(sorted(result))


def parse_litellm_embedding_models(body: object, listed: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Embedding models by ``/model/info`` mode; names decide when the proxy reports no mode."""
    items = body.get("data") if isinstance(body, dict) else None
    modes: dict[str, object] = {}
    for item in items or []:
        if isinstance(item, dict) and isinstance(item.get("model_name"), str):
            info = item.get("model_info")
            modes[item["model_name"]] = info.get("mode") if isinstance(info, dict) else None
    return tuple(sorted({
        model for model in {*listed, *modes}
        if modes.get(model) == "embedding" or (not modes.get(model) and "embed" in model.lower())
    }))


def parse_judge_models(value: object) -> tuple[str, ...]:
    """Normalize one judge model, a list, or the comma-joined form saved in results."""
    if value is None:
        return ()
    items = value.split(",") if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else [value]
    result: list[str] = []
    for item in items:
        model = str(item).strip().removeprefix(LEGACY_MODEL_PREFIX)
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
    """Display name of a judge or panel; results saved with the former prefix read the same."""
    models = parse_judge_models(model)
    return "、".join(models) if models else ("" if model is None else str(model))
