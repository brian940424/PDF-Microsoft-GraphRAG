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
ALLOWED_CHAT_MODELS = ("gpt-4o-mini", "gpt-4.1-mini")
ALLOWED_EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


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
