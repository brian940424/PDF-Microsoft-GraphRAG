"""Shared GraphRAG API connection settings."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .projects import ProjectError


API_KEY_ENVIRONMENT_VARIABLE = "GRAPHRAG_API_KEY"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"


@dataclass(frozen=True, slots=True)
class ConnectionTestResult:
    success: bool
    message: str


ConnectionTester = Callable[[str], ConnectionTestResult]


class ConnectionSettings:
    """Persist one API key shared by every project in a project root."""

    def __init__(self, project_root: str | Path, tester: ConnectionTester | None = None) -> None:
        self.path = Path(project_root) / ".connection.json"
        self.tester = tester or self._test_openai

    def has_api_key(self) -> bool:
        return self.get_api_key() is not None

    def masked_api_key(self) -> str:
        key = self.get_api_key()
        if not key:
            return "尚未設定"
        return f"已設定（••••{key[-4:]}）"

    def get_api_key(self) -> str | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return os.environ.get(API_KEY_ENVIRONMENT_VARIABLE) or None
        except (json.JSONDecodeError, OSError) as exc:
            raise ProjectError("共用連線設定無法讀取") from exc
        key = value.get("api_key")
        if not isinstance(key, str) or not key:
            raise ProjectError("共用連線設定缺少 API Key")
        return key

    def save_api_key(self, api_key: str) -> None:
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
                json.dump({"api_key": key}, temporary)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        os.environ[API_KEY_ENVIRONMENT_VARIABLE] = key

    def apply_to_environment(self) -> str:
        key = self.get_api_key()
        if not key:
            raise ProjectError("尚未設定共用 GRAPHRAG_API_KEY，請先到連線設定頁設定")
        os.environ[API_KEY_ENVIRONMENT_VARIABLE] = key
        return key

    def test(self, api_key: str | None = None) -> ConnectionTestResult:
        key = api_key.strip() if api_key else self.get_api_key()
        if not key:
            return ConnectionTestResult(False, "尚未設定 API Key")
        return self.tester(key)

    @staticmethod
    def _test_openai(api_key: str) -> ConnectionTestResult:
        request = urllib.request.Request(
            OPENAI_MODELS_URL,
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
