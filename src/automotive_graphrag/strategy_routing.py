"""LLM-based selection of a GraphRAG query strategy for one question."""

from __future__ import annotations

import inspect
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .connections import chat_completion_api_params
from .projects import ProjectError


ROUTABLE_METHODS = ("local", "global", "drift", "basic")
FALLBACK_METHOD = "local"

ROUTING_PROMPT = """You choose the Microsoft GraphRAG search method for a question about automotive service manuals.

Available methods:
- local: Starts from the entities most similar to the question and expands to their relationships, community reports and source text. Best for specific facts about particular parts, systems, specifications, symptoms or repair procedures.
- global: Map-reduce over community reports of the whole corpus. Best for corpus-wide questions that ask for overviews, themes, summaries, comparisons across many systems, or "what kinds of ..." questions. It does not cite exact source pages.
- drift: Starts from community reports, generates follow-up sub-questions and answers them with local search over several rounds. Best for complex multi-hop questions that need both broad context and specific details, such as causal chains across systems or diagnosis spanning several components. Slowest and most expensive.
- basic: Plain vector search over text chunks without the graph. Best for simple lookups of a single value, term, warning or sentence that likely appears verbatim in one passage.

Prefer local when unsure. Choose drift only when the question clearly needs multi-step reasoning.

Return a JSON object with exactly these keys:
- "method": one of "local", "global", "drift", "basic"
- "reason": one or two sentences in Traditional Chinese explaining which characteristics of the question led to this choice

Question:
{question}"""


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    method: str
    reason: str


ChatFunction = Callable[..., object]


class StrategyRouter:
    def __init__(self, chat: ChatFunction | None = None) -> None:
        self.chat = chat or self._openai_chat

    def route(
        self, question: str, model: str, api_base_url: str, api_key: str | None,
        reasoning_effort: str = "medium",
    ) -> StrategyDecision:
        """Return the selected method, falling back to Local instead of failing the query."""
        if not api_key:
            return self._fallback("尚未設定 API Key")
        try:
            args = (api_base_url, api_key, model, ROUTING_PROMPT.format(question=question))
            if "reasoning_effort" in inspect.signature(self.chat).parameters:
                response = self.chat(*args, reasoning_effort=reasoning_effort)
            else:
                response = self.chat(*args)
            return self._parse_response(response)
        except ProjectError as exc:
            return self._fallback(str(exc))

    @staticmethod
    def _parse_response(response: object) -> StrategyDecision:
        if not isinstance(response, dict):
            raise ProjectError("策略判斷回應不是 JSON 物件")
        method = str(response.get("method", "")).strip().lower()
        reason = str(response.get("reason", "")).strip()
        if method not in ROUTABLE_METHODS:
            raise ProjectError(f"策略判斷回傳不支援的方法：{method or '空白'}")
        return StrategyDecision(method, reason or "模型未提供原因")

    @staticmethod
    def _fallback(detail: str) -> StrategyDecision:
        return StrategyDecision(FALLBACK_METHOD, f"自動判斷失敗（{detail}），改用預設 Local")

    @staticmethod
    def _openai_chat(
        api_base_url: str, api_key: str, model: str, prompt: str,
        reasoning_effort: str = "medium", timeout_seconds: float = 60,
    ) -> object:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        payload.update(chat_completion_api_params(model, 2000, reasoning_effort))
        request = urllib.request.Request(
            f"{api_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ProjectError(f"策略判斷 API 回應 HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProjectError("策略判斷 API 無法連線或回應格式錯誤") from exc
        try:
            return json.loads(value["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProjectError("策略判斷 API 回應缺少有效 JSON 內容") from exc
