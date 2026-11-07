import csv
import io
import json
import re
import os
import tempfile
import threading
import time
import unittest
import urllib.error
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from automotive_graphrag.app import _autoqa_source_cell, create_app
from automotive_graphrag.automatic_evaluation import (
    PROMPT_VERSION,
    AutomaticEvaluationItem,
    AutomaticEvaluationService,
    JudgeScore,
    extract_json_object,
)
from automotive_graphrag.connections import ConnectionSettings, ConnectionTestResult, JudgeModelPreferences
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import BatchQuestion, GoldEvidence, QuestionSetService
from automotive_graphrag.reviews import ReviewService


def evidence(chunk_id: str, page: int) -> Evidence:
    return Evidence(
        evidence_id=f"E-{chunk_id}",
        rank=1,
        context_id="1",
        text_unit_id=f"tu-{chunk_id}",
        chunk_id=chunk_id,
        section_id="WW",
        section_name="Wiper",
        document_id="WW.pdf",
        page=page,
        block_id="b01",
        text=f"支援內容 {chunk_id}" * 20,
    )


class AnswerRunner:
    def ask(self, project_id: str, question: str, method: str = "local", source: str | None = None) -> QueryResult:
        number = 1 if question == "問題一" else 2
        now = datetime.now(timezone.utc).isoformat()
        return QueryResult(
            query_id=f"query-{number}",
            project_id=project_id,
            question=question,
            method=method,
            status="COMPLETED",
            answer=f"系統答案{number}",
            error=None,
            started_at=now,
            completed_at=now,
            duration_seconds=0.1,
            evidence=(evidence(f"gold-{number}", 20 + number),),
        )


def case_id(prompt: str) -> str:
    return json.loads(prompt.split("Evaluation case:", 1)[1])["question_id"]


class FakeJudge:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, str, str]] = []
        self.lock = threading.Lock()
        self.calls_by_question: dict[str, int] = {}

    def __call__(self, base_url: str, api_key: str, model: str, prompt: str) -> object:
        question_id = case_id(prompt)
        with self.lock:
            self.calls.append((base_url, api_key, model, prompt))
            response_index = self.calls_by_question.get(question_id, 0)
            self.calls_by_question[question_id] = response_index + 1
            response = self.responses[min(response_index, len(self.responses) - 1)]
        if isinstance(response, dict) and isinstance(response.get("items"), list):
            matching = [item for item in response["items"] if item.get("question_id") == question_id]
            return {"item": (matching or response["items"])[0]}
        return response


class AutomaticEvaluationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = ProjectStore(self.root / "projects")
        self.store.create(
            project_id="L33-SM3E",
            display_name="L33 / SM3E",
            vehicle_name="L33",
            manual_version="SM3E",
        )
        self.store.update_status("L33-SM3E", "INDEXED")
        source = self.root / "questions.json"
        source.write_text(
            json.dumps(
                {
                    "questions": [
                        {
                            "number": 1,
                            "question": "問題一",
                            "expected_answer": "參考答案一",
                            "question_sources": [],
                            "answer_sources": [{"document_name": "WW.pdf", "pages": [21]}],
                        },
                        {
                            "number": 2,
                            "question": "問題二",
                            "expected_answer": "參考答案二",
                            "question_sources": [],
                            "answer_sources": [{"document_name": "WW.pdf", "pages": [22]}],
                        },
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.question_sets = QuestionSetService(self.store, AnswerRunner())
        imported = self.question_sets.import_file("L33-SM3E", source)
        self.question_set_id = imported.question_set_id
        self.question_sets.run("L33-SM3E", self.question_set_id)
        self.connections = ConnectionSettings(self.store.root)
        self.connections.save("https://api.openai.com/v1", "test-key")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def judged(q1_score: int = 3, extra_content_error: bool = False) -> dict[str, object]:
        return {
            "items": [
                {
                    "question_id": "Q0001",
                    "level": 3 if extra_content_error else (0 if q1_score < 4 else 5),
                    "reason": "第一題評語",
                    "extra_content_error": extra_content_error,
                },
                {
                    "question_id": "Q0002",
                    "level": 5,
                    "reason": "第二題評語",
                    "extra_content_error": True,  # ignored: a correct answer never earns a bonus
                },
            ]
        }

    def test_evaluate_judges_each_answer_without_evidence(self) -> None:
        judge = FakeJudge([self.judged()])
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)

        result = service.evaluate("L33-SM3E", self.question_set_id)

        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(judge.calls[0][2], "gpt-6-luna")
        self.assertFalse(result.items[0].is_correct)
        self.assertTrue(result.items[1].is_correct)
        self.assertEqual(result.correct_count, 1)
        self.assertEqual(result.prompt_version, PROMPT_VERSION)
        self.assertIn("參考答案一", result.judge_prompt)
        self.assertNotIn('"retrieved_evidence"', result.judge_prompt)
        self.assertNotIn("汽車維修", result.judge_prompt)
        self.assertNotIn("answer_score", result.judge_prompt)
        self.assertNotIn("evidence_support_score", result.judge_prompt)
        self.assertIn("missing a required step", result.judge_prompt)
        self.assertIn("Mode: strict", result.judge_prompt)
        self.assertEqual(service.last_result("L33-SM3E", self.question_set_id), result)

        ReviewService(self.store, self.question_sets).save(
            "L33-SM3E", self.question_set_id, "Q0001", "partially_correct", "人工抽查"
        )
        json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-6-luna")
        self.assertEqual(payload["human_reviews"][0]["human_label"], "partially_correct")
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reviewer_note"], "人工抽查")

    def test_judge_uses_the_shared_connection_and_records_model(self) -> None:
        judge = FakeJudge([self.judged()])
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, judge
        )
        self.assertNotIn("claude-judge", service.judge_model_choices())
        with self.assertRaisesRegex(ProjectError, "可用清單"):
            service.evaluate("L33-SM3E", self.question_set_id, model="claude-judge")
        self.assertEqual(judge.calls, [])

        self.connections.save("http://localhost:4000", "sk-litellm", "claude-judge", "text-embedding-3-small", ["claude-judge"], [])
        self.assertEqual(service.judge_model_choices(), ("claude-judge",))
        result = service.evaluate("L33-SM3E", self.question_set_id, model="claude-judge")

        self.assertEqual(result.model, "claude-judge")
        self.assertEqual(
            {(url, key, model) for url, key, model, _prompt in judge.calls},
            {("http://localhost:4000/v1", "sk-litellm", "claude-judge")},
        )
        grade_judge = lambda *_args, **_kwargs: {"item": {"question_id": "Q0001", "level": 5, "reason": "ok"}}
        service.client = grade_judge
        grade = service.evaluate_single_answer_grade(
            "L33-SM3E", "Q0001", "問題一", "參考答案一", "系統答案1", "claude-judge"
        )
        self.assertEqual(grade.score, 2)

    def test_results_saved_with_the_former_prefix_validate_and_display_plainly(self) -> None:
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, FakeJudge([]))
        self.assertEqual(service.validate_judge_models("litellm:gpt-6-luna"), ("gpt-6-luna",))

    def test_judge_panel_scores_every_question_with_each_model_and_averages(self) -> None:
        verdicts = {
            ("gpt-4.1-mini", "Q0001"): 5, ("gpt-4.1-mini", "Q0002"): 5,
            ("claude-judge", "Q0001"): 0, ("claude-judge", "Q0002"): 5,
            ("gemini-flash", "Q0002"): 5,
        }
        prompts: dict[str, set[str]] = {}
        lock = threading.Lock()

        def client(_url, _key, model, prompt, reasoning_effort="medium"):
            question_id = case_id(prompt)
            with lock:
                prompts.setdefault(question_id, set()).add(prompt)
            if (model, question_id) not in verdicts:
                raise RuntimeError("proxy timeout")
            return {"item": {"question_id": question_id, "level": verdicts[(model, question_id)], "reason": model}}
        self.connections.save("http://localhost:4000", "sk", "claude-judge", "text-embedding-3-small", ["gpt-4.1-mini", "claude-judge", "gemini-flash"], [])
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, client
        )
        progress: list[tuple[int, int]] = []
        panel = ["gpt-4.1-mini", "claude-judge", "gemini-flash"]
        result = service.evaluate(
            "L33-SM3E", self.question_set_id, model=panel, progress_callback=lambda *args: progress.append(args)
        )

        # Every judge receives exactly the same grading prompt for a question.
        self.assertEqual({question_id: len(values) for question_id, values in prompts.items()}, {"Q0001": 1, "Q0002": 1})
        self.assertEqual(progress[-1], (6, 6))
        self.assertEqual(result.models, tuple(panel))
        self.assertEqual(result.model, "gpt-4.1-mini, claude-judge, gemini-flash")
        first, second = result.items
        self.assertEqual([judge.score for judge in first.judges], [1.0, 0.0, None])
        self.assertEqual(first.judges[2].error, "答案評判 API 呼叫失敗：proxy timeout")
        self.assertEqual(first.score, 0.5)
        self.assertEqual(second.score, 1.0)
        self.assertEqual(result.average_score, 0.75)
        self.assertEqual(
            result.judge_average_scores,
            {"gpt-4.1-mini": 1.0, "claude-judge": 0.5, "gemini-flash": 1.0},
        )
        self.assertEqual(result.disagreement_count, 1)
        self.assertIn("部分評審失敗", result.judge_error)
        self.assertIn("gemini-flash：失敗", first.judge_reason)
        self.assertEqual(service.last_result("L33-SM3E", self.question_set_id), result)

        _json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.assertEqual(rows[0]["average_score"], "0.5")
        self.assertEqual(rows[0]["judge_score:claude-judge"], "0.0")
        self.assertEqual(rows[0]["judge_reason:gemini-flash"], "答案評判 API 呼叫失敗：proxy timeout")

        with self.assertRaisesRegex(ProjectError, "最多選擇 5"):
            service.evaluate("L33-SM3E", self.question_set_id, model=["gpt-4o-mini", *(f"m{i}" for i in range(5))])

        app = create_app(self.store.root)
        detail_table = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "各評審評分細節"
        )
        self.assertEqual(
            detail_table.get_config()["headers"],
            ["題號", "來源", "題目", "系統回答", "正確答案", "評審分數", "評分原因", "實際檢索策略"],
        )
        handlers = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        _judge, detail_divider, model_update, detail_update, _divider, metric_update = handlers["autoqa_result_tables"](
            "L33-SM3E", self.question_set_id, [], "not-a-judge", True, False
        )
        # An unknown selection falls back to the first judge; every panel model is selectable.
        self.assertEqual([value for _label, value in model_update["choices"]], panel)
        self.assertEqual(model_update["value"], "gpt-4.1-mini")
        self.assertTrue(model_update["visible"])
        self.assertTrue(detail_divider["visible"])
        self.assertEqual([row[5:7] for row in detail_update["value"]], [["100%", "gpt-4.1-mini"]] * 2)
        self.assertFalse(metric_update["visible"])

        switched = handlers["autoqa_judge_detail_view"]("L33-SM3E", self.question_set_id, "gemini-flash")
        self.assertEqual(
            [row[5:7] for row in switched["value"]],
            [["失敗", "答案評判 API 呼叫失敗：proxy timeout"], ["100%", "gemini-flash"]],
        )

        _judge, detail_divider, model_update, detail_update, *_rest = handlers["autoqa_result_tables"](
            "L33-SM3E", self.question_set_id, [], "claude-judge", False, False
        )
        self.assertFalse(model_update["visible"])
        self.assertFalse(detail_divider["visible"])
        self.assertFalse(detail_update["visible"])

    def test_litellm_connection_test_does_not_preprocess_dynamic_model_choices(self) -> None:
        app = create_app(self.store.root)
        model_choices = next(
            component for component in app.blocks.values()
            if getattr(component, "label", None) == "可用模型（LiteLLM config.yaml 的 model_name）"
        )
        connection_test = next(
            dependency for dependency in app.config["dependencies"]
            if dependency.get("api_name") == "test_connection"
        )
        self.assertNotIn(model_choices._id, connection_test["inputs"])
        self.assertEqual(len(connection_test["inputs"]), 2)

    def test_connection_panel_is_a_single_litellm_section(self) -> None:
        app = create_app(self.store.root)
        labels = {getattr(component, "label", None) for component in app.blocks.values()}
        self.assertIn("LiteLLM 連線設定（所有專案共用）", labels)
        self.assertNotIn("GraphRAG 共用連線設定（所有專案共用）", labels)
        self.assertFalse(any(isinstance(label, str) and "評測模型設定" in label for label in labels))
        selector = next(
            component for component in app.blocks.values()
            if getattr(component, "label", None) == "評測模型（評審）"
        )
        self.assertFalse(any("LiteLLM" in label for label, _value in selector.choices))

    def test_connection_test_warns_when_proxy_has_no_embedding_model(self) -> None:
        app = create_app(self.store.root)
        handlers = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        result = ConnectionTestResult(True, "ok", ("claude", "gpt-6-luna"), ("claude",), ())
        with patch.object(ConnectionSettings, "test", return_value=result):
            message, models, chat, embedding, reasoning, embedding_state = handlers["test_connection"]("", "")
        self.assertIn("未提供 Embedding 模型", message)
        self.assertEqual(models["choices"], ["claude", "gpt-6-luna"])
        self.assertEqual(chat["value"], "gpt-6-luna")
        self.assertEqual(reasoning, ["claude"])
        self.assertNotIn("choices", embedding)

    def test_autoqa_litellm_judge_selection_persists_across_app_refresh(self) -> None:
        self.connections.save("http://localhost:4000", "sk-litellm", "gpt-6-luna", "text-embedding-3-small", ["gpt-6-luna"], [])
        app = create_app(self.store.root)
        handlers = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}

        save_message = handlers["save_autoqa_judge_model_selection"](["gpt-6-luna"])
        self.assertIn("✅ 評測模型選擇已儲存至", save_message)
        self.assertIn(".judge_model_preferences.json", save_message)

        refreshed_value, refreshed_status = handlers["refresh_autoqa_judge_model_selection"]()
        self.assertEqual(refreshed_value["value"], ["gpt-6-luna"])
        self.assertIn("gpt-6-luna", refreshed_status)

        refreshed = create_app(self.store.root)
        selector = next(
            component for component in refreshed.blocks.values()
            if getattr(component, "label", None) == "評測模型（評審）"
        )
        self.assertEqual(selector.get_config()["value"], ["gpt-6-luna"])
        self.assertIn(("gpt-6-luna", "gpt-6-luna"), selector.choices)
        save_event = next(
            dependency for dependency in refreshed.config["dependencies"]
            if dependency.get("api_name") == "save_autoqa_judge_model_selection"
        )
        self.assertIn((selector._id, "input"), save_event["targets"])
        self.assertEqual(save_event["inputs"], [selector._id])

    def test_saved_litellm_preference_missing_from_active_root_is_not_silently_replaced(self) -> None:
        preferences = JudgeModelPreferences(self.store.root)
        preferences.save_autoqa_models(["model-removed-from-proxy"])
        app = create_app(self.store.root)
        selector = next(
            component for component in app.blocks.values()
            if getattr(component, "label", None) == "評測模型（評審）"
        )
        status = next(
            component for component in app.blocks.values()
            if isinstance(getattr(component, "value", None), str)
            and component.value.startswith("⚠️ 已保存的評測模型")
        )
        self.assertEqual(selector.get_config()["value"], [])
        self.assertIn("不在目前設定中", status.value)
        self.assertIn(".judge_model_preferences.json", status.value)
        handlers = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        save_message = handlers["save_autoqa_judge_model_selection"](["model-removed-from-proxy"])
        self.assertIn("❌ 未儲存", save_message)
        self.assertEqual(preferences.get_autoqa_models(), ("model-removed-from-proxy",))

    def test_explicit_empty_judge_selection_does_not_fall_back_to_default_model(self) -> None:
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections)
        with self.assertRaisesRegex(ProjectError, "至少選擇一個評測模型"):
            service.validate_judge_models([])

    def test_question_fails_only_when_every_judge_fails(self) -> None:
        def client(_url, _key, model, prompt):
            question_id = case_id(prompt)
            if question_id == "Q0001":
                raise RuntimeError(f"{model} down")
            return {"item": {"question_id": question_id, "level": 5, "reason": "ok"}}

        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, client)
        with self.assertRaisesRegex(ProjectError, "Q0001: gpt-4o-mini .*gpt-4.1-mini"):
            service.evaluate("L33-SM3E", self.question_set_id, model="gpt-4o-mini, gpt-4.1-mini")

    def test_single_judge_result_matches_previous_format_and_legacy_file_loads(self) -> None:
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, FakeJudge([self.judged()])
        )
        result = service.evaluate("L33-SM3E", self.question_set_id, model="gpt-4o-mini")
        self.assertEqual(result.model, "gpt-4o-mini")
        self.assertEqual([item.judge_reason for item in result.items], ["第一題評語", "第二題評語"])
        self.assertEqual(result.average_score, 0.5)
        self.assertEqual(result.judge_average_scores, {"gpt-4o-mini": 0.5})
        self.assertEqual(result.disagreement_count, 0)

        path = service._path("L33-SM3E", self.question_set_id)
        legacy = json.loads(path.read_text(encoding="utf-8"))
        for key in ("models", "average_score", "judge_average_scores", "disagreement_count"):
            legacy.pop(key)
        legacy["items"] = [
            {"question_id": "Q0001", "is_correct": False, "judge_reason": "舊"},
            {"question_id": "Q0002", "is_correct": True, "judge_reason": "舊"},
        ]
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        loaded = service.last_result("L33-SM3E", self.question_set_id)
        self.assertEqual(loaded.models, ("gpt-4o-mini",))
        self.assertEqual(loaded.average_score, 0.5)
        self.assertEqual([item.score for item in loaded.items], [0.0, 1.0])
        self.assertEqual(loaded.items[0].judges, ())

    def test_litellm_reasoning_effort_is_only_sent_when_model_supports_it(self) -> None:
        self.connections.save("http://localhost:4000", "sk", "claude-judge", "text-embedding-3-small", ["claude-judge", "plain"], ["claude-judge"])
        efforts: list[tuple[str, object]] = []

        def client(_url, _key, model, _prompt, reasoning_effort="medium"):
            efforts.append((model, reasoning_effort))
            return {"item": {"question_id": "Q0001", "level": 2, "reason": "partial"}}

        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, client
        )
        self.assertTrue(service.supports_judge_reasoning("claude-judge"))
        self.assertFalse(service.supports_judge_reasoning("plain"))
        for model in ("claude-judge", "plain"):
            service.evaluate_single_answer_grade("L33-SM3E", "Q0001", "問", "答", "系統", model, reasoning_effort="high")
        self.assertEqual(efforts, [("claude-judge", "high"), ("plain", None)])

    def test_plain_text_judge_reply_extracts_first_json_object(self) -> None:
        reply = '判定如下：\n```json\n{"item": {"question_id": "Q0001", "level": 5, "reason": "ok"}}\n```'
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, lambda *_args, **_kwargs: reply
        )
        grade = service.evaluate_single_answer_grade("L33-SM3E", "Q0001", "問", "答", "系統", "gpt-4o-mini")
        self.assertEqual(grade.score, 2)
        self.assertIsNone(extract_json_object("no json {here"))
        self.assertEqual(extract_json_object('x {"a": {"b": 1}} y'), {"a": {"b": 1}})

    def test_chat_retries_without_json_mode(self) -> None:
        bodies: list[dict] = []
        urls: list[str] = []

        class Response:
            def __init__(self, value: dict) -> None:
                self.value = value

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(self.value).encode("utf-8")

        def fake_urlopen(request, timeout):
            urls.append(request.full_url)
            self.assertEqual(request.get_header("User-agent"), "automotive-graphrag/0.1")
            body = json.loads(request.data.decode("utf-8"))
            bodies.append(body)
            if request.get_header("Cf-access-client-id"):
                self.assertEqual(request.get_header("Cf-access-client-id"), "id")
                self.assertEqual(request.get_header("Cf-access-client-secret"), "secret")
            if "response_format" in body:
                raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)
            return Response({"choices": [{"message": {"content": 'Sure: {"items": []} done'}}]})

        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "id", "CF_ACCESS_CLIENT_SECRET": "secret",
        }), patch("urllib.request.urlopen", fake_urlopen):
            value = AutomaticEvaluationService._openai_chat(
                "http://localhost:4000/v1", "", "plain", "prompt", reasoning_effort=None,
            )
        self.assertEqual(value, {"items": []})
        self.assertEqual([body["model"] for body in bodies], ["plain", "plain"])
        self.assertEqual(urls, [
            "http://localhost:4000/v1/chat/completions",
            "http://localhost:4000/v1/chat/completions",
        ])
        self.assertNotIn("reasoning_effort", bodies[0])
        self.assertNotIn("response_format", bodies[1])

    def test_litellm_evaluation_passes_cloudflare_headers_from_environment(self) -> None:
        self.connections.save("http://localhost:4000", "sk-litellm", "proxy-judge", "text-embedding-3-small", ["proxy-judge"], [])
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections,
        )
        requests = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return b'{"choices":[{"message":{"content":"{\\"items\\":[]}"}}]}'

        def fake_urlopen(request, timeout):
            requests.append(request)
            return Response()

        with patch.dict(os.environ, {
            "CF_ACCESS_CLIENT_ID": "cf-client-id",
            "CF_ACCESS_CLIENT_SECRET": "cf-client-secret",
        }, clear=True), patch("urllib.request.urlopen", fake_urlopen):
            service._call_judge("L33-SM3E", "proxy-judge", "prompt", "medium")

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].get_header("Cf-access-client-id"), "cf-client-id")
        self.assertEqual(requests[0].get_header("Cf-access-client-secret"), "cf-client-secret")
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer sk-litellm")
        self.assertEqual(requests[0].get_header("User-agent"), "automotive-graphrag/0.1")
        self.assertEqual(requests[0].full_url, "http://localhost:4000/v1/chat/completions")

    def test_litellm_403_identifies_cloudflare_or_proxy_response(self) -> None:
        for headers, access_headers, expected, credential_status in (
            ({"server": "cloudflare", "cf-ray": "test-ray"}, {
                "CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "secret",
            }, "經過 Cloudflare", "本次請求已帶入"),
            ({"server": "cloudflare", "cf-ray": "test-ray"}, {}, "經過 Cloudflare", "本次請求未帶齊"),
            ({"server": "litellm"}, {}, "LiteLLM proxy 拒絕", None),
        ):
            def forbidden(request, timeout):
                raise urllib.error.HTTPError(request.full_url, 403, "forbidden", headers, None)

            with self.subTest(expected=expected, credential_status=credential_status), patch(
                "urllib.request.urlopen", forbidden
            ), patch("automotive_graphrag.connections._cloudflare_access_headers", return_value=access_headers):
                with self.assertRaisesRegex(ProjectError, expected):
                    try:
                        AutomaticEvaluationService._openai_chat(
                            "https://proxy.example/v1", "sk-test", "gpt-6-luna", "prompt",
                        )
                    except ProjectError as exc:
                        if credential_status:
                            self.assertIn(credential_status, str(exc))
                        if "cf-ray" in headers:
                            self.assertIn("test-ray", str(exc))
                            self.assertIn("無法確認是 Access policy 拒絕", str(exc))
                        raise

    def test_cloudflare_proxied_403_includes_sanitized_litellm_error_detail(self) -> None:
        def forbidden(request, timeout):
            body = json.dumps({
                "error": {"message": "virtual key is not allowed for this model; secret-key access-secret"}
            }).encode()
            raise urllib.error.HTTPError(
                request.full_url, 403, "forbidden", {"server": "cloudflare", "cf-ray": "ray-test"},
                io.BytesIO(body),
            )

        with self.assertLogs("automotive_graphrag.automatic_evaluation", level="ERROR") as captured_logs:
            with patch("urllib.request.urlopen", forbidden), patch(
                "automotive_graphrag.connections._cloudflare_access_headers",
                return_value={"CF-Access-Client-Id": "client-id", "CF-Access-Client-Secret": "access-secret"},
            ):
                with self.assertRaises(ProjectError) as caught:
                    AutomaticEvaluationService._openai_chat(
                        "https://proxy.example/v1", "secret-key", "gpt-6-luna", "prompt",
                    )
        message = str(caught.exception)
        self.assertIn("virtual key is not allowed for this model", message)
        self.assertIn("本次 LiteLLM Bearer Key 尾碼：••••-key", message)
        self.assertNotIn("secret-key", message)
        log_entry = "\n".join(captured_logs.output)
        self.assertIn("endpoint=https://proxy.example/v1/chat/completions", log_entry)
        self.assertIn("response_body=", log_entry)
        self.assertIn("[redacted]", log_entry)
        self.assertNotIn("secret-key", log_entry)
        self.assertNotIn("access-secret", log_entry)

    def test_cloudflare_error_1010_identifies_user_agent_signature_block(self) -> None:
        def forbidden(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 403, "forbidden", {"server": "cloudflare", "cf-ray": "ray-1010"},
                io.BytesIO(b"error code: 1010\n"),
            )

        with patch("urllib.request.urlopen", forbidden):
            with self.assertRaises(ProjectError) as caught:
                AutomaticEvaluationService._openai_chat(
                    "https://proxy.example/v1", "sk-test", "gpt-6-luna", "prompt",
                )
        self.assertIn("Cloudflare Error 1010", str(caught.exception))
        self.assertIn("User-Agent", str(caught.exception))

    def test_classic_only_uses_answer_aliases_without_calling_judge_and_exports_scores(self) -> None:
        question_set = self.question_sets.get("L33-SM3E", self.question_set_id)
        questions = tuple(
            replace(item, answer_aliases=(item.answer,)) for item in question_set.questions
        )
        self.question_sets.update_questions("L33-SM3E", self.question_set_id, questions)
        judge = FakeJudge([])
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)

        result = service.evaluate(
            "L33-SM3E", self.question_set_id,
            enable_llm_judge=False, enable_classic_metrics=True,
        )

        self.assertEqual(judge.calls, [])
        self.assertEqual(result.items, ())
        self.assertEqual([item.answer_em for item in result.classic_items], [1.0, 1.0])
        self.assertEqual([item.answer_f1 for item in result.classic_items], [1.0, 1.0])
        self.assertEqual([item.answer_term_hit for item in result.classic_items], [1.0, 1.0])
        self.assertEqual([item.answer_term_coverage for item in result.classic_items], [1.0, 1.0])
        self.assertEqual(result.average_answer_em, 1.0)
        self.assertEqual(result.average_answer_f1, 1.0)
        self.assertEqual(result.average_answer_term_hit, 1.0)
        self.assertEqual(result.average_answer_term_coverage, 1.0)
        self.assertEqual(service.last_result("L33-SM3E", self.question_set_id), result)

        json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertFalse(payload["enable_llm_judge"])
        self.assertTrue(payload["enable_classic_metrics"])
        self.assertEqual(payload["classic_items"][0]["answer_em"], 1.0)
        self.assertEqual(payload["question_results"][0]["answer_em"], 1.0)
        self.assertEqual(payload["question_results"][0]["answer_term_coverage"], 1.0)
        self.assertEqual(payload["summary"]["average_answer_f1"], 1.0)
        self.assertEqual(payload["summary"]["average_answer_term_hit"], 1.0)
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["answer_em"], "1.0")
        self.assertEqual(rows[0]["answer_term_hit"], "1.0")

    def test_classic_scores_survive_llm_judge_failures_when_both_are_enabled(self) -> None:
        class FailingJudge:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("judge offline")

        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, FailingJudge())
        result = service.evaluate(
            "L33-SM3E", self.question_set_id,
            enable_llm_judge=True, enable_classic_metrics=True,
        )
        self.assertTrue(result.enable_llm_judge)
        self.assertTrue(result.enable_classic_metrics)
        self.assertEqual(len(result.classic_items), 2)
        self.assertIn("judge offline", result.judge_error or "")

    def test_lenient_single_answer_uses_prompt_that_accepts_paraphrases_and_extras(self) -> None:
        judge = FakeJudge([self.judged(5)])
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)

        result = service.evaluate_single_answer(
            "L33-SM3E", "Q0001", "問題一", "核心正確答案", "改寫後的回答與額外補充", "gpt-4o-mini",
            evaluation_mode="lenient",
        )

        self.assertTrue(result.is_correct)
        prompt = judge.calls[0][3]
        self.assertIn("Mode: lenient", prompt)
        self.assertNotIn("Mode: strict", prompt)
        self.assertNotIn('"retrieved_evidence"', prompt)

    def test_strict_and_lenient_prompts_have_distinct_criteria(self) -> None:
        question = BatchQuestion("Q1", "問題", "標準答案", answer="回答")
        strict = AutomaticEvaluationService._build_prompt(question)
        lenient = AutomaticEvaluationService._build_prompt(question, "lenient")

        self.assertIn("Mode: strict", strict)
        self.assertIn("Mode: lenient", lenient)
        for prompt in (strict, lenient):
            for level in range(6):
                self.assertIn(f"Level {level} (", prompt)
            self.assertIn("Write \"reason\" in Traditional Chinese", prompt)
            self.assertIn("is a contradiction, not added content", prompt)

    def test_prompt_instructions_are_english_and_only_case_data_keeps_its_language(self) -> None:
        question = BatchQuestion("Q1", "問題", "標準答案", answer="回答")
        for mode in ("strict", "lenient"):
            prompt = AutomaticEvaluationService._build_prompt(question, mode)
            instructions = prompt.split("Evaluation case:", 1)[0]
            self.assertIsNone(re.search(r"[\u3400-\u9fff]", instructions))
            self.assertNotIn("retrieved_evidence", prompt)

    def test_levels_map_to_scores_and_correctness_starts_at_level_four(self) -> None:
        for level, score, grade, correct in (
            (5, 1.0, 2.0, True), (4, 0.8, 1.6, True), (3, 0.6, 1.2, False),
            (2, 0.4, 0.8, False), (1, 0.2, 0.4, False), (0, 0.0, 0.0, False),
        ):
            service = AutomaticEvaluationService(
                self.store, self.question_sets, self.connections,
                lambda *_args, **_kwargs: {"item": {"question_id": "Q1", "level": level, "reason": "依據"}},
            )
            args = ("L33-SM3E", "Q1", "題目", "答案", "回答", "gpt-4o-mini")
            item = service.evaluate_single_answer(*args)
            self.assertEqual((item.score, item.is_correct, item.judge_reason), (score, correct, "依據"))
            self.assertEqual(service.evaluate_single_answer_grade(*args).score, grade)

    def test_rejects_invalid_level_and_missing_reason(self) -> None:
        parse = AutomaticEvaluationService._parse_verdict
        self.assertEqual(parse({"item": {"question_id": "Q1", "level": "4", "reason": "r"}}, "Q1").level, 4)
        for level in (6, -1, 2.5, True, None, "high"):
            with self.assertRaisesRegex(ProjectError, "level 必須是 0 到 5"):
                parse({"item": {"question_id": "Q1", "level": level, "reason": "r"}}, "Q1")
        for reason in ("", "  ", None):
            with self.assertRaisesRegex(ProjectError, "缺少評分理由"):
                parse({"item": {"question_id": "Q1", "level": 5, "reason": reason}}, "Q1")

    def test_only_previous_failures_rejudges_subset_and_merges_result(self) -> None:
        second_response = {"items": [self.judged(5)["items"][0]]}
        judge = FakeJudge([self.judged(), second_response])
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)
        service.evaluate("L33-SM3E", self.question_set_id)

        result = service.evaluate("L33-SM3E", self.question_set_id, only_previous_failures=True)

        self.assertEqual(len(judge.calls), 3)
        rerun_prompt = next(
            prompt
            for _, _, _, prompt in judge.calls
            if '"question_id": "Q0001"' in prompt and "correct_answer" in prompt
        )
        self.assertIn("Q0001", rerun_prompt)
        self.assertNotIn("Q0002", rerun_prompt)
        self.assertEqual(result.question_count, 2)
        self.assertTrue(result.items[0].is_correct)
        self.assertTrue(result.items[1].is_correct)

    def test_rejects_invalid_or_invented_judge_result(self) -> None:
        response = {
            "items": [{"question_id": "INVENTED", "level": 5, "reason": "錯誤題號"}]
        }
        judge = FakeJudge([response])
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)

        with self.assertRaisesRegex(ProjectError, "正確的 question_id"):
            service.evaluate("L33-SM3E", self.question_set_id)

    def test_evaluation_concurrency_limits_parallel_requests(self) -> None:
        class SlowJudge:
            def __init__(self):
                self.active = 0
                self.maximum = 0
                self.lock = threading.Lock()

            def __call__(self, base_url, api_key, model, prompt):
                question_id = case_id(prompt)
                with self.lock:
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return {"item": {"question_id": question_id, "level": 5, "reason": "符合"}}

        judge = SlowJudge()
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)
        service.evaluate("L33-SM3E", self.question_set_id, concurrency=1)
        self.assertEqual(judge.maximum, 1)
        judge.maximum = 0
        service.evaluate("L33-SM3E", self.question_set_id, concurrency=2)
        self.assertEqual(judge.maximum, 2)
        with self.assertRaisesRegex(ProjectError, "並行數必須介於 1 到 32"):
            service.evaluate("L33-SM3E", self.question_set_id, concurrency=0)

    def test_cited_sources_only_lift_answers_wrong_because_of_added_content(self) -> None:
        class CitingRunner(AnswerRunner):
            def ask(self, *args, **kwargs):
                result = super().ask(*args, **kwargs)
                return replace(result, answer=result.answer + " [Data: Sources (1)]")

        question_sets = QuestionSetService(self.store, CitingRunner())
        question_sets.run("L33-SM3E", self.question_set_id)
        service = AutomaticEvaluationService(
            self.store, question_sets, self.connections,
            FakeJudge([self.judged(extra_content_error=True)]),
        )

        result = service.evaluate("L33-SM3E", self.question_set_id)

        # Both answers cite the gold page; Q1 is wrong only through added content, Q2 is correct.
        self.assertEqual([item.source_score for item in result.items], [1.0, 1.0])
        self.assertEqual([item.combined_score for item in result.items], [0.7, 1.0])
        self.assertEqual(result.source_weight, 0.1)
        self.assertEqual(result.average_source_score, 1.0)
        self.assertEqual(result.average_combined_score, 0.85)
        self.assertEqual(result.correct_count, 1)  # correctness still follows the judge only

        updated = service.update_manual_results("L33-SM3E", self.question_set_id, {"Q0001": 1})
        self.assertEqual(updated.items[0].combined_score, 1.0)
        self.assertEqual(updated.average_combined_score, 1.0)
        partial = service.update_manual_results("L33-SM3E", self.question_set_id, {"Q0001": 0.5})
        self.assertEqual(partial.items[0].combined_score, 0.5)  # a manual score is final

        weighted = service.evaluate("L33-SM3E", self.question_set_id, source_weight=0.5)
        self.assertEqual(weighted.items[0].combined_score, 1.0)
        with self.assertRaisesRegex(ProjectError, "0 到 1"):
            service.evaluate("L33-SM3E", self.question_set_id, source_weight=2)

        json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["average_combined_score"], 1.0)
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["source_score"], "1.0")
        self.assertEqual(rows[0]["source_bonus_share"], "1.0")

    def test_ordinary_wrong_and_correct_answers_ignore_sources(self) -> None:
        class CitingRunner(AnswerRunner):
            def ask(self, *args, **kwargs):
                result = super().ask(*args, **kwargs)
                return replace(result, answer=result.answer + " [Data: Sources (1)]")

        question_sets = QuestionSetService(self.store, CitingRunner())
        question_sets.run("L33-SM3E", self.question_set_id)
        service = AutomaticEvaluationService(
            self.store, question_sets, self.connections, FakeJudge([self.judged()])
        )
        result = service.evaluate("L33-SM3E", self.question_set_id)
        self.assertEqual([item.source_score for item in result.items], [1.0, 1.0])
        self.assertEqual([item.combined_score for item in result.items], [0.0, 1.0])

    def test_source_bonus_scales_with_share_of_flagging_judges(self) -> None:
        item = AutomaticEvaluationItem(
            "Q1", False, "reason", average_score=1 / 3, source_score=1.0,
            judges=(
                JudgeScore("a", 1.0, "ok"),
                JudgeScore("b", 0.0, "extra", extra_content_error=True),
                JudgeScore("c", 0.0, "extra", extra_content_error=True),
                JudgeScore("d", None, "", "failed", extra_content_error=True),
            ),
        )
        self.assertAlmostEqual(item.source_bonus_share, 2 / 3)
        self.assertEqual(AutomaticEvaluationService._with_combined(item, 0.1).combined_score, 0.4)

    def test_result_source_cell_shows_cited_and_gold_documents_with_pages(self) -> None:
        cited = Evidence(
            evidence_id="E1", rank=1, context_id="85", text_unit_id="tu", chunk_id="", section_id="",
            section_name="", document_id="A.pdf", page=86, block_id="", text="",
        )
        question = BatchQuestion(
            "Q1", "問題", "答案", answer="回答 [Data: Sources (85)]", retrieved_evidence=(cited,),
            answer_source_evidence=(GoldEvidence("A.pdf", (86, 87), (), "A.pdf"),),
        )
        self.assertEqual(_autoqa_source_cell(question), "系統引用：A.pdf p.86\n正確來源：A.pdf p.86, 87")
        unanswered = BatchQuestion("Q2", "問題", "答案")
        self.assertEqual(_autoqa_source_cell(unanswered), "系統引用：—\n正確來源：—")

    def test_judge_sees_answer_text_without_citations(self) -> None:
        question = BatchQuestion("Q1", "問題", "答案", answer="系統答案 [Data: Sources (85)]，另見 [E3]。")
        for mode in ("strict", "lenient"):
            prompt = AutomaticEvaluationService._build_prompt(question, mode)
            self.assertIn('"system_answer": "系統答案，另見。"', prompt)
            self.assertNotIn("Sources (85)", prompt)
            self.assertNotIn("[E3]", prompt)

    def test_prompts_ask_for_the_extra_content_flag(self) -> None:
        question = BatchQuestion("Q1", "題目", "答案", answer="回答")
        for mode in ("strict", "lenient"):
            self.assertIn("extra_content_error", AutomaticEvaluationService._build_prompt(question, mode))

    def test_extra_content_flag_only_counts_at_level_three(self) -> None:
        def respond(level, flag):
            return lambda *args: {"item": {
                "question_id": "Q1", "level": level, "reason": "r", "extra_content_error": flag,
            }}

        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, respond(5, True))
        args = ("L33-SM3E", "Q1", "題目", "答案", "回答", self.connections.get_chat_model())
        self.assertFalse(service.evaluate_single_answer_grade(*args).extra_content_error)
        service.client = respond(2, True)
        self.assertFalse(service.evaluate_single_answer_grade(*args).extra_content_error)
        service.client = respond(3, True)
        self.assertTrue(service.evaluate_single_answer_grade(*args).extra_content_error)

    def test_uncited_answers_score_zero_source_and_legacy_records_fall_back(self) -> None:
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, FakeJudge([self.judged()])
        )
        result = service.evaluate("L33-SM3E", self.question_set_id)
        self.assertEqual([item.source_score for item in result.items], [0.0, 0.0])
        self.assertEqual(result.average_combined_score, 0.5)

        path = service._path("L33-SM3E", self.question_set_id)
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy["average_combined_score"] = 0.123  # stale value from the former weighted formula
        for item in legacy["items"]:
            item.pop("source_score")
            item["combined_score"] = 0.123
            for judge in item["judges"]:
                judge.pop("extra_content_error")
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        loaded = service.last_result("L33-SM3E", self.question_set_id)
        self.assertEqual(loaded.average_combined_score, loaded.average_score)
        self.assertEqual([item.final_score for item in loaded.items], [0.0, 1.0])

    def test_manual_score_edit_overrides_average_and_keeps_judge_scores(self) -> None:
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, FakeJudge([self.judged()])
        )
        original = service.evaluate("L33-SM3E", self.question_set_id)

        updated = service.update_manual_results(
            "L33-SM3E", self.question_set_id, {"Q0001": "正確", "Q0002": 1}
        )

        self.assertEqual(updated.correct_count, 2)
        self.assertTrue(updated.items[0].is_correct)
        self.assertTrue(updated.items[0].manual)
        self.assertEqual(updated.items[0].score, 1.0)
        # The judge's own verdict and reason are kept; only the question score is overridden.
        self.assertEqual(updated.items[0].judges, original.items[0].judges)
        self.assertEqual(updated.items[0].judge_reason, "第一題評語")
        self.assertTrue(updated.items[1].is_correct)
        self.assertFalse(updated.items[1].manual)
        self.assertEqual(updated.items[1].judge_reason, "第二題評語")
        self.assertEqual(updated.average_score, 1.0)
        with self.assertRaisesRegex(ProjectError, "0 到 1"):
            service.update_manual_results("L33-SM3E", self.question_set_id, {"Q0001": 2})
        self.assertEqual(
            service.last_result("L33-SM3E", self.question_set_id),
            updated,
        )

    def test_finished_evaluation_rows_match_overview_table_columns(self) -> None:
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, FakeJudge([]))
        judge = service.evaluate(
            "L33-SM3E", self.question_set_id, enable_llm_judge=False, enable_classic_metrics=True,
        )
        judge = replace(judge, items=(AutomaticEvaluationItem(
            "Q0001", True, "評語", judges=(JudgeScore("gpt-6-luna", 0.8),),
            average_score=0.8, source_score=0.5, combined_score=0.85,
        ),))
        question_set = self.question_sets.get("L33-SM3E", self.question_set_id)
        report = SimpleNamespace(
            question_set=question_set, judge=judge, judge_error=None, generation_errors=(),
            retrieval=None, retrieval_error=None, answer_model="m", generation_model="—",
        )
        app = create_app(self.store.root)
        table = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "模型評測結果概覽"
        )
        handler = next(
            fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "evaluate_automatic_answers"
        )
        cells = dict(zip(handler.__code__.co_freevars, handler.__closure__))
        sources = lambda item: "；".join(
            f"{source.document_id}: {', '.join(str(page) for page in source.pages)}" for source in item.gold_evidence
        )
        question_rows = [
            [item.question_id, item.question, item.reference_answer, sources(item), sources(item)]
            for item in question_set.questions
        ]
        with patch.object(cells["automatic_qa"].cell_contents, "evaluate_existing", return_value=report):
            *_, final = handler(
                "L33-SM3E", self.question_set_id, question_rows, "m", [], "low", "local", 1,
                question_set.response_type, False, True,
            )
        rows = final[2]["value"]
        headers = table.get_config()["headers"]
        self.assertEqual({len(row) for row in rows}, {len(headers)})
        scored = dict(zip(headers, rows[0]))
        self.assertEqual(scored["LLM 分數"], "80%")
        self.assertEqual(scored["來源分數"], "50%")
        self.assertEqual(scored["綜合分數"], "85%")
        self.assertIn("gpt-6-luna", scored["各評審分數"])
        self.assertEqual(dict(zip(headers, rows[1]))["各評審分數"], "未啟用 LLM Judge")

        # A judge failure leaves no saved result, but every question still shows the error.
        failed = SimpleNamespace(**{**vars(report), "judge": None, "judge_error": "judge offline"})
        with patch.object(cells["automatic_qa"].cell_contents, "evaluate_existing", return_value=failed):
            *_, final = handler(
                "L33-SM3E", self.question_set_id, question_rows, "m", [], "low", "local", 1,
                question_set.response_type, False, True,
            )
        rows = final[2]["value"]
        self.assertEqual({len(row) for row in rows}, {len(headers)})
        self.assertEqual(dict(zip(headers, rows[0]))["各評審分數"], "judge offline")

    def test_app_shows_classic_metric_table_only_when_its_checkbox_is_enabled(self) -> None:
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, FakeJudge([]))
        service.evaluate(
            "L33-SM3E", self.question_set_id,
            enable_llm_judge=False, enable_classic_metrics=True,
        )
        app = create_app(self.store.root)
        metric_table = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "傳統指標結果"
        )
        self.assertEqual(
            metric_table.get_config()["headers"],
            ["題號", "來源", "題目", "系統回答", "正確答案", "EM", "F1", "Term Hit", "Term Coverage"],
        )
        refresh = next(
            fn.fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "autoqa_result_tables"
        )
        judge_rows = [["Q0001", "WW.pdf", "問題一", "系統答案1", "參考答案一", "100.0%", "", ""]]

        judge_update, _detail_divider, _model_update, _detail_update, divider_update, metric_update = refresh(
            "L33-SM3E", self.question_set_id, judge_rows, None, False, True
        )
        self.assertFalse(judge_update["visible"])
        self.assertTrue(divider_update["visible"])
        self.assertTrue(metric_update["visible"])
        self.assertEqual(
            metric_update["value"][0],
            ["Q0001", "系統引用：—\n正確來源：WW.pdf p.21", "問題一", "系統答案1", "參考答案一", "0.0%", "40.0%", "0.0%", "40.0%"],
        )

        judge_update, _detail_divider, _model_update, _detail_update, divider_update, metric_update = refresh(
            "L33-SM3E", self.question_set_id, judge_rows, None, True, False
        )
        self.assertTrue(judge_update["visible"])
        self.assertFalse(divider_update["visible"])
        self.assertFalse(metric_update["visible"])
        judge_update, *_others = refresh("L33-SM3E", self.question_set_id, [], None, True, False)
        self.assertFalse(judge_update["visible"])

    def test_app_builds_with_automatic_evaluation_controls(self) -> None:
        app = create_app(self.root / "ui-projects")

        labels = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }
        button_values = [
            component.get_config().get("value")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        ]
        self.assertIn("逐題自動評測結果", labels)
        result_table = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config")
            and (component.get_config().get("label") or "").startswith("模型評測結果")
        )
        self.assertFalse(result_table.get_config().get("interactive"))
        self.assertEqual(
            result_table.get_config()["headers"],
            ["題號", "來源", "題目", "系統回答", "正確答案", "LLM 分數", "來源分數", "綜合分數", "各評審分數", "實際檢索策略"],
        )
        self.assertIn("只重跑前次答錯題目", labels)
        self.assertIn("自動評測 JSON", labels)
        self.assertIn("評測結果 JSON（按匯出後顯示）", labels)
        self.assertIn("評測結果 CSV（按匯出後顯示）", labels)
        self.assertIn("檢索並生成回答", button_values)
        self.assertIn("評測回答", button_values)
        self.assertIn("匯出評測結果（JSON／CSV）", button_values)
        self.assertIn("評測請求並行數", labels)
        self.assertIn("本次回答模型", labels)
        self.assertEqual(sum(label == "回答格式" for label in labels), 1)
        compatibility_handler = next(
            fn.fn for fn in app.fns.values()
            if getattr(fn.fn, "__name__", "") == "update_autoqa_model_method_compatibility"
        )
        model_update, method_update = compatibility_handler("gpt-6-luna", "local")
        self.assertIn("drift", [value for _label, value in method_update["choices"]])
        self.assertEqual(method_update["value"], "local")
        drift_model_update, drift_update = compatibility_handler("gpt-4o-mini", "drift")
        self.assertIn("drift", [value for _label, value in drift_update["choices"]])
        self.assertIn("gpt-6-luna", drift_model_update["choices"])
        self.assertIn("尚未生成回答。完成前不能評測。", [
            component.get_config().get("value")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        ])
        ask_handler = next(fn for fn in app.fns.values() if getattr(fn.fn, "__name__", "") == "ask_question")
        # Single-query now also accepts the reasoning-effort control.
        self.assertEqual(len(ask_handler.inputs), 6)
        self.assertNotIn(
            "autosave_automatic_qa_judgements",
            [getattr(fn.fn, "__name__", "") for fn in app.fns.values()],
            "the result table is read-only again (F-21 ended)",
        )
        response_type_dropdowns = [
            component for component in app.blocks.values()
            if hasattr(component, "get_config")
            and component.get_config().get("label") == "回答格式"
        ]
        self.assertEqual(len(response_type_dropdowns), 2)

        export_button = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config")
            and component.get_config().get("value") == "匯出評測結果（JSON／CSV）"
        )
        export_event = next(
            dependency for dependency in app.config["dependencies"]
            if (export_button._id, "click") in dependency.get("targets", [])
        )
        export_handler = app.fns[export_event["id"]].fn
        self.assertEqual(getattr(export_handler, "__name__", ""), "export_automatic_evaluation")
        self.assertEqual(len(export_event["inputs"]), 2)
        self.assertEqual(len(export_event["outputs"]), 3)
        self.assertTrue(all(component.get_config()["value"] == "Single Paragraph"
                            for component in response_type_dropdowns))
        tabs = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }
        self.assertTrue({"0 專案與連線設定", "1 文件與建圖", "2 問答測試", "3 自動問答測試", "4 檢索實驗"}.issubset(tabs))
        self.assertNotIn("新題目集名稱", labels)
        self.assertNotIn("題目集內容（每題均包含問題與答案）", labels)
        self.assertNotIn("將目前問題與編輯後答案加入題目集", button_values)

    def test_autoqa_export_callback_stages_judge_json_and_csv_downloads(self) -> None:
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, FakeJudge([self.judged()])
        )
        service.evaluate("L33-SM3E", self.question_set_id, model="gpt-4o-mini", enable_classic_metrics=True)
        app = create_app(self.store.root)
        export = next(
            fn.fn for fn in app.fns.values()
            if getattr(fn.fn, "__name__", "") == "export_automatic_evaluation"
        )

        message, json_download, csv_download = export("L33-SM3E", self.question_set_id)

        self.assertIn("已匯出自動評測 JSON 與 CSV", message)
        json_path, csv_path = Path(json_download), Path(csv_download)
        self.assertTrue(json_path.is_file())
        self.assertTrue(csv_path.is_file())
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["question_results"]), 2)
        first = payload["question_results"][0]
        self.assertEqual(first["question"], "問題一")
        self.assertEqual(first["expected_answer"], "參考答案一")
        self.assertEqual(first["system_answer"], "系統答案1")
        self.assertEqual(first["raw_system_answer"], "系統答案1")
        self.assertEqual(first["system_answer_sources"], [])
        self.assertEqual(first["answer_sources"][0]["pages"], [21])
        self.assertTrue(payload["question_results"][0]["judges"])
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["question"], "問題一")
        self.assertEqual(rows[0]["expected_answer"], "參考答案一")
        self.assertIn("judge_reason:gpt-4o-mini", rows[0])

        message, json_download, csv_download = export(None, self.question_set_id)
        self.assertIn("請先選擇專案與題目集", message)
        self.assertIsNone(json_download)
        self.assertIsNone(csv_download)


if __name__ == "__main__":
    unittest.main()
