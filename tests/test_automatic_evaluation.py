import csv
import json
import tempfile
import threading
import time
import unittest
import urllib.error
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from automotive_graphrag.app import create_app
from automotive_graphrag.automatic_evaluation import AutomaticEvaluationService, PROMPT_VERSION, extract_json_object
from automotive_graphrag.connections import ConnectionSettings, JudgeConnectionSettings
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import BatchQuestion, QuestionSetService
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


class FakeJudge:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str, str, str]] = []
        self.lock = threading.Lock()
        self.calls_by_question: dict[str, int] = {}

    def __call__(self, base_url: str, api_key: str, model: str, prompt: str) -> object:
        cases = json.loads(prompt.split("評測案例：", 1)[1])
        question_id = cases[0]["question_id"]
        with self.lock:
            self.calls.append((base_url, api_key, model, prompt))
            response_index = self.calls_by_question.get(question_id, 0)
            self.calls_by_question[question_id] = response_index + 1
            response = self.responses[min(response_index, len(self.responses) - 1)]
        if isinstance(response, dict) and isinstance(response.get("items"), list):
            matching = [item for item in response["items"] if item.get("question_id") == question_id]
            return {"items": matching or response["items"]}
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
    def judged(q1_score: int = 3) -> dict[str, object]:
        return {
            "items": [
                {
                    "question_id": "Q0001",
                    "result": "incorrect" if q1_score < 4 else "correct",
                    "reason": "第一題評語",
                },
                {
                    "question_id": "Q0002",
                    "result": "correct",
                    "reason": "第二題評語",
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
        self.assertIn("必要步驟", result.judge_prompt)
        self.assertIn("額外步驟", result.judge_prompt)
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

    def test_litellm_judge_uses_judge_connection_and_records_model(self) -> None:
        judge = FakeJudge([self.judged()])
        judge_connections = JudgeConnectionSettings(self.store.root)
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, judge, judge_connections=judge_connections
        )
        self.assertNotIn("litellm:claude-judge", service.judge_model_choices())
        with self.assertRaisesRegex(ProjectError, "LiteLLM"):
            service.evaluate("L33-SM3E", self.question_set_id, model="litellm:claude-judge")
        self.assertEqual(judge.calls, [])

        judge_connections.save("http://localhost:4000", "sk-litellm", ["claude-judge"])
        self.assertEqual(
            service.judge_model_choices(),
            ("gpt-4o-mini", "gpt-4.1-mini", "gpt-6-luna", "litellm:claude-judge"),
        )
        result = service.evaluate("L33-SM3E", self.question_set_id, model="litellm:claude-judge")

        self.assertEqual(result.model, "litellm:claude-judge")
        self.assertEqual(
            {(url, key, model) for url, key, model, _prompt in judge.calls},
            {("http://localhost:4000", "sk-litellm", "litellm:claude-judge")},
        )
        grade_judge = lambda *_args, **_kwargs: {"item": {"question_id": "Q0001", "result": 2, "reason": "ok"}}
        service.client = grade_judge
        grade = service.evaluate_single_answer_grade(
            "L33-SM3E", "Q0001", "問題一", "參考答案一", "系統答案1", "litellm:claude-judge"
        )
        self.assertEqual(grade.score, 2)

        openai_judge = FakeJudge([self.judged()])
        service.client = openai_judge
        service.evaluate("L33-SM3E", self.question_set_id, model="gpt-4o-mini")
        self.assertEqual(openai_judge.calls[0][:3], ("https://api.openai.com/v1", "test-key", "gpt-4o-mini"))

    def test_judge_panel_scores_every_question_with_each_model_and_averages(self) -> None:
        verdicts = {
            ("gpt-4.1-mini", "Q0001"): "correct", ("gpt-4.1-mini", "Q0002"): "correct",
            ("litellm:claude-judge", "Q0001"): "incorrect", ("litellm:claude-judge", "Q0002"): "correct",
            ("litellm:gemini-flash", "Q0002"): "correct",
        }
        prompts: dict[str, set[str]] = {}
        lock = threading.Lock()

        def client(_url, _key, model, prompt, reasoning_effort="medium"):
            question_id = json.loads(prompt.split("評測案例：", 1)[1])[0]["question_id"]
            with lock:
                prompts.setdefault(question_id, set()).add(prompt)
            if (model, question_id) not in verdicts:
                raise RuntimeError("proxy timeout")
            return {"items": [{"question_id": question_id, "result": verdicts[(model, question_id)], "reason": model}]}

        judge_connections = JudgeConnectionSettings(self.store.root)
        judge_connections.save("http://localhost:4000", "sk", ["claude-judge", "gemini-flash"])
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, client, judge_connections=judge_connections
        )
        progress: list[tuple[int, int]] = []
        panel = ["gpt-4.1-mini", "litellm:claude-judge", "litellm:gemini-flash"]
        result = service.evaluate(
            "L33-SM3E", self.question_set_id, model=panel, progress_callback=lambda *args: progress.append(args)
        )

        # Every judge receives exactly the same grading prompt for a question.
        self.assertEqual({question_id: len(values) for question_id, values in prompts.items()}, {"Q0001": 1, "Q0002": 1})
        self.assertEqual(progress[-1], (6, 6))
        self.assertEqual(result.models, tuple(panel))
        self.assertEqual(result.model, "gpt-4.1-mini, litellm:claude-judge, litellm:gemini-flash")
        first, second = result.items
        self.assertEqual([judge.score for judge in first.judges], [1.0, 0.0, None])
        self.assertEqual(first.judges[2].error, "答案評判 API 呼叫失敗：proxy timeout")
        self.assertEqual(first.score, 0.5)
        self.assertEqual(second.score, 1.0)
        self.assertEqual(result.average_score, 0.75)
        self.assertEqual(
            result.judge_average_scores,
            {"gpt-4.1-mini": 1.0, "litellm:claude-judge": 0.5, "litellm:gemini-flash": 1.0},
        )
        self.assertEqual(result.disagreement_count, 1)
        self.assertIn("部分評審失敗", result.judge_error)
        self.assertIn("LiteLLM／gemini-flash：失敗", first.judge_reason)
        self.assertEqual(service.last_result("L33-SM3E", self.question_set_id), result)

        _json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        self.assertEqual(rows[0]["average_score"], "0.5")
        self.assertEqual(rows[0]["judge_score:litellm:claude-judge"], "0.0")
        self.assertEqual(rows[0]["judge_reason:litellm:gemini-flash"], "答案評判 API 呼叫失敗：proxy timeout")

        with self.assertRaisesRegex(ProjectError, "最多選擇 5"):
            service.evaluate("L33-SM3E", self.question_set_id, model=["gpt-4o-mini", *(f"m{i}" for i in range(5))])

        app = create_app(self.store.root)
        detail_table = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("label") == "各評審評分細節"
        )
        self.assertEqual(
            detail_table.get_config()["headers"],
            ["題號", "來源文件", "題目", "系統回答", "正確答案", "評審分數", "評分原因", "實際檢索策略"],
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

        switched = handlers["autoqa_judge_detail_view"]("L33-SM3E", self.question_set_id, "litellm:gemini-flash")
        self.assertEqual(
            [row[5:7] for row in switched["value"]],
            [["失敗", "答案評判 API 呼叫失敗：proxy timeout"], ["100%", "litellm:gemini-flash"]],
        )

        _judge, detail_divider, model_update, detail_update, *_rest = handlers["autoqa_result_tables"](
            "L33-SM3E", self.question_set_id, [], "litellm:claude-judge", False, False
        )
        self.assertFalse(model_update["visible"])
        self.assertFalse(detail_divider["visible"])
        self.assertFalse(detail_update["visible"])

    def test_litellm_connection_test_does_not_preprocess_dynamic_model_choices(self) -> None:
        app = create_app(self.store.root)
        model_choices = next(
            component for component in app.blocks.values()
            if getattr(component, "label", None) == "可選評測模型（LiteLLM config.yaml 的 model_name 別名）"
        )
        connection_test = next(
            dependency for dependency in app.config["dependencies"]
            if dependency.get("api_name") == "test_judge_connection"
        )
        self.assertNotIn(model_choices._id, connection_test["inputs"])
        self.assertEqual(len(connection_test["inputs"]), 2)

    def test_question_fails_only_when_every_judge_fails(self) -> None:
        def client(_url, _key, model, prompt):
            question_id = json.loads(prompt.split("評測案例：", 1)[1])[0]["question_id"]
            if question_id == "Q0001":
                raise RuntimeError(f"{model} down")
            return {"items": [{"question_id": question_id, "result": "correct", "reason": "ok"}]}

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
        judge_connections = JudgeConnectionSettings(self.store.root)
        judge_connections.save("http://localhost:4000", "sk", ["claude-judge", "plain"], ["claude-judge"])
        efforts: list[tuple[str, object]] = []

        def client(_url, _key, model, _prompt, reasoning_effort="medium"):
            efforts.append((model, reasoning_effort))
            return {"item": {"question_id": "Q0001", "result": 1, "reason": "partial"}}

        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, client, judge_connections=judge_connections
        )
        self.assertTrue(service.supports_judge_reasoning("litellm:claude-judge"))
        self.assertFalse(service.supports_judge_reasoning("litellm:plain"))
        self.assertTrue(service.supports_judge_reasoning("gpt-6-luna"))
        for model in ("litellm:claude-judge", "litellm:plain"):
            service.evaluate_single_answer_grade("L33-SM3E", "Q0001", "問", "答", "系統", model, reasoning_effort="high")
        self.assertEqual(efforts, [("litellm:claude-judge", "high"), ("litellm:plain", None)])

    def test_plain_text_judge_reply_extracts_first_json_object(self) -> None:
        reply = '判定如下：\n```json\n{"item": {"question_id": "Q0001", "result": 2, "reason": "ok"}}\n```'
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, lambda *_args, **_kwargs: reply
        )
        grade = service.evaluate_single_answer_grade("L33-SM3E", "Q0001", "問", "答", "系統", "gpt-4o-mini")
        self.assertEqual(grade.score, 2)
        self.assertIsNone(extract_json_object("no json {here"))
        self.assertEqual(extract_json_object('x {"a": {"b": 1}} y'), {"a": {"b": 1}})

    def test_litellm_chat_strips_prefix_and_retries_without_json_mode(self) -> None:
        bodies: list[dict] = []

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
            body = json.loads(request.data.decode("utf-8"))
            bodies.append(body)
            if request.get_header("Cf-access-client-id"):
                self.assertEqual(request.get_header("Cf-access-client-id"), "id")
                self.assertEqual(request.get_header("Cf-access-client-secret"), "secret")
            if "response_format" in body:
                raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)
            return Response({"choices": [{"message": {"content": 'Sure: {"items": []} done'}}]})

        with patch("urllib.request.urlopen", fake_urlopen):
            value = AutomaticEvaluationService._openai_chat(
                "http://localhost:4000", "", "litellm:plain", "prompt", reasoning_effort=None,
                extra_headers={"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "secret"},
            )
        self.assertEqual(value, {"items": []})
        self.assertEqual([body["model"] for body in bodies], ["plain", "plain"])
        self.assertNotIn("reasoning_effort", bodies[0])
        self.assertNotIn("response_format", bodies[1])

        bodies.clear()
        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(ProjectError, "HTTP 400"):
                AutomaticEvaluationService._openai_chat("https://api.openai.com/v1", "k", "gpt-4o-mini", "p")
        self.assertEqual(len(bodies), 1)

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
        self.assertIn("使用寬鬆的語意比對", prompt)
        self.assertIn("必須接受同義詞", prompt)
        self.assertIn("不要只因回答比正確答案更詳細", prompt)
        self.assertNotIn("額外步驟／實質資訊，判為 incorrect", prompt)
        self.assertNotIn('"retrieved_evidence"', prompt)

    def test_strict_and_lenient_prompts_have_distinct_criteria(self) -> None:
        question = BatchQuestion("Q1", "問題", "標準答案", answer="回答")
        strict = AutomaticEvaluationService._build_prompt([question])
        lenient = AutomaticEvaluationService._build_prompt([question], "lenient")

        self.assertIn("加入正確答案未支持的額外步驟／實質資訊", strict)
        self.assertIn("合理補充", lenient)
        self.assertIn("只有在回答漏掉核心資訊", lenient)
        self.assertNotIn("額外步驟／實質資訊，判為 incorrect", lenient)

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
            "items": [{"question_id": "INVENTED", "result": "correct", "reason": "錯誤題號"}]
        }
        judge = FakeJudge([response])
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)

        with self.assertRaisesRegex(ProjectError, "未知或重複"):
            service.evaluate("L33-SM3E", self.question_set_id)

    def test_evaluation_concurrency_limits_parallel_requests(self) -> None:
        class SlowJudge:
            def __init__(self):
                self.active = 0
                self.maximum = 0
                self.lock = threading.Lock()

            def __call__(self, base_url, api_key, model, prompt):
                cases = json.loads(prompt.split("評測案例：", 1)[1])
                question_id = cases[0]["question_id"]
                with self.lock:
                    self.active += 1
                    self.maximum = max(self.maximum, self.active)
                time.sleep(0.02)
                with self.lock:
                    self.active -= 1
                return {"items": [{"question_id": question_id, "result": "correct", "reason": "符合"}]}

        judge = SlowJudge()
        service = AutomaticEvaluationService(self.store, self.question_sets, self.connections, judge)
        service.evaluate("L33-SM3E", self.question_set_id, concurrency=1)
        self.assertEqual(judge.maximum, 1)
        judge.maximum = 0
        service.evaluate("L33-SM3E", self.question_set_id, concurrency=2)
        self.assertEqual(judge.maximum, 2)
        with self.assertRaisesRegex(ProjectError, "並行數必須介於 1 到 32"):
            service.evaluate("L33-SM3E", self.question_set_id, concurrency=0)

    def test_combined_score_mixes_judge_and_cited_source_scores(self) -> None:
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

        # Both answers cite the gold page; Q1 is judged incorrect, Q2 correct.
        self.assertEqual([item.source_score for item in result.items], [1.0, 1.0])
        self.assertEqual([item.combined_score for item in result.items], [0.1, 1.0])
        self.assertEqual(result.source_weight, 0.1)
        self.assertEqual(result.average_source_score, 1.0)
        self.assertEqual(result.average_combined_score, 0.55)
        self.assertEqual(result.correct_count, 1)  # correctness still follows the judge only

        updated = service.update_manual_results("L33-SM3E", self.question_set_id, {"Q0001": 1})
        self.assertEqual(updated.items[0].combined_score, 1.0)
        self.assertEqual(updated.average_combined_score, 1.0)

        weighted = service.evaluate("L33-SM3E", self.question_set_id, source_weight=0.5)
        self.assertEqual(weighted.items[0].combined_score, 0.5)
        with self.assertRaisesRegex(ProjectError, "0 到 1"):
            service.evaluate("L33-SM3E", self.question_set_id, source_weight=2)

        json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["average_combined_score"], 0.75)
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["source_score"], "1.0")

    def test_uncited_answers_score_zero_source_and_legacy_records_fall_back(self) -> None:
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, FakeJudge([self.judged()])
        )
        result = service.evaluate("L33-SM3E", self.question_set_id)
        self.assertEqual([item.source_score for item in result.items], [0.0, 0.0])
        self.assertEqual(result.average_combined_score, 0.45)

        path = service._path("L33-SM3E", self.question_set_id)
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy.pop("average_combined_score")
        for item in legacy["items"]:
            item.pop("source_score")
            item.pop("combined_score")
        path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
        loaded = service.last_result("L33-SM3E", self.question_set_id)
        self.assertEqual(loaded.average_combined_score, loaded.average_score)

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
            ["題號", "來源文件", "題目", "系統回答", "正確答案", "EM", "F1", "Term Hit", "Term Coverage"],
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
            ["Q0001", "WW.pdf", "問題一", "系統答案1", "參考答案一", "0.0%", "40.0%", "0.0%", "40.0%"],
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
            ["題號", "來源文件", "題目", "系統回答", "正確答案", "LLM 分數", "來源分數", "綜合分數", "各評審分數", "實際檢索策略"],
        )
        self.assertIn("只重跑前次答錯題目", labels)
        self.assertIn("自動評測 JSON", labels)
        self.assertIn("檢索並生成回答", button_values)
        self.assertIn("評測回答", button_values)
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


if __name__ == "__main__":
    unittest.main()
