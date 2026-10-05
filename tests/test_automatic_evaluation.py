import csv
import json
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from automotive_graphrag.app import create_app
from automotive_graphrag.automatic_evaluation import AutomaticEvaluationService, PROMPT_VERSION
from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.querying import QueryResult
from automotive_graphrag.question_sets import QuestionSetService
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
    def ask(self, project_id: str, question: str, method: str = "local") -> QueryResult:
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
                    "name": "自動評測集",
                    "questions": [
                        {
                            "question_id": "Q001",
                            "question": "問題一",
                            "reference_answer": "參考答案一",
                            "gold_evidence": [
                                {"document_id": "WW.pdf", "pages": [21], "chunk_ids": ["gold-1"]}
                            ],
                        },
                        {
                            "question_id": "Q002",
                            "question": "問題二",
                            "reference_answer": "參考答案二",
                            "gold_evidence": [
                                {"document_id": "WW.pdf", "pages": [22], "chunk_ids": ["gold-2"]}
                            ],
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
                    "question_id": "Q001",
                    "result": "incorrect" if q1_score < 4 else "correct",
                    "reason": "第一題評語",
                },
                {
                    "question_id": "Q002",
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
        self.assertEqual(judge.calls[0][2], "gpt-4o-mini")
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
            "L33-SM3E", self.question_set_id, "Q001", "partially_correct", "人工抽查"
        )
        json_path, csv_path = service.export("L33-SM3E", self.question_set_id)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["model"], "gpt-4o-mini")
        self.assertEqual(payload["human_reviews"][0]["human_label"], "partially_correct")
        with csv_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reviewer_note"], "人工抽查")

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
            if '"question_id": "Q001"' in prompt and "correct_answer" in prompt
        )
        self.assertIn("Q001", rerun_prompt)
        self.assertNotIn("Q002", rerun_prompt)
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

    def test_manual_judgement_edit_clears_stale_reason_and_recalculates_count(self) -> None:
        service = AutomaticEvaluationService(
            self.store, self.question_sets, self.connections, FakeJudge([self.judged()])
        )
        service.evaluate("L33-SM3E", self.question_set_id)

        updated = service.update_manual_results(
            "L33-SM3E", self.question_set_id, {"Q001": "正確", "Q002": "正確"}
        )

        self.assertEqual(updated.correct_count, 2)
        self.assertTrue(updated.items[0].is_correct)
        self.assertEqual(updated.items[0].judge_reason, "")
        self.assertTrue(updated.items[1].is_correct)
        self.assertEqual(updated.items[1].judge_reason, "第二題評語")
        self.assertEqual(
            service.last_result("L33-SM3E", self.question_set_id),
            updated,
        )

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
        self.assertIn("只重跑前次答錯題目", labels)
        self.assertIn("自動評測 JSON", labels)
        self.assertIn("檢索並生成回答", button_values)
        self.assertIn("評測回答", button_values)
        self.assertIn("評測請求並行數", labels)
        self.assertIn("尚未生成回答。完成前不能評測。", [
            component.get_config().get("value")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        ])
        tabs = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }
        self.assertTrue({"0-0 專案設定", "0-1 連線設定", "0-2 文件與建圖", "0-3 問答測試", "0-4 自動問答測試", "0-5 檢索實驗"}.issubset(tabs))
        self.assertNotIn("新題目集名稱", labels)
        self.assertNotIn("題目集內容（每題均包含問題與答案）", labels)
        self.assertNotIn("將目前問題與編輯後答案加入題目集", button_values)


if __name__ == "__main__":
    unittest.main()
