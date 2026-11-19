import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from automotive_graphrag.app import create_app
from automotive_graphrag.automatic_evaluation import AutomaticEvaluationGrade, AutomaticEvaluationService
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore


def evidence(context_id: str, document_id: str, page: int) -> dict:
    return asdict(Evidence(
        evidence_id=f"E{context_id}", rank=1, context_id=context_id, text_unit_id="tu", chunk_id="c",
        section_id="S", section_name="", document_id=document_id, page=page, block_id="b", text="內文",
    ))


class ManualGradingUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "projects"
        ProjectStore(self.root).create(
            project_id="L33-SM3E", display_name="L33", vehicle_name="L33", manual_version="SM3E"
        )
        app = create_app(self.root)
        functions = {getattr(fn.fn, "__name__", ""): fn.fn for fn in app.fns.values()}
        self.grade = functions["grade_manual_answer"]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def judges(self, verdicts: dict):
        """Patch the per-model judge call; a verdict that is an exception is raised for that model."""
        def judge(_self, project_id, question_id, question, correct, actual, model, mode, effort):
            verdict = verdicts[model]
            if isinstance(verdict, Exception):
                raise verdict
            return verdict
        supported = mock.patch.object(AutomaticEvaluationService, "supports_judge_model", return_value=True)
        supported.start()
        self.addCleanup(supported.stop)
        return mock.patch.object(AutomaticEvaluationService, "evaluate_single_answer_grade", autospec=True, side_effect=judge)

    def test_panel_score_is_average_of_judges(self) -> None:
        verdicts = {
            "judge-a": AutomaticEvaluationGrade("MANUAL", 5, "5 級：完整"),
            "judge-b": AutomaticEvaluationGrade("MANUAL", 3, "3 級：缺少扭力值"),
        }
        with self.judges(verdicts) as judge:
            markup = self.grade(
                "L33-SM3E", " 問題 ", "系統回答", "正確答案", "", ["judge-a", "judge-b"], "lenient", "high", [],
            )

        self.assertEqual(judge.call_count, 2)
        self.assertEqual(judge.call_args.args[1:5], ("L33-SM3E", "MANUAL", "問題", "正確答案"))
        self.assertEqual(judge.call_args.args[7:], ("lenient", "high"))
        self.assertIn("LLM 分數：Level 4（80%）｜✅ 正確", markup)
        self.assertIn("未填答案來源", markup)
        self.assertIn("* 系統引用：—", markup)
        self.assertIn("`judge-a`**：Level 5（100%）｜5 級：完整", markup)
        self.assertIn("`judge-b`**：Level 3（60%）｜3 級：缺少扭力值", markup)

    def test_failed_judge_is_excluded_from_average(self) -> None:
        verdicts = {"judge-a": AutomaticEvaluationGrade("MANUAL", 2, "2 級"), "judge-b": ProjectError("逾時")}
        with self.judges(verdicts):
            markup = self.grade("L33-SM3E", "問題", "答", "正解", "", ["judge-a", "judge-b"], "strict")

        self.assertIn("LLM 分數：Level 2（40%）｜❌ 不正確", markup)
        self.assertIn("`judge-b`**：⚠️ 失敗（不計入平均）｜逾時", markup)

    def test_all_judges_failing_is_reported(self) -> None:
        with self.judges({"judge-a": ProjectError("JSON 無效")}):
            markup = self.grade("L33-SM3E", "問題", "答", "正解", "", ["judge-a"], "strict")

        self.assertEqual(markup, "❌ 評分失敗：JSON 無效")

    def test_source_score_compares_cited_pages_with_answer_sources(self) -> None:
        retrieved = [evidence("12", "manual.pdf", 40), evidence("13", "manual.pdf", 90)]
        verdicts = {"judge-a": AutomaticEvaluationGrade("MANUAL", 3, "3 級：多出步驟", extra_content_error=True)}
        with self.judges(verdicts):
            markup = self.grade(
                "L33-SM3E", "問題", "先拆卸 [Data: Sources (12)]", "正解", "manual.pdf: 41, 70",
                ["judge-a"], "strict", "medium", retrieved,
            )

        # Page 41 is hit by cited page 40 (±1), page 70 is not: source 50%; the extra-content
        # verdict earns the source bonus 0.6 + 0.1 × 0.5.
        self.assertIn("**來源分數：** 50%｜**綜合分數：** 65%", markup)
        self.assertIn("含正確答案未提及的內容", markup)
        self.assertIn("* 系統引用：manual.pdf p.40", markup)
        self.assertIn("* 正確來源：manual.pdf p.41, 70", markup)

    def test_invalid_inputs_do_not_call_judge(self) -> None:
        with mock.patch.object(AutomaticEvaluationService, "evaluate_single_answer_grade") as judge:
            self.assertIn("請先選擇", self.grade(None, "問題", "答", "正解", "", ["m"], "strict"))
            self.assertIn("系統回答、正確答案", self.grade("L33-SM3E", "問題", " ", "", "", ["m"], "strict"))
            self.assertIn("至少選擇一個評測模型", self.grade("L33-SM3E", "問題", "答", "正解", "", [], "strict"))
            self.assertIn("來源格式", self.grade("L33-SM3E", "問題", "答", "正解", "沒有頁碼", ["m"], "strict"))
        judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
