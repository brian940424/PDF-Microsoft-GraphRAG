import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
import numpy as np

from automotive_graphrag.app import create_app
from automotive_graphrag.automatic_evaluation import AutomaticEvaluationItem, AutomaticEvaluationService
from automotive_graphrag.connections import ConnectionSettings
from automotive_graphrag.projects import ProjectStore
from automotive_graphrag.question_sets import BatchQuestion, QuestionSetService
from automotive_graphrag.retrieval_experiments import ExperimentGroup, RetrievalExperimentService


class FakeJudge:
    def evaluate_single_answer(self, project_id, question_id, question, correct, actual, model):
        return AutomaticEvaluationItem(question_id, actual == correct, f"model={model}")


class RetrievalExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.projects = ProjectStore(self.root / "projects")
        self.projects.create(
            project_id="project", display_name="測試專案", vehicle_name="一般", manual_version="v1"
        )
        self.projects.update_status("project", "INDEXED")
        self.question_sets = QuestionSetService(self.projects)
        self.connections = ConnectionSettings(self.projects.root)
        self.connections.save("https://api.openai.com/v1", "test-key")
        self.calls = []

        def query(project_id, question, method, model):
            self.calls.append((project_id, question, method, model))
            return "先確認設備已關閉，準備指定工具，並依序完成安全檢查。", {}

        self.service = RetrievalExperimentService(
            self.projects,
            self.question_sets,
            self.connections,
            FakeJudge(),  # type: ignore[arg-type]
            query_function=query,
        )
        self.question_file = self.root / "questions.json"
        example = Path(__file__).parents[1] / "docs" / "檢索實驗格式範例.json"
        self.question_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_import_schema_v1_question_set_with_cross_page_source(self):
        question_set = self.service.import_question_set("project", self.question_file)
        question = question_set.questions[0]

        self.assertEqual(question.question_id, "Q0001")
        self.assertEqual(len(question.question_source_evidence), 1)
        self.assertEqual(question.question_source_evidence[0].pages, (330,))
        self.assertEqual(len(question.answer_source_evidence), 1)
        self.assertEqual(self.service.load("project")["question_set_id"], question_set.question_set_id)

    def test_group_settings_autosave_reload_and_only_explicit_remove_deletes(self):
        self.service.add_group("project")
        state = self.service.add_group("project")
        original = [dict(item) for item in state["groups"]]
        rows = [[original[0]["group_id"], "", "", "", ""]]
        state = self.service.update_group_fields("project", np.array(rows, dtype=object))
        groups = [ExperimentGroup(**item) for item in state["groups"]]
        self.service.save_configuration("project", groups, "question-set-id", 5)

        reloaded = RetrievalExperimentService(
            self.projects, self.question_sets, self.connections, FakeJudge(), query_function=lambda *args: ("", {})
        ).load("project")
        self.assertEqual(len(reloaded["groups"]), 2)
        self.assertEqual(reloaded["groups"][0]["name"], original[0]["name"])
        self.assertEqual(reloaded["question_set_id"], "question-set-id")
        self.service.remove_group("project", original[0]["group_id"])
        self.assertEqual(len(self.service.load("project")["groups"]), 1)

    def test_run_judges_each_group_and_persists_summary_and_results(self):
        question_set = self.service.import_question_set("project", self.question_file)
        groups = [
            ExperimentGroup("G01", "Local baseline", "gpt-4o-mini", "gpt-4.1-mini", "local"),
            ExperimentGroup("G02", "Basic baseline", "gpt-4.1-mini", "gpt-4o-mini", "basic"),
        ]
        self.service.save_configuration("project", groups, question_set.question_set_id, 2)

        run = self.service.run("project", question_set.question_set_id, groups, max_concurrency=2)
        loaded = self.service.load("project")["run"]

        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.results), 4)
        self.assertEqual({call[2] for call in self.calls}, {"local", "basic"})
        self.assertEqual({call[3] for call in self.calls}, {"gpt-4o-mini", "gpt-4.1-mini"})
        self.assertTrue(all(result.evaluation_result == "正確" for result in run.results))
        self.assertTrue(all(result.answer_source_rank is None for result in run.results))
        self.assertTrue(all(group.recall_at_5 is None and group.mrr is None for group in run.groups))
        self.assertEqual(loaded["status"], "completed")
        payload = json.loads(self.service.export("project").read_text(encoding="utf-8"))
        self.assertEqual(payload["execution_status"], "completed")
        self.assertEqual(len(payload["experiment_groups"]), 2)
        self.assertEqual(len(payload["question_results"]), 4)
        self.assertEqual(payload["experiment_groups"][0]["answer_model"], "gpt-4o-mini")
        self.assertEqual(payload["experiment_groups"][0]["judge_model"], "gpt-4.1-mini")
        self.assertEqual(payload["question_results"][0]["answer_model"], "gpt-4o-mini")
        self.assertEqual(payload["question_results"][0]["judge_model"], "gpt-4.1-mini")

    def test_stop_keeps_completed_results_and_saves_partial_summary(self):
        question_set = self.question_sets.create_with_questions(
            "project", "many questions", [
                BatchQuestion(f"Q{i}", f"問題{i}", reference_answer="標準答案") for i in range(1, 5)
            ],
        )
        group = ExperimentGroup("G01", "Local", "gpt-4o-mini", "gpt-4o-mini", "local")

        def stop_after_first(run):
            if run.results:
                self.service.stop("project")

        run = self.service.run(
            "project", question_set.question_set_id, [group], max_concurrency=1,
            update_callback=stop_after_first,
        )

        self.assertEqual(run.status, "stopped")
        self.assertEqual(len(run.results), 1)
        self.assertEqual(self.service.load("project")["run"]["groups"][0]["completed_count"], 1)

    def test_each_strategy_uses_its_graph_rag_32_cli_query_api(self):
        graph_root = self.projects.path_for("project") / "graphrag"
        (graph_root / "output").mkdir(parents=True)
        (graph_root / "settings.yaml").write_text(
            "completion_models:\n  default:\n    model: old-model\n    api_base: old-url\n"
            "embedding_models:\n  default:\n    model: old-embedding\n    api_base: old-url\n",
            encoding="utf-8",
        )
        calls = {}
        def capture(name):
            def run(**kwargs):
                root = kwargs["root_dir"]
                settings = yaml.safe_load((root / "settings.yaml").read_text(encoding="utf-8"))
                calls[name] = {
                    **kwargs,
                    "output_is_symlink": (root / "output").is_symlink(),
                    "completion_model": settings["completion_models"]["default"]["model"],
                }
                return "answer", {}
            return run

        with (
            patch("graphrag.cli.query.run_local_search", side_effect=capture("local")) as local,
            patch("graphrag.cli.query.run_global_search", side_effect=capture("global")) as global_search,
            patch("graphrag.cli.query.run_drift_search", side_effect=capture("drift")) as drift,
            patch("graphrag.cli.query.run_basic_search", side_effect=capture("basic")) as basic,
        ):
            for method in ("local", "global", "drift", "basic"):
                answer, _ = self.service._graphrag_query("project", "查詢", method, "gpt-4.1-mini")
                self.assertEqual(answer, "answer")

        self.assertTrue(all(call.call_count == 1 for call in (local, global_search, drift, basic)))
        self.assertEqual(calls["local"]["community_level"], 2)
        self.assertFalse(calls["global"]["dynamic_community_selection"])
        self.assertEqual(calls["drift"]["community_level"], 2)
        self.assertNotIn("community_level", calls["basic"])
        for kwargs in calls.values():
            self.assertEqual(kwargs["response_type"], "Multiple Paragraphs")
            self.assertFalse(kwargs["streaming"])
            self.assertTrue(kwargs["output_is_symlink"])
            self.assertEqual(kwargs["completion_model"], "gpt-4.1-mini")

    def test_app_exposes_retrieval_experiment_page_and_controls(self):
        app = create_app(self.root / "ui-projects")
        labels = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }

        self.assertIn("實驗組", labels)
        self.assertIn("GraphRAG 檢索策略", labels)
        self.assertIn("測試最大並行請求數", labels)
        self.assertIn("實驗組摘要", labels)
        self.assertIn("逐題實驗結果", labels)


if __name__ == "__main__":
    unittest.main()
