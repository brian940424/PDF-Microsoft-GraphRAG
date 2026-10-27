import json
import subprocess
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
import numpy as np
import gradio as gr

from automotive_graphrag.app import _experiment_dataset_summary_rows, _sort_experiment_results, create_app
from automotive_graphrag.automatic_evaluation import AutomaticEvaluationItem, AutomaticEvaluationService
from automotive_graphrag.connections import ConnectionSettings, DEFAULT_REASONING_EFFORT
from automotive_graphrag.evidence import Evidence
from automotive_graphrag.projects import ProjectError, ProjectStore
from automotive_graphrag.question_sets import BatchQuestion, QuestionSetService
from automotive_graphrag.retrieval_experiments import (
    DEFAULT_EXPERIMENT_RESPONSE_TYPE,
    EXPERIMENT_REASONING_EFFORT_OPTIONS,
    EXPERIMENT_RESPONSE_TYPE_OPTIONS,
    ExperimentGroup,
    RetrievalExperimentService,
)
from automotive_graphrag.source_metadata import SourceMetadata
from automotive_graphrag.strategy_routing import StrategyDecision


class FakeJudge:
    def __init__(self, lenient_result: bool | None = None):
        self.calls = []
        self.lenient_result = lenient_result

    def evaluate_single_answer(self, project_id, question_id, question, correct, actual, model, evaluation_mode="strict"):
        self.calls.append((project_id, question_id, question, correct, actual, model, evaluation_mode))
        is_correct = (
            self.lenient_result if evaluation_mode == "lenient" and self.lenient_result is not None
            else actual == correct
        )
        reason = f"寬鬆評分 model={model}" if evaluation_mode == "lenient" else f"model={model}"
        return AutomaticEvaluationItem(question_id, is_correct, reason)

    def evaluate_single_answer_grade(
        self, project_id, question_id, question, correct, actual, model,
        evaluation_mode="strict", reasoning_effort="medium",
    ):
        self.calls.append((project_id, question_id, question, correct, actual, model, evaluation_mode))
        score = 2 if actual == correct else (1 if actual and correct and actual in correct else 0)
        if evaluation_mode == "lenient" and self.lenient_result is not None:
            score = 2 if self.lenient_result else 0
        reason = f"寬鬆評分 model={model}" if evaluation_mode == "lenient" else f"model={model}"
        return SimpleNamespace(question_id=question_id, score=score, judge_reason=reason)


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

        def query(project_id, question, method, model, response_type, reasoning_effort):
            self.calls.append((project_id, question, method, model, response_type, reasoning_effort))
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

    def test_experiment_results_sort_by_group_then_question_set_order(self):
        groups = [{"group_id": "G01"}, {"group_id": "G02"}]
        questions = [
            SimpleNamespace(question_id="Q1"),
            SimpleNamespace(question_id="Q2"),
        ]
        completed_out_of_order = [
            {"group_id": "G02", "question_id": "Q2"},
            {"group_id": "G01", "question_id": "Q2"},
            {"group_id": "G02", "question_id": "Q1"},
            {"group_id": "G01", "question_id": "Q1"},
        ]

        ordered = _sort_experiment_results(completed_out_of_order, groups, questions)

        self.assertEqual(
            [(item["group_id"], item["question_id"]) for item in ordered],
            [("G01", "Q1"), ("G01", "Q2"), ("G02", "Q1"), ("G02", "Q2")],
        )

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

    def test_classic_only_scores_aliases_and_exports_per_question_and_average_metrics(self):
        question_set = self.service.import_question_set("project", self.question_file)
        first = replace(
            question_set.questions[0],
            reference_answer="主要答案以外的文字",
            answer_aliases=("先確認設備已關閉，準備指定工具，並依序完成安全檢查。",),
        )
        self.question_sets.update_questions(
            "project", question_set.question_set_id, (first, *question_set.questions[1:])
        )
        group = ExperimentGroup("G01", "實驗組1", "gpt-4o-mini", "local")
        self.service.save_configuration(
            "project", [group], question_set.question_set_id, 2, "gpt-4.1-mini"
        )
        self.service.generate_answers("project", question_set.question_set_id, [group], max_concurrency=2)

        run = self.service.evaluate_answers(
            "project", question_set.question_set_id,
            enable_llm_judge=False, enable_classic_metrics=True,
        )

        self.assertEqual(self.service.judging.calls, [])
        self.assertFalse(run.enable_llm_judge)
        self.assertTrue(run.enable_classic_metrics)
        self.assertEqual(run.results[0].answer_em, 1.0)
        self.assertEqual(run.results[0].answer_f1, 1.0)
        self.assertEqual(run.results[0].answer_term_hit, 1.0)
        self.assertEqual(run.results[0].answer_term_coverage, 1.0)
        self.assertEqual(run.results[0].evaluation_result, "未啟用")
        self.assertEqual(run.groups[0].average_answer_em, 1.0)
        self.assertEqual(run.groups[0].average_answer_f1, 1.0)
        self.assertEqual(run.groups[0].average_answer_term_hit, 1.0)
        self.assertEqual(run.groups[0].average_answer_term_coverage, 1.0)

        summary_path, details_path = self.service.export("project")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertFalse(summary["evaluation"]["enable_llm_judge"])
        self.assertTrue(summary["evaluation"]["enable_classic_metrics"])
        self.assertEqual(summary["summary"]["average_answer_em"], 1.0)
        self.assertEqual(summary["summary"]["average_answer_term_hit"], 1.0)
        self.assertEqual(details["question_results"][0]["answer_em"], 1.0)
        self.assertEqual(details["question_results"][0]["answer_term_coverage"], 1.0)

    def test_failed_replacement_keeps_existing_sets_when_source_pages_are_empty(self):
        existing = self.service.import_question_set("project", self.question_file)
        invalid_file = self.root / "invalid-pages.json"
        invalid_payload = json.loads(self.question_file.read_text(encoding="utf-8"))
        invalid_payload["questions"][0]["question_sources"][0]["pages"] = []
        invalid_file.write_text(json.dumps(invalid_payload), encoding="utf-8")

        imported, errors = self.service.import_question_sets_with_errors(
            "project", [invalid_file, self.question_file]
        )

        self.assertEqual(imported, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid-pages.json", errors[0])
        self.assertIn("question_sources[0].pages", errors[0])
        self.assertEqual(self.service.load("project")["question_set_ids"], [existing.question_set_id])
        self.assertEqual(self.question_sets.get("project", existing.question_set_id), existing)

    def test_importing_new_sets_replaces_and_deletes_old_sets(self):
        old_set = self.service.import_question_set("project", self.question_file)
        second_file = self.root / "second-dataset.json"
        second_file.write_text(self.question_file.read_text(encoding="utf-8"), encoding="utf-8")

        imported = self.service.import_question_sets("project", [self.question_file, second_file])

        imported_ids = [item.question_set_id for item in imported]
        self.assertEqual(self.service.load("project")["question_set_ids"], imported_ids)
        self.assertFalse(
            (self.projects.path_for("project") / "question_sets" / f"{old_set.question_set_id}.json").exists()
        )

    def test_removing_question_set_deletes_it_and_clears_saved_run(self):
        question_set = self.service.import_question_set("project", self.question_file)
        state = self.service.load("project")
        state["run"] = {"status": "stopped"}
        self.service._write("project", state)

        updated = self.service.remove_question_set("project", question_set.question_set_id)

        self.assertEqual(updated["question_set_ids"], [])
        self.assertIsNone(updated["run"])
        self.assertFalse(
            (self.projects.path_for("project") / "question_sets" / f"{question_set.question_set_id}.json").exists()
        )

    def test_run_multiple_question_sets_keeps_dataset_identity_on_every_result(self):
        second_file = self.root / "second-dataset.json"
        second_file.write_text(self.question_file.read_text(encoding="utf-8"), encoding="utf-8")
        imported = self.service.import_question_sets("project", [self.question_file, second_file])
        dataset_ids = [item.question_set_id for item in imported]
        self.assertEqual(self.service.load("project")["question_set_ids"], dataset_ids)

        group = ExperimentGroup("G01", "實驗組1", "gpt-4o-mini", "local")
        self.service.save_configuration("project", [group], dataset_ids, 2, "gpt-4.1-mini")
        generated = self.service.generate_answers("project", dataset_ids, [group], max_concurrency=2)
        self.assertEqual(generated.status, "answers_completed")
        run = self.service.evaluate_answers(
            "project", dataset_ids, judge_model="gpt-4.1-mini", max_concurrency=2
        )

        self.assertEqual(run.question_count, 4)
        self.assertEqual(len(run.results), 4)
        self.assertEqual({item.dataset_id for item in run.results}, set(dataset_ids))
        self.assertEqual({item.dataset_name for item in run.results}, {"questions", "second-dataset"})
        self.assertTrue(all(item.evaluation_result == 2 for item in run.results))
        dataset_summaries = _experiment_dataset_summary_rows(
            self.service.load("project")["run"],
            [asdict(group)],
            [self.question_sets.get("project", item) for item in dataset_ids],
        )
        self.assertEqual(len(dataset_summaries), 2)
        self.assertEqual({row[1].split(" (")[0] for row in dataset_summaries}, {"questions", "second-dataset"})
        self.assertTrue(all(row[4:12] == [2, 2, 2, 2, 0, 0, "100.0%", "100.0%"] for row in dataset_summaries))

        self.service.update_manual_judgments(
            "project", [("實驗組1", dataset_ids[0], run.results[0].question_id, False)]
        )
        manually_updated = self.service.load("project")["run"]["results"]
        first_dataset_results = [item for item in manually_updated if item["dataset_id"] == dataset_ids[0]]
        second_dataset_results = [item for item in manually_updated if item["dataset_id"] == dataset_ids[1]]
        self.assertEqual(sum(item["evaluation_result"] == 0 for item in first_dataset_results), 1)
        self.assertTrue(all(item["evaluation_result"] == 2 for item in second_dataset_results))

        _summary_path, details_path = self.service.export("project")
        details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertEqual(set(item["dataset_id"] for item in details["question_results"]), set(dataset_ids))
        self.assertEqual(set(item["dataset_name"] for item in details["question_results"]), {"questions", "second-dataset"})

    def test_group_settings_autosave_reload_and_only_explicit_remove_deletes(self):
        state = self.service.add_group("project")
        self.assertEqual(state["groups"][0]["name"], "實驗組1")
        self.assertEqual(state["groups"][0]["answer_model"], self.connections.get_chat_model())
        self.assertEqual(state["groups"][0]["method"], "local")
        self.assertEqual(state["groups"][0]["response_type"], DEFAULT_EXPERIMENT_RESPONSE_TYPE)
        self.assertEqual(state["groups"][0]["answer_reasoning_effort"], DEFAULT_REASONING_EFFORT)
        state = self.service.add_group("project")
        self.assertEqual(state["groups"][1]["name"], "實驗組2")
        original = [dict(item) for item in state["groups"]]
        rows = [[original[0]["group_id"], "實驗組1", "gpt-4o-mini", "local"]]
        state = self.service.update_group_fields("project", np.array(rows, dtype=object))
        self.assertEqual(state["groups"][0]["method"], "local")
        self.assertNotIn("judge_model", state["groups"][0])
        state = self.service.set_group_answer_model("project", original[0]["group_id"], "gpt-4.1-mini")
        self.assertEqual(state["groups"][0]["answer_model"], "gpt-4.1-mini")
        state = self.service.set_group_response_type(
            "project", original[0]["group_id"], "List of 3-7 Points"
        )
        self.assertEqual(state["groups"][0]["response_type"], "List of 3-7 Points")
        self.service.set_group_answer_model("project", original[0]["group_id"], "gpt-6-luna")
        state = self.service.set_group_reasoning_effort("project", original[0]["group_id"], "high")
        self.assertEqual(state["groups"][0]["answer_reasoning_effort"], "high")
        groups = [ExperimentGroup(**item) for item in state["groups"]]
        self.service.save_configuration("project", groups, "question-set-id", 5, "gpt-4.1-mini")

        reloaded = RetrievalExperimentService(
            self.projects, self.question_sets, self.connections, FakeJudge(), query_function=lambda *args: ("", {})
        ).load("project")
        self.assertEqual(len(reloaded["groups"]), 2)
        self.assertEqual(reloaded["groups"][0]["name"], original[0]["name"])
        self.assertEqual(reloaded["groups"][0]["response_type"], "List of 3-7 Points")
        self.assertEqual(reloaded["groups"][0]["answer_reasoning_effort"], "high")
        self.assertEqual(reloaded["question_set_id"], "question-set-id")
        self.assertEqual(reloaded["judge_model"], "gpt-4.1-mini")
        self.service.remove_group("project", original[0]["group_id"])
        self.assertEqual(len(self.service.load("project")["groups"]), 1)

    def test_drift_accepts_gpt6_luna_and_preserves_saved_configuration(self):
        state = self.service.add_group("project")
        group_id = state["groups"][0]["group_id"]
        self.service.set_group_answer_model("project", group_id, "gpt-6-luna")
        self.service.set_group_method("project", group_id, "drift")
        saved = self.service.load("project")["groups"][0]
        self.assertEqual(saved["method"], "drift")
        self.assertEqual(saved["answer_model"], "gpt-6-luna")

        state = self.service.load("project")
        state["groups"][0]["method"] = "local"
        self.service._write("project", state)
        reloaded = self.service.load("project")["groups"][0]
        self.assertEqual(reloaded["method"], "local")
        self.assertEqual(reloaded["answer_model"], "gpt-6-luna")

    def test_drift_nested_sources_are_mapped_and_included_in_group_metrics(self):
        question_set = self.service.import_question_set("project", self.question_file)
        question = question_set.questions[0]
        gold = question.gold_evidence[0]
        page = gold.pages[0]
        evidence = Evidence(
            evidence_id="E1", rank=1, context_id="7", text_unit_id="tu-7",
            chunk_id="chunk-7", section_id="section", section_name="Section",
            document_id=gold.document_id, page=page, block_id="block", text="gold source",
        )
        group = ExperimentGroup("G01", "DRIFT", "gpt-4o-mini", "drift")
        self.service.query_function = lambda *args: (
            "回答", {"子查詢一": {"sources": [{"id": "7", "text": "source"}]}}
        )

        with patch.object(self.service, "_resolved_evidence", return_value=(evidence,)):
            result = self.service._generate_case("project", group, question, "gpt-4.1-mini")
            self.assertEqual(result.retrieved_evidence, (evidence,))
            self.assertIn("已映射", result.retrieval_metrics_status)
            saved_result = replace(result, evaluation_result=2)
            run = self.service._persist_run(
                "project", question_set.question_set_id, [group], "gpt-4.1-mini", 1,
                "started", "completed", {(group.group_id, question.question_id): saved_result},
            )

        summary = run.groups[0]
        self.assertEqual(summary.evidence_hit_rate, 1.0)
        self.assertEqual(summary.evidence_precision, 1.0)
        self.assertEqual(summary.document_hit_rate, 1.0)

    def test_auto_group_routes_each_question_and_uses_routed_strategy_parameters(self):
        question_set = self.service.import_question_set("project", self.question_file)
        question = question_set.questions[0]
        routed = []

        class FakeRouter:
            def route(self, question_text, model, api_base_url, api_key):
                routed.append((question_text, model, api_key))
                return StrategyDecision("basic", "單一數值查詢")

        received = []
        self.service.router = FakeRouter()  # type: ignore[assignment]
        self.service.query_function = lambda *args: (received.append(args) or "回答", {})
        state = self.service.add_group("project")
        group_id = state["groups"][0]["group_id"]
        self.service.set_group_method("project", group_id, "auto")
        self.service.set_group_strategy_parameter("project", group_id, "basic", "k", 3)
        group = ExperimentGroup(**self.service.load("project")["groups"][0])

        result = self.service._generate_case("project", group, question, "gpt-4.1-mini")

        self.assertEqual(routed, [(question.question, group.answer_model, "test-key")])
        self.assertEqual(received[0][2], "basic")
        self.assertEqual(received[0][6]["k"], 3)
        self.assertEqual(result.method, "auto")
        self.assertEqual(result.routed_method, "basic")
        self.assertEqual(result.effective_method, "basic")
        self.assertEqual(result.routing_reason, "單一數值查詢")
        self.assertIn("已映射", result.retrieval_metrics_status)

    def test_auto_group_metrics_skip_questions_routed_to_global(self):
        question_set = self.service.import_question_set("project", self.question_file)
        question = question_set.questions[0]
        gold = question.gold_evidence[0]
        evidence = Evidence(
            evidence_id="E1", rank=1, context_id="7", text_unit_id="tu-7",
            chunk_id="chunk-7", section_id="section", section_name="Section",
            document_id=gold.document_id, page=gold.pages[0], block_id="block", text="gold source",
        )
        group = ExperimentGroup("G01", "自動", "gpt-4o-mini", "auto")
        base = self.service._failed_result(group, question, "gpt-4.1-mini", "", question_set.question_set_id)
        local_result = replace(
            base, status="completed", error=None, actual_answer="回答", evaluation_result=2,
            routed_method="local", retrieved_evidence=(evidence,),
            retrieval_metrics_status="已映射至原始文件／頁碼",
        )
        global_result = replace(
            base, question_id="other", status="completed", error=None, actual_answer="回答",
            evaluation_result=0, routed_method="global",
            retrieval_metrics_status="已映射至原始文件／頁碼",
        )

        run = self.service._persist_run(
            "project", question_set.question_set_id, [group], "gpt-4.1-mini", 1, "started", "completed",
            {("G01", "a"): local_result, ("G01", "b"): global_result},
        )

        summary = run.groups[0]
        self.assertEqual(summary.evidence_hit_rate, 1.0)
        self.assertIn("Global 的題目不計入", summary.retrieval_metrics_status)
        self.assertEqual(set(summary.strategy_parameters), {"local", "global", "drift", "basic"})

    def test_legacy_per_group_judge_model_migrates_to_global_setting(self):
        path = self.service._path("project")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "groups": [{
                "group_id": "G01", "name": "舊組別", "answer_model": "gpt-4o-mini",
                "judge_model": "gpt-4.1-mini", "method": "local",
            }],
        }), encoding="utf-8")

        migrated = self.service.load("project")

        self.assertEqual(migrated["judge_model"], "gpt-4.1-mini")
        self.assertNotIn("judge_model", migrated["groups"][0])

    def test_run_judges_each_group_and_persists_summary_and_results(self):
        question_set = self.service.import_question_set("project", self.question_file)
        groups = [
            ExperimentGroup("G01", "Local baseline", "gpt-4o-mini", "local"),
            ExperimentGroup("G02", "Basic baseline", "gpt-4.1-mini", "basic"),
        ]
        self.service.save_configuration("project", groups, question_set.question_set_id, 2, "gpt-4.1-mini")

        run = self.service.run("project", question_set.question_set_id, groups, max_concurrency=2)
        loaded = self.service.load("project")["run"]

        self.assertEqual(run.status, "completed")
        self.assertEqual(len(run.results), 4)
        self.assertEqual({call[2] for call in self.calls}, {"local", "basic"})
        self.assertEqual({call[3] for call in self.calls}, {"gpt-4o-mini", "gpt-4.1-mini"})
        self.assertEqual({call[4] for call in self.calls}, {"Single Paragraph"})
        self.assertEqual({call[5] for call in self.calls}, {DEFAULT_REASONING_EFFORT})
        self.assertTrue(all(result.evaluation_result == 2 for result in run.results))
        self.assertTrue(all(result.judge_model == "gpt-4.1-mini" for result in run.results))
        self.assertTrue(all(result.answer_source_rank is None for result in run.results))
        local_summary = next(group for group in run.groups if group.method == "local")
        basic_summary = next(group for group in run.groups if group.method == "basic")
        self.assertEqual(local_summary.evidence_hit_rate, 0.0)
        self.assertEqual(basic_summary.document_hit_rate, 0.0)
        self.assertIn("不使用排名", local_summary.retrieval_metrics_status)
        self.assertEqual(loaded["status"], "completed")
        summary_path, details_path = self.service.export("project")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["format"], "manual-graphrag-experiment-summary")
        self.assertEqual(payload["project"], {"project_id": "project", "name": "測試專案"})
        self.assertEqual(payload["max_concurrent_requests"], 2)
        self.assertEqual(payload["evaluation"]["judge_model"], "gpt-4.1-mini")
        self.assertIsNone(payload["evaluation"]["judge_reasoning_effort"])
        self.assertEqual(payload["summary"]["question_count"], 4)
        self.assertEqual(payload["summary"]["fully_correct_total"], "4 / 4")
        self.assertEqual(payload["summary"]["partial_correct_count"], 0)
        self.assertEqual(payload["summary"]["score_accuracy"], 1.0)
        self.assertEqual(details["summary"]["score_accuracy"], 1.0)
        self.assertEqual(len(payload["groups"]), 2)
        self.assertNotIn("question_results", payload)
        self.assertEqual(payload["groups"][0]["parameters"]["answer_model"], "gpt-4o-mini")
        self.assertEqual(payload["groups"][0]["parameters"]["retrieval_mode"], "Microsoft GraphRAG Local")
        self.assertIsNone(payload["groups"][0]["parameters"]["answer_reasoning_effort"])
        self.assertEqual(payload["groups"][0]["summary"]["judge_model"], "gpt-4.1-mini")
        self.assertNotIn("top_k", payload["groups"][0]["parameters"])
        self.assertNotIn("use_reranker", payload["groups"][0]["parameters"])
        self.assertEqual(details["format"], "manual-graphrag-experiment-question-results")
        self.assertEqual(details["execution_status"], "completed")
        self.assertEqual(len(details["question_results"]), 4)
        self.assertEqual(details["question_results"][0]["answer_model"], "gpt-4o-mini")
        self.assertEqual(details["question_results"][0]["judge_model"], "gpt-4.1-mini")

        self.service.update_manual_judgments(
            "project", [("Local baseline", "Q0001", 0), ("Local baseline", "Q0002", 1)]
        )
        reloaded = RetrievalExperimentService(
            self.projects, self.question_sets, self.connections, FakeJudge(), query_function=lambda *args: ("", {})
        ).load("project")["run"]
        local_group = next(group for group in reloaded["groups"] if group["group_name"] == "Local baseline")
        edited_result = next(
            item for item in reloaded["results"]
            if item["group_name"] == "Local baseline" and item["question_id"] == "Q0001"
        )
        self.assertEqual(local_group["correct_count"], 0)
        self.assertEqual(local_group["partial_correct_count"], 1)
        self.assertEqual(local_group["incorrect_count"], 1)
        self.assertEqual(local_group["accuracy"], 0.0)
        self.assertEqual(local_group["score_accuracy"], 0.25)
        self.assertEqual(edited_result["evaluation_result"], 0)
        self.assertEqual(edited_result["evaluation_reason"], "")
        updated_summary, updated_details = self.service.export("project")
        updated_summary = json.loads(updated_summary.read_text(encoding="utf-8"))
        updated_details = json.loads(updated_details.read_text(encoding="utf-8"))
        updated_edited_result = next(
            item for item in updated_details["question_results"]
            if item["group_name"] == "Local baseline" and item["question_id"] == "Q0001"
        )
        self.assertEqual(updated_summary["summary"]["fully_correct_total"], "2 / 4")
        self.assertEqual(updated_summary["summary"]["partial_correct_count"], 1)
        self.assertEqual(updated_summary["summary"]["score_accuracy"], 0.625)
        self.assertEqual(updated_summary["groups"][0]["summary"]["fully_correct_total"], "0 / 2")
        self.assertEqual(updated_summary["groups"][0]["summary"]["partial_correct_count"], 1)
        self.assertEqual(updated_edited_result["evaluation_result"], 0)

    def test_answer_generation_and_evaluation_are_separate_persisted_steps(self):
        judge = FakeJudge()
        self.service.judging = judge
        question_set = self.question_sets.create_with_questions(
            "project", "split flow", [BatchQuestion("Q1", "問題", "標準答案")]
        )
        group = ExperimentGroup("G01", "Local", "gpt-4o-mini", "local")
        self.service.save_configuration(
            "project", [group], question_set.question_set_id, 1, "gpt-4o-mini"
        )

        generated = self.service.generate_answers(
            "project", question_set.question_set_id, [group], max_concurrency=1
        )

        self.assertEqual(generated.status, "answers_completed")
        self.assertEqual(generated.results[0].status, "answered")
        self.assertEqual(generated.results[0].evaluation_result, "待評測")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(judge.calls, [])
        self.assertEqual(self.service.load("project")["run"]["status"], "answers_completed")

        # Older combined runs could mark a generated answer failed when judging failed.
        saved_state = self.service.load("project")
        saved_state["run"]["results"][0].update(
            status="failed", evaluation_result="評判失敗", evaluation_reason="舊版評判錯誤"
        )
        self.service._write("project", saved_state)

        evaluated = self.service.evaluate_answers(
            "project", question_set.question_set_id, judge_model="gpt-4o-mini", max_concurrency=1
        )

        self.assertEqual(evaluated.status, "completed")
        self.assertEqual(evaluated.results[0].status, "completed")
        self.assertIn(evaluated.results[0].evaluation_result, {0, 1, 2})
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(len(self.calls), 1, "evaluation must not repeat GraphRAG retrieval")
        self.assertEqual(self.service.load("project")["run"]["status"], "completed")

    def test_lenient_evaluation_uses_lenient_judge_prompt_and_persists_mode(self):
        judge = FakeJudge(lenient_result=True)
        self.service.judging = judge
        question_set = self.question_sets.create_with_questions(
            "project", "lenient", [
                BatchQuestion("Q1", "問題", "先確認設備已關閉，\n準備指定工具")
            ],
        )
        group = ExperimentGroup("G01", "Basic", "gpt-4o-mini", "basic")
        self.service.save_configuration(
            "project", [group], question_set.question_set_id, 1, "gpt-4o-mini"
        )
        self.service.generate_answers(
            "project", question_set.question_set_id, [group], max_concurrency=1
        )

        run = self.service.evaluate_answers(
            "project", question_set.question_set_id, judge_model="gpt-4o-mini",
            max_concurrency=1, evaluation_mode="lenient",
        )

        self.assertEqual(run.results[0].evaluation_result, 2)
        self.assertIn("寬鬆評分", run.results[0].evaluation_reason)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(judge.calls[0][-1], "lenient")
        self.assertEqual(self.service.load("project")["evaluation_mode"], "lenient")
        self.service.set_evaluation_mode("project", "strict")
        summary_path, details_path = self.service.export("project")
        self.assertEqual(json.loads(summary_path.read_text())["evaluation"]["mode"], "lenient")
        self.assertEqual(json.loads(details_path.read_text())["evaluation_mode"], "lenient")

    def test_evaluation_mode_defaults_to_strict_and_rejects_unknown_values(self):
        self.assertEqual(self.service.load("project")["evaluation_mode"], "strict")
        with self.assertRaisesRegex(ProjectError, "不支援的評分方式"):
            self.service.set_evaluation_mode("project", "guess")

    def test_evaluation_is_rejected_before_answers_exist(self):
        question_set = self.service.import_question_set("project", self.question_file)
        self.service.add_group("project")

        with self.assertRaisesRegex(ProjectError, "執行「檢索並生成答案」"):
            self.service.evaluate_answers("project", question_set.question_set_id)

    def test_evaluation_requires_answers_to_match_current_group_settings(self):
        question_set = self.question_sets.create_with_questions(
            "project", "changed config", [BatchQuestion("Q1", "問題", "標準答案")]
        )
        group = ExperimentGroup("G01", "Local", "gpt-4o-mini", "local")
        self.service.save_configuration(
            "project", [group], question_set.question_set_id, 1, "gpt-4o-mini"
        )
        self.service.generate_answers(
            "project", question_set.question_set_id, [group], max_concurrency=1
        )
        self.service.set_group_method("project", group.group_id, "basic")

        with self.assertRaisesRegex(ProjectError, "檢索策略或回答格式已變更"):
            self.service.evaluate_answers("project", question_set.question_set_id)

    def test_stop_keeps_completed_results_and_saves_partial_summary(self):
        question_set = self.question_sets.create_with_questions(
            "project", "many questions", [
                BatchQuestion(f"Q{i}", f"問題{i}", reference_answer="標準答案") for i in range(1, 5)
            ],
        )
        group = ExperimentGroup("G01", "Local", "gpt-4o-mini", "local")

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

        def run_worker(command, **kwargs):
            payload = json.loads(kwargs["input"])
            root = Path(payload["root_dir"])
            settings = yaml.safe_load((root / "settings.yaml").read_text(encoding="utf-8"))
            calls[payload["method"]] = {
                **payload,
                "output_is_symlink": (root / "output").is_symlink(),
                "completion_model": settings["completion_models"]["default"]["model"],
                "cwd": str(kwargs["cwd"]),
            }
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"answer": "answer", "context": {}}),
                "",
            )

        with patch(
            "automotive_graphrag.retrieval_experiments.subprocess.run",
            side_effect=run_worker,
        ) as worker:
            for method in ("local", "global", "drift", "basic"):
                answer, _ = self.service._graphrag_query(
                    "project", "查詢", method, "gpt-4.1-mini", "List of 3-7 Points"
                )
                self.assertEqual(answer, "answer")

        self.assertEqual(worker.call_count, 4)
        self.assertEqual(set(calls), {"local", "global", "drift", "basic"})
        self.assertTrue(all(call["output_is_symlink"] for call in calls.values()))
        self.assertTrue(all(call["cwd"] == str(Path(__file__).parents[1]) for call in calls.values()))
        self.assertTrue(all(call["completion_model"] == "gpt-4.1-mini" for call in calls.values()))
        self.assertTrue(all(call["response_type"] == "List of 3-7 Points" for call in calls.values()))

    def test_strategy_parameters_are_saved_per_group_and_applied_to_settings(self):
        self.service.add_group("project")
        self.service.add_group("project")
        first, second = self.service.load("project")["groups"]
        self.service.set_group_strategy_parameter("project", first["group_id"], "basic", "k", 17)
        self.service.set_group_strategy_parameter("project", first["group_id"], "local", "top_k_entities", 22)
        saved = self.service.load("project")["groups"]
        first = next(item for item in saved if item["group_id"] == first["group_id"])
        second = next(item for item in saved if item["group_id"] == second["group_id"])
        self.assertEqual(first["strategy_parameters"]["basic"]["k"], 17)
        self.assertEqual(first["strategy_parameters"]["local"]["top_k_entities"], 22)
        self.assertEqual(second["strategy_parameters"]["basic"]["k"], 10)
        with self.assertRaisesRegex(ProjectError, "沒有該設定參數"):
            self.service.set_group_strategy_parameter("project", first["group_id"], "basic", "n_depth", 3)

        graph_root = self.projects.path_for("project") / "graphrag"
        (graph_root / "output").mkdir(parents=True)
        (graph_root / "settings.yaml").write_text("completion_models: {default: {model: old}}\n", encoding="utf-8")
        captured = {}

        def run_worker(command, **kwargs):
            payload = json.loads(kwargs["input"])
            root = Path(payload["root_dir"])
            captured["settings"] = yaml.safe_load((root / "settings.yaml").read_text(encoding="utf-8"))
            return subprocess.CompletedProcess(command, 0, json.dumps({"answer": "ok", "context": {}}), "")

        with patch("automotive_graphrag.retrieval_experiments.subprocess.run", side_effect=run_worker):
            self.service._graphrag_query(
                "project", "q", "basic", "gpt-4.1-mini",
                strategy_parameters={"k": 17, "max_context_tokens": 8000},
            )
        self.assertEqual(captured["settings"]["basic_search"]["k"], 17)
        self.assertEqual(captured["settings"]["basic_search"]["max_context_tokens"], 8000)

    def test_luna_drift_uses_sampling_parameter_compatibility_adapter(self):
        graph_root = self.projects.path_for("project") / "graphrag"
        (graph_root / "output").mkdir(parents=True)
        (graph_root / "settings.yaml").write_text(
            "completion_models:\n  default:\n    model: old-model\n    call_args: {temperature: 0, top_p: 1}\n"
            "embedding_models:\n  default:\n    model: old-embedding\n",
            encoding="utf-8",
        )
        captured = {}

        def run_worker(command, **kwargs):
            payload = json.loads(kwargs["input"])
            root = Path(payload["root_dir"])
            settings = yaml.safe_load((root / "settings.yaml").read_text(encoding="utf-8"))
            captured["model"] = settings["completion_models"]["default"]["model"]
            captured["args"] = settings["completion_models"]["default"]["call_args"]
            captured["output_is_symlink"] = (root / "output").is_symlink()
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"answer": "Luna DRIFT answer", "context": {"sources": []}}),
                "",
            )

        with patch(
            "automotive_graphrag.retrieval_experiments.subprocess.run",
            side_effect=run_worker,
        ) as adapter:
            answer, _context = self.service._graphrag_query(
                "project", "測試", "drift", "gpt-6-luna", reasoning_effort="high"
            )

        self.assertEqual(answer, "Luna DRIFT answer")
        self.assertEqual(captured["model"], "gpt-6-luna")
        self.assertNotIn("temperature", captured["args"])
        self.assertNotIn("top_p", captured["args"])
        self.assertEqual(captured["args"]["reasoning_effort"], "high")
        self.assertTrue(captured["output_is_symlink"])
        adapter.assert_called_once()

    def test_resolved_sources_reads_document_id_from_source_metadata(self):
        source = SourceMetadata(
            text_unit_id="unit-1", chunk_id="chunk-1", project_id="project",
            section_id="S1", section_name="Section", document_id="manual.pdf",
            page=12, block_id="b1", text="來源文字",
        )
        evidence = SimpleNamespace(text_unit_id="unit-1", document_id="graph-document-id")
        with patch("automotive_graphrag.retrieval_experiments.EvidenceService") as evidence_service:
            evidence_service.return_value.from_context.return_value = [evidence]
            evidence_service.return_value.metadata.load.return_value = [source]

            resolved = self.service._resolved_sources("project", {"sources": ["unit-1"]})

        self.assertEqual(resolved, ("manual.pdf",))

    def test_app_exposes_retrieval_experiment_page_and_controls(self):
        from gradio.context import LocalContext

        ui_root = self.root / "ui-projects"
        ui_projects = ProjectStore(ui_root)
        ui_projects.create(
            project_id="ui-project", display_name="UI 測試", vehicle_name="測試", manual_version="v1"
        )
        ui_sets = QuestionSetService(ui_projects)
        ui_connections = ConnectionSettings(ui_root)
        ui_service = RetrievalExperimentService(
            ui_projects, ui_sets, ui_connections,
            AutomaticEvaluationService(ui_projects, ui_sets, ui_connections),
        )
        ui_service.add_group("ui-project")
        ui_question_set = ui_service.import_question_set("ui-project", self.question_file)
        ui_group_id = ui_service.load("ui-project")["groups"][0]["group_id"]
        ui_service.set_group_method("ui-project", ui_group_id, "drift")
        app = create_app(ui_root)
        generate_answers_handler = next(
            fn.fn for fn in app.fns.values()
            if getattr(fn.fn, "__name__", "") == "generate_retrieval_experiment_answers"
        )
        resolve_question_set_id = next(
            cell.cell_contents for cell in generate_answers_handler.__closure__ or ()
            if getattr(cell.cell_contents, "__name__", "") == "resolve_experiment_question_set_id"
        )
        self.assertEqual(resolve_question_set_id("ui-project", None), [ui_question_set.question_set_id])
        self.assertEqual(resolve_question_set_id("ui-project", ["stale-browser-value"]), [ui_question_set.question_set_id])
        labels = {
            component.get_config().get("label")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        }

        self.assertIn("回答模型", labels)
        self.assertIn("答案評分方式", labels)
        self.assertIn("啟用 LLM Judge", labels)
        self.assertIn("啟用程式化答案指標（EM／F1／詞項）", labels)
        self.assertEqual(len(app.renderables), 1)
        self.assertIn("測試最大並行請求數", labels)
        self.assertIn("回答生成進度", labels)
        self.assertIn("實驗組摘要", labels)
        self.assertIn("實驗組 × 題目集摘要", labels)
        self.assertIn(
            "逐題實驗結果（不執行模型評測也可手動編輯等級與品質欄位；拒答欄填正確／錯誤／不適用）",
            labels,
        )
        button_values = [
            component.get_config().get("value")
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        ]
        self.assertIn("檢索並生成答案", button_values)
        self.assertIn("評測答案", button_values)
        progress_bar = next(
            component for component in app.blocks.values()
            if hasattr(component, "get_config")
            and component.get_config().get("label") == "回答生成進度"
        )
        generation_event = next(
            fn for fn in app.fns.values()
            if getattr(fn.fn, "__name__", "") == "generate_retrieval_experiment_answers"
        )
        self.assertIn(progress_bar, generation_event.outputs)
        component_configs = [
            component.get_config()
            for component in app.blocks.values()
            if hasattr(component, "get_config")
        ]
        question_preview = next(
            config for config in component_configs if config.get("label") == "目前匯入資料集與題目"
        )
        self.assertEqual(question_preview["headers"][0], "資料集")
        question_uploader = next(
            config for config in component_configs
            if config.get("label") == "選擇要取代目前資料集的 JSON 題目集"
        )
        self.assertEqual(question_uploader["file_count"], "multiple")
        experiment_results = next(
            config for config in component_configs if str(config.get("label", "")).startswith("逐題實驗結果")
        )
        self.assertEqual(
            experiment_results["headers"],
            ["實驗組名稱", "資料集", "題號", "來源文件", "題目", "系統回答", "正確答案", "評測等級（0/1/2）", "評判理由", "實際檢索策略", "Agentic 步驟與停止原因", "答案要點覆蓋率 %", "無根據主張數", "無答案題拒答判定", "Answer EM", "Answer F1", "Answer Term Hit", "Answer Term Coverage"],
        )
        self.assertEqual(experiment_results["datatype"][7], "str")
        self.assertTrue(experiment_results["interactive"])
        self.assertEqual(
            experiment_results["column_widths"],
            ["6%", "8%", "4%", "8%", "9%", "13%", "12%", "5%", "7%", "7%", "9%", "5%", "4%", "5%", "4%", "4%", "5%", "6%"],
        )
        block_values = list(app.blocks.values())
        result_table_position = next(
            index for index, component in enumerate(block_values)
            if hasattr(component, "get_config")
            and str(component.get_config().get("label", "")).startswith("逐題實驗結果")
        )
        export_button_position = next(
            index for index, component in enumerate(block_values)
            if hasattr(component, "get_config")
            and component.get_config().get("value") == "匯出兩種實驗結果 JSON"
        )
        self.assertGreater(export_button_position, result_table_position)
        self.assertIn("精簡摘要 JSON", labels)
        self.assertIn("逐題結果 JSON", labels)
        evaluation_mode_dropdown = next(
            component for component in app.blocks.values()
            if isinstance(component, gr.Dropdown) and component.label == "答案評分方式"
        )
        self.assertEqual(evaluation_mode_dropdown.value, "strict")
        self.assertEqual(
            [value for _label, value in evaluation_mode_dropdown.choices],
            ["strict", "lenient"],
        )

        renderer = app.renderables[0]
        LocalContext.blocks_config.set(app.default_config)
        LocalContext.blocks.set(app)
        try:
            renderer.apply("ui-project", 0)
        finally:
            LocalContext.blocks_config.set(None)
            LocalContext.blocks.set(None)
        dynamic_handlers = [
            fn for fn in app.default_config.fns.values() if fn.rendered_in is renderer
        ]
        dynamic_handler_ids = {fn.key: fn._id for fn in dynamic_handlers}
        dynamic_handler_inputs = {fn.key: list(fn.inputs) for fn in dynamic_handlers}
        dynamic_dropdowns = [
            component for component in app.default_config.blocks.values()
            if isinstance(component, gr.Dropdown) and component.rendered_in is renderer
        ]
        self.assertTrue(any(
            component.get_config().get("value") == "移除此組"
            for component in app.default_config.blocks.values()
            if getattr(component, "rendered_in", None) is renderer
            and hasattr(component, "get_config")
        ))
        dynamic_handlers_after_rerender = [
            fn for fn in app.default_config.fns.values() if fn.rendered_in is renderer
        ]
        self.assertEqual(
            {fn.key: fn._id for fn in dynamic_handlers_after_rerender},
            dynamic_handler_ids,
        )
        self.assertEqual(
            {fn.key: list(fn.inputs) for fn in dynamic_handlers_after_rerender},
            dynamic_handler_inputs,
        )
        self.assertEqual(
            {component.label for component in dynamic_dropdowns},
            {"回答模型", "GraphRAG 檢索策略", "回答格式", "推理深度"},
        )
        self.assertTrue(all(component.interactive is True for component in dynamic_dropdowns))
        answer_dropdown = next(component for component in dynamic_dropdowns if component.label == "回答模型")
        self.assertIn("gpt-6-luna", [value for _label, value in answer_dropdown.choices])
        # Model and strategy choices remain independent; Luna + DRIFT is supported.
        ui_service.set_group_method("ui-project", ui_group_id, "local")
        ui_service.set_group_answer_model("ui-project", ui_group_id, "gpt-6-luna")
        LocalContext.blocks_config.set(app.default_config)
        LocalContext.blocks.set(app)
        try:
            renderer.apply("ui-project", 1)
        finally:
            LocalContext.blocks_config.set(None)
            LocalContext.blocks.set(None)
        dynamic_dropdowns = [
            component for component in app.default_config.blocks.values()
            if isinstance(component, gr.Dropdown) and component.rendered_in is renderer
        ]
        strategy_dropdown = next(component for component in dynamic_dropdowns if component.label == "GraphRAG 檢索策略")
        answer_dropdown = next(component for component in dynamic_dropdowns if component.label == "回答模型")
        response_type_dropdown = next(component for component in dynamic_dropdowns if component.label == "回答格式")
        reasoning_effort_dropdown = next(component for component in dynamic_dropdowns if component.label == "推理深度")
        self.assertIn("gpt-6-luna", [value for _label, value in answer_dropdown.choices])
        self.assertEqual(response_type_dropdown.value, DEFAULT_EXPERIMENT_RESPONSE_TYPE)
        self.assertEqual(response_type_dropdown.choices, list(EXPERIMENT_RESPONSE_TYPE_OPTIONS))
        self.assertEqual(reasoning_effort_dropdown.choices, list(EXPERIMENT_REASONING_EFFORT_OPTIONS))
        self.assertEqual(reasoning_effort_dropdown.value, DEFAULT_REASONING_EFFORT)
        self.assertIn("drift", [value for _label, value in strategy_dropdown.choices])
        self.assertEqual(response_type_dropdown.value, DEFAULT_EXPERIMENT_RESPONSE_TYPE)
        self.assertEqual(
            {fn.fn.__name__ for fn in dynamic_handlers},
            {
                "save_experiment_answer_model", "save_experiment_method",
                "save_experiment_response_type", "save_experiment_reasoning_effort",
                "save_experiment_strategy_parameter", "remove_experiment_group",
            },
        )
        self.assertTrue(all(len(fn.inputs) == 3 for fn in dynamic_handlers if fn.fn.__name__.startswith("save_experiment_")))
        drift_parameter_controls = [
            component for component in app.default_config.blocks.values()
            if getattr(component, "rendered_in", None) is renderer
            and hasattr(component, "get_config")
            and str(component.get_config().get("key", "")).startswith(f"experiment-param-{ui_group_id}-drift-")
        ]
        self.assertTrue(drift_parameter_controls)
        self.assertTrue(all(component.get_config().get("interactive") for component in drift_parameter_controls))
        remove_handler = next(fn for fn in dynamic_handlers if fn.fn.__name__ == "remove_experiment_group")
        self.assertEqual(len(remove_handler.inputs), 2)
        effort_handler = next(
            fn for fn in dynamic_handlers
            if fn.fn.__name__ == "save_experiment_reasoning_effort"
        )
        effort_status, _revision = effort_handler.fn("ui-project", "high", 0)
        self.assertIn("已儲存", effort_status)
        self.assertEqual(
            ui_service.load("ui-project")["groups"][0]["answer_reasoning_effort"], "high"
        )
        removal_status, _revision = remove_handler.fn("ui-project", 0)
        self.assertIn(ui_group_id, removal_status)
        self.assertEqual(ui_service.load("ui-project")["groups"], [])


if __name__ == "__main__":
    unittest.main()
