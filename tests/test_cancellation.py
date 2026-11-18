"""F-45: long-running jobs can be stopped."""

import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pymupdf

from automotive_graphrag.app import JOB_BUTTON_LABELS, JOB_STOPPING_LABEL, create_app
from automotive_graphrag.automatic_evaluation import AutomaticEvaluationItem, AutomaticEvaluationService
from automotive_graphrag.automatic_qa import AutomaticQATestService
from automotive_graphrag.cancellation import JobAlreadyRunning, JobRegistry, JobStopped
from automotive_graphrag.connections import ConnectionSettings, ConnectionTestResult
from automotive_graphrag.documents import DocumentService, SECTIONING_FORMAT_VERSION
from automotive_graphrag.indexing import IndexingService
from automotive_graphrag.projects import ProjectError, ProjectStore, indexing_lock_held, indexing_lock_path
from automotive_graphrag.querying import QueryResult, QueryService
from automotive_graphrag.question_generation import GeneratedQuestion
from automotive_graphrag.question_sets import BatchQuestion, GoldEvidence, QuestionSetService
from automotive_graphrag.source_sampling import SourceSample
from automotive_graphrag.vision import VisionSettings


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed child that has not been reaped yet is a zombie, which no longer runs.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return False
    return state != "Z"


class JobRegistryTests(unittest.TestCase):
    def test_same_kind_runs_once_per_project(self) -> None:
        jobs = JobRegistry()
        job = jobs.begin("P", "index")
        with self.assertRaises(JobAlreadyRunning):
            jobs.begin("P", "index")
        # Other kinds and other projects are independent.
        jobs.begin("P", "preprocess").release()
        jobs.begin("Q", "index").release()
        job.release()
        jobs.begin("P", "index").release()

    def test_exception_releases_registration(self) -> None:
        jobs = JobRegistry()
        with self.assertRaises(RuntimeError), jobs.running("P", "index"):
            raise RuntimeError("boom")
        self.assertFalse(jobs.is_running("P", "index"))

        job = jobs.begin("P", "qa_answer")

        def fail() -> None:
            raise RuntimeError("worker failed")

        with mock.patch("threading.excepthook"):
            job.run_in_thread(fail).join()
        self.assertFalse(jobs.is_running("P", "qa_answer"))

    def test_handed_off_job_stays_registered_until_worker_ends(self) -> None:
        jobs = JobRegistry()
        job = jobs.begin("P", "index")
        release = threading.Event()
        thread = job.run_in_thread(release.wait)
        # The page that started the job goes away; the worker keeps the registration.
        job.release_unless_handed_off()
        self.assertTrue(jobs.is_running("P", "index"))
        self.assertTrue(jobs.stop("P", "index"))
        self.assertTrue(job.stop_requested)
        release.set()
        thread.join()
        self.assertFalse(jobs.is_running("P", "index"))
        self.assertFalse(jobs.stop("P", "index"))


class PreprocessStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(project_id="P", display_name="P", vehicle_name="V", manual_version="1")
        source = self.store.path_for("P") / "source"
        for name in ("A.pdf", "B.pdf"):
            document = pymupdf.open()
            for page_number in range(3):
                document.new_page().insert_text((72, 72), f"{name} page {page_number}")
            document.save(source / name)
            document.close()
        self.documents = DocumentService(self.store)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def processed_records(self) -> list[dict]:
        path = self.store.path_for("P") / "processed" / "input.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_stop_saves_finished_pdfs_and_keeps_previous_output_of_the_rest(self) -> None:
        self.documents.preprocess("P")
        previous_b = [item for item in self.processed_records() if item["document_id"] == "B.pdf"]
        previous_report = json.loads((self.store.path_for("P") / "processed" / "report.json").read_text())
        stop = threading.Event()

        def stop_inside_b(done: int, total: int, name: str) -> None:
            if name == "B.pdf" and done > 3:
                stop.set()

        report = self.documents.preprocess(
            "P", options_by_filename={"A.pdf": {"start_page": 1}}, progress_callback=stop_inside_b, stop_event=stop,
        )

        self.assertTrue(report.stopped)
        self.assertEqual(report.completed_documents, ("A.pdf",))
        self.assertEqual(report.unprocessed_documents, ("B.pdf",))
        records = self.processed_records()
        self.assertEqual(sorted({item["page"] for item in records if item["document_id"] == "A.pdf"}), [2, 3])
        self.assertEqual([item for item in records if item["document_id"] == "B.pdf"], previous_b)
        saved = {item["filename"]: item for item in json.loads(
            (self.store.path_for("P") / "processed" / "report.json").read_text()
        )["documents"]}
        previous_documents = {item["filename"]: item for item in previous_report["documents"]}
        self.assertEqual(saved["B.pdf"], previous_documents["B.pdf"])
        self.assertEqual(self.store.get("P").status, "READY")

    def test_stop_before_any_pdf_finishes_changes_nothing(self) -> None:
        stop = threading.Event()
        stop.set()
        status = self.store.get("P").status

        report = self.documents.preprocess("P", stop_event=stop)

        self.assertTrue(report.stopped)
        self.assertEqual(report.completed_documents, ())
        self.assertEqual(report.unprocessed_documents, ("A.pdf", "B.pdf"))
        self.assertFalse((self.store.path_for("P") / "processed" / "input.jsonl").exists())
        self.assertEqual(self.store.get("P").status, status)

    def test_stop_sends_no_new_vision_requests(self) -> None:
        stop = threading.Event()
        calls: list[int] = []

        def analyzer(image, page_number, language, vision) -> str:
            calls.append(page_number)
            stop.set()
            return "vision text"

        documents = DocumentService(self.store, page_analyzer=analyzer)
        vision = VisionSettings(model="m", scope="all", api_base_url="http://x", api_key="k", max_workers=1)

        report = documents.preprocess("P", filenames=["A.pdf"], vision=vision, stop_event=stop)

        self.assertTrue(report.stopped)
        self.assertEqual(calls, [1])
        self.assertEqual(report.unprocessed_documents, ("A.pdf",))


class IndexingStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(project_id="P", display_name="P", vehicle_name="V", manual_version="1")
        self.project_path = self.store.path_for("P")
        (self.project_path / "processed" / "input.jsonl").write_text(json.dumps({
            "id": "p1", "title": "Page", "text": "content",
            "sectioning_version": SECTIONING_FORMAT_VERSION,
        }) + "\n")
        self.connections = ConnectionSettings(self.store.root, lambda base_url, key: ConnectionTestResult(True, "ok"))
        self.connections.save("https://api.openai.com/v1", "test-key")
        self.pid_file = Path(self.temporary_directory.name) / "index.pid"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def service(self) -> IndexingService:
        service = IndexingService(
            self.store, connection_settings=self.connections, metadata_builder=lambda project_id: None,
        )
        script = (
            "import os, sys, time\n"
            f"open({str(self.pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "print('Starting workflow: load_input_documents', flush=True)\n"
            "time.sleep(60)\n"
        )

        def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
            root = Path(command[command.index("--root") + 1])
            if "init" in command:
                (root / "settings.yaml").write_text(
                    "input: {}\ncompletion_models:\n  default_completion_model: {}\n"
                    "embedding_models:\n  default_embedding_model: {}\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")
            # A fake long-running `graphrag index` run through the real process handling.
            (root / "output").mkdir(exist_ok=True)
            (root / "output" / "partial.parquet").write_bytes(b"partial")
            return service._run([sys.executable, "-c", script])

        service.runner = runner
        return service

    def build_and_stop(self, service: IndexingService):
        stop = threading.Event()

        def on_log(text: str) -> None:
            if "Starting workflow" in text:
                stop.set()

        started = time.monotonic()
        result = service.build("P", log_callback=on_log, stop_event=stop)
        return result, time.monotonic() - started

    def test_stop_terminates_index_process_and_restores_previous_graph(self) -> None:
        output = self.project_path / "graphrag" / "output"
        output.mkdir(parents=True)
        (output / "entities.parquet").write_bytes(b"previous")
        self.store.update_status("P", "STALE")

        result, elapsed = self.build_and_stop(self.service())

        self.assertEqual(result.status, "STOPPED")
        self.assertLess(elapsed, 10)
        self.assertFalse(process_alive(int(self.pid_file.read_text())))
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["entities.parquet"])
        self.assertEqual(self.store.get("P").status, "STALE")
        saved = json.loads((self.project_path / "graphrag" / "last_index_run.json").read_text())
        self.assertEqual(saved["status"], "STOPPED")
        self.assertIn("使用者停止", (self.project_path / "graphrag" / "indexing.log").read_text())
        self.assertFalse(indexing_lock_held(indexing_lock_path(self.project_path)))

    def test_stop_without_previous_graph_marks_failed_and_allows_rebuild(self) -> None:
        self.store.update_status("P", "READY")

        result, _ = self.build_and_stop(self.service())

        self.assertEqual(result.status, "STOPPED")
        self.assertEqual(self.store.get("P").status, "FAILED")
        self.assertFalse((self.project_path / "graphrag" / "output").exists())
        # The lock is released, so the same job can start again right away.
        result, _ = self.build_and_stop(self.service())
        self.assertEqual(result.status, "STOPPED")


class QueryStopTests(unittest.TestCase):
    def test_stop_terminates_running_query_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "worker.pid"
            script = (
                "import os, sys, time\n"
                f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
                "sys.stdin.read()\ntime.sleep(60)\n"
            )
            stop = threading.Event()
            threading.Timer(0.5, stop.set).start()
            started = time.monotonic()
            with mock.patch(
                "automotive_graphrag.querying._query_worker_command",
                return_value=[sys.executable, "-c", script],
            ), self.assertRaises(JobStopped):
                QueryService._run_worker(dict(os.environ), Path(temporary), "{}", None, stop)
            self.assertLess(time.monotonic() - started, 10)
            self.assertFalse(process_alive(int(pid_file.read_text())))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StoppingQuery:
    """Answers the first question, then stops the job; later questions see the stop."""

    def __init__(self, stop: threading.Event) -> None:
        self.stop = stop
        self.asked: list[str] = []

    def ask(self, project_id, question, method="local", chat_model=None, response_type="Single Paragraph",
            source=None, reasoning_effort="medium", stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise JobStopped("stopped")
        self.asked.append(question)
        self.stop.set()
        return QueryResult(
            query_id=question, project_id=project_id, question=question, method=method,
            status="COMPLETED", answer="新回答", error=None, started_at=now(), completed_at=now(),
            duration_seconds=0.01,
        )


class AutomaticQAStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self.temporary_directory.name) / "projects")
        self.store.create(project_id="P", display_name="P", vehicle_name="V", manual_version="1")
        self.store.update_status("P", "INDEXED")
        self.sets = QuestionSetService(self.store)
        self.connections = ConnectionSettings(self.store.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_stopped_answers_keep_finished_and_reset_the_rest(self) -> None:
        question_set = self.sets.create_with_questions("P", "set", [
            BatchQuestion(f"Q{index}", f"問題{index}", "答案", status="COMPLETED", answer="舊回答")
            for index in range(1, 5)
        ])
        stop = threading.Event()
        query = StoppingQuery(stop)
        judging = AutomaticEvaluationService(self.store, self.sets, self.connections)
        service = AutomaticQATestService(self.store, object(), object(), query, self.sets, judging, object())

        report = service.answer_existing(
            "P", question_set.question_set_id, "gpt-4o-mini", "local", 1, stop_event=stop,
        )

        self.assertTrue(report.stopped)
        self.assertEqual(len(query.asked), 1)
        saved = self.sets.get("P", question_set.question_set_id).questions
        self.assertEqual([item.status for item in saved], ["COMPLETED", "PENDING", "PENDING", "PENDING"])
        self.assertEqual(saved[0].answer, "新回答")
        self.assertEqual({item.answer for item in saved[1:]}, {""})
        # Judging still requires every answer.
        with self.assertRaisesRegex(ProjectError, "完成所有題目"):
            service.evaluate_existing("P", question_set.question_set_id, "m", "gpt-4o-mini", "local")

    def test_stopped_generation_saves_partial_question_set(self) -> None:
        source = self.store.path_for("P") / "source"
        names = ("one.pdf", "two.pdf", "three.pdf")
        for name in names:
            (source / name).write_bytes(b"")
        stop = threading.Event()

        class Generation:
            calls = 0

            def generate_for_document(self, project_id, document_id, samples, count, model, **kwargs):
                Generation.calls += 1
                stop.set()
                return (GeneratedQuestion(
                    question_id=document_id, question=f"{document_id} 的問題？", reference_answer="答案",
                    gold_evidence=(GoldEvidence(document_id, (1,), ("c",)),), source_sample_ids=("s",),
                    difficulty="simple", generation_status="pending_review",
                ),)

        class Sampling:
            def scan(self, project_id, minimum_characters=1):
                return [SourceSample(
                    sample_id=name, chunk_id=name, project_id="P", section_id=name, section_name=name,
                    document_id=name, page=1, block_id="b1", content_type="general", character_count=10, text="x",
                ) for name in names]

        service = AutomaticQATestService(self.store, Sampling(), Generation(), object(), self.sets, object(), object())

        question_set = service.generate_question_set("P", 1, False, "gpt-4o-mini", stop_event=stop)

        self.assertEqual(Generation.calls, 1)
        self.assertEqual(len(question_set.questions), 1)
        self.assertIn("已停止，部分題目", question_set.description)

        stop.set()
        before = len(self.sets.list("P"))
        with self.assertRaises(JobStopped):
            service.generate_question_set("P", 1, False, "gpt-4o-mini", stop_event=stop)
        self.assertEqual(len(self.sets.list("P")), before)

    def test_stopped_judging_saves_finished_scores_and_skips_retrieval(self) -> None:
        question_set = self.sets.create_with_questions("P", "set", [
            BatchQuestion(f"Q{index}", f"問題{index}", "答案", status="COMPLETED", answer="回答")
            for index in range(1, 4)
        ])
        stop = threading.Event()
        judged: list[str] = []

        class Judging(AutomaticEvaluationService):
            def evaluate_single_answer(self, project_id, question_id, *args, **kwargs):
                judged.append(question_id)
                stop.set()
                return AutomaticEvaluationItem(question_id, True, "ok", average_score=1.0)

        class Retrieval:
            def evaluate(self, *args, **kwargs):
                raise AssertionError("retrieval metrics must not run after a stop")

        judging = Judging(self.store, self.sets, self.connections)
        service = AutomaticQATestService(self.store, object(), object(), object(), self.sets, judging, Retrieval())

        report = service.evaluate_existing(
            "P", question_set.question_set_id, "m", "gpt-4o-mini", "local", concurrency=1, stop_event=stop,
        )

        self.assertTrue(report.stopped)
        self.assertEqual(judged, ["Q1"])
        self.assertEqual([item.question_id for item in report.judge.items], ["Q1"])
        saved = judging.last_result("P", question_set.question_set_id)
        self.assertTrue(saved.stopped)
        self.assertEqual(len(saved.items), 1)


class JobButtonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name) / "projects"
        ProjectStore(self.root).create(project_id="P", display_name="P", vehicle_name="V", manual_version="1")
        self.app = create_app(self.root)
        functions = {getattr(fn.fn, "__name__", ""): fn for fn in self.app.fns.values()}
        self.events = functions
        self.running_job_views = functions["running_job_views"].fn
        self.jobs = dict(zip(
            self.running_job_views.__code__.co_freevars,
            (cell.cell_contents for cell in self.running_job_views.__closure__),
        ))["jobs"]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def button_values(self, label: str) -> list:
        return [
            component for component in self.app.blocks.values()
            if hasattr(component, "get_config") and component.get_config().get("value") == label
        ]

    def test_each_job_has_hidden_stop_button_and_old_stop_button_is_gone(self) -> None:
        for start_label, stop_label in JOB_BUTTON_LABELS.values():
            (stop_button,) = self.button_values(stop_label)
            self.assertFalse(stop_button.get_config()["visible"])
            self.assertEqual(stop_button.get_config()["variant"], "stop")
        self.assertEqual(self.button_values("停止實驗"), [])
        stop_events = [
            event for event in self.app.fns.values()
            if any(output.get_config().get("value") == JOB_BUTTON_LABELS["index"][1] for output in event.outputs)
            and len(event.outputs) == 2 and event.concurrency_limit is None
        ]
        self.assertTrue(stop_events)

    def test_running_job_views_follow_registry(self) -> None:
        *buttons, notice, timer = self.running_job_views("P")
        self.assertTrue(all(update["visible"] for update in buttons[0::2]))
        self.assertFalse(any(update["visible"] for update in buttons[1::2]))
        self.assertFalse(notice["visible"])

        job = self.jobs.begin("P", "index")
        *buttons, notice, timer = self.running_job_views("P")
        index_start, index_stop = buttons[2:4]
        self.assertFalse(index_start["visible"])
        self.assertTrue(index_stop["visible"])
        self.assertTrue(index_stop["interactive"])
        self.assertIn("執行中（重新整理前啟動）", notice["value"])
        self.assertTrue(timer.active)
        # Other buttons keep their state.
        self.assertTrue(buttons[0]["visible"])

        job.stop()
        *buttons, _notice, _timer = self.running_job_views("P")
        self.assertEqual(buttons[3]["value"], JOB_STOPPING_LABEL)
        self.assertFalse(buttons[3]["interactive"])

        job.release()
        *buttons, notice, timer = self.running_job_views("P")
        self.assertTrue(buttons[2]["visible"])
        self.assertFalse(buttons[3]["visible"])
        self.assertFalse(timer.active)

    def test_wrapped_handler_swaps_buttons_and_releases_job(self) -> None:
        build = self.events["build_index"].fn
        self.assertIs(inspect.unwrap(build).__name__, "build_index")

        outputs = list(build("P", None, [], None))
        self.assertIn("沒有已前處理的 PDF", outputs[0][0])
        start, stop = outputs[0][-2:]
        self.assertFalse(start["visible"])
        self.assertTrue(stop["visible"])
        # A failed start releases the job so the restore step shows the start button again.
        self.assertFalse(self.jobs.is_running("P", "index"))

    def test_gradio_passes_only_event_inputs_to_job_handlers(self) -> None:
        from gradio.helpers import special_args

        wrapped = [
            event for event in self.app.fns.values()
            if getattr(event.fn, "__wrapped__", None) is not None
            and any(output.get_config().get("value") in [labels[1] for labels in JOB_BUTTON_LABELS.values()]
                    for output in event.outputs)
            and len(event.outputs) > 2
        ]
        self.assertEqual(len(wrapped), len(JOB_BUTTON_LABELS))
        for event in wrapped:
            with self.subTest(handler=event.fn.__name__):
                # Gradio fills defaulted parameters it sees; `job` must not be one of them.
                args, *_ = special_args(event.fn, [None] * len(event.inputs))
                self.assertEqual(len(args), len(event.inputs))
                first = next(iter(event.fn(*args)))
                self.assertEqual(len(first), len(event.outputs))

    def test_second_start_while_running_is_rejected(self) -> None:
        job = self.jobs.begin("P", "index")
        build = self.events["build_index"].fn

        (output,) = list(build("P", None, [], None))

        self.assertIn("相同的工作正在執行", output[0])
        self.assertTrue(output[-1]["visible"])
        self.assertIs(self.jobs.get("P", "index"), job)
        job.release()


if __name__ == "__main__":
    unittest.main()
