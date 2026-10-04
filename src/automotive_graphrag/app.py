"""Gradio entry point for the v1.0 project management interface."""

from __future__ import annotations

import json
import os
import queue
import threading
from dataclasses import asdict
from pathlib import Path

import gradio as gr

from .automatic_evaluation import AutomaticEvaluationService
from .automatic_qa import AutomaticQATestService
from .connections import ALLOWED_CHAT_MODELS, ALLOWED_EMBEDDING_MODELS, ConnectionSettings
from .documents import DocumentInfo, DocumentService
from .downloads import stage_downloads
from .ground_truth import GroundTruthService
from .indexing import IndexingService
from .projects import ProjectError, ProjectStore
from .querying import QueryService
from .question_generation import GeneratedQuestion, QuestionGenerationService
from .question_sets import BatchQuestion, GoldEvidence, QuestionSet, QuestionSetService
from .reviews import ReviewService
from .retrieval_evaluation import RetrievalEvaluationService
from .source_sampling import SourceSample, SourceSamplingService


PROJECT_COLUMNS = ["Project ID", "顯示名稱", "文件數", "索引狀態", "啟用", "更新時間"]
DOCUMENT_COLUMNS = ["檔名", "頁數", "大小 (bytes)", "前處理狀態", "空白頁", "錯誤頁", "錯誤"]
PROCESSING_COLUMNS = ["PDF", "總頁數", "忽略 Header (%)", "忽略 Footer (%)", "起始頁索引", "結束頁索引"]
EVIDENCE_COLUMNS = ["Rank", "Evidence", "PDF", "Page", "Chunk ID", "Score"]
SOURCE_COLUMNS = ["PDF", "Page", "Chunk ID", "Section", "Text"]
RETRIEVAL_COLUMNS = ["題號", "Gold Evidence", "Retrieved", "First Relevant Rank", "Pass@K"]
SAMPLE_COLUMNS = ["PDF", "Page", "Section", "Content Type", "Characters", "Chunk ID"]
GENERATED_QUESTION_COLUMNS = ["Question ID", "Question", "Difficulty", "Status", "Sources"]
AUTOMATIC_EVALUATION_COLUMNS = [
    "Question ID",
    "Retrieval Pass",
    "Answer Score",
    "Evidence Support",
    "Confidence",
    "Human Review",
    "Judge Reason",
]
INDEXING_LOG_AUTOSCROLL_JS = """() => {
    let previousLog = null;
    window.setInterval(() => {
        const log = document.querySelector("#indexing-log textarea");
        if (!log || log.value === previousLog) return;
        previousLog = log.value;
        log.scrollTop = log.scrollHeight;
    }, 100);
}"""


def create_app(project_root: str | Path | None = None) -> gr.Blocks:
    store = ProjectStore(project_root or os.environ.get("PROJECTS_ROOT", "projects"))
    connections = ConnectionSettings(store.root)
    documents = DocumentService(store)
    indexing = IndexingService(store, connection_settings=connections)
    querying = QueryService(store, connection_settings=connections)
    question_sets = QuestionSetService(store, querying)
    reviews = ReviewService(store, question_sets)
    ground_truth = GroundTruthService(store, question_sets)
    retrieval_evaluation = RetrievalEvaluationService(store, question_sets)
    source_sampling = SourceSamplingService(store)
    question_generation = QuestionGenerationService(store, source_sampling, connections)
    automatic_evaluation = AutomaticEvaluationService(store, question_sets, connections)
    automatic_qa = AutomaticQATestService(
        store, source_sampling, question_generation, querying, question_sets,
        automatic_evaluation, retrieval_evaluation,
    )

    def connection_status(project_id: str | None = None) -> str:
        return (
            f"目前狀態：Base URL `{connections.get_api_base_url()}`；API Key {connections.masked_api_key(project_id)}；"
            f"Chat `{connections.get_chat_model()}`；Embedding `{connections.get_embedding_model()}`"
        )

    def save_connection(api_base_url: str, api_key: str, chat_model: str, embedding_model: str):
        try:
            connections.save(api_base_url, api_key, chat_model, embedding_model)
        except ProjectError as exc:
            return f"❌ {exc}", api_base_url, api_key, chat_model, embedding_model
        return (
            f"✅ 連線設定已儲存。{connection_status()}",
            api_base_url,
            api_key,
            chat_model,
            embedding_model,
        )

    def test_connection(api_base_url: str, api_key: str):
        try:
            result = connections.test(api_base_url or None, api_key or None)
        except ProjectError as exc:
            return f"❌ {exc}"
        icon = "✅" if result.success else "❌"
        return f"{icon} {result.message}"

    def refresh_projects() -> list[list[str | int]]:
        return store.table_rows()

    def project_choices() -> list[tuple[str, str]]:
        return [(project.display_name, project.project_id) for project in store.list()]

    def active_project_label(project_id: str | None) -> str:
        if not project_id:
            return "目前沒有開啟的專案。請先到「專案設定」選取並開啟專案。"
        try:
            project = store.get(project_id)
        except ProjectError:
            return "目前開啟的專案已不存在，請重新選取專案。"
        return (
            f"目前開啟專案：**{project.display_name}** (`{project.project_id}`)｜"
            f"狀態：{project.status}"
        )

    def open_project(project_id: str | None):
        if not project_id:
            return None, active_project_label(None), {}, connections.get_api_key() or "", "請先選取專案", connection_status()
        try:
            project = store.get(project_id)
        except ProjectError as exc:
            return None, active_project_label(None), {}, connections.get_api_key() or "", f"❌ {exc}", connection_status()
        key = connections.get_api_key(project.project_id) or ""
        return (
            project.project_id,
            active_project_label(project.project_id),
            asdict(project),
            key,
            f"✅ 已開啟專案 {project.project_id}，後續分頁將使用此專案",
            connection_status(project.project_id),
        )

    def ask_question(project_id: str | None, question: str, method: str):
        yield "", "⏳ 正在查詢 GraphRAG，完成後將顯示回答與 Evidence…", [], gr.Dropdown(choices=[]), "", {}, [], "", ""
        if not project_id:
            yield "", "❌ 請先選擇已完成建圖的專案", [], gr.Dropdown(choices=[]), "", {}, [], "", ""
            return
        try:
            query_result = querying.ask(project_id, question, method)
        except ProjectError as exc:
            yield "", f"❌ {exc}", [], gr.Dropdown(choices=[]), "", {}, [], "", ""
            return
        summary = (
            f"狀態：{query_result.status}｜方法：{query_result.method}｜"
            f"耗時：{query_result.duration_seconds:.3f} 秒｜執行時間：{query_result.completed_at}"
        )
        if query_result.status == "FAILED":
            yield "", f"❌ {summary}｜{query_result.error}", [], gr.Dropdown(choices=[]), "", {}, [], "", ""
            return
        evidence_values = [asdict(item) for item in query_result.evidence]
        evidence_rows = [
            [item.rank, item.evidence_id, item.document_id, item.page, item.chunk_id, item.score]
            for item in query_result.evidence
        ]
        answer = query_result.answer
        if query_result.evidence:
            selected = query_result.evidence[0].evidence_id
            detail = evidence_markdown(selected, evidence_values)
        else:
            selected = None
            detail = ""
        selector = gr.Dropdown(
            choices=[item.evidence_id for item in query_result.evidence],
            value=selected,
        )
        yield (
            answer,
            f"✅ {summary}",
            evidence_rows,
            selector,
            detail,
            querying.context_summary(query_result.context),
            evidence_values,
            question,
            query_result.answer,
        )

    def evidence_markdown(evidence_id: str | None, evidence_values: list[dict[str, object]]) -> str:
        for item in evidence_values or []:
            if item.get("evidence_id") == evidence_id:
                return (
                    f"### [{item['evidence_id']}] {item['document_id']} — 第 {item['page']} 頁\n\n"
                    f"章節：{item['section_name']} (`{item['section_id']}`)  \n"
                    f"Chunk：`{item['chunk_id']}`  \n"
                    f"Text Unit：`{item['text_unit_id']}`\n\n{item['text']}"
                )
        return ""

    def question_set_choices(project_id: str | None) -> list[tuple[str, str]]:
        if not project_id:
            return []
        try:
            return [(item.name, item.question_set_id) for item in question_sets.list(project_id)]
        except ProjectError:
            return []

    def answered_question_rows(question_set: QuestionSet | None) -> list[list[str]]:
        if question_set is None:
            return []
        return [
            [item.question_id, item.question, item.reference_answer]
            for item in question_set.questions
        ]

    def saved_question_set_view(project_id: str | None, question_set_id: str | None):
        choices = question_set_choices(project_id)
        if not project_id or not question_set_id:
            return gr.Dropdown(choices=choices, value=None), [], "請選取或建立題目集"
        try:
            question_set = question_sets.get(project_id, question_set_id)
        except ProjectError as exc:
            return gr.Dropdown(choices=choices, value=None), [], f"❌ {exc}"
        return (
            gr.Dropdown(choices=choices, value=question_set.question_set_id),
            answered_question_rows(question_set),
            f"目前題目集：**{question_set.name}**｜共 {len(question_set.questions)} 題",
        )

    def load_saved_question_set(project_id: str | None, question_set_id: str | None):
        selector, rows, summary = saved_question_set_view(project_id, question_set_id)
        if not project_id or not question_set_id:
            result = "❌ 請選擇題目集"
        elif summary.startswith("❌"):
            result = summary
        else:
            result = "✅ 已載入題目集"
        return result, selector, rows, summary

    def create_saved_question_set(project_id: str | None, name: str, description: str):
        if not project_id:
            return "❌ 請先開啟專案", gr.Dropdown(choices=[]), [], ""
        try:
            question_set = question_sets.create(project_id, name, description)
        except ProjectError as exc:
            return f"❌ {exc}", gr.Dropdown(choices=question_set_choices(project_id)), [], ""
        selector, rows, status = saved_question_set_view(project_id, question_set.question_set_id)
        return f"✅ 已建立題目集「{question_set.name}」", selector, rows, status

    def add_current_answer_to_question_set(
        project_id: str | None,
        question_set_id: str | None,
        current_question: str,
        answered_question: str,
        system_answer: str,
        edited_answer: str,
        method: str,
    ):
        if not project_id or not question_set_id:
            return "❌ 請先選取題目集", gr.Dropdown(choices=question_set_choices(project_id)), [], ""
        current_question_text = (current_question or "").strip()
        if not current_question_text or current_question_text != (answered_question or "").strip():
            return "❌ 請先對目前這個問題完成問答，再加入題目集", *saved_question_set_view(project_id, question_set_id)
        if not (system_answer or "").strip():
            return "❌ 請先完成問答，再修改及加入回答", *saved_question_set_view(project_id, question_set_id)
        try:
            question_set = question_sets.append_answered_question(
                project_id,
                question_set_id,
                answered_question,
                edited_answer,
                method,
            )
        except ProjectError as exc:
            return f"❌ {exc}", *saved_question_set_view(project_id, question_set_id)
        return "✅ 已將問題與答案一併加入題目集", *saved_question_set_view(
            project_id,
            question_set.question_set_id,
        )

    def review_view(project_id: str | None, question_set_id: str | None, index: int = 0):
        if not project_id or not question_set_id:
            return "", "", None, "", 0, gr.Number(value=1, maximum=1), "尚未選擇題目集"
        records = reviews.records(project_id, question_set_id)
        if not records:
            return "", "", None, "", 0, gr.Number(value=1, maximum=1), "題目集沒有題目"
        safe_index = min(max(int(index), 0), len(records) - 1)
        record = records[safe_index]
        progress = reviews.progress(project_id, question_set_id)
        question_text = f"### {record.question_id}\n\n{record.question}"
        answer_text = record.answer or f"*沒有回答（狀態：{record.answer_status}）*"
        progress_text = f"審查進度：{progress.reviewed} / {progress.total}｜目前第 {safe_index + 1} 題"
        return (
            question_text,
            answer_text,
            record.human_label,
            record.reviewer_note,
            safe_index,
            gr.Number(value=safe_index + 1, minimum=1, maximum=len(records)),
            progress_text,
        )

    def save_current_review(
        project_id: str,
        question_set_id: str,
        index: int,
        label: str | None,
        note: str,
        require_label: bool,
    ) -> str:
        if not label:
            if require_label:
                raise ProjectError("請先選擇正確性標籤")
            return ""
        records = reviews.records(project_id, question_set_id)
        if not records:
            raise ProjectError("題目集沒有題目")
        safe_index = min(max(int(index), 0), len(records) - 1)
        reviews.save(project_id, question_set_id, records[safe_index].question_id, label, note)
        return "已自動儲存目前評測"

    def navigate_review(
        project_id: str | None,
        question_set_id: str | None,
        index: int,
        label: str | None,
        note: str,
        offset: int,
        require_label: bool = False,
    ):
        if not project_id or not question_set_id:
            return *review_view(None, None), "❌ 請先選擇專案與題目集"
        try:
            saved = save_current_review(project_id, question_set_id, index, label, note, require_label)
            view = review_view(project_id, question_set_id, int(index) + offset)
        except ProjectError as exc:
            return *review_view(project_id, question_set_id, index), f"❌ {exc}"
        return *view, f"✅ {saved}" if saved else ""

    def jump_review(
        project_id: str | None,
        question_set_id: str | None,
        index: int,
        label: str | None,
        note: str,
        target_number: float,
    ):
        target_index = max(int(target_number or 1) - 1, 0)
        return navigate_review(project_id, question_set_id, index, label, note, target_index - int(index))

    def export_reviews(project_id: str | None, question_set_id: str | None):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", None, None
        try:
            json_path, csv_path = reviews.export(project_id, question_set_id)
            json_path, csv_path = stage_downloads((json_path, csv_path))
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出人工評測 JSON 與 CSV", str(json_path), str(csv_path)

    def source_rows(project_id: str | None) -> list[list[str | int]]:
        if not project_id:
            return []
        try:
            sources = ground_truth.available_sources(project_id)
        except ProjectError:
            return []
        seen: set[str] = set()
        rows: list[list[str | int]] = []
        for source in sources:
            if source.chunk_id in seen:
                continue
            seen.add(source.chunk_id)
            rows.append(
                [source.document_id, source.page, source.chunk_id, source.section_name, source.text]
            )
        return rows

    def ground_truth_set_view(project_id: str | None, question_set_id: str | None):
        if not project_id or not question_set_id:
            return gr.Dropdown(choices=[]), "", "[]", source_rows(project_id), ""
        try:
            question_set = question_sets.get(project_id, question_set_id)
        except ProjectError as exc:
            return gr.Dropdown(choices=[]), "", "[]", source_rows(project_id), f"❌ {exc}"
        choices = [(item.question_id, item.question_id) for item in question_set.questions]
        return gr.Dropdown(choices=choices), "", "[]", source_rows(project_id), ""

    def ground_truth_question_view(
        project_id: str | None,
        question_set_id: str | None,
        question_id: str | None,
    ):
        if not project_id or not question_set_id or not question_id:
            return "", "[]"
        question_set = question_sets.get(project_id, question_set_id)
        item = next((item for item in question_set.questions if item.question_id == question_id), None)
        if item is None:
            return "", "[]"
        evidence_json = json.dumps([asdict(value) for value in item.gold_evidence], ensure_ascii=False, indent=2)
        return f"### {item.question_id}\n\n{item.question}", evidence_json

    def save_ground_truth(
        project_id: str | None,
        question_set_id: str | None,
        question_id: str | None,
        evidence_json: str,
    ):
        if not project_id or not question_set_id or not question_id:
            return "❌ 請先選擇專案、題目集與題目", evidence_json
        try:
            value = json.loads(evidence_json)
        except json.JSONDecodeError as exc:
            return f"❌ JSON 格式錯誤：第 {exc.lineno} 行第 {exc.colno} 欄，{exc.msg}", evidence_json
        try:
            updated = ground_truth.save(project_id, question_set_id, question_id, value)
        except ProjectError as exc:
            return f"❌ {exc}", evidence_json
        item = next(item for item in updated.questions if item.question_id == question_id)
        normalized = json.dumps([asdict(gold) for gold in item.gold_evidence], ensure_ascii=False, indent=2)
        return f"✅ 已儲存 {len(item.gold_evidence)} 組 Gold Evidence", normalized

    def run_retrieval_evaluation(
        project_id: str | None,
        question_set_id: str | None,
        top_k: float,
        rerun_queries: bool,
        method: str,
    ):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", "", [], gr.Dropdown(choices=[]), []
        try:
            result = retrieval_evaluation.evaluate(
                project_id,
                question_set_id,
                top_k=int(top_k),
                rerun_queries=rerun_queries,
                method=method,
            )
        except ProjectError as exc:
            return f"❌ {exc}", "", [], gr.Dropdown(choices=[]), []
        summary = (
            f"Recall@1 `{result.recall_at_1:.3f}`｜Recall@3 `{result.recall_at_3:.3f}`｜"
            f"Recall@5 `{result.recall_at_5:.3f}`｜MRR `{result.mrr:.3f}`｜"
            f"Avg First Rank `{result.average_first_relevant_rank if result.average_first_relevant_rank is not None else 'N/A'}`｜"
            f"Source Accuracy `{result.evidence_source_accuracy:.3f}`｜"
            f"Avg Latency `{result.average_latency_seconds if result.average_latency_seconds is not None else 'N/A'}` 秒"
        )
        item_values = [asdict(item) for item in result.items]
        rows = [
            [
                item.question_id,
                sum(len(gold.chunk_ids) or len(gold.pages) for gold in item.gold_evidence),
                len(item.retrieved_evidence),
                item.first_relevant_rank,
                item.passed_at_k,
            ]
            for item in result.items
        ]
        selector = gr.Dropdown(
            choices=[(item.question_id, item.question_id) for item in result.items],
            value=result.items[0].question_id if result.items else None,
        )
        return "✅ Retrieval 評估完成", summary, rows, selector, item_values

    def retrieval_item_detail(question_id: str | None, items: list[dict[str, object]]):
        for item in items or []:
            if item.get("question_id") == question_id:
                return item
        return {}

    def export_retrieval_evaluation(project_id: str | None, question_set_id: str | None):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", None, None
        try:
            json_path, csv_path = retrieval_evaluation.export(project_id, question_set_id)
            json_path, csv_path = stage_downloads((json_path, csv_path))
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出 Retrieval 評估 JSON 與 CSV", str(json_path), str(csv_path)

    def run_automatic_evaluation(
        project_id: str | None,
        question_set_id: str | None,
        top_k: float,
        rerun_answers: bool,
        only_failures: bool,
        evidence_limit: float,
    ):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", "", [], gr.Dropdown(choices=[]), []
        try:
            result = automatic_evaluation.evaluate(
                project_id,
                question_set_id,
                top_k=int(top_k),
                rerun_answers=rerun_answers,
                only_previous_failures=only_failures,
                maximum_evidence_characters=int(evidence_limit),
            )
        except ProjectError as exc:
            return f"❌ {exc}", "", [], gr.Dropdown(choices=[]), []
        rows = [
            [
                item.question_id,
                item.retrieval_pass,
                item.answer_score,
                item.evidence_support_score,
                item.judge_confidence,
                item.needs_human_review,
                item.judge_reason,
            ]
            for item in result.items
        ]
        values = [asdict(item) for item in result.items]
        first_id = result.items[0].question_id if result.items else None
        selector = gr.Dropdown(
            choices=[(item.question_id, item.question_id) for item in result.items],
            value=first_id,
        )
        summary = (
            f"模型 `{result.model}`｜Prompt `{result.prompt_version}`｜題數 {result.question_count}｜"
            f"平均答案分數 `{result.average_answer_score:.2f}`｜平均證據支持分數 "
            f"`{result.average_evidence_support_score:.2f}`｜待人工審查 {result.human_review_count}"
        )
        return "✅ 自動評測完成（LLM Judge 結果仍需人工抽查）", summary, rows, selector, values

    def automatic_evaluation_detail(question_id: str | None, items: list[dict[str, object]]):
        return next((item for item in items or [] if item.get("question_id") == question_id), {})

    def export_automatic_evaluation(project_id: str | None, question_set_id: str | None):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", None, None
        try:
            json_path, csv_path = automatic_evaluation.export(project_id, question_set_id)
            json_path, csv_path = stage_downloads((json_path, csv_path))
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出自動評測 JSON 與 CSV", str(json_path), str(csv_path)

    def automatic_qa_rows(report):
        questions = {item.question_id: item for item in report.question_set.questions}
        judges = {item.question_id: item for item in report.judge.items} if report.judge else {}
        rows = []
        for question_id, question in questions.items():
            judged = judges.get(question_id)
            evidence_docs = ", ".join(dict.fromkeys(item.document_id for item in question.gold_evidence)) or "—"
            rows.append([
                evidence_docs, question.question, question.answer, question.reference_answer,
                "答對" if judged and judged.answer_score >= 4 else ("未評判" if not judged else "答錯"),
                judged.answer_score if judged else None,
                judged.evidence_support_score if judged else None,
                judged.judge_reason if judged else (question.error or "尚無評判結果"),
            ])
        return rows

    def autoqa_question_rows(question_set):
        rows = []
        for item in question_set.questions:
            question_sources = item.question_source_evidence or item.gold_evidence
            answer_sources = item.answer_source_evidence or item.gold_evidence
            rows.append([
                item.question_id,
                item.question,
                item.reference_answer,
                QuestionSetService._format_source_pages(question_sources),
                QuestionSetService._format_source_pages(answer_sources),
            ])
        return rows

    def parse_autoqa_sources(value, existing):
        text = str(value or "").strip()
        if not text:
            return ()
        try:
            records = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProjectError("來源欄位必須是 JSON 陣列，每筆包含文件 ID、文件名稱及頁碼") from exc
        parsed = QuestionSetService._parse_source_pages(records, "source_records")
        result = []
        for item in parsed:
            previous = next(
                (source for source in existing if source.document_id == item.document_id and source.pages == item.pages),
                None,
            )
            result.append(GoldEvidence(
                item.document_id,
                item.pages,
                previous.chunk_ids if previous else item.chunk_ids,
                item.document_name,
            ))
        return tuple(result)

    def save_autoqa_edits(project_id, question_set_id, rows):
        if not project_id or not question_set_id:
            raise ProjectError("請先生成或匯入題目集")
        current = question_sets.get(project_id, question_set_id)
        by_id = {item.question_id: item for item in current.questions}
        updated = []
        if rows is None:
            editable_rows = []
        elif hasattr(rows, "itertuples"):
            editable_rows = rows.itertuples(index=False, name=None)
        else:
            editable_rows = list(rows)
        for row in editable_rows:
            if len(row) < 5:
                raise ProjectError("編輯表格欄位不完整")
            question_id = str(row[0])
            if question_id not in by_id:
                raise ProjectError(f"題目集找不到題號：{question_id}")
            old = by_id[question_id]
            question_sources = parse_autoqa_sources(
                row[3], old.question_source_evidence or old.gold_evidence
            )
            answer_sources = parse_autoqa_sources(
                row[4], old.answer_source_evidence or old.gold_evidence
            )
            source_documents = tuple(dict.fromkeys(
                source.document_id for source in (*question_sources, *answer_sources)
            ))
            updated.append(BatchQuestion(
                question_id=question_id,
                question=str(row[1] or "").strip(),
                reference_answer=str(row[2] or "").strip(),
                status=old.status,
                answer=old.answer,
                error=old.error,
                duration_seconds=old.duration_seconds,
                completed_at=old.completed_at,
                gold_evidence=answer_sources,
                retrieved_evidence=old.retrieved_evidence,
                question_source_evidence=question_sources,
                answer_source_evidence=answer_sources,
                source_documents=source_documents,
            ))
        return question_sets.update_questions(project_id, question_set_id, updated)

    def automatic_qa_summary(report):
        total = len(report.question_set.questions)
        judged = {item.question_id: item for item in report.judge.items} if report.judge else {}
        correct = sum(item.answer_score >= 4 for item in judged.values())
        lines = [
            "**題目集：** " + report.question_set.name + " (" + report.question_set.question_set_id + ")｜"
            + "**答對：** " + str(correct) + "/" + str(total) + "（" + format(correct / total, ".1%") + "）｜評判 "
            + str(len(judged)) + "/" + str(total) + " 題",
            "**模型：** 生題 " + report.generation_model + "｜回答 " + report.answer_model + "｜評判 " + report.judge_model,
        ]
        if report.retrieval:
            result = report.retrieval
            lines.append(
                "**Retrieval：** Recall@1 " + format(result.recall_at_1, ".1%")
                + "｜Recall@3 " + format(result.recall_at_3, ".1%")
                + "｜Recall@5 " + format(result.recall_at_5, ".1%")
                + "｜MRR " + format(result.mrr, ".3f")
                + "｜Evidence 命中率 " + format(result.evidence_source_accuracy, ".1%")
                + "｜平均延遲 " + format(result.average_latency_seconds or 0, ".2f") + "s"
            )
        elif report.retrieval_error:
            lines.append("**Retrieval 指標：** " + report.retrieval_error)
        if report.judge_error:
            lines.append("**評判提醒：** " + report.judge_error)
        if report.generation_errors:
            lines.append("**部分 PDF 生題未完成：** " + "；".join(report.generation_errors))
        return "\n\n".join(lines)

    def generate_automatic_qa(project_id, count, parallel_generation, generation_model, method):
        if not project_id:
            return "❌ 請先開啟專案", [], ""
        try:
            question_set = automatic_qa.generate_question_set(
                project_id, int(count), parallel_generation, generation_model, method,
            )
        except (ProjectError, ValueError) as exc:
            return "❌ " + str(exc), [], ""
        return (
            f"✅ 已生成 {len(question_set.questions)} 題。請檢查並可直接編輯下表，再選擇匯出或開始測試。\n\n{question_set.description}",
            autoqa_question_rows(question_set),
            question_set.question_set_id,
        )

    def run_automatic_qa(project_id, question_set_id, rows, answer_model, judge_model, method, concurrency):
        if not project_id or not question_set_id:
            return "❌ 請先生成或匯入題目集", "", []
        try:
            save_autoqa_edits(project_id, question_set_id, rows)
            report = automatic_qa.run_existing(
                project_id, question_set_id, answer_model, judge_model, method, int(concurrency)
            )
        except (ProjectError, ValueError) as exc:
            return "❌ " + str(exc), "", []
        return "✅ 自動問答測試完成", automatic_qa_summary(report), automatic_qa_rows(report)

    def import_automatic_question_set(project_id, uploaded):
        if not project_id or not uploaded:
            return "❌ 請先開啟專案並選擇 JSON 題目集", [], ""
        source = uploaded if isinstance(uploaded, (str, Path)) else getattr(uploaded, "name", None)
        if not source:
            return "❌ 無法讀取上傳檔案", [], ""
        try:
            question_set = question_sets.import_file(project_id, source)
        except ProjectError as exc:
            return "❌ " + str(exc), [], ""
        return (
            "✅ 已匯入題目集「" + question_set.name + "」，可先編輯再測試。",
            autoqa_question_rows(question_set),
            question_set.question_set_id,
        )

    def export_automatic_question_set(project_id, question_set_id, rows):
        if not project_id or not question_set_id:
            return "❌ 請先生成或匯入題目集", None
        try:
            save_autoqa_edits(project_id, question_set_id, rows)
            json_path = question_sets.export_json(project_id, question_set_id)
            (json_path,) = stage_downloads((json_path,))
        except (ProjectError, ValueError) as exc:
            return "❌ " + str(exc), None
        return "✅ 已匯出目前編輯後的題目集 JSON", str(json_path)

    def sampling_section_choices(project_id: str | None):
        if not project_id:
            return gr.Dropdown(choices=[], value=None)
        try:
            sections = source_sampling.available_sections(project_id)
        except ProjectError:
            sections = []
        return gr.Dropdown(
            choices=[(f"{section_id} — {name}", section_id) for section_id, name in sections],
            value=[],
        )

    @staticmethod
    def sample_rows(samples: list[SourceSample]) -> list[list[str | int]]:
        return [
            [
                item.document_id,
                item.page,
                item.section_name,
                item.content_type,
                item.character_count,
                item.chunk_id,
            ]
            for item in samples
        ]

    def source_sample_detail(sample_id: str | None, values: list[dict[str, object]]) -> str:
        for item in values or []:
            if item.get("sample_id") == sample_id:
                return (
                    f"### {item['document_id']} — 第 {item['page']} 頁\n\n"
                    f"類型：{item['content_type']}  \nChunk：`{item['chunk_id']}`\n\n{item['text']}"
                )
        return ""

    def scan_source_samples(
        project_id: str | None,
        section_ids: list[str] | None,
        page_from: float | None,
        page_to: float | None,
        content_type: str,
        minimum_characters: float,
    ):
        if not project_id:
            return "❌ 請先選擇專案", [], gr.Dropdown(choices=[]), "", []
        try:
            samples = source_sampling.scan(
                project_id,
                section_ids,
                int(page_from) if page_from is not None else None,
                int(page_to) if page_to is not None else None,
                content_type,
                int(minimum_characters),
            )
        except ProjectError as exc:
            return f"❌ {exc}", [], gr.Dropdown(choices=[]), "", []
        values = [asdict(item) for item in samples]
        selector = gr.Dropdown(
            choices=[(f"{item.document_id} p.{item.page} — {item.content_type}", item.sample_id) for item in samples],
            value=samples[0].sample_id if samples else None,
        )
        detail = source_sample_detail(samples[0].sample_id, values) if samples else ""
        return f"✅ 找到 {len(samples)} 筆候選原文", sample_rows(samples), selector, detail, values

    def create_source_sample_batch(
        project_id: str | None,
        section_ids: list[str] | None,
        page_from: float | None,
        page_to: float | None,
        content_type: str,
        minimum_characters: float,
        count: float,
        seed: float,
    ):
        if not project_id:
            return "❌ 請先選擇專案", [], gr.Dropdown(choices=[]), "", [], ""
        try:
            batch = source_sampling.sample(
                project_id,
                int(count),
                section_ids,
                int(page_from) if page_from is not None else None,
                int(page_to) if page_to is not None else None,
                content_type,
                int(minimum_characters),
                int(seed),
            )
        except ProjectError as exc:
            return f"❌ {exc}", [], gr.Dropdown(choices=[]), "", [], ""
        samples = list(batch.samples)
        values = [asdict(item) for item in samples]
        selector = gr.Dropdown(
            choices=[(f"{item.document_id} p.{item.page} — {item.content_type}", item.sample_id) for item in samples],
            value=samples[0].sample_id if samples else None,
        )
        detail = source_sample_detail(samples[0].sample_id, values) if samples else ""
        return (
            f"✅ 已建立取樣批次 {batch.sample_batch_id}，共 {len(samples)} 筆",
            sample_rows(samples),
            selector,
            detail,
            values,
            batch.sample_batch_id,
        )

    def export_source_samples(project_id: str | None, sample_batch_id: str):
        if not project_id or not sample_batch_id:
            return "❌ 尚未建立取樣批次", None, None
        try:
            json_path, csv_path = source_sampling.export(project_id, sample_batch_id)
            json_path, csv_path = stage_downloads((json_path, csv_path))
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出原文取樣 JSON 與 CSV", str(json_path), str(csv_path)

    @staticmethod
    def generated_question_rows(questions: list[GeneratedQuestion]) -> list[list[str | int]]:
        return [
            [
                item.question_id,
                item.question,
                item.difficulty,
                item.generation_status,
                len(item.source_sample_ids),
            ]
            for item in questions
        ]

    def generated_question_detail(question_id: str | None, values: list[dict[str, object]]):
        for item in values or []:
            if item.get("question_id") == question_id:
                return (
                    str(item.get("question", "")),
                    str(item.get("reference_answer", "")),
                    str(item.get("generation_status", "pending_review")),
                    {
                        "source_sample_ids": item.get("source_sample_ids", []),
                        "gold_evidence": item.get("gold_evidence", []),
                    },
                )
        return "", "", "pending_review", {}

    def generate_questions(
        project_id: str | None,
        sample_batch_id: str,
        count: float,
        difficulty: str,
        maximum_source_characters: float,
    ):
        empty = ([], gr.Dropdown(choices=[]), "", "", "pending_review", {}, [], "")
        if not project_id or not sample_batch_id:
            return ("❌ 請先選擇專案並建立取樣批次", *empty)
        try:
            batch = question_generation.generate(
                project_id,
                sample_batch_id,
                int(count),
                difficulty,
                int(maximum_source_characters),
            )
        except ProjectError as exc:
            return (f"❌ {exc}", *empty)
        questions = list(batch.questions)
        values = [asdict(item) for item in questions]
        first_id = questions[0].question_id if questions else None
        selector = gr.Dropdown(
            choices=[(item.question_id, item.question_id) for item in questions],
            value=first_id,
        )
        detail = generated_question_detail(first_id, values)
        return (
            f"✅ 已用 {batch.model} 單次呼叫生成 {len(questions)} 題；請逐題人工審核",
            generated_question_rows(questions),
            selector,
            *detail,
            values,
            batch.generation_batch_id,
        )

    def update_generated_question(
        project_id: str | None,
        generation_batch_id: str,
        question_id: str | None,
        question: str,
        reference_answer: str,
        generation_status: str,
    ):
        if not project_id or not generation_batch_id or not question_id:
            return "❌ 尚未選擇生成題目", [], gr.Dropdown(choices=[]), [], ""
        try:
            batch = question_generation.update_question(
                project_id,
                generation_batch_id,
                question_id,
                question,
                reference_answer,
                generation_status,
            )
        except ProjectError as exc:
            return f"❌ {exc}", [], gr.Dropdown(), [], generation_batch_id
        questions = list(batch.questions)
        values = [asdict(item) for item in questions]
        selector = gr.Dropdown(
            choices=[(item.question_id, item.question_id) for item in questions],
            value=question_id,
        )
        return "✅ 題目與審核狀態已儲存", generated_question_rows(questions), selector, values, generation_batch_id

    def export_generated_question_set(project_id: str | None, generation_batch_id: str):
        if not project_id or not generation_batch_id:
            return "❌ 尚未建立題目生成批次", None
        try:
            path = question_generation.export_question_set(project_id, generation_batch_id)
            (path,) = stage_downloads((path,))
        except ProjectError as exc:
            return f"❌ {exc}", None
        return "✅ 已匯出所有已核准題目的 question_set JSON", str(path)

    def project_details(project_id: str | None) -> dict[str, str]:
        if not project_id:
            return {}
        try:
            return asdict(store.get(project_id))
        except ProjectError:
            return {}

    def project_enabled_value(project_id: str | None) -> bool:
        try:
            return store.get(project_id).enabled if project_id else True
        except ProjectError:
            return True

    def update_project_enabled(project_id: str | None, enabled: bool):
        if not project_id:
            return "❌ 請先選擇專案", store.table_rows(), {}
        try:
            project = store.set_enabled(project_id, enabled)
        except ProjectError as exc:
            return f"❌ {exc}", store.table_rows(), project_details(project_id)
        state = "啟用" if project.enabled else "停用"
        return f"✅ 已{state}專案 {project.project_id}", store.table_rows(), asdict(project)

    def refresh_project_views(project_id: str | None, active_id: str | None):
        available_ids = {project.project_id for project in store.list()}
        selected = project_id if project_id in available_ids else None
        active = active_id if active_id in available_ids else None
        choices = project_choices()
        return (
            store.table_rows(),
            gr.Dropdown(choices=choices, value=selected, allow_custom_value=True),
            project_details(selected),
            active,
            active_project_label(active),
        )

    def delete_project(project_id: str | None, confirmed: bool, active_id: str | None):
        if not project_id:
            return "❌ 請先選擇專案", *refresh_project_views(None, active_id), False
        if not confirmed:
            return "❌ 請勾選刪除確認", *refresh_project_views(project_id, active_id), False
        try:
            store.delete(project_id)
        except ProjectError as exc:
            return f"❌ {exc}", *refresh_project_views(project_id, active_id), False
        next_active = None if project_id == active_id else active_id
        return f"✅ 已刪除專案 {project_id}", *refresh_project_views(None, next_active), False

    def document_rows(project_id: str | None) -> list[list[str | int | None]]:
        return [document_row(document) for document in documents.list_documents_if_available(project_id)]

    def processing_option_rows(project_id: str | None) -> list[list[str | int | float]]:
        if not project_id:
            return []
        saved = documents.processing_options(project_id)
        rows: list[list[str | int | float]] = []
        for document in documents.list_documents_if_available(project_id):
            options = saved.get(document.filename, {})
            pages = document.pages or 0
            rows.append(
                [
                    document.filename,
                    pages,
                    float(options.get("header_ignore_percent", 0)),
                    float(options.get("footer_ignore_percent", 0)),
                    int(options.get("start_page", 0)),
                    int(options.get("end_page", max(pages - 1, 0))),
                ]
            )
        return rows

    def parse_processing_options(rows: object | None) -> dict[str, dict[str, object]]:
        options: dict[str, dict[str, object]] = {}
        if rows is None:
            return options
        # Gradio's Dataframe(type="array") may submit a NumPy array; avoid
        # truth-value checks, which are ambiguous for arrays with multiple rows.
        for row in rows:
            if len(row) < 6 or row[0] in (None, ""):
                continue
            filename = Path(str(row[0])).name
            options[filename] = {
                "header_ignore_percent": row[2] if row[2] not in (None, "") else 0,
                "footer_ignore_percent": row[3] if row[3] not in (None, "") else 0,
                "start_page": row[4] if row[4] not in (None, "") else 0,
                "end_page": row[5],
            }
        return options

    def document_view(project_id: str | None):
        rows = document_rows(project_id)
        choices = [row[0] for row in rows]
        return rows, processing_option_rows(project_id), gr.Dropdown(choices=choices, value=None)

    def active_project_views(project_id: str | None):
        documents_view = document_view(project_id)
        choices = question_set_choices(project_id) if project_id else []
        return (
            *documents_view,
            gr.Dropdown(choices=choices, value=None),
            gr.Dropdown(choices=choices, value=None),
            sampling_section_choices(project_id),
            [],
            "請選取或建立題目集",
        )

    def import_documents(project_id: str | None, files: list[str] | None):
        if not project_id:
            return "❌ 請先到「專案設定」開啟專案", [], [], gr.Dropdown(choices=[])
        try:
            imported = documents.import_pdfs(project_id, files or [])
        except ProjectError as exc:
            rows, settings, selector = document_view(project_id)
            return f"❌ {exc}", rows, settings, selector
        rows = [document_row(item) for item in imported]
        return (
            f"✅ 已匯入 {len(files or [])} 份 PDF",
            rows,
            processing_option_rows(project_id),
            gr.Dropdown(choices=[row[0] for row in rows]),
        )

    def preprocess_documents(project_id: str | None, option_rows: list[list[object]] | None):
        if not project_id:
            return "❌ 請先到「專案設定」開啟專案", [], [], gr.Dropdown(choices=[])
        try:
            report = documents.preprocess(project_id, parse_processing_options(option_rows))
        except (ProjectError, OSError, TypeError, ValueError) as exc:
            rows, settings, selector = document_view(project_id)
            return f"❌ {exc}", rows, settings, selector
        message = (
            f"✅ 前處理完成：成功 {report.successful_pages} 頁、"
            f"無文字 {report.empty_pages} 頁、錯誤 {report.error_pages} 頁"
        )
        rows, settings, selector = document_view(project_id)
        return message, rows, settings, selector

    def remove_document(project_id: str | None, filename: str | None, confirmed: bool):
        if not project_id or not filename:
            rows, settings, selector = document_view(project_id)
            return "❌ 請先選擇 PDF", rows, settings, selector, False
        if not confirmed:
            rows, settings, selector = document_view(project_id)
            return "❌ 請勾選移除確認", rows, settings, selector, False
        try:
            documents.remove_pdf(project_id, filename)
        except ProjectError as exc:
            rows, settings, selector = document_view(project_id)
            return f"❌ {exc}", rows, settings, selector, False
        rows, settings, selector = document_view(project_id)
        return f"✅ 已移除 {filename}；請重新執行前處理與建圖", rows, settings, selector, False

    def build_index(project_id: str | None):
        if not project_id:
            yield "❌ 請先到「專案設定」開啟專案", ""
            return
        chunks: queue.Queue[str] = queue.Queue()
        result_holder: list[object] = []
        error_holder: list[Exception] = []

        def run_index() -> None:
            try:
                result_holder.append(indexing.build(project_id, log_callback=chunks.put))
            except Exception as exc:
                error_holder.append(exc)

        worker = threading.Thread(target=run_index, daemon=True)
        worker.start()
        log = ""
        yield "⏳ GraphRAG 建圖執行中…", log
        while worker.is_alive() or not chunks.empty():
            try:
                log += chunks.get(timeout=0.25)
            except queue.Empty:
                pass
            yield "⏳ GraphRAG 建圖執行中…", log
        worker.join()
        if error_holder:
            yield f"❌ {error_holder[0]}", log
            return
        result = result_holder[0]
        icon = "✅" if result.status == "INDEXED" else "❌"
        summary = f"{icon} {result.status}｜耗時 {result.duration_seconds:.1f} 秒｜{result.last_message}"
        yield summary, log

    def create_project(
        project_id: str,
        display_name: str,
        vehicle_name: str,
        manual_version: str,
        description: str,
    ):
        try:
            project = store.create(
                project_id=project_id,
                display_name=display_name,
                vehicle_name=vehicle_name,
                manual_version=manual_version,
                description=description,
            )
        except ProjectError as exc:
            selector = gr.Dropdown(choices=project_choices(), allow_custom_value=True)
            return (
                f"❌ {exc}",
                store.table_rows(),
                project_id,
                display_name,
                vehicle_name,
                manual_version,
                description,
                selector,
                {},
            )
        selector = gr.Dropdown(
            choices=project_choices(),
            value=project.project_id,
            allow_custom_value=True,
        )
        return (
            f"✅ 已建立專案 {project.project_id}",
            store.table_rows(),
            "",
            "",
            "",
            "",
            "",
            selector,
            asdict(project),
        )

    with gr.Blocks(title="汽車維修 GraphRAG 管理後台") as demo:
        gr.Markdown("# 汽車維修 GraphRAG 管理後台")
        active_project_id = gr.State(value=None)
        active_project_banner = gr.Markdown(active_project_label(None))
        document_project = active_project_id
        query_project = active_project_id
        batch_project = active_project_id
        review_project = active_project_id
        gold_project = active_project_id
        retrieval_project = active_project_id
        automatic_project = active_project_id
        sampling_project = active_project_id
        with gr.Tab("專案設定"):
            with gr.Row():
                selected_project = gr.Dropdown(
                    choices=project_choices(),
                    label="專案",
                    allow_custom_value=True,
                )
                open_project_button = gr.Button("開啟專案", variant="primary")
            open_project_result = gr.Markdown()
            selected_project_details = gr.JSON(label="專案資料")
            project_table = gr.Dataframe(
                headers=PROJECT_COLUMNS,
                value=refresh_projects,
                interactive=False,
                datatype=["str", "str", "number", "str", "bool", "str"],
                label="Projects",
            )
            with gr.Row():
                refresh_button = gr.Button("重新整理")
                project_enabled = gr.Checkbox(label="啟用一般使用者查詢", value=True)
                update_project_enabled_button = gr.Button("更新啟用狀態")
            project_activation_result = gr.Markdown()
            gr.Markdown("## 建立新專案")
            project_id = gr.Textbox(label="Project ID", placeholder="L33-SM3E")
            display_name = gr.Textbox(label="顯示名稱", placeholder="L33 / SM3E")
            with gr.Row():
                vehicle_name = gr.Textbox(label="車型", placeholder="L33")
                manual_version = gr.Textbox(label="手冊版本", placeholder="SM3E")
            description = gr.Textbox(label="說明", lines=3)
            create_button = gr.Button("建立專案", variant="primary")
            result = gr.Markdown()
            gr.Markdown("## 刪除專案")
            delete_confirmation = gr.Checkbox(label="我確認要永久刪除選取的專案及其所有資料")
            delete_button = gr.Button("刪除選取專案", variant="stop")

        with gr.Tab("連線設定"):
            gr.Markdown("## GraphRAG 共用連線設定")
            gr.Markdown("API Base URL 與 API Key 由所有 Project 共用。")
            connection_state = gr.Markdown(value=connection_status)
            api_base_url = gr.Textbox(
                label="API Base URL",
                value=connections.get_api_base_url,
                placeholder="https://api.openai.com/v1",
            )
            api_key = gr.Textbox(
                label="API Key",
                value=connections.get_api_key() or "",
                type="password",
                placeholder="輸入 API Key",
            )
            with gr.Row():
                chat_model = gr.Dropdown(
                    choices=list(ALLOWED_CHAT_MODELS),
                    value=connections.get_chat_model,
                    label="Chat 模型",
                )
                embedding_model = gr.Dropdown(
                    choices=list(ALLOWED_EMBEDDING_MODELS),
                    value=connections.get_embedding_model,
                    label="Embedding 模型",
                )
            gr.Markdown("開發測試建議使用 `gpt-4o-mini` 與 `text-embedding-3-small` 以降低成本。")
            with gr.Row():
                test_connection_button = gr.Button("測試連線")
                save_connection_button = gr.Button("儲存連線設定", variant="primary")
            connection_result = gr.Markdown()

        with gr.Tab("文件與建圖"):
            uploaded_files = gr.File(file_count="multiple", file_types=[".pdf"], type="filepath", label="匯入 PDF")
            with gr.Row():
                upload_button = gr.Button("上傳")
                preprocess_button = gr.Button("開始前處理", variant="primary")
                index_button = gr.Button("建立 Graph", variant="primary")
                document_refresh_button = gr.Button("重新整理")
            document_result = gr.Markdown()
            document_table = gr.Dataframe(
                headers=DOCUMENT_COLUMNS,
                interactive=False,
                datatype=["str", "number", "number", "str", "number", "number", "str"],
                label="Documents",
            )
            processing_options_table = gr.Dataframe(
                headers=PROCESSING_COLUMNS,
                interactive=True,
                datatype=["str", "number", "number", "number", "number", "number"],
                type="array",
                static_columns=[0, 1],
                label="各 PDF 前處理設定（頁面索引從 0 開始，結束頁包含在範圍內）",
            )
            with gr.Row():
                removable_pdf = gr.Dropdown(label="選擇要移除的 PDF")
                remove_pdf_confirmation = gr.Checkbox(label="我確認要移除選取的 PDF")
                remove_pdf_button = gr.Button("移除 PDF", variant="stop")
            indexing_log = gr.Textbox(
                label="建圖日誌",
                lines=12,
                interactive=False,
                autoscroll=True,
                elem_id="indexing-log",
            )

        with gr.Tab("自動問答測試"):
            gr.Markdown(
                "依已開啟專案中的每份 PDF 自動生成題目、執行 GraphRAG 問答及 LLM 評判。"
                "系統會跨 PDF 去重；勾選平行生題時不同 PDF 可併行，同一 PDF 的請求仍會逐次執行。"
            )
            with gr.Row():
                autoqa_questions_per_pdf = gr.Number(label="每份 PDF 題數", value=3, minimum=1, maximum=30, precision=0)
                autoqa_parallel_generation = gr.Checkbox(label="允許不同 PDF 平行生題", value=False)
                autoqa_generation_model = gr.Dropdown(
                    choices=list(ALLOWED_CHAT_MODELS), value=connections.get_chat_model(), label="生題模型"
                )
            autoqa_generate_button = gr.Button("生成題目", variant="primary")
            with gr.Row():
                autoqa_answer_model = gr.Dropdown(
                    choices=list(ALLOWED_CHAT_MODELS), value=connections.get_chat_model(), label="回答模型"
                )
                autoqa_judge_model = gr.Dropdown(
                    choices=list(ALLOWED_CHAT_MODELS), value=connections.get_chat_model(), label="評判模型"
                )
                autoqa_method = gr.Dropdown(
                    choices=[("Local", "local"), ("Global", "global"), ("DRIFT", "drift"), ("Basic", "basic")],
                    value="local", label="檢索模式",
                )
                autoqa_answer_concurrency = gr.Number(label="回答請求並行數", value=2, minimum=1, maximum=32, precision=0)
            autoqa_test_button = gr.Button("開始自動問答測試", variant="primary")
            with gr.Row():
                autoqa_import_file = gr.File(label="匯入題目集 JSON", file_types=[".json"], type="filepath")
                autoqa_import_button = gr.Button("匯入題目集")
                autoqa_export_button = gr.Button("匯出目前題目集")
                autoqa_json_export = gr.File(label="題目集 JSON（匯出）")
            autoqa_result = gr.Markdown()
            autoqa_question_set_state = gr.State("")
            autoqa_questions_table = gr.Dataframe(
                headers=["題號", "題目", "正確答案", "題目來源（文件 ID／名稱／頁碼 JSON 陣列）", "答案來源（文件 ID／名稱／頁碼 JSON 陣列）"],
                interactive=True,
                datatype=["str", "str", "str", "str", "str"],
                label="生成題目與來源（可直接編輯）",
                wrap=True,
            )
            autoqa_summary = gr.Markdown()
            autoqa_table = gr.Dataframe(
                headers=["來源文件", "題目", "系統回答", "正確答案", "評判結果", "答案分數", "證據支持", "評判理由"],
                interactive=False,
                datatype=["str", "str", "str", "str", "str", "number", "number", "str"],
                label="逐題自動問答測試結果",
                wrap=True,
            )

        with gr.Tab("問答測試"):
            with gr.Row():
                with gr.Column():
                    question = gr.Textbox(label="問題", lines=5)
                    query_method = gr.Dropdown(
                        choices=[("Local", "local"), ("Global", "global"), ("DRIFT", "drift"), ("Basic", "basic")],
                        value="local",
                        label="查詢方法",
                    )
                    with gr.Row():
                        ask_button = gr.Button("送出問題", variant="primary")
                        clear_question_button = gr.Button("清除")
                with gr.Column():
                    query_answer = gr.Textbox(
                label="系統回答（可編輯，加入題目集時會保存修改後內容）",
                        lines=10,
                        interactive=True,
                    )
                    query_summary = gr.Markdown(label="查詢資訊")
            gr.Markdown("## 題目集")
            with gr.Row():
                question_set_name = gr.Textbox(label="新題目集名稱")
                question_set_description = gr.Textbox(label="說明（選填）")
                create_question_set_button = gr.Button("建立題目集", variant="primary")
            question_set_selector = gr.Dropdown(label="題目集")
            with gr.Row():
                refresh_question_sets_button = gr.Button("重新整理")
                load_question_set_button = gr.Button("載入題目集")
                add_qa_to_question_set_button = gr.Button("將目前問題與編輯後答案加入題目集")
            question_set_result = gr.Markdown()
            question_set_summary = gr.Markdown("請建立或載入題目集")
            saved_questions_table = gr.Dataframe(
                headers=["題號", "問題", "答案"],
                interactive=False,
                datatype=["str", "str", "str"],
                label="題目集內容（每題均包含問題與答案）",
            )
            gr.Markdown("### Evidence 詳情")
            query_evidence_state = gr.State([])
            query_question_state = gr.State("")
            query_answer_state = gr.State("")
            query_evidence_table = gr.Dataframe(
                headers=EVIDENCE_COLUMNS,
                interactive=False,
                datatype=["number", "str", "str", "number", "str", "number"],
                label="Top Evidence",
            )
            query_evidence_selector = gr.Dropdown(label="選取證據全文")
            query_evidence_detail = gr.Markdown()
            query_context = gr.JSON(label="原始 GraphRAG Query Context")

        with gr.Tab("自動評測（已整合）", visible=False):
            gr.Markdown(
                "## 回答與證據自動評測\n"
                "每批 Judge 僅呼叫 API 一次，預設使用 `gpt-4o-mini`；低分、Retrieval 失敗或低信心結果會標記待人工審查。"
            )
            with gr.Row():
                automatic_project_refresh = gr.Button("重新整理題目集")
                automatic_question_set = gr.Dropdown(label="題目集")
            with gr.Row():
                automatic_top_k = gr.Number(label="Top-K", value=5, minimum=1, precision=0)
                automatic_evidence_limit = gr.Number(
                    label="每筆 Evidence 字數上限",
                    value=1200,
                    minimum=200,
                    precision=0,
                )
                automatic_rerun_answers = gr.Checkbox(
                    label="先重新執行系統回答（增加 API 成本）",
                    value=False,
                )
                automatic_only_failures = gr.Checkbox(
                    label="只重跑前次待人工審查題目",
                    value=False,
                )
            with gr.Row():
                run_automatic_button = gr.Button("執行回答與評測", variant="primary")
                export_automatic_button = gr.Button("匯出評測報告")
            automatic_result = gr.Markdown()
            automatic_summary = gr.Markdown()
            automatic_table = gr.Dataframe(
                headers=AUTOMATIC_EVALUATION_COLUMNS,
                interactive=False,
                datatype=["str", "bool", "number", "number", "str", "bool", "str"],
                label="逐題自動評測結果",
            )
            automatic_items_state = gr.State([])
            automatic_item_selector = gr.Dropdown(label="查看逐題 Judge 詳情")
            automatic_item_json = gr.JSON(label="自動評測詳情")
            with gr.Row():
                automatic_json_export = gr.File(label="自動評測 JSON")
                automatic_csv_export = gr.File(label="自動評測 CSV")

        with gr.Tab("題目生成（已整合）", visible=False):
            gr.Markdown("## 原文取樣器")
            with gr.Row():
                sampling_project_refresh = gr.Button("重新整理章節")
                sampling_sections = gr.Dropdown(multiselect=True, label="章節（空白代表全部）")
            with gr.Row():
                sampling_page_from = gr.Number(label="起始頁碼", value=1, minimum=1, precision=0)
                sampling_page_to = gr.Number(label="結束頁碼", value=1, minimum=1, precision=0)
                sampling_content_type = gr.Dropdown(
                    choices=[
                        ("全部", "all"),
                        ("診斷", "diagnostic"),
                        ("程序", "procedure"),
                        ("規格", "specification"),
                        ("一般", "general"),
                    ],
                    value="all",
                    label="內容類型",
                )
                sampling_minimum_characters = gr.Number(
                    label="最小字數",
                    value=200,
                    minimum=1,
                    precision=0,
                )
            with gr.Row():
                sampling_count = gr.Number(label="取樣數量", value=10, minimum=1, precision=0)
                sampling_seed = gr.Number(label="Random Seed", value=42, precision=0)
                scan_sources_button = gr.Button("掃描可用原文")
                sample_sources_button = gr.Button("建立取樣批次", variant="primary")
                export_samples_button = gr.Button("匯出取樣")
            sampling_result = gr.Markdown()
            sampling_table = gr.Dataframe(
                headers=SAMPLE_COLUMNS,
                interactive=False,
                datatype=["str", "number", "str", "str", "number", "str"],
                label="候選／取樣原文",
            )
            sampling_values_state = gr.State([])
            sampling_batch_state = gr.State("")
            sampling_preview_selector = gr.Dropdown(label="預覽原文")
            sampling_preview = gr.Markdown()
            with gr.Row():
                sampling_json_export = gr.File(label="取樣 JSON")
                sampling_csv_export = gr.File(label="取樣 CSV")

            gr.Markdown("## AI 題目生成與人工審核")
            gr.Markdown(
                "每個批次只呼叫 API 一次，使用連線設定中的 Chat Model（預設 `gpt-4o-mini`）。"
                "問題與參考答案固定使用繁體中文，技術代號可保留英文。"
                "生成結果只會引用上述取樣原文，且匯出前必須人工核准。"
            )
            with gr.Row():
                generation_count = gr.Number(label="生成題數", value=3, minimum=1, precision=0)
                generation_difficulty = gr.Dropdown(
                    choices=[
                        ("簡單（單一來源）", "simple"),
                        ("中等（至少兩筆來源）", "medium"),
                        ("跨章節（至少兩個章節）", "cross_section"),
                    ],
                    value="simple",
                    label="難度",
                )
                generation_source_limit = gr.Number(
                    label="每筆來源字數上限",
                    value=2500,
                    minimum=200,
                    precision=0,
                )
                generate_questions_button = gr.Button("呼叫 API 生成候選題", variant="primary")
            generation_result = gr.Markdown()
            generated_question_table = gr.Dataframe(
                headers=GENERATED_QUESTION_COLUMNS,
                interactive=False,
                datatype=["str", "str", "str", "str", "number"],
                label="生成候選題",
            )
            generated_values_state = gr.State([])
            generation_batch_state = gr.State("")
            generated_question_selector = gr.Dropdown(label="選擇要審核的題目")
            generated_question_editor = gr.Textbox(label="問題", lines=3)
            generated_answer_editor = gr.Textbox(label="參考答案", lines=5)
            with gr.Row():
                generated_status = gr.Dropdown(
                    choices=[
                        ("待審核", "pending_review"),
                        ("核准", "approved"),
                        ("淘汰", "rejected"),
                    ],
                    value="pending_review",
                    label="審核狀態",
                )
                save_generated_question_button = gr.Button("儲存編輯與審核狀態")
                export_generated_questions_button = gr.Button("匯出已核准 question_set")
            generated_evidence = gr.JSON(label="引用來源與 Gold Evidence")
            generated_question_export = gr.File(label="Question Set JSON")

        refresh_button.click(
            refresh_project_views,
            inputs=[selected_project, active_project_id],
            outputs=[project_table, selected_project, selected_project_details, active_project_id, active_project_banner],
        )
        open_project_button.click(
            open_project,
            inputs=selected_project,
            outputs=[active_project_id, active_project_banner, selected_project_details, api_key, open_project_result, connection_state],
        ).then(
            active_project_views,
            inputs=active_project_id,
            outputs=[document_table, processing_options_table, removable_pdf, question_set_selector, automatic_question_set, sampling_sections, saved_questions_table, question_set_summary],
        )
        selected_project.change(
            lambda project_id: (project_details(project_id), project_enabled_value(project_id)),
            inputs=selected_project,
            outputs=[selected_project_details, project_enabled],
        )
        update_project_enabled_button.click(
            update_project_enabled,
            inputs=[selected_project, project_enabled],
            outputs=[project_activation_result, project_table, selected_project_details],
        )
        create_button.click(
            create_project,
            inputs=[project_id, display_name, vehicle_name, manual_version, description],
            outputs=[
                result,
                project_table,
                project_id,
                display_name,
                vehicle_name,
                manual_version,
                description,
                selected_project,
                selected_project_details,
            ],
        )
        delete_button.click(
            delete_project,
            inputs=[selected_project, delete_confirmation, active_project_id],
            outputs=[
                result,
                project_table,
                selected_project,
                selected_project_details,
                active_project_id,
                active_project_banner,
                delete_confirmation,
            ],
        ).then(
            active_project_views,
            inputs=active_project_id,
            outputs=[document_table, processing_options_table, removable_pdf, question_set_selector, automatic_question_set, sampling_sections, saved_questions_table, question_set_summary],
        )
        upload_button.click(
            import_documents,
            inputs=[document_project, uploaded_files],
            outputs=[document_result, document_table, processing_options_table, removable_pdf],
        )
        preprocess_button.click(
            preprocess_documents,
            inputs=[document_project, processing_options_table],
            outputs=[document_result, document_table, processing_options_table, removable_pdf],
        )
        index_button.click(build_index, inputs=document_project, outputs=[document_result, indexing_log])
        document_refresh_button.click(
            document_view,
            inputs=document_project,
            outputs=[document_table, processing_options_table, removable_pdf],
        )
        remove_pdf_button.click(
            remove_document,
            inputs=[document_project, removable_pdf, remove_pdf_confirmation],
            outputs=[document_result, document_table, processing_options_table, removable_pdf, remove_pdf_confirmation],
        )
        autoqa_generate_button.click(
            generate_automatic_qa,
            inputs=[
                automatic_project, autoqa_questions_per_pdf, autoqa_parallel_generation,
                autoqa_generation_model, autoqa_method,
            ],
            outputs=[
                autoqa_result, autoqa_questions_table, autoqa_question_set_state,
            ],
        )
        autoqa_import_button.click(
            import_automatic_question_set,
            inputs=[automatic_project, autoqa_import_file],
            outputs=[autoqa_result, autoqa_questions_table, autoqa_question_set_state],
        )
        autoqa_test_button.click(
            run_automatic_qa,
            inputs=[
                automatic_project, autoqa_question_set_state, autoqa_questions_table, autoqa_answer_model,
                autoqa_judge_model, autoqa_method, autoqa_answer_concurrency,
            ],
            outputs=[autoqa_result, autoqa_summary, autoqa_table],
        )
        autoqa_export_button.click(
            export_automatic_question_set,
            inputs=[automatic_project, autoqa_question_set_state, autoqa_questions_table],
            outputs=[autoqa_result, autoqa_json_export],
        )
        ask_button.click(
            ask_question,
            inputs=[query_project, question, query_method],
            outputs=[
                query_answer,
                query_summary,
                query_evidence_table,
                query_evidence_selector,
                query_evidence_detail,
                query_context,
                query_evidence_state,
                query_question_state,
                query_answer_state,
            ],
        )
        query_evidence_selector.input(
            evidence_markdown,
            inputs=[query_evidence_selector, query_evidence_state],
            outputs=query_evidence_detail,
        )
        clear_question_button.click(
            lambda: ("", "", "", [], gr.Dropdown(choices=[]), "", {}, [], "", ""),
            outputs=[
                question,
                query_answer,
                query_summary,
                query_evidence_table,
                query_evidence_selector,
                query_evidence_detail,
                query_context,
                query_evidence_state,
                query_question_state,
                query_answer_state,
            ],
        )
        create_question_set_button.click(
            create_saved_question_set,
            inputs=[query_project, question_set_name, question_set_description],
            outputs=[question_set_result, question_set_selector, saved_questions_table, question_set_summary],
        )
        refresh_question_sets_button.click(
            lambda project_id: saved_question_set_view(project_id, None),
            inputs=query_project,
            outputs=[question_set_selector, saved_questions_table, question_set_summary],
        )
        load_question_set_button.click(
            load_saved_question_set,
            inputs=[query_project, question_set_selector],
            outputs=[question_set_result, question_set_selector, saved_questions_table, question_set_summary],
        )
        add_qa_to_question_set_button.click(
            add_current_answer_to_question_set,
            inputs=[
                query_project,
                question_set_selector,
                question,
                query_question_state,
                query_answer_state,
                query_answer,
                query_method,
            ],
            outputs=[question_set_result, question_set_selector, saved_questions_table, question_set_summary],
        )
        automatic_project_refresh.click(
            lambda project_id: gr.Dropdown(choices=question_set_choices(project_id), value=None),
            inputs=automatic_project,
            outputs=automatic_question_set,
        )
        run_automatic_button.click(
            run_automatic_evaluation,
            inputs=[
                automatic_project,
                automatic_question_set,
                automatic_top_k,
                automatic_rerun_answers,
                automatic_only_failures,
                automatic_evidence_limit,
            ],
            outputs=[
                automatic_result,
                automatic_summary,
                automatic_table,
                automatic_item_selector,
                automatic_items_state,
            ],
        ).then(
            automatic_evaluation_detail,
            inputs=[automatic_item_selector, automatic_items_state],
            outputs=automatic_item_json,
        )
        automatic_item_selector.input(
            automatic_evaluation_detail,
            inputs=[automatic_item_selector, automatic_items_state],
            outputs=automatic_item_json,
        )
        export_automatic_button.click(
            export_automatic_evaluation,
            inputs=[automatic_project, automatic_question_set],
            outputs=[automatic_result, automatic_json_export, automatic_csv_export],
        )
        sampling_project_refresh.click(
            sampling_section_choices,
            inputs=sampling_project,
            outputs=sampling_sections,
        )
        scan_sources_button.click(
            scan_source_samples,
            inputs=[
                sampling_project,
                sampling_sections,
                sampling_page_from,
                sampling_page_to,
                sampling_content_type,
                sampling_minimum_characters,
            ],
            outputs=[
                sampling_result,
                sampling_table,
                sampling_preview_selector,
                sampling_preview,
                sampling_values_state,
            ],
        )
        sample_sources_button.click(
            create_source_sample_batch,
            inputs=[
                sampling_project,
                sampling_sections,
                sampling_page_from,
                sampling_page_to,
                sampling_content_type,
                sampling_minimum_characters,
                sampling_count,
                sampling_seed,
            ],
            outputs=[
                sampling_result,
                sampling_table,
                sampling_preview_selector,
                sampling_preview,
                sampling_values_state,
                sampling_batch_state,
            ],
        )
        sampling_preview_selector.input(
            source_sample_detail,
            inputs=[sampling_preview_selector, sampling_values_state],
            outputs=sampling_preview,
        )
        export_samples_button.click(
            export_source_samples,
            inputs=[sampling_project, sampling_batch_state],
            outputs=[sampling_result, sampling_json_export, sampling_csv_export],
        )
        generate_questions_button.click(
            generate_questions,
            inputs=[
                sampling_project,
                sampling_batch_state,
                generation_count,
                generation_difficulty,
                generation_source_limit,
            ],
            outputs=[
                generation_result,
                generated_question_table,
                generated_question_selector,
                generated_question_editor,
                generated_answer_editor,
                generated_status,
                generated_evidence,
                generated_values_state,
                generation_batch_state,
            ],
        )
        generated_question_selector.input(
            generated_question_detail,
            inputs=[generated_question_selector, generated_values_state],
            outputs=[
                generated_question_editor,
                generated_answer_editor,
                generated_status,
                generated_evidence,
            ],
        )
        save_generated_question_button.click(
            update_generated_question,
            inputs=[
                sampling_project,
                generation_batch_state,
                generated_question_selector,
                generated_question_editor,
                generated_answer_editor,
                generated_status,
            ],
            outputs=[
                generation_result,
                generated_question_table,
                generated_question_selector,
                generated_values_state,
                generation_batch_state,
            ],
        )
        export_generated_questions_button.click(
            export_generated_question_set,
            inputs=[sampling_project, generation_batch_state],
            outputs=[generation_result, generated_question_export],
        )
        test_connection_button.click(
            test_connection,
            inputs=[api_base_url, api_key],
            outputs=connection_result,
        )
        save_connection_button.click(
            save_connection,
            inputs=[api_base_url, api_key, chat_model, embedding_model],
            outputs=[connection_result, api_base_url, api_key, chat_model, embedding_model],
        ).then(connection_status, outputs=connection_state)
    return demo


def document_row(document: DocumentInfo) -> list[str | int | None]:
    return [
        document.filename,
        document.pages,
        document.size_bytes,
        document.status,
        document.empty_pages,
        document.error_pages,
        document.error,
    ]


def main() -> None:
    create_app().queue(default_concurrency_limit=1).launch(js=INDEXING_LOG_AUTOSCROLL_JS)


if __name__ == "__main__":
    main()
