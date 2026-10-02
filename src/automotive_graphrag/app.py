"""Gradio entry point for the v1.0 project management interface."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import gradio as gr

from .connections import ALLOWED_CHAT_MODELS, ALLOWED_EMBEDDING_MODELS, ConnectionSettings
from .documents import DocumentInfo, DocumentService
from .indexing import IndexingService
from .projects import ProjectError, ProjectStore
from .querying import QueryService
from .question_sets import QuestionSet, QuestionSetService
from .reviews import ReviewService


PROJECT_COLUMNS = ["Project ID", "顯示名稱", "文件數", "索引狀態", "更新時間"]
DOCUMENT_COLUMNS = ["檔名", "頁數", "大小 (bytes)", "前處理狀態", "空白頁", "錯誤頁", "錯誤"]
BATCH_COLUMNS = ["題號", "問題", "狀態", "耗時 (秒)", "錯誤"]
EVIDENCE_COLUMNS = ["Rank", "Evidence", "PDF", "Page", "Chunk ID", "Score"]


def create_app(project_root: str | Path | None = None) -> gr.Blocks:
    store = ProjectStore(project_root or os.environ.get("PROJECTS_ROOT", "projects"))
    connections = ConnectionSettings(store.root)
    documents = DocumentService(store)
    indexing = IndexingService(store, connection_settings=connections)
    querying = QueryService(store, connection_settings=connections)
    question_sets = QuestionSetService(store, querying)
    reviews = ReviewService(store, question_sets)

    def connection_status() -> str:
        return (
            f"目前狀態：Base URL `{connections.get_api_base_url()}`；API Key {connections.masked_api_key()}；"
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

    def query_project_choices() -> list[tuple[str, str]]:
        return [
            (project.display_name, project.project_id)
            for project in store.list()
            if project.status == "INDEXED"
        ]

    def refresh_query_projects(project_id: str | None):
        choices = query_project_choices()
        available_ids = {value for _, value in choices}
        return gr.Dropdown(choices=choices, value=project_id if project_id in available_ids else None)

    def ask_question(project_id: str | None, question: str, method: str):
        if not project_id:
            return "", "❌ 請先選擇已完成建圖的專案", [], gr.Dropdown(choices=[]), "", {}, []
        try:
            query_result = querying.ask(project_id, question, method)
        except ProjectError as exc:
            return "", f"❌ {exc}", [], gr.Dropdown(choices=[]), "", {}, []
        summary = (
            f"狀態：{query_result.status}｜方法：{query_result.method}｜"
            f"耗時：{query_result.duration_seconds:.3f} 秒｜執行時間：{query_result.completed_at}"
        )
        if query_result.status == "FAILED":
            return "", f"❌ {summary}｜{query_result.error}", [], gr.Dropdown(choices=[]), "", {}, []
        evidence_values = [asdict(item) for item in query_result.evidence]
        evidence_rows = [
            [item.rank, item.evidence_id, item.document_id, item.page, item.chunk_id, item.score]
            for item in query_result.evidence
        ]
        if query_result.evidence:
            references = "\n".join(
                f"- [{item.evidence_id}] {item.document_id}，第 {item.page} 頁，Chunk `{item.chunk_id}`"
                for item in query_result.evidence
            )
            answer = f"{query_result.answer}\n\n---\n### Evidence\n{references}"
            selected = query_result.evidence[0].evidence_id
            detail = evidence_markdown(selected, evidence_values)
        else:
            answer = f"{query_result.answer}\n\n> ⚠️ 未驗證：Query Context 中沒有可回連的來源證據。"
            selected = None
            detail = ""
        selector = gr.Dropdown(
            choices=[item.evidence_id for item in query_result.evidence],
            value=selected,
        )
        return answer, f"✅ {summary}", evidence_rows, selector, detail, query_result.context, evidence_values

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

    def batch_rows(question_set: QuestionSet | None) -> list[list[str | float | None]]:
        if question_set is None:
            return []
        return [
            [item.question_id, item.question, item.status, item.duration_seconds, item.error]
            for item in question_set.questions
        ]

    def batch_summary(question_set: QuestionSet) -> str:
        summary = question_sets.summary(question_set)
        return (
            f"題目集：{question_set.name}｜總計 {summary.total}｜完成 {summary.completed}｜"
            f"失敗 {summary.failed}｜待執行 {summary.pending}｜執行中 {summary.running}"
        )

    def question_set_view(project_id: str | None, question_set_id: str | None):
        choices = question_set_choices(project_id)
        if not project_id or not question_set_id:
            return gr.Dropdown(choices=choices), [], gr.Dropdown(choices=[]), ""
        try:
            question_set = question_sets.get(project_id, question_set_id)
        except ProjectError:
            return gr.Dropdown(choices=choices), [], gr.Dropdown(choices=[]), ""
        question_choices = [(item.question_id, item.question_id) for item in question_set.questions]
        return (
            gr.Dropdown(choices=choices, value=question_set_id),
            batch_rows(question_set),
            gr.Dropdown(choices=question_choices, value=[]),
            batch_summary(question_set),
        )

    def import_question_set(project_id: str | None, source: str | None):
        if not project_id or not source:
            return "❌ 請選擇專案與題目集 JSON", {"errors": ["缺少專案或檔案"]}, *question_set_view(project_id, None)
        try:
            imported = question_sets.import_file(project_id, source)
        except ProjectError as exc:
            return f"❌ {exc}", {"errors": str(exc).splitlines()}, *question_set_view(project_id, None)
        view = question_set_view(project_id, imported.question_set_id)
        return f"✅ 已匯入 {len(imported.questions)} 題", {}, *view

    def run_batch(
        project_id: str | None,
        question_set_id: str | None,
        method: str,
        selected_question_ids: list[str] | None,
        mode: str,
    ):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", [], ""
        if mode == "selected" and not selected_question_ids:
            return "❌ 請至少選擇一題", batch_rows(question_sets.get(project_id, question_set_id)), ""
        try:
            result = question_sets.run(
                project_id,
                question_set_id,
                method,
                selected_question_ids=selected_question_ids if mode == "selected" else None,
                resume=mode == "resume",
            )
        except ProjectError as exc:
            return f"❌ {exc}", [], ""
        return f"✅ 批次執行完成｜{batch_summary(result)}", batch_rows(result), batch_summary(result)

    def export_batch(project_id: str | None, question_set_id: str | None):
        if not project_id or not question_set_id:
            return "❌ 請先選擇專案與題目集", None, None
        try:
            json_path, csv_path = question_sets.export(project_id, question_set_id)
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出 JSON 與 CSV", str(json_path), str(csv_path)

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
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出人工評測 JSON 與 CSV", str(json_path), str(csv_path)

    def project_details(project_id: str | None) -> dict[str, str]:
        if not project_id:
            return {}
        try:
            return asdict(store.get(project_id))
        except ProjectError:
            return {}

    def refresh_project_views(project_id: str | None):
        available_ids = {project.project_id for project in store.list()}
        selected = project_id if project_id in available_ids else None
        choices = project_choices()
        return (
            store.table_rows(),
            gr.Dropdown(choices=choices, value=selected),
            project_details(selected),
            gr.Dropdown(choices=choices, value=selected),
        )

    def delete_project(project_id: str | None, confirmed: bool):
        if not project_id:
            return "❌ 請先選擇專案", *refresh_project_views(None), False
        if not confirmed:
            return "❌ 請勾選刪除確認", *refresh_project_views(project_id), False
        try:
            store.delete(project_id)
        except ProjectError as exc:
            return f"❌ {exc}", *refresh_project_views(project_id), False
        return f"✅ 已刪除專案 {project_id}", *refresh_project_views(None), False

    def document_rows(project_id: str | None) -> list[list[str | int | None]]:
        if not project_id:
            return []
        return [document_row(document) for document in documents.list_documents(project_id)]

    def document_view(project_id: str | None):
        rows = document_rows(project_id)
        choices = [row[0] for row in rows]
        return rows, gr.Dropdown(choices=choices, value=None)

    def import_documents(project_id: str | None, files: list[str] | None):
        if not project_id:
            return "❌ 請先選擇專案", [], gr.Dropdown(choices=[])
        try:
            imported = documents.import_pdfs(project_id, files or [])
        except ProjectError as exc:
            rows, selector = document_view(project_id)
            return f"❌ {exc}", rows, selector
        rows = [document_row(item) for item in imported]
        return f"✅ 已匯入 {len(files or [])} 份 PDF", rows, gr.Dropdown(choices=[row[0] for row in rows])

    def preprocess_documents(
        project_id: str | None,
        header_ignore_percent: float,
        footer_ignore_percent: float,
    ):
        if not project_id:
            return "❌ 請先選擇專案", [], gr.Dropdown(choices=[])
        try:
            report = documents.preprocess(project_id, header_ignore_percent, footer_ignore_percent)
        except (ProjectError, OSError) as exc:
            rows, selector = document_view(project_id)
            return f"❌ {exc}", rows, selector
        message = (
            f"✅ 前處理完成：成功 {report.successful_pages} 頁、"
            f"無文字 {report.empty_pages} 頁、錯誤 {report.error_pages} 頁"
        )
        rows, selector = document_view(project_id)
        return message, rows, selector

    def remove_document(project_id: str | None, filename: str | None, confirmed: bool):
        if not project_id or not filename:
            rows, selector = document_view(project_id)
            return "❌ 請先選擇專案與 PDF", rows, selector, False
        if not confirmed:
            rows, selector = document_view(project_id)
            return "❌ 請勾選移除確認", rows, selector, False
        try:
            documents.remove_pdf(project_id, filename)
        except ProjectError as exc:
            rows, selector = document_view(project_id)
            return f"❌ {exc}", rows, selector, False
        rows, selector = document_view(project_id)
        return f"✅ 已移除 {filename}；請重新執行前處理與建圖", rows, selector, False

    def build_index(project_id: str | None):
        if not project_id:
            return "❌ 請先選擇專案", ""
        try:
            result = indexing.build(project_id)
        except ProjectError as exc:
            return f"❌ {exc}", ""
        icon = "✅" if result.status == "INDEXED" else "❌"
        summary = f"{icon} {result.status}｜耗時 {result.duration_seconds:.1f} 秒｜{result.last_message}"
        log_path = store.path_for(project_id) / "graphrag" / result.log_file
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        return summary, log

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
            selector = gr.Dropdown(choices=project_choices())
            return (
                f"❌ {exc}",
                store.table_rows(),
                project_id,
                display_name,
                vehicle_name,
                manual_version,
                description,
                selector,
                selector,
                {},
            )
        selector = gr.Dropdown(choices=project_choices(), value=project.project_id)
        return (
            f"✅ 已建立專案 {project.project_id}",
            store.table_rows(),
            "",
            "",
            "",
            "",
            "",
            selector,
            selector,
            asdict(project),
        )

    with gr.Blocks(title="汽車維修 GraphRAG 平台") as demo:
        gr.Markdown("# 汽車維修 GraphRAG 平台")
        with gr.Tab("專案設定"):
            selected_project = gr.Dropdown(choices=project_choices(), label="選擇專案紀錄")
            selected_project_details = gr.JSON(label="專案資料")
            project_table = gr.Dataframe(
                headers=PROJECT_COLUMNS,
                value=refresh_projects,
                interactive=False,
                datatype=["str", "str", "number", "str", "str"],
                label="Projects",
            )
            with gr.Row():
                refresh_button = gr.Button("重新整理")
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
            api_key = gr.Textbox(label="API Key", type="password", placeholder="輸入 API Key")
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
            document_project = gr.Dropdown(choices=project_choices(), label="專案")
            uploaded_files = gr.File(file_count="multiple", file_types=[".pdf"], type="filepath", label="匯入 PDF")
            with gr.Row():
                header_ignore_percent = gr.Number(
                    label="忽略頁首高度 (%)",
                    value=0,
                    minimum=0,
                    maximum=99,
                )
                footer_ignore_percent = gr.Number(
                    label="忽略頁尾高度 (%)",
                    value=0,
                    minimum=0,
                    maximum=99,
                )
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
            with gr.Row():
                removable_pdf = gr.Dropdown(label="選擇要移除的 PDF")
                remove_pdf_confirmation = gr.Checkbox(label="我確認要移除選取的 PDF")
                remove_pdf_button = gr.Button("移除 PDF", variant="stop")
            indexing_log = gr.Textbox(label="建圖日誌", lines=12, interactive=False)

        with gr.Tab("問答測試"):
            with gr.Row():
                query_project = gr.Dropdown(choices=query_project_choices(), label="已建圖專案")
                query_project_refresh = gr.Button("重新整理專案")
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
                    query_answer = gr.Markdown(label="系統回答")
                    query_summary = gr.Markdown(label="查詢資訊")
            gr.Markdown("### Evidence 詳情")
            query_evidence_state = gr.State([])
            query_evidence_table = gr.Dataframe(
                headers=EVIDENCE_COLUMNS,
                interactive=False,
                datatype=["number", "str", "str", "number", "str", "number"],
                label="Top Evidence",
            )
            query_evidence_selector = gr.Dropdown(label="選取證據全文")
            query_evidence_detail = gr.Markdown()
            query_context = gr.JSON(label="原始 GraphRAG Query Context")
            gr.Markdown("## 題目集批次問答")
            with gr.Row():
                batch_project = gr.Dropdown(choices=project_choices(), label="專案")
                batch_project_refresh = gr.Button("重新整理專案")
                question_set_selector = gr.Dropdown(label="題目集")
            question_set_file = gr.File(file_types=[".json"], type="filepath", label="匯入 question_set.json")
            with gr.Row():
                import_question_set_button = gr.Button("匯入並驗證")
                batch_method = gr.Dropdown(
                    choices=[("Local", "local"), ("Global", "global"), ("DRIFT", "drift"), ("Basic", "basic")],
                    value="local",
                    label="查詢方法",
                )
            question_set_validation = gr.JSON(label="驗證錯誤")
            selected_batch_questions = gr.Dropdown(multiselect=True, label="選擇題目（選題執行用）")
            with gr.Row():
                run_all_button = gr.Button("執行全部", variant="primary")
                run_selected_button = gr.Button("執行選取題目")
                resume_batch_button = gr.Button("繼續未完成")
                export_batch_button = gr.Button("匯出結果")
            batch_result = gr.Markdown()
            batch_status = gr.Markdown()
            batch_table = gr.Dataframe(
                headers=BATCH_COLUMNS,
                interactive=False,
                datatype=["str", "str", "str", "number", "str"],
                label="批次題目狀態",
            )
            with gr.Row():
                batch_json_export = gr.File(label="JSON 匯出")
                batch_csv_export = gr.File(label="CSV 匯出")

        with gr.Tab("人工檢查"):
            with gr.Row():
                review_project = gr.Dropdown(choices=project_choices(), label="專案")
                review_project_refresh = gr.Button("重新整理專案")
                review_question_set = gr.Dropdown(label="題目集")
            review_progress = gr.Markdown("尚未選擇題目集")
            review_question = gr.Markdown()
            gr.Markdown("### 系統回答")
            review_answer = gr.Markdown()
            review_label = gr.Radio(
                choices=[
                    ("正確", "correct"),
                    ("部分正確", "partially_correct"),
                    ("錯誤", "incorrect"),
                    ("資料不足", "insufficient"),
                ],
                label="正確性",
            )
            review_note = gr.Textbox(label="審查備註", lines=3)
            review_index = gr.State(0)
            with gr.Row():
                previous_review_button = gr.Button("上一題")
                save_next_review_button = gr.Button("儲存並下一題", variant="primary")
                review_number = gr.Number(label="跳到題號", value=1, minimum=1, maximum=1)
                jump_review_button = gr.Button("跳轉")
                export_reviews_button = gr.Button("匯出評測結果")
            review_result = gr.Markdown()
            with gr.Row():
                review_json_export = gr.File(label="人工評測 JSON")
                review_csv_export = gr.File(label="人工評測 CSV")

        refresh_button.click(
            refresh_project_views,
            inputs=selected_project,
            outputs=[project_table, selected_project, selected_project_details, document_project],
        )
        selected_project.change(project_details, inputs=selected_project, outputs=selected_project_details)
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
                document_project,
                selected_project_details,
            ],
        )
        delete_button.click(
            delete_project,
            inputs=[selected_project, delete_confirmation],
            outputs=[
                result,
                project_table,
                selected_project,
                selected_project_details,
                document_project,
                delete_confirmation,
            ],
        )
        document_project.change(
            document_view,
            inputs=document_project,
            outputs=[document_table, removable_pdf],
        )
        upload_button.click(
            import_documents,
            inputs=[document_project, uploaded_files],
            outputs=[document_result, document_table, removable_pdf],
        )
        preprocess_button.click(
            preprocess_documents,
            inputs=[document_project, header_ignore_percent, footer_ignore_percent],
            outputs=[document_result, document_table, removable_pdf],
        )
        index_button.click(build_index, inputs=document_project, outputs=[document_result, indexing_log])
        document_refresh_button.click(
            document_view,
            inputs=document_project,
            outputs=[document_table, removable_pdf],
        )
        remove_pdf_button.click(
            remove_document,
            inputs=[document_project, removable_pdf, remove_pdf_confirmation],
            outputs=[document_result, document_table, removable_pdf, remove_pdf_confirmation],
        )
        query_project_refresh.click(
            refresh_query_projects,
            inputs=query_project,
            outputs=query_project,
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
            ],
        )
        query_evidence_selector.input(
            evidence_markdown,
            inputs=[query_evidence_selector, query_evidence_state],
            outputs=query_evidence_detail,
        )
        clear_question_button.click(
            lambda: ("", "", "", [], gr.Dropdown(choices=[]), "", {}, []),
            outputs=[
                question,
                query_answer,
                query_summary,
                query_evidence_table,
                query_evidence_selector,
                query_evidence_detail,
                query_context,
                query_evidence_state,
            ],
        )
        batch_project_refresh.click(
            lambda: gr.Dropdown(choices=project_choices()),
            outputs=batch_project,
        )
        batch_project.change(
            lambda project_id: question_set_view(project_id, None),
            inputs=batch_project,
            outputs=[question_set_selector, batch_table, selected_batch_questions, batch_status],
        )
        question_set_selector.input(
            question_set_view,
            inputs=[batch_project, question_set_selector],
            outputs=[question_set_selector, batch_table, selected_batch_questions, batch_status],
        )
        import_question_set_button.click(
            import_question_set,
            inputs=[batch_project, question_set_file],
            outputs=[
                batch_result,
                question_set_validation,
                question_set_selector,
                batch_table,
                selected_batch_questions,
                batch_status,
            ],
        )
        run_all_button.click(
            lambda project_id, question_set_id, method, selected: run_batch(
                project_id, question_set_id, method, selected, "all"
            ),
            inputs=[batch_project, question_set_selector, batch_method, selected_batch_questions],
            outputs=[batch_result, batch_table, batch_status],
        )
        run_selected_button.click(
            lambda project_id, question_set_id, method, selected: run_batch(
                project_id, question_set_id, method, selected, "selected"
            ),
            inputs=[batch_project, question_set_selector, batch_method, selected_batch_questions],
            outputs=[batch_result, batch_table, batch_status],
        )
        resume_batch_button.click(
            lambda project_id, question_set_id, method, selected: run_batch(
                project_id, question_set_id, method, selected, "resume"
            ),
            inputs=[batch_project, question_set_selector, batch_method, selected_batch_questions],
            outputs=[batch_result, batch_table, batch_status],
        )
        export_batch_button.click(
            export_batch,
            inputs=[batch_project, question_set_selector],
            outputs=[batch_result, batch_json_export, batch_csv_export],
        )
        review_project_refresh.click(
            lambda: gr.Dropdown(choices=project_choices()),
            outputs=review_project,
        )
        review_project.change(
            lambda project_id: gr.Dropdown(choices=question_set_choices(project_id), value=None),
            inputs=review_project,
            outputs=review_question_set,
        )
        review_question_set.input(
            lambda project_id, question_set_id: review_view(project_id, question_set_id, 0),
            inputs=[review_project, review_question_set],
            outputs=[
                review_question,
                review_answer,
                review_label,
                review_note,
                review_index,
                review_number,
                review_progress,
            ],
        )
        previous_review_button.click(
            lambda project_id, question_set_id, index, label, note: navigate_review(
                project_id, question_set_id, index, label, note, -1
            ),
            inputs=[review_project, review_question_set, review_index, review_label, review_note],
            outputs=[
                review_question,
                review_answer,
                review_label,
                review_note,
                review_index,
                review_number,
                review_progress,
                review_result,
            ],
        )
        save_next_review_button.click(
            lambda project_id, question_set_id, index, label, note: navigate_review(
                project_id, question_set_id, index, label, note, 1, True
            ),
            inputs=[review_project, review_question_set, review_index, review_label, review_note],
            outputs=[
                review_question,
                review_answer,
                review_label,
                review_note,
                review_index,
                review_number,
                review_progress,
                review_result,
            ],
        )
        jump_review_button.click(
            jump_review,
            inputs=[
                review_project,
                review_question_set,
                review_index,
                review_label,
                review_note,
                review_number,
            ],
            outputs=[
                review_question,
                review_answer,
                review_label,
                review_note,
                review_index,
                review_number,
                review_progress,
                review_result,
            ],
        )
        export_reviews_button.click(
            export_reviews,
            inputs=[review_project, review_question_set],
            outputs=[review_result, review_json_export, review_csv_export],
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
    create_app().queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
