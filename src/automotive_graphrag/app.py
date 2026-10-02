"""Gradio entry point for the v1.0 project management interface."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import gradio as gr

from .connections import ConnectionSettings
from .documents import DocumentInfo, DocumentService
from .indexing import IndexingService
from .projects import ProjectError, ProjectStore


PROJECT_COLUMNS = ["Project ID", "顯示名稱", "文件數", "索引狀態", "更新時間"]
DOCUMENT_COLUMNS = ["檔名", "頁數", "大小 (bytes)", "前處理狀態", "空白頁", "錯誤頁", "錯誤"]


def create_app(project_root: str | Path | None = None) -> gr.Blocks:
    store = ProjectStore(project_root or os.environ.get("PROJECTS_ROOT", "projects"))
    connections = ConnectionSettings(store.root)
    documents = DocumentService(store)
    indexing = IndexingService(store, connection_settings=connections)

    def connection_status() -> str:
        return f"目前狀態：Base URL `{connections.get_api_base_url()}`；API Key {connections.masked_api_key()}"

    def save_connection(api_base_url: str, api_key: str):
        try:
            connections.save(api_base_url, api_key)
        except ProjectError as exc:
            return f"❌ {exc}", api_base_url, api_key
        return f"✅ 連線設定已儲存。{connection_status()}", api_base_url, api_key

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

    def import_documents(project_id: str | None, files: list[str] | None):
        if not project_id:
            return "❌ 請先選擇專案", []
        try:
            imported = documents.import_pdfs(project_id, files or [])
        except ProjectError as exc:
            return f"❌ {exc}", document_rows(project_id)
        return f"✅ 已匯入 {len(files or [])} 份 PDF", [document_row(item) for item in imported]

    def preprocess_documents(
        project_id: str | None,
        header_ignore_percent: float,
        footer_ignore_percent: float,
    ):
        if not project_id:
            return "❌ 請先選擇專案", []
        try:
            report = documents.preprocess(project_id, header_ignore_percent, footer_ignore_percent)
        except (ProjectError, OSError) as exc:
            return f"❌ {exc}", document_rows(project_id)
        message = (
            f"✅ 前處理完成：成功 {report.successful_pages} 頁、"
            f"無文字 {report.empty_pages} 頁、錯誤 {report.error_pages} 頁"
        )
        return message, document_rows(project_id)

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
            indexing_log = gr.Textbox(label="建圖日誌", lines=12, interactive=False)

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
        document_project.change(document_rows, inputs=document_project, outputs=document_table)
        upload_button.click(
            import_documents,
            inputs=[document_project, uploaded_files],
            outputs=[document_result, document_table],
        )
        preprocess_button.click(
            preprocess_documents,
            inputs=[document_project, header_ignore_percent, footer_ignore_percent],
            outputs=[document_result, document_table],
        )
        index_button.click(build_index, inputs=document_project, outputs=[document_result, indexing_log])
        document_refresh_button.click(document_rows, inputs=document_project, outputs=document_table)
        test_connection_button.click(
            test_connection,
            inputs=[api_base_url, api_key],
            outputs=connection_result,
        )
        save_connection_button.click(
            save_connection,
            inputs=[api_base_url, api_key],
            outputs=[connection_result, api_base_url, api_key],
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
