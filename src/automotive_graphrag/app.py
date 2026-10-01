"""Gradio entry point for the v1.0 project management interface."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .documents import DocumentInfo, DocumentService
from .projects import ProjectError, ProjectStore


PROJECT_COLUMNS = ["Project ID", "顯示名稱", "文件數", "索引狀態", "更新時間"]
DOCUMENT_COLUMNS = ["檔名", "頁數", "大小 (bytes)", "前處理狀態", "空白頁", "錯誤頁", "錯誤"]


def create_app(project_root: str | Path | None = None) -> gr.Blocks:
    store = ProjectStore(project_root or os.environ.get("PROJECTS_ROOT", "projects"))
    documents = DocumentService(store)

    def refresh_projects() -> list[list[str | int]]:
        return store.table_rows()

    def project_choices() -> list[tuple[str, str]]:
        return [(project.display_name, project.project_id) for project in store.list()]

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

    def preprocess_documents(project_id: str | None):
        if not project_id:
            return "❌ 請先選擇專案", []
        try:
            report = documents.preprocess(project_id)
        except (ProjectError, OSError) as exc:
            return f"❌ {exc}", document_rows(project_id)
        message = (
            f"✅ 前處理完成：成功 {report.successful_pages} 頁、"
            f"無文字 {report.empty_pages} 頁、錯誤 {report.error_pages} 頁"
        )
        return message, document_rows(project_id)

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
            )
        selector = gr.Dropdown(choices=project_choices(), value=project.project_id)
        return f"✅ 已建立專案 {project.project_id}", store.table_rows(), "", "", "", "", "", selector

    with gr.Blocks(title="汽車維修 GraphRAG 平台") as demo:
        gr.Markdown("# 汽車維修 GraphRAG 平台")
        with gr.Tab("專案列表"):
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

        with gr.Tab("文件與建圖"):
            document_project = gr.Dropdown(choices=project_choices(), label="專案")
            uploaded_files = gr.File(file_count="multiple", file_types=[".pdf"], type="filepath", label="匯入 PDF")
            with gr.Row():
                upload_button = gr.Button("上傳")
                preprocess_button = gr.Button("開始前處理", variant="primary")
                document_refresh_button = gr.Button("重新整理")
            document_result = gr.Markdown()
            document_table = gr.Dataframe(
                headers=DOCUMENT_COLUMNS,
                interactive=False,
                datatype=["str", "number", "number", "str", "number", "number", "str"],
                label="Documents",
            )

        refresh_button.click(refresh_projects, outputs=project_table)
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
                document_project,
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
            inputs=document_project,
            outputs=[document_result, document_table],
        )
        document_refresh_button.click(document_rows, inputs=document_project, outputs=document_table)
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
