"""Gradio entry point for the v1.0 project management interface."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from .projects import ProjectError, ProjectStore


PROJECT_COLUMNS = ["Project ID", "顯示名稱", "文件數", "索引狀態", "更新時間"]


def create_app(project_root: str | Path | None = None) -> gr.Blocks:
    store = ProjectStore(project_root or os.environ.get("PROJECTS_ROOT", "projects"))

    def refresh_projects() -> list[list[str | int]]:
        return store.table_rows()

    def create_project(
        project_id: str,
        display_name: str,
        vehicle_name: str,
        manual_version: str,
        description: str,
    ) -> tuple[str, list[list[str | int]], str, str, str, str, str]:
        try:
            project = store.create(
                project_id=project_id,
                display_name=display_name,
                vehicle_name=vehicle_name,
                manual_version=manual_version,
                description=description,
            )
        except ProjectError as exc:
            return f"❌ {exc}", store.table_rows(), project_id, display_name, vehicle_name, manual_version, description
        return f"✅ 已建立專案 {project.project_id}", store.table_rows(), "", "", "", "", ""

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

        refresh_button.click(refresh_projects, outputs=project_table)
        create_button.click(
            create_project,
            inputs=[project_id, display_name, vehicle_name, manual_version, description],
            outputs=[result, project_table, project_id, display_name, vehicle_name, manual_version, description],
        )
    return demo


def main() -> None:
    create_app().queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
