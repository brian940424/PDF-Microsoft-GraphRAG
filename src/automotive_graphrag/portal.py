"""Focused end-user vehicle support portal."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import gradio as gr

from .cases import CaseService
from .connections import ConnectionSettings
from .downloads import stage_downloads
from .projects import Project, ProjectError, ProjectStore
from .querying import QueryResult, QueryService


EVIDENCE_COLUMNS = ["Rank", "Evidence", "PDF", "Page", "Chunk ID", "Score"]


class UserPortalService:
    """Validate user-facing project selection before dispatching a query."""

    def __init__(self, projects: ProjectStore, queries: QueryService | None = None) -> None:
        self.projects = projects
        self.queries = queries or QueryService(projects)

    def choices(self) -> list[tuple[str, str]]:
        return [
            (
                f"{project.display_name}｜{project.vehicle_name}｜{project.manual_version}",
                project.project_id,
            )
            for project in self.projects.queryable_projects()
        ]

    def selected_project(self, project_id: str) -> Project:
        project = self.projects.get(project_id)
        if project.status != "INDEXED" or not project.enabled:
            raise ProjectError("此專案目前未啟用或索引不可用，請聯絡管理者")
        return project

    def ask(self, project_id: str, question: str) -> QueryResult:
        self.selected_project(project_id)
        return self.queries.ask(project_id, question, "local")


def create_portal_app(project_root: str | Path | None = None) -> gr.Blocks:
    store = ProjectStore(project_root or os.environ.get("PROJECTS_ROOT", "projects"))
    connections = ConnectionSettings(store.root)
    queries = QueryService(store, connection_settings=connections)
    portal = UserPortalService(store, queries)
    cases = CaseService(store, queries)

    def evidence_detail(evidence_id: str | None, values: list[dict[str, object]]) -> str:
        for item in values or []:
            if item.get("evidence_id") == evidence_id:
                return (
                    f"### [{item['evidence_id']}] {item['document_id']} — 第 {item['page']} 頁\n\n"
                    f"章節：{item['section_name']} (`{item['section_id']}`)  \n"
                    f"Chunk：`{item['chunk_id']}`\n\n{item['text']}"
                )
        return ""

    def project_view(project_id: str | None):
        empty = ("", "", [], gr.Dropdown(choices=[]), "", [], "", gr.Button(interactive=False))
        if not project_id:
            return ("### 尚未選擇車型／專案", *empty)
        try:
            project = portal.selected_project(project_id)
        except ProjectError as exc:
            return ("### 專案無法使用", f"❌ {exc}", "", [], gr.Dropdown(choices=[]), "", [], "", gr.Button(interactive=False))
        header = (
            f"### {project.display_name}\n"
            f"車型：`{project.vehicle_name}`｜手冊版本：`{project.manual_version}`"
        )
        return (header, *empty[:-1], gr.Button(interactive=True))

    def ask(project_id: str | None, question: str):
        empty = ("", [], gr.Dropdown(choices=[]), "", [], "")
        yield ("⏳ 正在查詢維修手冊，完成後將顯示回答與 Evidence…", *empty)
        if not project_id:
            yield ("❌ 請先選擇車型／專案", *empty)
            return
        try:
            result = portal.ask(project_id, question)
        except ProjectError as exc:
            yield (f"❌ {exc}", *empty)
            return
        if result.status != "COMPLETED":
            yield (f"❌ 查詢失敗：{result.error}", *empty)
            return
        values = [asdict(item) for item in result.evidence]
        rows = [
            [item.rank, item.evidence_id, item.document_id, item.page, item.chunk_id, item.score]
            for item in result.evidence
        ]
        selected = result.evidence[0].evidence_id if result.evidence else None
        selector = gr.Dropdown(
            choices=[item.evidence_id for item in result.evidence],
            value=selected,
        )
        detail = evidence_detail(selected, values) if selected else ""
        warning = "" if result.evidence else "\n\n> ⚠️ 此回答沒有可回連的來源證據，請人工確認。"
        yield (
            f"✅ 回答完成｜耗時 {result.duration_seconds:.3f} 秒",
            f"{result.answer}{warning}",
            rows,
            selector,
            detail,
            values,
            result.query_id,
        )

    def save_case(project_id: str | None, query_id: str, note: str):
        if not project_id or not query_id:
            return "❌ 請先完成一筆問答"
        try:
            record = cases.save_query(project_id, query_id, note)
        except ProjectError as exc:
            return f"❌ {exc}"
        return f"✅ 已加入案例紀錄 {record.case_id}"

    def export_cases(project_id: str | None):
        if not project_id:
            return "❌ 請先選擇車型／專案", None, None
        try:
            json_path, csv_path = cases.export(project_id)
            json_path, csv_path = stage_downloads((json_path, csv_path))
        except ProjectError as exc:
            return f"❌ {exc}", None, None
        return "✅ 已匯出案例紀錄", str(json_path), str(csv_path)

    with gr.Blocks(title="汽車維修手冊問答助手") as demo:
        gr.Markdown("# 汽車維修手冊問答助手")
        gr.Markdown("選擇車型與手冊版本後輸入維修問題；回答下方會列出採用的原始手冊證據。")
        with gr.Row():
            project_selector = gr.Dropdown(
                choices=portal.choices(),
                label="車型／專案",
                allow_custom_value=True,
            )
            refresh_button = gr.Button("重新整理可用專案")
        project_header = gr.Markdown("### 尚未選擇車型／專案")
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="維修問題", lines=6)
                with gr.Row():
                    ask_button = gr.Button("送出問題", variant="primary", interactive=False)
                    clear_button = gr.Button("清除")
                status = gr.Markdown()
            with gr.Column():
                answer = gr.Markdown(label="系統回答")
        evidence_state = gr.State([])
        query_id_state = gr.State("")
        evidence_table = gr.Dataframe(
            headers=EVIDENCE_COLUMNS,
            interactive=False,
            datatype=["number", "str", "str", "number", "str", "number"],
            label="回答採用的 Evidence",
        )
        evidence_selector = gr.Dropdown(label="查看證據全文")
        evidence_markdown = gr.Markdown()
        gr.Markdown("### 案例紀錄")
        case_note = gr.Textbox(label="案例備註", lines=2)
        with gr.Row():
            save_case_button = gr.Button("將本次問答加入案例")
            export_cases_button = gr.Button("匯出此專案案例")
        case_result = gr.Markdown()
        with gr.Row():
            cases_json = gr.File(label="案例 JSON")
            cases_csv = gr.File(label="案例 CSV")
        with gr.Accordion("維修步驟與記錄表（未來擴充）", open=False):
            gr.Markdown("此區預留給結構化維修步驟、技師處置與完工紀錄。")

        refresh_button.click(
            lambda current: gr.Dropdown(
                choices=portal.choices(),
                value=current if current in {value for _, value in portal.choices()} else None,
                allow_custom_value=True,
            ),
            inputs=project_selector,
            outputs=project_selector,
        )
        project_selector.change(
            project_view,
            inputs=project_selector,
            outputs=[
                project_header,
                status,
                answer,
                evidence_table,
                evidence_selector,
                evidence_markdown,
                evidence_state,
                query_id_state,
                ask_button,
            ],
        )
        ask_button.click(
            ask,
            inputs=[project_selector, question],
            outputs=[status, answer, evidence_table, evidence_selector, evidence_markdown, evidence_state, query_id_state],
        )
        evidence_selector.input(
            evidence_detail,
            inputs=[evidence_selector, evidence_state],
            outputs=evidence_markdown,
        )
        clear_button.click(
            lambda: ("", "", "", [], gr.Dropdown(choices=[]), "", [], "", ""),
            outputs=[question, status, answer, evidence_table, evidence_selector, evidence_markdown, evidence_state, query_id_state, case_note],
        )
        save_case_button.click(
            save_case,
            inputs=[project_selector, query_id_state, case_note],
            outputs=case_result,
        )
        export_cases_button.click(
            export_cases,
            inputs=project_selector,
            outputs=[case_result, cases_json, cases_csv],
        )
    return demo


def main() -> None:
    create_portal_app().queue(default_concurrency_limit=1).launch()


if __name__ == "__main__":
    main()
