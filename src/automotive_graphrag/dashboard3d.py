"""3D 知識圖譜儀表板：把專案的 GraphRAG 索引輸出呈現為明亮等角風格的 three.js 場景。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .dashboard_world import WORLD_JS
from .projects import Project

THREE_VERSION = "0.160.0"
DASHBOARD_HEIGHT_PX = 820
DESCRIPTION_LIMIT = 320
LINK_DESCRIPTION_LIMIT = 160
SUMMARY_LIMIT = 420
RECENT_EVENT_LIMIT = 12


def build_dashboard_data(project: Project, project_path: str | Path) -> dict[str, Any] | None:
    """讀取索引輸出並整理成前端場景需要的資料；尚未建圖時回傳 None。"""
    output = Path(project_path) / "graphrag" / "output"
    if not (output / "entities.parquet").exists():
        return None
    entities = pd.read_parquet(output / "entities.parquet")
    relationships = _read_optional(output / "relationships.parquet")
    communities = _read_optional(output / "communities.parquet")
    reports = _read_optional(output / "community_reports.parquet")

    regions, region_of_entity = _top_level_regions(communities, reports)
    index_of_title: dict[str, int] = {}
    nodes = []
    for position, row in enumerate(entities.itertuples(index=False)):
        index_of_title[str(row.title)] = position
        nodes.append({
            "t": str(row.title),
            "y": str(row.type or "UNKNOWN"),
            "d": _clip(row.description, DESCRIPTION_LIMIT),
            "deg": int(row.degree or 0),
            "freq": int(row.frequency or 0),
            "r": region_of_entity.get(str(row.id), -1),
        })
    links = []
    for row in relationships.itertuples(index=False):
        source, target = index_of_title.get(str(row.source)), index_of_title.get(str(row.target))
        if source is None or target is None:
            continue
        links.append([source, target, round(float(row.weight or 0), 2), _clip(row.description, LINK_DESCRIPTION_LIMIT)])

    events = _recent_queries(Path(project_path) / "runs" / "queries.jsonl")
    index_stats = _read_json(output / "stats.json")
    type_counts = entities["type"].fillna("UNKNOWN").value_counts()
    return {
        "project": {"id": project.project_id, "name": project.display_name, "status": project.status},
        "stats": {
            "entities": len(entities),
            "relationships": len(links),
            "communities": len(communities),
            "regions": len(regions),
            "documents": _row_count(output / "documents.parquet"),
            "text_units": _row_count(output / "text_units.parquet"),
            "index_runtime_s": round(float(index_stats.get("total_runtime") or 0), 1),
            "queries": events["total"],
            "queries_completed": events["completed"],
        },
        "types": [{"name": str(name), "count": int(count)} for name, count in type_counts.items()],
        "regions": regions,
        "nodes": nodes,
        "links": links,
        "events": events["recent"],
        "alerts": _alerts(nodes, regions, events["recent"]),
    }


def render_dashboard_html(data: dict[str, Any]) -> str:
    """產生自足的 HTML 頁面（three.js 由 CDN 載入）。"""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        DASHBOARD_TEMPLATE
        .replace("__THREE_VERSION__", THREE_VERSION)
        .replace("__WORLD_JS__", WORLD_JS)
        .replace("__DASHBOARD_DATA__", payload)
    )


def dashboard_iframe(project: Project | None, project_path: str | Path | None) -> str:
    """供 Gradio gr.HTML 使用：以 iframe srcdoc 嵌入，腳本才能在頁面中執行。"""
    if project is None or project_path is None:
        return _placeholder("請先在「0 專案與連線設定」開啟專案。")
    data = build_dashboard_data(project, project_path)
    if data is None:
        return _placeholder(f"專案 {html.escape(project.project_id)} 尚未建圖，完成「1 文件與建圖」後即可檢視 3D 儀表板。")
    page = html.escape(render_dashboard_html(data), quote=True)
    return (
        f'<iframe title="3D 儀表板" srcdoc="{page}" '
        f'style="width:100%;height:{DASHBOARD_HEIGHT_PX}px;border:0;border-radius:16px;background:#e9eef7" '
        'allow="fullscreen"></iframe>'
    )


def _top_level_regions(
    communities: pd.DataFrame, reports: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if communities.empty:
        return [], {}
    reports_by_community = {}
    if not reports.empty:
        reports_by_community = {int(row.community): row for row in reports.itertuples(index=False)}
    regions: list[dict[str, Any]] = []
    region_of_entity: dict[str, int] = {}
    top_level = communities[communities["level"] == communities["level"].min()]
    for row in top_level.sort_values("community").itertuples(index=False):
        report = reports_by_community.get(int(row.community))
        region_index = len(regions)
        entity_ids = [str(entity_id) for entity_id in (row.entity_ids if row.entity_ids is not None else [])]
        for entity_id in entity_ids:
            region_of_entity.setdefault(entity_id, region_index)
        regions.append({
            "c": int(row.community),
            "t": str(report.title) if report is not None else str(row.title),
            "rank": round(float(report.rank), 1) if report is not None and pd.notna(report.rank) else None,
            "s": _clip(report.summary, SUMMARY_LIMIT) if report is not None else "",
            "n": len(entity_ids),
        })
    return regions, region_of_entity


def _alerts(nodes: list[dict[str, Any]], regions: list[dict[str, Any]], recent: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """鈴鐺清單：未完成的問答、孤立實體、缺少報告的社群、未歸入社群的實體，嚴重的排前面。"""
    alerts = [
        {"tone": "bad", "title": f"問答未完成：{event['s'] or '未知'}", "body": event["q"], "sel": {"kind": "query", "k": k}}
        for k, event in enumerate(recent)
        if event["s"] != "COMPLETED"
    ]
    isolated = [i for i, node in enumerate(nodes) if node["deg"] == 0]
    if isolated:
        alerts.append({
            "tone": "warn", "title": f"{len(isolated)} 個孤立實體", "body": f"例如 {nodes[isolated[0]]['t']}，沒有任何關係",
            "sel": {"kind": "node", "i": isolated[0]},
        })
    unreported = [r for r, region in enumerate(regions) if region["rank"] is None]
    if unreported:
        alerts.append({
            "tone": "warn", "title": f"{len(unreported)} 個社群缺少報告", "body": regions[unreported[0]]["t"],
            "sel": {"kind": "region", "r": unreported[0]},
        })
    frontier = sum(1 for node in nodes if node["r"] == -1)
    if frontier and regions:
        alerts.append({
            "tone": "info", "title": f"{frontier} 個實體未歸入社群", "body": "位於地圖的邊境分區", "sel": {"kind": "region", "r": -1},
        })
    return alerts


def _recent_queries(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"total": 0, "completed": 0, "recent": []}
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    recent = [
        {
            "q": _clip(record.get("question"), 80),
            "m": str(record.get("method") or ""),
            "s": str(record.get("status") or ""),
            "at": str(record.get("completed_at") or record.get("started_at") or ""),
            "sec": record.get("duration_seconds"),
        }
        for record in records[-RECENT_EVENT_LIMIT:]
    ]
    return {
        "total": len(records),
        "completed": sum(1 for record in records if record.get("status") == "COMPLETED"),
        "recent": list(reversed(recent)),
    }


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _row_count(path: Path) -> int:
    return len(pd.read_parquet(path, columns=["id"])) if path.exists() else 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _placeholder(message: str) -> str:
    return (
        '<div style="padding:48px 24px;text-align:center;border:1px dashed #c9d4ea;background:#f4f6fb;'
        f'border-radius:16px;color:#6b7280">{message}</div>'
    )


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphRAG 知識圖譜地圖</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<style>
  :root {
    --bg: #e9eef7; --card: #ffffff; --text: #111827; --muted: #6b7280; --line: #e7ebf3; --soft: #f4f6fb;
    --blue: #3563f0; --blue-soft: #e6edff; --good: #22a06b; --bad: #e5484d;
    --radius: 16px; --shadow: 0 1px 2px rgba(17, 24, 39, 0.04), 0 10px 30px rgba(31, 52, 102, 0.1);
    --gutter: 16px; --nav: 60px; --detail-w: 340px; --bottom-h: 128px; --tabs-h: 268px;
    --font: Inter, "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: var(--bg); color: var(--text);
    font: 14px/1.45 var(--font); -webkit-font-smoothing: antialiased; }
  button { font: inherit; color: inherit; background: none; border: 0; cursor: pointer; text-align: left; padding: 0; }
  .mono { font-family: ui-monospace, "SF Mono", "Cascadia Mono", monospace; }
  .muted { color: var(--muted); font-weight: 400; }
  #scene { position: fixed; inset: 0; }
  .card { position: absolute; z-index: 20; background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow); }

  /* ---------- 導覽列 ---------- */
  .topnav { position: absolute; z-index: 30; top: 0; left: 0; right: 0; height: var(--nav); display: flex; align-items: center; gap: 14px;
    padding: 0 var(--gutter); background: rgba(255, 255, 255, 0.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(8px); }
  .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 19px; letter-spacing: -0.02em; margin-right: 18px; white-space: nowrap; }
  .logo-mark { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; color: #fff;
    background: linear-gradient(145deg, #4f7bff, #2447c4); box-shadow: 0 6px 14px rgba(53, 99, 240, 0.35); }
  .search { position: relative; display: flex; align-items: center; gap: 10px; width: min(400px, 32vw); height: 40px; padding: 0 12px 0 14px;
    border: 1px solid var(--line); border-radius: 12px; color: var(--muted); background: #fff; }
  .search input { flex: 1; border: 0; outline: none; font: inherit; color: var(--text); background: none; min-width: 0; }
  .search:focus-within { border-color: #b9c7f5; box-shadow: 0 0 0 3px var(--blue-soft); }
  .search kbd { font: 12px var(--font); padding: 1px 7px; border: 1px solid var(--line); border-radius: 6px; color: var(--muted); }
  .search-results { position: absolute; top: 46px; left: 0; right: 0; background: #fff; border-radius: 12px; box-shadow: var(--shadow); padding: 6px; display: none; }
  .search-results.open { display: block; }
  .search-results button { width: 100%; display: flex; justify-content: space-between; gap: 12px; padding: 8px 10px; border-radius: 8px; color: var(--text); }
  .search-results button:hover, .search-results button.on { background: var(--soft); }
  .search-results span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .search-results small { color: var(--muted); white-space: nowrap; }
  .site-pill { margin-left: auto; display: flex; align-items: center; gap: 10px; height: 44px; padding: 0 16px 0 6px; border: 1px solid var(--line); border-radius: 12px; max-width: 300px; }
  .site-pill b { display: block; font-size: 14px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .site-pill small { display: block; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .site-pill > div { min-width: 0; }
  .site-badge { display: grid; place-items: center; width: 32px; height: 32px; border-radius: 9px; background: var(--blue); color: #fff; flex: none; }
  .live { display: flex; align-items: center; gap: 8px; height: 32px; padding: 0 14px; border-radius: 999px; background: #e7f7ee; color: #15803d; font-weight: 600; white-space: nowrap; }
  .live i { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.18); }
  .bell { position: relative; display: grid; place-items: center; width: 40px; height: 40px; border-radius: 10px; color: #374151; flex: none; }
  .bell:hover, .bell.on { background: var(--soft); color: var(--blue); }
  .bell i { position: absolute; top: 3px; right: 2px; min-width: 18px; height: 18px; padding: 0 4px; border-radius: 999px; border: 2px solid #fff;
    background: var(--bad); color: #fff; font: 700 10px/14px var(--font); font-style: normal; text-align: center; }
  .bell i.warn { background: #f59e0b; }
  .bell-menu { z-index: 40; top: calc(var(--nav) - 4px); right: var(--gutter); width: 340px; max-height: calc(100% - var(--nav) - 24px); overflow-y: auto; padding: 8px; display: none; scrollbar-width: thin; }
  .bell-menu.open { display: block; }
  .dd-head { display: flex; justify-content: space-between; align-items: baseline; padding: 8px 8px 4px; font-size: 13px; }
  .notice { width: 100%; display: flex; align-items: flex-start; gap: 10px; padding: 8px; border-radius: 10px; }
  .notice:hover { background: var(--soft); }
  .notice-icon { flex: none; display: grid; place-items: center; width: 26px; height: 26px; border-radius: 8px; color: #fff; background: var(--blue); }
  .notice.bad .notice-icon { background: var(--bad); }
  .notice.warn .notice-icon { background: #f59e0b; }
  .notice-text { flex: 1; min-width: 0; }
  .notice-text b { display: block; font-size: 13px; font-weight: 600; }
  .notice-text small { display: block; color: var(--muted); font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* ---------- KPI ---------- */
  .kpis { position: absolute; z-index: 20; top: calc(var(--nav) + 14px); left: var(--gutter); display: flex; gap: 12px; }
  .kpi { position: relative; display: flex; align-items: center; gap: 12px; padding: 12px 16px 12px 12px; min-width: 176px; }
  .kpi-icon { display: grid; place-items: center; width: 42px; height: 42px; border-radius: 12px; color: var(--blue); background: var(--blue-soft); flex: none; }
  .kpi-label { font-size: 13px; color: #374151; }
  .kpi-value { display: flex; align-items: baseline; gap: 6px; white-space: nowrap; }
  .kpi-value b { font-size: 22px; letter-spacing: -0.02em; }
  .kpi-value .unit { color: var(--muted); font-size: 13px; }
  .delta { font-size: 12px; font-weight: 600; padding: 1px 7px; border-radius: 999px; margin-left: 2px; }
  .delta.good { color: #15803d; background: #e3f6ea; }
  .delta.bad { color: #c43d43; background: #fde8e8; }
  .kpi-sub { display: block; font-size: 12px; color: var(--muted); white-space: nowrap; }

  /* ---------- 地圖控制 ---------- */
  .map-controls { top: calc(var(--nav) + 14px); right: calc(var(--gutter) + var(--detail-w) + 12px); display: flex; flex-direction: column; padding: 6px; gap: 2px; border-radius: 14px; }
  .map-controls button { display: grid; place-items: center; width: 36px; height: 36px; border-radius: 10px; color: #374151; }
  .map-controls button:hover { background: var(--soft); color: var(--blue); }
  .map-controls button.on { background: var(--blue-soft); color: var(--blue); }
  .map-controls button:last-child { border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; margin-top: 4px; padding-top: 4px; }
  .map-controls button:nth-child(3) { border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; margin-top: 4px; padding-top: 4px; }

  /* ---------- 右側詳細卡 ---------- */
  .detail { top: calc(var(--nav) + 14px); right: var(--gutter); width: var(--detail-w); bottom: calc(var(--tabs-h) + var(--gutter) + 12px); display: flex; flex-direction: column; }
  .detail-scroll { overflow-y: auto; overflow-x: hidden; padding: 18px; scrollbar-width: thin; }
  .detail-close { position: absolute; top: 14px; right: 14px; display: none; place-items: center; width: 30px; height: 30px; border-radius: 8px; border: 1px solid var(--line); color: var(--muted); background: #fff; z-index: 1; }
  .detail-close.show { display: grid; }
  .detail-header { display: flex; gap: 14px; align-items: flex-start; padding-bottom: 14px; border-bottom: 1px solid var(--line); padding-right: 30px; }
  .detail-icon { flex: none; display: grid; place-items: center; width: 48px; height: 48px; border-radius: 12px; color: var(--blue); background: linear-gradient(145deg, #eef3ff, #dfe7ff); }
  .detail-header > div { min-width: 0; }
  .eyebrow { display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: var(--blue); text-transform: uppercase; }
  .detail-header h2 { margin: 2px 0; font-size: 17px; letter-spacing: -0.01em; line-height: 1.3; word-break: break-word; }
  .detail-header .muted { font-size: 13px; }
  .status-row { display: flex; align-items: center; gap: 10px; padding: 12px 0; flex-wrap: wrap; }
  .pill { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 8px; font-size: 12.5px; font-weight: 600; white-space: nowrap; }
  .pill.big { padding: 5px 12px; font-size: 13.5px; }
  .pill-good { color: #15803d; background: #e3f6ea; }
  .pill-bad { color: #c43d43; background: #fde8e8; }
  .pill-blue { color: #2c56e0; background: #e6edff; }
  .pill-grey { color: #5b6476; background: #eef1f6; }
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat { padding: 10px 12px; border-radius: 12px; background: var(--soft); min-width: 0; }
  .stat-label { display: block; font-size: 12px; color: var(--muted); }
  .stat b { font-size: 17px; }
  .track, .mini-track { height: 5px; margin-top: 8px; border-radius: 3px; background: #dde3ee; overflow: hidden; }
  .track i, .mini-track i { display: block; height: 100%; border-radius: 3px; background: linear-gradient(90deg, #3563f0, #22a06b); }
  .mini-track i { background: var(--blue); }
  h4 { margin: 16px 0 8px; font-size: 13.5px; display: flex; justify-content: space-between; }
  .inventory-row { width: 100%; display: grid; grid-template-columns: 30px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 7px 4px; border-bottom: 1px solid var(--line); border-radius: 6px; }
  .inventory-row:hover { background: var(--soft); }
  .inventory-row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .inv-icon { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 8px; }
  .follow-button { display: flex; align-items: center; gap: 8px; width: 100%; margin-bottom: 12px; padding: 9px 12px; border-radius: 10px; border: 1px solid var(--line); font-weight: 600; color: #374151; }
  .follow-button:hover { border-color: #b9c7f5; color: var(--blue); }
  .follow-button.on { background: var(--blue); border-color: var(--blue); color: #fff; }
  .follow-button kbd { margin-left: auto; font: 11px var(--font); padding: 0 6px; border: 1px solid currentColor; border-radius: 5px; opacity: 0.6; }
  .prose { margin: 0 0 6px; color: #374151; font-size: 13.5px; white-space: pre-wrap; word-break: break-word; }

  /* ---------- 左下：建圖流程 ---------- */
  .tracker { left: var(--gutter); bottom: var(--gutter); height: var(--bottom-h); width: min(900px, calc(100% - 2 * var(--gutter) - var(--detail-w) - 16px));
    display: flex; gap: 16px; padding: 14px 14px 14px 20px; }
  .tracker-main { flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: space-between; }
  .tracker-head { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
  .tracker-title { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 15px; }
  .tracker-title svg { color: var(--blue); }
  .tracker-head .muted { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .stepper { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(5, 1fr); }
  .stepper li { position: relative; display: flex; flex-direction: column; align-items: center; text-align: center; gap: 1px; }
  .stepper li::before { content: ""; position: absolute; top: 16px; left: -50%; width: 100%; height: 3px; background: #dfe5f0; z-index: 0; }
  .stepper li:first-child::before { display: none; }
  .stepper li.done::before, .stepper li.current::before { background: var(--blue); }
  .step-dot { position: relative; z-index: 1; display: grid; place-items: center; width: 34px; height: 34px; border-radius: 50%; background: #eef1f6; color: #9aa3b2; margin-bottom: 3px; }
  .stepper li.done .step-dot { background: var(--blue); color: #fff; }
  .stepper li.current .step-dot { background: #fff; color: var(--blue); box-shadow: 0 0 0 3px var(--blue), 0 0 0 8px var(--blue-soft); }
  .stepper b { font-size: 13px; }
  .stepper small { font-size: 12px; color: var(--muted); }
  .next-card { width: 280px; flex: none; padding: 12px 14px; border-radius: 12px; background: var(--soft); display: flex; flex-direction: column; gap: 6px; }
  .next-head { display: flex; justify-content: space-between; align-items: center; }
  .pager { display: flex; align-items: center; gap: 4px; font-size: 12px; color: var(--muted); }
  .pager button { display: grid; place-items: center; width: 22px; height: 22px; border-radius: 6px; }
  .pager button:hover { background: #fff; }
  .next-card p { margin: 0; font-weight: 600; font-size: 13.5px; line-height: 1.45; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; overflow-wrap: anywhere; -webkit-box-orient: vertical; cursor: pointer; }
  .next-card .pills { display: flex; gap: 6px; }

  /* ---------- 右下：分頁表格 ---------- */
  .tabs-card { right: var(--gutter); bottom: var(--gutter); width: var(--detail-w); height: var(--tabs-h); display: flex; flex-direction: column; padding: 12px 14px 8px; }
  .segmented { display: flex; gap: 2px; padding: 3px; border-radius: 10px; background: var(--soft); align-self: flex-start; }
  .segmented button { padding: 5px 12px; border-radius: 8px; font-weight: 600; color: #374151; font-size: 13.5px; }
  .segmented button span { color: var(--muted); font-weight: 500; margin-left: 2px; }
  .segmented button.on { background: #fff; box-shadow: 0 1px 3px rgba(17, 24, 39, 0.1); color: var(--text); }
  .filters { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; padding: 10px 0 4px; }
  .pill.filter { cursor: pointer; }
  .pill.filter.off { opacity: 0.35; }
  .pill i { width: 8px; height: 8px; border-radius: 2px; }
  .table { overflow-y: auto; flex: 1; scrollbar-width: thin; margin-top: 6px; }
  .table-row { width: 100%; display: flex; align-items: center; gap: 10px; padding: 8px 4px; border-bottom: 1px solid var(--line); }
  .table-row:hover { background: var(--soft); }
  .cell-id { flex: none; width: 44px; font-weight: 700; font-size: 13px; color: var(--blue); }
  .cell-id.mono { width: 82px; font-weight: 500; font-size: 12px; color: var(--muted); }
  .cell-main { flex: 1; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cell-num { flex: none; width: 44px; text-align: right; color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums; }
  .mini-track { width: 56px; margin: 0; flex: none; }
  .chev { color: #9aa3b2; flex: none; }
  .pad { padding: 10px 4px; }

  /* ---------- 3D 場景內的標籤 ---------- */
  .zone-label { display: flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 8px; background: #fff; box-shadow: 0 2px 8px rgba(31, 52, 102, 0.12);
    font: 600 12px var(--font); color: #1f2937; white-space: nowrap; max-width: 220px; pointer-events: auto; cursor: pointer; user-select: none; }
  .zone-label b { font-weight: 600; overflow: hidden; text-overflow: ellipsis; }
  .zone-label span { color: var(--muted); font-weight: 500; }
  .zone-label.on { background: var(--blue); color: #fff; }
  .zone-label.on span { color: #dbe4ff; }
  .tooltip3d { position: absolute; z-index: 25; display: none; padding: 6px 12px; border-radius: 10px; background: #fff; box-shadow: var(--shadow); font: 12.5px var(--font); color: var(--text);
    white-space: nowrap; max-width: 340px; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  /* 選取物件上方的藍色標籤（WareTrack 的 tag3d），每幀投影到螢幕座標 */
  .tag3d { position: absolute; left: 0; top: 0; z-index: 24; display: none; padding: 4px 9px; border-radius: 7px; background: var(--blue); color: #fff; box-shadow: var(--shadow);
    font: 700 11.5px ui-monospace, "SF Mono", "Cascadia Mono", monospace; white-space: nowrap; max-width: 280px; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  .tag3d small { margin-left: 6px; font: 500 11px var(--font); opacity: 0.8; }
  #error { position: absolute; inset: 0; z-index: 40; display: none; align-items: center; justify-content: center; text-align: center; padding: 24px; color: var(--muted); background: var(--bg); }

  @media (max-width: 1320px) { .kpi:nth-child(4) { display: none; } .next-card { display: none; } }
  @media (max-width: 1120px) { :root { --detail-w: 300px; } .kpi { min-width: 0; } .kpi:nth-child(3), .live { display: none; } .logo span:last-child { display: none; } }
  @media (max-width: 860px) {
    .search, .site-pill, .kpis, .map-controls, .tracker { display: none; }
    .bell-menu { left: var(--gutter); width: auto; }
    .detail { left: var(--gutter); width: auto; top: auto; height: 34%; bottom: calc(var(--tabs-h) + var(--gutter) + 12px); }
    .tabs-card { left: var(--gutter); width: auto; }
  }
</style>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE_VERSION__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE_VERSION__/examples/jsm/"
} }
</script>
</head>
<body>
<div id="scene"></div>
<nav class="topnav">
  <button class="logo" id="logo"><span class="logo-mark" data-icon="box" data-size="22"></span><span>GraphSite</span></button>
  <div class="search">
    <span data-icon="search"></span>
    <input id="search" placeholder="搜尋實體、社群…" autocomplete="off">
    <kbd>/</kbd>
    <div class="search-results" id="searchResults"></div>
  </div>
  <div class="site-pill">
    <span class="site-badge" data-icon="branch" data-size="14"></span>
    <div><b id="projectName"></b><small id="projectMeta"></small></div>
  </div>
  <div class="live" id="live"></div>
  <button class="bell" id="bell" aria-label="通知" aria-expanded="false"></button>
</nav>
<div class="card bell-menu" id="bellMenu" role="dialog" aria-label="通知"></div>
<div class="kpis" id="kpis"></div>
<div class="card map-controls" id="mapControls"></div>
<aside class="card detail">
  <button class="detail-close" id="detailClose" aria-label="關閉" data-icon="close" data-size="16"></button>
  <div class="detail-scroll" id="detail"></div>
</aside>
<section class="card tracker" id="tracker"></section>
<section class="card tabs-card" id="tabs"></section>
<div class="tooltip3d" id="tooltip"></div>
<div class="tag3d" id="tag3d"></div>
<div id="error">無法載入 three.js（需要可連線至 cdn.jsdelivr.net）。</div>
<script type="application/json" id="data">__DASHBOARD_DATA__</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// 城鎮、車流與物流倉庫的共用模擬：由 progress-dashboard/lib/world.ts 產生（dashboard_world.py），包在命名空間內避免撞名
const World = (() => {
__WORLD_JS__
return { packBlocks, createTraffic, stepTraffic, vehiclePose, bayState, layoutTown, streetParts, signalLamps, lightState };
})();

const DATA = JSON.parse(document.getElementById("data").textContent);
const S = DATA.stats;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const FRONTIER = -1;

const ICONS = {
  search: "M11 4a7 7 0 1 0 4.2 12.6l4.1 4.1 1.4-1.4-4.1-4.1A7 7 0 0 0 11 4Zm0 2a5 5 0 1 1 0 10 5 5 0 0 1 0-10Z",
  plus: "M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z",
  minus: "M5 11h14v2H5z",
  rotateLeft: "M12 5a7 7 0 1 1-6.6 9.3l1.9-.6A5 5 0 1 0 8.5 8.5L11 11H4V4l3.1 3.1A7 7 0 0 1 12 5Z",
  rotateRight: "M12 5a7 7 0 1 0 6.6 9.3l-1.9-.6a5 5 0 1 1-1.2-5.2L13 11h7V4l-3.1 3.1A7 7 0 0 0 12 5Z",
  home: "M12 3 3 10.5V21h7v-6h4v6h7V10.5L12 3Zm0 2.6 7 5.8V19h-3v-6H8v6H5v-7.6l7-5.8Z",
  branch: "M7 3a3 3 0 0 0-1 5.8v6.4A3 3 0 1 0 8 15.2V13a2 2 0 0 1 2-2h4a4 4 0 0 0 4-4v-.2a3 3 0 1 0-2 0V7a2 2 0 0 1-2 2h-4a4 4 0 0 0-2 .5V8.8A3 3 0 0 0 7 3Z",
  box: "M12 2 3 7v10l9 5 9-5V7l-9-5Zm0 2.3L18.6 8 12 11.6 5.4 8 12 4.3ZM5 9.7l6 3.3v6.6l-6-3.3V9.7Zm8 9.9V13l6-3.3v6.6l-6 3.3Z",
  check: "M9.5 16.6 5.4 12.5 4 13.9l5.5 5.5L20 8.9l-1.4-1.4z",
  truck: "M2 6h12v9h1.2a3 3 0 0 1 5.6 0H22v-4l-3-4h-4V6H2v9h.2a3 3 0 0 1 5.6 0H14V6Zm16 9a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3ZM5 15a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Zm10-6h3l2 2.7V13h-5V9Z",
  crane: "M4 21V5h2v2h14v2H11l-3 3v9H4Zm2-12v1.2L7.2 9H6Zm11 1h2v6h-2v-6Z",
  flag: "M5 3h2v1h11l-2 4 2 4H7v9H5V3Z",
  chevronLeft: "M15.4 6.6 14 5.2 7.2 12l6.8 6.8 1.4-1.4L10 12z",
  chevronRight: "M8.6 6.6 10 5.2l6.8 6.8-6.8 6.8-1.4-1.4L14 12z",
  close: "M6.4 5 5 6.4 10.6 12 5 17.6 6.4 19l5.6-5.6 5.6 5.6 1.4-1.4-5.6-5.6L19 6.4 17.6 5 12 10.6z",
  pin: "M12 2a7 7 0 0 0-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 0 0-7-7Zm0 4a3 3 0 1 1 0 6 3 3 0 0 1 0-6Z",
  doc: "M6 2h8l6 6v14H6V2Zm7 1.5V9h5.5L13 3.5ZM8 12v2h8v-2H8Zm0 4v2h8v-2H8Z",
  bell: "M12 3a6 6 0 0 0-6 6v4l-2 3v1h16v-1l-2-3V9a6 6 0 0 0-6-6Zm-2 16a2 2 0 0 0 4 0h-4Z",
  play: "M8 5v14l11-7z",
  pause: "M7 5h4v14H7zm6 0h4v14h-4z",
  target: "M11 2h2v5h-2zm0 15h2v5h-2zM2 11h5v2H2zm15 0h5v2h-5zm-5-2a3 3 0 1 1 0 6a3 3 0 1 1 0-6Z",
};
const icon = (name, size = 18) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="${ICONS[name]}"/></svg>`;
document.querySelectorAll("[data-icon]").forEach((el) => { el.innerHTML = icon(el.dataset.icon, Number(el.dataset.size || 18)); });

const PALETTE = {
  blue: "#3563f0", blueLight: "#6f91ff", white: "#f8fafc", grey: "#9aa5b8", dark: "#2b3140",
  leaf: "#6cc785", leafDark: "#4fae6b", trunk: "#a8754a",
};
const TYPE_COLORS = ["#4570f5", "#1fb5a8", "#f6a821", "#8b5cf6", "#ec5a5f", "#ff8a3d", "#22a06b", "#64748b"];
const typeColor = {};
DATA.types.forEach((t, i) => { typeColor[t.name] = TYPE_COLORS[i % TYPE_COLORS.length]; });
const METHOD_COLORS = { local: "#3563f0", global: "#8b5cf6", drift: "#1fb5a8", basic: "#f6a821", auto: "#ff8a3d" };
const methodColor = (m) => METHOD_COLORS[m] || "#64748b";
const methodLabel = (m) => (m || "?").toUpperCase();
const regionTitle = (r) => r === FRONTIER ? "未歸入社群" : DATA.regions[r].t;
const shortTime = (iso) => iso ? iso.slice(5, 16).replace("T", " ") : "—";
const queryPill = (ev) => ev.s === "COMPLETED" ? '<span class="pill pill-good">完成</span>' : `<span class="pill pill-bad">${esc(ev.s || "未知")}</span>`;

// ---------- 鄰接與統計 ----------
const neighbors = DATA.nodes.map(() => []);
DATA.links.forEach(([s, t, w, d], li) => { neighbors[s].push({ o: t, w, d, li }); neighbors[t].push({ o: s, w, d, li }); });
neighbors.forEach((list) => list.sort((a, b) => b.w - a.w));
const maxDeg = Math.max(1, ...DATA.nodes.map((n) => n.deg));
const maxFreq = Math.max(1, ...DATA.nodes.map((n) => n.freq));
const isolated = DATA.nodes.filter((n) => n.deg === 0).length;
const avgDeg = DATA.nodes.length ? (DATA.links.length * 2 / DATA.nodes.length).toFixed(1) : "0";

// ---------- 版面：社群 = 分區，實體 = 地塊上的建築；分區以格狀重劃排進街廓（四周都是街道） ----------
const LOT = 3.4, ZONE_PADDING = 1.5, LABEL_SPACE = 1.6, BASE_Y = 0.12;
const groups = new Map();
DATA.nodes.forEach((n, i) => { if (!groups.has(n.r)) groups.set(n.r, []); groups.get(n.r).push(i); });
for (const members of groups.values()) members.sort((a, b) => DATA.nodes[b].deg - DATA.nodes[a].deg);
const blocks = [...groups.keys()]
  .sort((a, b) => (a === FRONTIER) - (b === FRONTIER) || groups.get(b).length - groups.get(a).length)
  .map((r) => {
    const count = groups.get(r).length, cols = Math.ceil(Math.sqrt(count)), rows = Math.ceil(count / cols);
    return { r, cols, w: cols * LOT + ZONE_PADDING * 2, d: rows * LOT + ZONE_PADDING * 2 + LABEL_SPACE };
  });
const packed = World.packBlocks(blocks);
const positions = new Array(DATA.nodes.length);
const zones = blocks.map((b, index) => {
  const { x, z } = packed.centers[index];
  const left = x - b.w / 2 + ZONE_PADDING, top = z - b.d / 2 + ZONE_PADDING + LABEL_SPACE;
  groups.get(b.r).forEach((nodeIndex, k) => {
    positions[nodeIndex] = new THREE.Vector3(left + (k % b.cols) * LOT + LOT / 2, BASE_Y, top + Math.floor(k / b.cols) * LOT + LOT / 2);
  });
  return { r: b.r, x, z, w: b.w, d: b.d };
});
const zoneOf = new Map(zones.map((zone) => [zone.r, zone]));
const road = packed.bounds; // 分區街廓的外框（最外圈街道中心線）
const radius = Math.hypot(road.w, road.d) / 2 + 6;

const footprint = (n) => 1.5 + 0.9 * Math.log2(n.freq + 1) / Math.log2(maxFreq + 1);
const heightOf = (n) => 0.7 + Math.log2(n.deg + 1) * 0.9;
const roofRise = (n) => footprint(n) * 0.85 * 0.3;
const roofTop = (i) => positions[i].y + heightOf(DATA.nodes[i]) + roofRise(DATA.nodes[i]);

// ---------- 場景 ----------
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (err) {
  $("error").style.display = "flex"; $("error").textContent = "此瀏覽器不支援 WebGL，無法顯示 3D 儀表板。";
  throw err;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
$("scene").appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
Object.assign(labelRenderer.domElement.style, { position: "absolute", top: "0", pointerEvents: "none" });
$("scene").appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#e9eef7");
const VIEW_DIRECTION = new THREE.Vector3(1, 0.95, 1).normalize();
const VIEW_DISTANCE = radius * 2 + 40;
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, VIEW_DISTANCE * 2 + radius * 2);
camera.position.copy(VIEW_DIRECTION).multiplyScalar(VIEW_DISTANCE);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.minPolarAngle = Math.PI * 0.12;
controls.maxPolarAngle = Math.PI * 0.38;
controls.screenSpacePanning = false;

scene.add(new THREE.AmbientLight("#ffffff", 1.15));
scene.add(new THREE.HemisphereLight("#ffffff", "#c9d3e6", 0.7));
const sun = new THREE.DirectionalLight("#ffffff", 1.5);
sun.position.set(-0.5 * radius, radius, 0.66 * radius);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.02;
const shadowRange = radius + 22; // 涵蓋道路外側的城鎮與倉庫
Object.assign(sun.shadow.camera, { left: -shadowRange, right: shadowRange, top: shadowRange, bottom: -shadowRange, far: radius * 4 + 60 });
scene.add(sun);

const mat = (color, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness: 0.75, metalness: 0.05, ...extra });
const box = (size, position, color) => {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), mat(color));
  mesh.position.set(...position); mesh.castShadow = true; mesh.receiveShadow = true;
  return mesh;
};
function plane(w, d, color, y, basic = false) {
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, d), basic ? new THREE.MeshBasicMaterial({ color }) : mat(color));
  mesh.rotation.x = -Math.PI / 2; mesh.position.y = y; mesh.receiveShadow = true; scene.add(mesh);
  return mesh;
}
plane(radius * 8, radius * 8, "#e3e9f4", -0.03);
plane(road.w + 44, road.d + 44, "#d9eedb", -0.02); // 城鎮綠地
plane(road.w, road.d, "#f1f4fa", -0.01); // 分區街廓的淺色鋪面

// ---------- 城市：路網與號誌、外圍住宅街廓、車流、物流倉庫（同 progress-dashboard） ----------
const truckEvents = DATA.events.slice(0, 8);
const traffic = World.createTraffic(packed, truckEvents.map((_, k) => `q${k}`));
const town = World.layoutTown(traffic, packed.parks);
const noRaycast = () => {};
const UP = new THREE.Vector3(0, 1, 0);
// 實例化繪製：上百棟建物與樹木只用少數幾次 draw call
function instanced(parts, geometry, shadows = true) {
  const mesh = new THREE.InstancedMesh(geometry, mat("#ffffff", { roughness: 0.8, metalness: 0.02 }), Math.max(1, parts.length));
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), c = new THREE.Color();
  parts.forEach((p, i) => {
    mesh.setMatrixAt(i, m.compose(new THREE.Vector3(p.x, p.y, p.z), q.setFromAxisAngle(UP, p.rotation), new THREE.Vector3(p.sx, p.sy, p.sz)));
    mesh.setColorAt(i, c.set(p.color));
  });
  mesh.count = parts.length; mesh.castShadow = shadows; mesh.receiveShadow = true; mesh.raycast = noRaycast;
  scene.add(mesh);
  return mesh;
}
const gableGeo = (() => { // 單位山形屋頂：底部 y=0、屋脊高 1、屋脊沿 x
  const shape = new THREE.Shape(); shape.moveTo(-0.5, 0); shape.lineTo(0.5, 0); shape.lineTo(0, 1); shape.closePath();
  const geo = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false }); geo.translate(0, 0, -0.5); geo.rotateY(Math.PI / 2);
  return geo;
})();
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const streets = World.streetParts(traffic);
instanced(streets.sidewalks, unitBox, false);
instanced(streets.asphalt, unitBox, false);
instanced(streets.markings, unitBox, false);
instanced(streets.poles, unitBox);
// 號誌燈：每個路口每個進入方向一盞，每幀依燈號變色
const lamps = World.signalLamps(traffic);
const LIGHT_COLORS = { green: new THREE.Color("#22c55e"), amber: new THREE.Color("#f59e0b"), red: new THREE.Color("#ef4444") };
const lampMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.17, 10, 8), new THREE.MeshBasicMaterial({ toneMapped: false }), Math.max(1, lamps.length));
lamps.forEach((lamp, i) => { lampMesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(lamp.x, lamp.y, lamp.z)); lampMesh.setColorAt(i, LIGHT_COLORS.red); });
lampMesh.count = lamps.length; lampMesh.raycast = noRaycast; scene.add(lampMesh);
function updateLamps() {
  if (!lamps.length) return;
  lamps.forEach((lamp, i) => lampMesh.setColorAt(i, LIGHT_COLORS[World.lightState(traffic, lamp.node, lamp.lane)]));
  lampMesh.instanceColor.needsUpdate = true;
}
instanced(town.lawns, unitBox, false);
instanced(town.boxes, unitBox);
instanced(town.roofs, gableGeo);
{ // 住宅街廓後院與公園的樹木
  const all = town.trees;
  const parts = (color) => all.map((t) => ({ x: t.x, y: 0, z: t.z, sx: t.scale, sy: t.scale, sz: t.scale, rotation: 0, color }));
  instanced(parts(PALETTE.trunk), new THREE.CylinderGeometry(0.12, 0.16, 0.7, 8).translate(0, 0.35, 0));
  instanced(parts(PALETTE.leafDark), new THREE.ConeGeometry(0.75, 1.6, 8).translate(0, 1.3, 0));
  instanced(parts(PALETTE.leaf), new THREE.ConeGeometry(0.55, 1.1, 8).translate(0, 1.95, 0));
}

// 一般車流：小客車與廂型車，雙向行駛、保持跟車距離
const CAR_COLORS = ["#ec5a5f", "#f6c544", "#f8fafc", "#1fb5a8", "#3a4256", "#6f91ff", "#ff8a3d"];
const wheelGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.16, 12), wheelMat = mat(PALETTE.dark);
const cars = traffic.vehicles.filter((v) => v.kind !== "truck").map((vehicle) => {
  const g = new THREE.Group(), color = CAR_COLORS[vehicle.color % CAR_COLORS.length];
  if (vehicle.kind === "van") {
    g.add(box([2.5, 1.05, 1.1], [0, 0.78, 0], color), box([0.06, 0.42, 0.9], [1.26, 1.0, 0], "#1c2433"),
      box([1.0, 0.36, 0.04], [0.55, 1.0, 0.56], "#1c2433"), box([1.0, 0.36, 0.04], [0.55, 1.0, -0.56], "#1c2433"));
  } else {
    g.add(box([2.1, 0.5, 1.0], [0, 0.5, 0], color), box([1.1, 0.42, 0.9], [-0.15, 0.95, 0], color), box([1.12, 0.3, 0.92], [-0.15, 0.97, 0], "#2b3550"));
  }
  for (const x of [-0.7, 0.7]) for (const z of [-0.5, 0.5]) {
    const wheel = new THREE.Mesh(wheelGeo, wheelMat); wheel.rotation.x = Math.PI / 2; wheel.position.set(x, 0.24, z); g.add(wheel);
  }
  g.traverse((o) => { o.raycast = noRaycast; });
  scene.add(g);
  return { vehicle, group: g };
});

// 物流倉庫：兩條穿越式卸貨道，卡車從一端捲門開進去、卸貨、從另一端開出來（門上燈號：綠＝空、黃＝進出、紅＝卸貨）
const depot = traffic.depot, depotDoors = [], depotLamps = [];
const LAMP = { free: "#24b46e", moving: "#f0a623", unloading: "#e0564b" };
const DOOR_HEIGHT = 2.6, WALL_HEIGHT = 3.3, OPENING = 2.3;
let depotSign = null;
if (depot) {
  for (const bay of depot.bays) { // 卸貨道路面
    const pos = [], index = [], half = 1.2;
    bay.path.forEach((p, k) => {
      const a = bay.path[Math.max(0, k - 1)], b = bay.path[Math.min(bay.path.length - 1, k + 1)];
      const len = Math.hypot(b.x - a.x, b.z - a.z) || 1, nx = -(b.z - a.z) / len, nz = (b.x - a.x) / len;
      pos.push(p.x + nx * half, 0.004, p.z + nz * half, p.x - nx * half, 0.004, p.z - nz * half);
      if (k < bay.path.length - 1) index.push(k * 2, k * 2 + 2, k * 2 + 1, k * 2 + 1, k * 2 + 2, k * 2 + 3);
    });
    const geo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pos, 3)); geo.setIndex(index); geo.computeVertexNormals();
    const strip = new THREE.Mesh(geo, mat("#c7cfe3", { side: THREE.DoubleSide })); strip.receiveShadow = true; scene.add(strip);
  }
  const g = new THREE.Group(); g.position.set(depot.x, 0, depot.z); g.rotation.y = depot.rotation; scene.add(g);
  const halfL = depot.length / 2, halfD = depot.depth / 2, center = (depot.bays[0].v + depot.bays[depot.bays.length - 1].v) / 2;
  const aisles = depot.bays.map((bay) => bay.v - center);
  g.add(box([depot.length + 6, 0.06, depot.depth + 1.2], [0, 0.03, 0], "#dde3ee"),
    box([depot.length, WALL_HEIGHT, 0.25], [0, WALL_HEIGHT / 2, -halfD], "#4570f5"),
    box([depot.length, WALL_HEIGHT, 0.25], [0, WALL_HEIGHT / 2, halfD], "#4570f5"),
    box([depot.length + 0.5, 0.3, depot.depth + 0.5], [0, WALL_HEIGHT + 0.15, 0], "#2447c4")); // 屋頂遮住倉庫內的卡車
  for (const x of [-3, 0, 3]) g.add(box([1.6, 0.12, depot.depth * 0.7], [x, WALL_HEIGHT + 0.36, 0], "#cfe0ff"));
  for (const x of [-halfL, halfL]) { // 兩端牆面：卸貨道處留出門洞
    const edges = [-halfD, ...aisles.flatMap((z) => [z - OPENING / 2, z + OPENING / 2]), halfD];
    for (let k = 0; k < edges.length; k += 2) if (edges[k + 1] - edges[k] > 0.01) g.add(box([0.25, WALL_HEIGHT, edges[k + 1] - edges[k]], [x, WALL_HEIGHT / 2, (edges[k] + edges[k + 1]) / 2], "#eef2f9"));
    for (const z of aisles) g.add(box([0.25, WALL_HEIGHT - DOOR_HEIGHT, OPENING], [x, (WALL_HEIGHT + DOOR_HEIGHT) / 2, z], "#eef2f9"));
  }
  aisles.forEach((z, bay) => [-1, 1].forEach((side, sideIndex) => {
    const door = new THREE.Mesh(new THREE.BoxGeometry(0.1, 1, OPENING - 0.1), mat("#d6dbe5", { roughness: 0.6 }));
    door.position.set(side * (halfL + 0.05), DOOR_HEIGHT / 2, z); door.castShadow = true; g.add(door); depotDoors[bay * 2 + sideIndex] = door;
    const lampMat = mat(LAMP.free, { emissive: LAMP.free, emissiveIntensity: 0.8 });
    const lamp = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.22, 0.22), lampMat);
    lamp.position.set(side * (halfL + 0.2), DOOR_HEIGHT + 0.35, z + OPENING / 2 - 0.2); g.add(lamp); depotLamps[bay * 2 + sideIndex] = lampMat;
    const guide = new THREE.Mesh(new THREE.PlaneGeometry(2.6, 0.12), new THREE.MeshBasicMaterial({ color: "#f6c544" }));
    guide.rotation.x = -Math.PI / 2; guide.position.set(side * (halfL + 1.6), 0.07, z); g.add(guide);
  }));
  for (const [x, layers] of [[-2, 3], [-0.6, 2], [0.8, 1]]) { // 門邊的棧板
    for (let k = 0; k < layers; k++) g.add(box([1.1, 0.42, 1.1], [x, 0.27 + k * 0.44, halfD + 1.1], k % 2 ? "#c98f55" : "#e3b27c"));
  }
  g.traverse((o) => { o.raycast = noRaycast; });
  // 屋頂看板：canvas 貼圖的 sprite，卸貨數變化時重畫
  const canvas = document.createElement("canvas"); canvas.width = 420; canvas.height = 112;
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
  sprite.position.set(depot.x, WALL_HEIGHT + 2.2, depot.z); sprite.scale.set(10.5, 10.5 * 112 / 420, 1); sprite.raycast = noRaycast; scene.add(sprite);
  depotSign = { canvas, texture, sprite, busy: -1 };
}
function drawDepotSign() {
  const busy = traffic.occupants.filter(Boolean).length;
  if (!depotSign || busy === depotSign.busy) return;
  depotSign.busy = busy;
  const { canvas, texture, sprite } = depotSign, label = "索引倉庫 · 卸貨", count = `${busy}/${depot.bays.length}`;
  const font = '700 50px Inter, "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif';
  let ctx = canvas.getContext("2d"); ctx.font = font;
  // 依字寬決定看板大小，避免名稱與數字重疊
  canvas.width = Math.ceil(ctx.measureText(label).width + ctx.measureText("8/8").width + 96);
  ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  sprite.scale.set(2.8 * w / h, 2.8, 1);
  const rounded = (x, y, rw, rh, r) => { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + rw, y, x + rw, y + rh, r); ctx.arcTo(x + rw, y + rh, x, y + rh, r); ctx.arcTo(x, y + rh, x, y, r); ctx.arcTo(x, y, x + rw, y, r); ctx.closePath(); };
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(31, 52, 102, 0.16)"; rounded(6, 10, w - 12, h - 14, 22); ctx.fill();
  ctx.fillStyle = "#ffffff"; rounded(4, 4, w - 8, h - 14, 22); ctx.fill();
  ctx.font = font; ctx.textBaseline = "middle";
  ctx.fillStyle = "#1f2937"; ctx.textAlign = "left"; ctx.fillText(label, 28, h / 2 - 4);
  ctx.fillStyle = busy ? "#c2410c" : "#6b7280"; ctx.textAlign = "right"; ctx.fillText(count, w - 28, h / 2 - 4);
  texture.needsUpdate = true;
}

// 分區底座、邊框與標籤
const zoneLabels = new Map();
for (const zone of zones) {
  const frontierZone = zone.r === FRONTIER;
  const pad = new THREE.Mesh(new THREE.BoxGeometry(zone.w, 0.12, zone.d), mat(frontierZone ? "#f2f4f9" : "#fbfcfe"));
  pad.position.set(zone.x, 0.06, zone.z); pad.receiveShadow = true; scene.add(pad);
  const borderMat = new THREE.MeshBasicMaterial({ color: frontierZone ? "#d5dbe7" : "#c9d4ea" });
  [[0, -zone.d / 2, zone.w, 0.16], [0, zone.d / 2, zone.w, 0.16], [-zone.w / 2, 0, 0.16, zone.d], [zone.w / 2, 0, 0.16, zone.d]].forEach(([px, pz, w, d]) => {
    const border = new THREE.Mesh(new THREE.PlaneGeometry(w, d), borderMat);
    border.rotation.x = -Math.PI / 2; border.position.set(zone.x + px, 0.125, zone.z + pz); scene.add(border);
  });
  const div = document.createElement("div");
  div.className = "zone-label";
  div.innerHTML = `<b>${esc(frontierZone ? "邊境 · 未歸入社群" : DATA.regions[zone.r].t)}</b><span>${groups.get(zone.r).length}</span>`;
  div.title = frontierZone ? "未歸入任何頂層社群的實體" : DATA.regions[zone.r].t;
  div.addEventListener("pointerdown", (e) => e.stopPropagation());
  div.addEventListener("click", () => select({ kind: "region", r: zone.r }));
  const label = new CSS2DObject(div);
  label.position.set(zone.x - zone.w / 2 + 0.4, 0.2, zone.z - zone.d / 2 + 0.9);
  label.center.set(0, 0.5);
  scene.add(label);
  zoneLabels.set(zone.r, div);
  zone.label = label;
}

// 地塊與建築（InstancedMesh：一次繪製全部實體）
const count = DATA.nodes.length;
const lotGeo = new THREE.PlaneGeometry(LOT - 0.5, LOT - 0.5); lotGeo.rotateX(-Math.PI / 2);
const lots = new THREE.InstancedMesh(lotGeo, new THREE.MeshStandardMaterial({ roughness: 0.9 }), count);
lots.receiveShadow = true;
const bodyGeo = new THREE.BoxGeometry(1, 1, 1); bodyGeo.translate(0, 0.5, 0);
const bodies = new THREE.InstancedMesh(bodyGeo, mat("#ffffff"), count);
const roofShape = new THREE.Shape(); roofShape.moveTo(-0.5, 0); roofShape.lineTo(0.5, 0); roofShape.lineTo(0, 1); roofShape.closePath();
const roofGeo = new THREE.ExtrudeGeometry(roofShape, { depth: 1, bevelEnabled: false }); roofGeo.translate(0, 0, -0.5);
const roofs = new THREE.InstancedMesh(roofGeo, mat("#ffffff", { roughness: 0.6 }), count);
const doorGeo = new THREE.BoxGeometry(1, 1, 1); doorGeo.translate(0, 0.5, 0);
const doors = new THREE.InstancedMesh(doorGeo, mat(PALETTE.white), count);
for (const mesh of [bodies, roofs, doors]) { mesh.castShadow = true; mesh.receiveShadow = true; }
const matrix = new THREE.Matrix4(), noRotation = new THREE.Quaternion(), tmpColor = new THREE.Color(), tiny = new THREE.Vector3(1e-4, 1e-4, 1e-4);
const visibleTypes = new Set(DATA.types.map((t) => t.name));
let selection = null, hovered = null;
const selectedNode = () => selection?.kind === "node" ? selection.i : null;
function placeBuildings() {
  DATA.nodes.forEach((n, i) => {
    const p = positions[i], visible = visibleTypes.has(n.y);
    const w = footprint(n), d = w * 0.85, h = heightOf(n);
    lots.setMatrixAt(i, matrix.compose(new THREE.Vector3(p.x, p.y + 0.006, p.z), noRotation, new THREE.Vector3(1, 1, 1)));
    bodies.setMatrixAt(i, matrix.compose(p, noRotation, visible ? new THREE.Vector3(w, h, d) : tiny));
    roofs.setMatrixAt(i, matrix.compose(new THREE.Vector3(p.x, p.y + h, p.z), noRotation, visible ? new THREE.Vector3(w + 0.25, roofRise(n), d + 0.25) : tiny));
    doors.setMatrixAt(i, matrix.compose(new THREE.Vector3(p.x, p.y, p.z + d / 2 + 0.02), noRotation, visible ? new THREE.Vector3(w * 0.42, Math.min(h * 0.6, 1.1), 0.06) : tiny));
  });
  for (const mesh of [lots, bodies, roofs, doors]) mesh.instanceMatrix.needsUpdate = true;
  paintBuildings();
}
function paintBuildings() {
  const focus = selectedNode(), focusNeighbors = focus == null ? null : new Set(neighbors[focus].map((x) => x.o));
  DATA.nodes.forEach((n, i) => {
    tmpColor.set(typeColor[n.y]);
    if (i === focus || i === hovered) tmpColor.lerp(new THREE.Color("#ffffff"), 0.3);
    bodies.setColorAt(i, tmpColor);
    roofs.setColorAt(i, tmpColor.clone().multiplyScalar(0.8));
    lots.setColorAt(i, tmpColor.set(i === focus ? PALETTE.blue : focusNeighbors?.has(i) ? "#c9d6ff" : n.r === FRONTIER ? "#e8ecf3" : "#eef2f9"));
    doors.setColorAt(i, tmpColor.set(PALETTE.white));
  });
  for (const mesh of [lots, bodies, roofs, doors]) mesh.instanceColor.needsUpdate = true;
}
placeBuildings();
scene.add(lots, bodies, roofs, doors);

// 關係：屋頂之間的弧線
function arcPoints(a, b, segments = 14) {
  const p0 = positions[a].clone().setY(roofTop(a)), p2 = positions[b].clone().setY(roofTop(b));
  const mid = p0.clone().add(p2).multiplyScalar(0.5);
  mid.y += Math.min(10, 1.5 + p0.distanceTo(p2) * 0.18);
  return new THREE.QuadraticBezierCurve3(p0, mid, p2).getPoints(segments);
}
const curves = DATA.links.map(([s, t]) => arcPoints(s, t));
function linkGeometry(indices) {
  const verts = [];
  for (const li of indices) {
    if (!visibleTypes.has(DATA.nodes[DATA.links[li][0]].y) || !visibleTypes.has(DATA.nodes[DATA.links[li][1]].y)) continue;
    const pts = curves[li];
    for (let k = 0; k < pts.length - 1; k++) verts.push(pts[k].x, pts[k].y, pts[k].z, pts[k + 1].x, pts[k + 1].y, pts[k + 1].z);
  }
  return new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
}
const allLinks = DATA.links.map((_, i) => i);
const linkMaterial = new THREE.LineBasicMaterial({ color: PALETTE.blue, transparent: true, opacity: 0.2, depthWrite: false });
const linkLines = new THREE.LineSegments(linkGeometry(allLinks), linkMaterial);
const highlight = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: PALETTE.blue, transparent: true, opacity: 0.95, depthWrite: false }));
scene.add(linkLines, highlight);

// 沿關係弧線移動的資料封包
const PACKETS = Math.min(80, DATA.links.length);
const packetGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(PACKETS * 3), 3));
const packets = new THREE.Points(packetGeo, new THREE.PointsMaterial({ color: PALETTE.blue, size: 5, sizeAttenuation: false, transparent: true, opacity: 0.85, depthWrite: false }));
packets.visible = PACKETS > 0;
scene.add(packets);
const randomLink = (pool) => pool[Math.floor(Math.random() * pool.length)];
const packetState = Array.from({ length: PACKETS }, () => ({ li: Math.floor(Math.random() * DATA.links.length), t: Math.random(), v: 0.15 + Math.random() * 0.25 }));
let packetPool = allLinks;

// 選取圖釘
function makePin(color) {
  const g = new THREE.Group();
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 16), mat(color));
  head.position.y = 1.15; head.castShadow = true;
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12), mat("#ffffff"));
  dot.position.set(0.18, 1.2, 0.38);
  const tip = new THREE.Mesh(new THREE.ConeGeometry(0.34, 0.95, 16), mat(color));
  tip.rotation.x = Math.PI; tip.position.y = 0.55; tip.castShadow = true;
  g.add(head, dot, tip); g.visible = false; scene.add(g);
  return g;
}
const pin = makePin(PALETTE.blue);

// 選取框：WareTrack 風格的藍色角框＋淡藍填色＋地面光暈（不在 pick() 的判定清單內，不會擋住點擊）
const brackets = new THREE.Group(); brackets.visible = false; scene.add(brackets);
const bracketMats = [
  new THREE.LineBasicMaterial({ color: PALETTE.blue }),
  new THREE.MeshBasicMaterial({ color: PALETTE.blue, transparent: true, opacity: 0.07, depthWrite: false }),
  new THREE.MeshBasicMaterial({ color: PALETTE.blue, transparent: true, opacity: 0.14, depthWrite: false }),
];
function setBrackets(size) {
  brackets.children.forEach((child) => child.geometry.dispose()); brackets.clear();
  brackets.visible = Boolean(size);
  if (!size) return;
  const [w, h, d] = size, arm = Math.min(w, h, d) * 0.28 + 0.18, pts = [];
  for (const x of [-w / 2, w / 2]) for (const y of [0, h]) for (const z of [-d / 2, d / 2]) {
    const sx = x < 0 ? 1 : -1, sy = y === 0 ? 1 : -1, sz = z < 0 ? 1 : -1;
    pts.push(x, y, z, x + sx * arm, y, z, x, y, z, x, y + sy * arm, z, x, y, z, x, y, z + sz * arm);
  }
  brackets.add(new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3)), bracketMats[0]));
  const fill = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), bracketMats[1]); fill.position.y = h / 2;
  const glow = new THREE.Mesh(new THREE.PlaneGeometry(w + 1.4, d + 1.4), bracketMats[2]); glow.rotation.x = -Math.PI / 2; glow.position.y = 0.02;
  brackets.add(fill, glow);
}

// 道路上的卡車：最近的問答（貨櫃色帶＝檢索方法），會開進物流倉庫卸貨
const trucks = truckEvents.map((ev, k) => {
  const outer = new THREE.Group(), body = new THREE.Group();
  body.add(
    box([3.2, 0.3, 1.3], [0, 0.42, 0], PALETTE.dark),
    box([2.1, 1.2, 1.3], [-0.45, 1.17, 0], "#ffffff"),
    box([2.12, 0.26, 1.32], [-0.45, 1.17, 0], methodColor(ev.m)),
    box([0.9, 1.0, 1.25], [1.1, 1.07, 0], PALETTE.blue),
    box([0.05, 0.4, 1.0], [1.56, 1.27, 0], "#cfe0ff"),
  );
  for (const [x, z] of [[-1, -0.6], [-1, 0.6], [1, -0.6], [1, 0.6]]) {
    const wheel = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.2, 12), mat(PALETTE.dark));
    wheel.rotation.x = Math.PI / 2; wheel.position.set(x, 0.28, z); body.add(wheel);
  }
  body.traverse((o) => { o.userData.truck = k; });
  outer.add(body); scene.add(outer);
  return { ev, group: outer, body, vehicle: traffic.vehicles.find((v) => v.id === `q${k}`) };
});
const truckPin = makePin(PALETTE.blue);

// ---------- 鏡頭：等角正交，平滑移動 ----------
let fitZoom = 10, hudRight = 0, goal = null;
// 跟隨：鏡頭目標每幀追著選取的卡車；followZoom 為 null 代表使用者已自行縮放，不再拉回
let following = false, followZoom = null;
// 偏好減少動態的使用者預設暫停動畫
let paused = Boolean(window.matchMedia?.("(prefers-reduced-motion: reduce)").matches);
const currentDirection = () => camera.position.clone().sub(controls.target).normalize();
function setGoal(target, zoom, direction = currentDirection()) { goal = { target: target.clone(), zoom, direction }; }
controls.addEventListener("start", () => { if (following) followZoom = null; else goal = null; });
function cameraCommand(kind) {
  const base = goal ?? { target: controls.target.clone(), direction: currentDirection(), zoom: camera.zoom };
  if (kind === "zoomIn") goal = { ...base, zoom: Math.min(fitZoom * 8, base.zoom * 1.4) };
  if (kind === "zoomOut") goal = { ...base, zoom: Math.max(fitZoom * 0.5, base.zoom / 1.4) };
  if (kind === "rotateLeft" || kind === "rotateRight") {
    goal = { ...base, direction: base.direction.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), kind === "rotateLeft" ? Math.PI / 4 : -Math.PI / 4) };
  }
  if (kind === "home") goal = { target: new THREE.Vector3(), direction: VIEW_DIRECTION.clone(), zoom: fitZoom };
  if (following) followZoom = goal.zoom;
}
$("mapControls").innerHTML = [["plus", "zoomIn", "放大"], ["minus", "zoomOut", "縮小"], ["rotateLeft", "rotateLeft", "向左旋轉"], ["rotateRight", "rotateRight", "向右旋轉"], ["home", "home", "回到全景"]]
  .map(([name, kind, label]) => `<button data-cmd="${kind}" title="${label}" aria-label="${label}">${icon(name)}</button>`).join("")
  + '<button data-cmd="pause" id="pauseButton"></button>';
function setPaused(on) {
  paused = on;
  const button = $("pauseButton"), label = paused ? "繼續動畫" : "暫停動畫";
  button.innerHTML = icon(paused ? "play" : "pause");
  button.classList.toggle("on", paused);
  button.title = `${label} (Space)`; button.setAttribute("aria-label", label); button.setAttribute("aria-pressed", String(paused));
}
setPaused(paused);
function setFollow(on) {
  following = Boolean(on) && selection?.kind === "query" && Boolean(trucks[selection.k]);
  followZoom = following ? Math.max(fitZoom * 2.4, camera.zoom) : null;
  if (!following) goal = null;
  renderDetail();
}
$("mapControls").addEventListener("click", (e) => {
  const button = e.target.closest("[data-cmd]"); if (!button) return;
  if (button.dataset.cmd === "pause") { setPaused(!paused); return; }
  if (button.dataset.cmd === "home") select(null, false);
  cameraCommand(button.dataset.cmd);
});

// ---------- 選取 ----------
const sameSelection = (a, b) => Boolean(a && b) && a.kind === b.kind && a.i === b.i && a.r === b.r && a.k === b.k;
function select(next, fly = true) {
  if (!sameSelection(next, selection)) { following = false; followZoom = null; }
  selection = next;
  const node = selectedNode();
  packetPool = node != null && neighbors[node].length ? neighbors[node].map((x) => x.li) : allLinks;
  highlight.geometry.dispose();
  highlight.geometry = node != null ? linkGeometry(neighbors[node].map((x) => x.li)) : new THREE.BufferGeometry();
  linkMaterial.opacity = node != null ? 0.07 : 0.2;
  pin.visible = node != null;
  if (node != null) pin.position.set(positions[node].x, roofTop(node) + 0.1, positions[node].z);
  if (node != null) {
    const n = DATA.nodes[node], w = footprint(n) + 0.5;
    setBrackets([w, heightOf(n) + roofRise(n) + 0.15, footprint(n) * 0.85 + 0.5]);
    brackets.position.copy(positions[node]); brackets.rotation.y = 0;
  } else setBrackets(next?.kind === "query" && trucks[next.k] ? [3.6, 1.9, 1.7] : null);
  paintBuildings();
  zoneLabels.forEach((div, r) => div.classList.toggle("on", selection?.kind === "region" && selection.r === r));
  renderDetail(); renderTracker(); renderTabs();
  if (!fly || !next) return;
  if (next.kind === "node") setGoal(positions[next.i].clone().setY(0), Math.max(fitZoom * 2.4, camera.zoom));
  if (next.kind === "region") {
    const zone = zoneOf.get(next.r);
    const zoom = Math.min((innerWidth - hudRight) / (zone.w * 1.9), innerHeight / (zone.d * 1.6), fitZoom * 6);
    setGoal(new THREE.Vector3(zone.x, 0, zone.z), Math.max(fitZoom * 1.2, zoom));
  }
  if (next.kind === "query" && trucks[next.k]) {
    const point = trucks[next.k].group.position;
    setGoal(new THREE.Vector3(point.x, 0, point.z), Math.max(fitZoom * 2.4, camera.zoom));
  }
}
$("detailClose").onclick = () => select(null);
$("detail").addEventListener("click", (e) => { if (e.target.closest("[data-follow]")) setFollow(!following); });
$("logo").onclick = () => { select(null, false); cameraCommand("home"); };
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-node],[data-region],[data-query]");
  if (!target || target.closest(".zone-label")) return;
  if (target.dataset.node != null) select({ kind: "node", i: Number(target.dataset.node) });
  else if (target.dataset.region != null) select({ kind: "region", r: Number(target.dataset.region) });
  else select({ kind: "query", k: Number(target.dataset.query) });
});

// ---------- HUD：導覽列與 KPI ----------
$("projectName").textContent = DATA.project.name;
$("projectMeta").textContent = `${DATA.project.id} · ${DATA.project.status}`;
$("live").innerHTML = `<i></i>建圖 ${Math.round(S.index_runtime_s)}s`;
const failedQueries = S.queries - S.queries_completed;
const kpi = (iconName, label, value, unit, delta, tone, sub) => `
  <div class="card kpi"><span class="kpi-icon">${icon(iconName, 22)}</span><div>
    <span class="kpi-label">${label}</span>
    <div class="kpi-value"><b>${esc(value)}</b><span class="unit">${esc(unit)}</span>${delta ? `<span class="delta ${tone}">${esc(delta)}</span>` : ""}</div>
    <span class="kpi-sub">${esc(sub)}</span></div></div>`;
$("kpis").innerHTML = [
  kpi("box", "實體", S.entities, "個", isolated ? `${isolated} 孤立` : "皆有連結", isolated ? "bad" : "good", `${DATA.types.length} 種類型`),
  kpi("branch", "關係", S.relationships, "條", `平均 ${avgDeg}`, "good", "每個實體的平均連結數"),
  kpi("pin", "社群", S.regions, `/ ${S.communities}`, `覆蓋 ${Math.round(((count - (groups.get(FRONTIER)?.length ?? 0)) / Math.max(1, count)) * 100)}%`, "good", "頂層分區 / 全部階層"),
  kpi("truck", "問答", S.queries_completed, `/ ${S.queries}`, S.queries ? (failedQueries ? `${failedQueries} 未完成` : "全數完成") : "", failedQueries ? "bad" : "good", `最新 ${shortTime(DATA.events[0]?.at)}`),
].join("");

// ---------- 通知鈴鐺：未完成問答、孤立實體、缺少報告的社群 ----------
const ALERT_ICON = { bad: "close", warn: "flag", info: "bell" };
const badAlerts = DATA.alerts.filter((a) => a.tone === "bad").length, warnAlerts = DATA.alerts.filter((a) => a.tone === "warn").length;
const badge = badAlerts || warnAlerts;
$("bell").innerHTML = icon("bell", 20) + (badge ? `<i class="${badAlerts ? "" : "warn"}">${badge > 9 ? "9+" : badge}</i>` : "");
$("bell").title = DATA.alerts.length ? `${DATA.alerts.length} 項需要注意` : "通知";
$("bellMenu").innerHTML = `<div class="dd-head"><b>需要注意</b><span class="muted">${DATA.alerts.length}</span></div>`
  + (DATA.alerts.map((a, k) => `<button class="notice ${a.tone}" data-alert="${k}"><span class="notice-icon">${icon(ALERT_ICON[a.tone], 14)}</span>
    <span class="notice-text"><b>${esc(a.title)}</b>${a.body ? `<small>${esc(a.body)}</small>` : ""}</span></button>`).join("")
    || '<p class="muted pad">索引與問答紀錄沒有需要處理的事項。</p>');
const bellOpen = () => $("bellMenu").classList.contains("open");
function setBell(open) { $("bellMenu").classList.toggle("open", open); $("bell").classList.toggle("on", open); $("bell").setAttribute("aria-expanded", String(open)); }
$("bell").onclick = () => setBell(!bellOpen());
$("bellMenu").addEventListener("click", (e) => {
  const button = e.target.closest("[data-alert]"); if (!button) return;
  const alert = DATA.alerts[Number(button.dataset.alert)];
  setBell(false);
  if (alert.sel) select(alert.sel);
});
document.addEventListener("pointerdown", (e) => { if (bellOpen() && !e.target.closest("#bellMenu, #bell")) setBell(false); });

// ---------- 搜尋 ----------
const searchInput = $("search"), searchResults = $("searchResults");
let results = [];
function renderSearch() {
  const q = searchInput.value.trim().toLowerCase();
  results = !q ? [] : [
    ...DATA.regions.map((r, i) => ({ label: r.t, hint: "社群", sel: { kind: "region", r: i } })).filter((x) => x.label.toLowerCase().includes(q)),
    ...DATA.nodes.map((n, i) => ({ label: n.t, hint: n.y, sel: { kind: "node", i }, exact: n.t.toLowerCase() === q })).filter((x) => x.label.toLowerCase().includes(q)),
  ].sort((a, b) => (b.exact ? 1 : 0) - (a.exact ? 1 : 0)).slice(0, 8);
  searchResults.innerHTML = results.map((x, k) => `<button data-result="${k}" class="${k ? "" : "on"}"><span>${esc(x.label)}</span><small>${esc(x.hint)}</small></button>`).join("");
  searchResults.classList.toggle("open", results.length > 0 && document.activeElement === searchInput);
}
const pickResult = (k) => { if (!results[k]) return; select(results[k].sel); searchInput.value = ""; searchInput.blur(); renderSearch(); };
searchInput.addEventListener("input", renderSearch);
searchInput.addEventListener("focus", renderSearch);
searchInput.addEventListener("blur", () => setTimeout(() => searchResults.classList.remove("open"), 150));
searchInput.addEventListener("keydown", (e) => { if (e.key === "Enter") pickResult(0); if (e.key === "Escape") searchInput.blur(); });
searchResults.addEventListener("mousedown", (e) => { const b = e.target.closest("[data-result]"); if (b) pickResult(Number(b.dataset.result)); });
// 快捷鍵（同 WareTrack）：/ 搜尋、Esc 關閉選單或取消選取、Space 暫停動畫、F 跟隨選取的卡車
addEventListener("keydown", (e) => {
  if (document.activeElement === searchInput || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "/") { e.preventDefault(); searchInput.focus(); }
  else if (e.key === "Escape") { if (bellOpen()) setBell(false); else select(null, false); }
  else if (e.key === " ") { e.preventDefault(); setPaused(!paused); }
  else if (e.key === "f" || e.key === "F") setFollow(!following);
});

// ---------- 右側詳細卡 ----------
const header = (eyebrow, title, subtitle, iconName) => `
  <div class="detail-header"><span class="detail-icon">${icon(iconName, 26)}</span><div>
    <span class="eyebrow">${esc(eyebrow)}</span><h2>${esc(title)}</h2><span class="muted">${esc(subtitle)}</span></div></div>`;
const statusRow = (pillHtml, text) => `<div class="status-row">${pillHtml}<span class="muted">${esc(text)}</span></div>`;
const stat = (label, value, unit = "", ratio = null) => `
  <div class="stat"><span class="stat-label">${label}</span><div><b>${esc(value)}</b> <span class="muted">${esc(unit)}</span></div>
  ${ratio == null ? "" : `<div class="track"><i style="width:${Math.max(3, Math.min(100, ratio * 100))}%"></i></div>`}</div>`;
const typePill = (type) => `<span class="pill" style="color:${typeColor[type]};background:${typeColor[type]}1f">${esc(type)}</span>`;
const nodeRow = (i, right) => {
  const n = DATA.nodes[i];
  return `<button class="inventory-row" data-node="${i}"><span class="inv-icon" style="color:${typeColor[n.y]};background:${typeColor[n.y]}1f">${icon("box", 16)}</span>
    <span class="name" title="${esc(n.t)}">${esc(n.t)}</span><span class="muted">${esc(right)}</span></button>`;
};
function renderDetail() {
  $("detailClose").classList.toggle("show", Boolean(selection));
  let html = "";
  if (selection?.kind === "node") {
    const i = selection.i, n = DATA.nodes[i];
    html = header(`ENTITY · ${n.y}`, n.t, regionTitle(n.r), "box")
      + statusRow(typePill(n.y), `連結 ${n.deg} · 出現 ${n.freq} 次`)
      + `<div class="stat-grid">${stat("連結度", n.deg, `/ ${maxDeg}`, n.deg / maxDeg)}${stat("出現次數", n.freq, `/ ${maxFreq}`, n.freq / maxFreq)}</div>`
      + `<h4>描述</h4><p class="prose">${esc(n.d) || '<span class="muted">無描述</span>'}</p>`
      + `<h4>關聯實體 <span class="muted">${neighbors[i].length}</span></h4>`
      + (neighbors[i].slice(0, 40).map((x) => nodeRow(x.o, `w ${x.w}`)).join("") || '<p class="muted">沒有關聯實體。</p>');
  } else if (selection?.kind === "region") {
    const r = selection.r, members = groups.get(r) || [];
    const region = r === FRONTIER ? { t: "邊境 · 未歸入社群", c: "—", rank: null, s: "" } : DATA.regions[r];
    html = header(`COMMUNITY · ${region.c}`, region.t, `${members.length} 個實體`, "pin")
      + statusRow(region.rank != null ? `<span class="pill big pill-blue">重要度 ${region.rank}</span>` : '<span class="pill big pill-grey">未評分</span>',
        `${members.filter((m) => DATA.nodes[m].deg === 0).length} 個孤立實體`)
      + `<div class="stat-grid">${stat("實體數", members.length, "個", members.length / Math.max(1, count))}${stat("重要度", region.rank ?? "—", "/ 10", region.rank != null ? region.rank / 10 : null)}</div>`
      + (region.s ? `<h4>社群摘要</h4><p class="prose">${esc(region.s)}</p>` : "")
      + `<h4>主要實體 <span class="muted">依連結度</span></h4>`
      + members.slice(0, 25).map((m) => nodeRow(m, `deg ${DATA.nodes[m].deg}`)).join("");
  } else if (selection?.kind === "query") {
    const ev = DATA.events[selection.k];
    html = header(`QUERY · ${methodLabel(ev.m)}`, ev.q, shortTime(ev.at), "truck")
      + statusRow(queryPill(ev).replace('class="pill', 'class="pill big'), ev.s === "COMPLETED" ? "已完成問答" : "問答未完成")
      + (trucks[selection.k] ? `<button class="follow-button ${following ? "on" : ""}" data-follow aria-pressed="${following}">${icon("target", 16)} ${following ? "停止跟隨" : "鏡頭跟隨卡車"} <kbd>F</kbd></button>` : "")
      + `<div class="stat-grid">${stat("耗時", ev.sec != null ? Math.round(ev.sec) : "—", "秒")}${stat("檢索方法", methodLabel(ev.m))}</div>`
      + '<p class="muted">完整回答與來源請到「2 問答測試」的問答紀錄查看。</p>';
  } else {
    const covered = count - (groups.get(FRONTIER)?.length ?? 0);
    const hubs = DATA.nodes.map((n, i) => i).sort((a, b) => DATA.nodes[b].deg - DATA.nodes[a].deg).slice(0, 8);
    html = header(`PROJECT · ${DATA.project.id}`, DATA.project.name, "知識圖譜總覽", "crane")
      + statusRow('<span class="pill big pill-good">已建圖</span>', `${S.regions} 個分區 · ${DATA.types.length} 種類型 · ${isolated} 個孤立實體`)
      + `<div class="stat-grid">${stat("社群覆蓋", covered, `/ ${count}`, covered / Math.max(1, count))}${stat("問答完成", S.queries_completed, `/ ${S.queries}`, S.queries ? S.queries_completed / S.queries : null)}
        ${stat("文件頁", S.documents, "頁")}${stat("Chunk", S.text_units, "個")}</div>`
      + `<h4>樞紐實體 <span class="muted">連結度 Top 8</span></h4>` + hubs.map((i) => nodeRow(i, `deg ${DATA.nodes[i].deg}`)).join("");
  }
  $("detail").innerHTML = html;
  $("detail").parentElement.scrollTop = 0; $("detail").scrollTop = 0;
}

// ---------- 左下：建圖流程與最近問答 ----------
const STAGES = [["文件頁", S.documents, "doc"], ["Chunk", S.text_units, "box"], ["實體", S.entities, "pin"], ["關係", S.relationships, "branch"], ["社群", S.communities, "crane"]];
let eventPage = 0;
function renderTracker() {
  const node = selectedNode();
  const current = selection?.kind === "node" ? 2 : selection?.kind === "region" ? 4 : -1;
  const context = node != null ? `${DATA.nodes[node].t} · ${regionTitle(DATA.nodes[node].r)}` : selection?.kind === "region" ? (selection.r === FRONTIER ? "未歸入社群" : DATA.regions[selection.r].t) : `建圖耗時 ${Math.round(S.index_runtime_s)}s`;
  const ev = DATA.events.length ? DATA.events[eventPage % DATA.events.length] : null;
  $("tracker").innerHTML = `
    <div class="tracker-main">
      <div class="tracker-head"><span class="tracker-title">${icon("flag")} 建圖流程</span><span class="muted">${esc(context)}</span></div>
      <ol class="stepper">${STAGES.map(([label, value, iconName], k) => `
        <li class="${k === current ? "current" : value ? "done" : ""}"><span class="step-dot">${icon(value && k !== current ? "check" : iconName, 16)}</span>
        <b>${label}</b><small>${value} ${k === 0 ? "頁" : "個"}</small></li>`).join("")}</ol>
    </div>
    <div class="next-card">
      <div class="next-head"><span class="eyebrow">最近問答</span>${DATA.events.length > 1 ? `<span class="pager">
        <button data-page="-1" aria-label="上一則">${icon("chevronLeft", 16)}</button>${eventPage % DATA.events.length + 1}/${DATA.events.length}
        <button data-page="1" aria-label="下一則">${icon("chevronRight", 16)}</button></span>` : ""}</div>
      ${ev ? `<p data-query="${eventPage % DATA.events.length}">${esc(ev.q)}</p>
        <div class="pills"><span class="pill" style="color:${methodColor(ev.m)};background:${methodColor(ev.m)}1a">${methodLabel(ev.m)}</span>${queryPill(ev)}</div>`
        : '<p style="cursor:default">尚無問答紀錄，到「2 問答測試」提問後會出現在這裡。</p>'}
    </div>`;
}
$("tracker").addEventListener("click", (e) => {
  const b = e.target.closest("[data-page]"); if (!b) return;
  eventPage = (eventPage + Number(b.dataset.page) + DATA.events.length) % DATA.events.length; renderTracker();
});

// ---------- 右下：分頁表格 ----------
let tab = "regions";
const regionOrder = DATA.regions.map((_, i) => i).sort((a, b) => (DATA.regions[b].rank ?? 0) - (DATA.regions[a].rank ?? 0) || DATA.regions[b].n - DATA.regions[a].n);
function renderTabs() {
  const tabButton = (key, label, n) => `<button data-tab="${key}" class="${tab === key ? "on" : ""}">${label} <span>${n}</span></button>`;
  let body = "";
  if (tab === "regions") {
    body = '<div class="table">' + regionOrder.map((r) => {
      const region = DATA.regions[r], active = selection?.kind === "region" && selection.r === r;
      return `<button class="table-row" data-region="${r}" style="${active ? "background:var(--blue-soft)" : ""}"><span class="cell-id">#${region.c}</span>
        <span class="cell-main" title="${esc(region.t)}">${esc(region.t)}</span>
        <span class="mini-track"><i style="width:${((region.rank ?? 0) / 10) * 100}%"></i></span><span class="cell-num">${groups.get(r)?.length ?? 0}</span>${icon("chevronRight", 16).replace("<svg", '<svg class="chev"')}</button>`;
    }).join("") + (groups.has(FRONTIER) ? `<button class="table-row" data-region="${FRONTIER}"><span class="cell-id">—</span><span class="cell-main muted">邊境 · 未歸入社群</span><span class="cell-num">${groups.get(FRONTIER).length}</span></button>` : "")
      + (DATA.regions.length ? "" : '<p class="muted pad">尚無社群資料。</p>') + "</div>";
  } else if (tab === "types") {
    body = `<div class="filters">${DATA.types.map((t) => `<button class="pill filter ${visibleTypes.has(t.name) ? "" : "off"}" data-type="${esc(t.name)}" style="color:${typeColor[t.name]};background:${typeColor[t.name]}1f"><i style="background:${typeColor[t.name]}"></i>${esc(t.name)} ${t.count}</button>`).join("")}</div>
      <div class="table">${DATA.nodes.map((n, i) => i).filter((i) => visibleTypes.has(DATA.nodes[i].y)).sort((a, b) => DATA.nodes[b].deg - DATA.nodes[a].deg).slice(0, 60)
        .map((i) => `<button class="table-row" data-node="${i}"><span class="cell-main" title="${esc(DATA.nodes[i].t)}">${esc(DATA.nodes[i].t)}</span>${typePill(DATA.nodes[i].y)}<span class="cell-num">${DATA.nodes[i].deg}</span></button>`).join("")}</div>`;
  } else {
    body = '<div class="table">' + DATA.events.map((ev, k) => `<button class="table-row" data-query="${k}"><span class="cell-id mono">${esc(shortTime(ev.at))}</span>
      <span class="cell-main" title="${esc(ev.q)}">${esc(ev.q)}</span><span class="pill" style="color:${methodColor(ev.m)};background:${methodColor(ev.m)}1a">${methodLabel(ev.m)}</span>${queryPill(ev)}</button>`).join("")
      + (DATA.events.length ? "" : '<p class="muted pad">尚無問答紀錄。到「2 問答測試」提問後，每筆問答會成為道路上的一台卡車。</p>') + "</div>";
  }
  const scroll = $("tabs").querySelector(".table")?.scrollTop ?? 0;
  $("tabs").innerHTML = `<div class="segmented">${tabButton("regions", "社群", DATA.regions.length)}${tabButton("types", "類型", DATA.types.length)}${tabButton("queries", "問答", S.queries)}</div>${body}`;
  const table = $("tabs").querySelector(".table"); if (table) table.scrollTop = scroll;
}
$("tabs").addEventListener("click", (e) => {
  const tabButton = e.target.closest("[data-tab]");
  if (tabButton) { tab = tabButton.dataset.tab; renderTabs(); $("tabs").querySelector(".table").scrollTop = 0; return; }
  const typeButton = e.target.closest("[data-type]");
  if (!typeButton) return;
  const type = typeButton.dataset.type;
  visibleTypes.has(type) ? visibleTypes.delete(type) : visibleTypes.add(type);
  if (selectedNode() != null && !visibleTypes.has(DATA.nodes[selectedNode()].y)) selection = null;
  placeBuildings();
  linkLines.geometry.dispose(); linkLines.geometry = linkGeometry(allLinks);
  select(selection, false);
});

// ---------- 滑鼠互動 ----------
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2(), tooltip = $("tooltip");
function pick(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const truckHit = raycaster.intersectObjects(trucks.map((t) => t.body), true)[0];
  if (truckHit) return { kind: "query", k: truckHit.object.userData.truck };
  const hit = raycaster.intersectObjects([bodies, roofs], false).find((h) => visibleTypes.has(DATA.nodes[h.instanceId].y));
  return hit ? { kind: "node", i: hit.instanceId } : null;
}
let downAt = null, hoveredTruck = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5 || e.button !== 0) return;
  select(pick(e), true);
});
let lastHover = 0;
renderer.domElement.addEventListener("pointermove", (e) => {
  // 拖曳平移超過 6px 視為手動操作，結束跟隨
  if (e.buttons && downAt && following && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 6) setFollow(false);
  // 拖曳中不做滑過判定；其餘以 50ms 節流，避免每次移動都對全部實體做 raycast
  const now = performance.now();
  if (e.buttons || now - lastHover < 50) return;
  lastHover = now;
  const hit = pick(e);
  const nextHovered = hit?.kind === "node" ? hit.i : null;
  hoveredTruck = hit?.kind === "query" ? hit.k : null;
  if (nextHovered !== hovered) { hovered = nextHovered; paintBuildings(); }
  renderer.domElement.style.cursor = hit ? "pointer" : "";
  if (!hit) { tooltip.style.display = "none"; return; }
  if (hit.kind === "node") {
    const n = DATA.nodes[hit.i];
    tooltip.innerHTML = `<b>${esc(n.t)}</b> <span style="color:${typeColor[n.y]}">${esc(n.y)}</span> <span class="muted">deg ${n.deg}</span>`;
  } else {
    const ev = DATA.events[hit.k];
    tooltip.innerHTML = `<b>${methodLabel(ev.m)}</b> ${esc(ev.q)}`;
  }
  tooltip.style.display = "block";
  tooltip.style.left = `${Math.min(e.clientX + 14, innerWidth - tooltip.offsetWidth - 8)}px`; tooltip.style.top = `${e.clientY + 12}px`;
});
renderer.domElement.addEventListener("pointerleave", () => { tooltip.style.display = "none"; if (hovered != null) { hovered = null; paintBuildings(); } });

// ---------- 迴圈 ----------
function resize() {
  const w = innerWidth, h = innerHeight;
  // 右側卡片遮住部分畫面：以 view offset 讓場景中心落在可見區域中央
  hudRight = w > 860 ? parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--detail-w")) + 32 : 0;
  const nextFit = Math.min((w - hudRight) / (radius * 2.1), h / (radius * 1.75));
  if (camera.zoom === 1 || Math.abs(camera.zoom - fitZoom) < 1e-6) camera.zoom = nextFit;
  fitZoom = nextFit;
  controls.minZoom = fitZoom * 0.5; controls.maxZoom = fitZoom * 8;
  Object.assign(camera, { left: -w / 2, right: w / 2, top: h / 2, bottom: -h / 2 });
  camera.setViewOffset(w, h, hudRight / 2, 0, w, h);
  camera.updateProjectionMatrix();
  renderer.setSize(w, h); labelRenderer.setSize(w, h);
}
addEventListener("resize", resize); resize();
select(null, false);
const clock = new THREE.Clock();
const tag = $("tag3d"), tagAnchor = new THREE.Vector3();
// 選取物件上方的藍色標籤：每幀投影到螢幕座標
function updateTag() {
  let html = "";
  if (selection?.kind === "node") {
    const i = selection.i;
    tagAnchor.set(positions[i].x, roofTop(i) + 2.3, positions[i].z);
    html = `${esc(DATA.nodes[i].t)}<small>${esc(DATA.nodes[i].y)}</small>`;
  } else if (selection?.kind === "query" && trucks[selection.k]) {
    trucks[selection.k].body.getWorldPosition(tagAnchor); tagAnchor.y = 4;
    html = `${methodLabel(DATA.events[selection.k].m)}<small>${following ? "跟隨中" : esc(shortTime(DATA.events[selection.k].at))}</small>`;
  }
  if (!html) { tag.style.display = "none"; return; }
  tagAnchor.project(camera);
  if (tag.dataset.html !== html) { tag.innerHTML = html; tag.dataset.html = html; }
  tag.style.display = "block";
  tag.style.transform = `translate(${(tagAnchor.x + 1) / 2 * innerWidth}px, ${(1 - tagAnchor.y) / 2 * innerHeight - 6}px) translate(-50%, -100%)`;
}
function tick() {
  const dt = Math.min(clock.getDelta(), 0.05), time = clock.elapsedTime, moveDt = paused ? 0 : dt;
  // 跟隨中的卡車繼續行駛；否則選取或滑過時停下來方便查看（後方車輛會排隊）
  trucks.forEach((truck, k) => { truck.vehicle.held = !following && ((selection?.kind === "query" && selection.k === k) || hoveredTruck === k); });
  World.stepTraffic(traffic, moveDt);
  updateLamps();
  for (const item of [...trucks, ...cars]) {
    const pose = World.vehiclePose(traffic, item.vehicle);
    item.group.position.set(pose.x, 0, pose.z); item.group.rotation.y = pose.heading;
  }
  if (depot) {
    depot.bays.forEach((_, bay) => [0, 1].forEach((side) => {
      const door = depotDoors[bay * 2 + side], height = DOOR_HEIGHT * (1 - traffic.doors[bay][side] * 0.9);
      door.scale.y = height; door.position.y = DOOR_HEIGHT - height / 2;
      const color = LAMP[World.bayState(traffic, bay)];
      depotLamps[bay * 2 + side].color.set(color); depotLamps[bay * 2 + side].emissive.set(color);
    }));
    drawDepotSign();
  }
  const followed = selection?.kind === "query" ? trucks[selection.k] : null;
  if (following && followed) {
    const p = followed.group.position;
    goal = { target: new THREE.Vector3(p.x, 0, p.z), direction: goal?.direction ?? currentDirection(), zoom: followZoom ?? camera.zoom };
  }
  if (goal) {
    const k = 1 - Math.exp(-dt * 4);
    controls.target.lerp(goal.target, k);
    const direction = currentDirection().lerp(goal.direction, k).normalize();
    camera.position.copy(controls.target).add(direction.multiplyScalar(VIEW_DISTANCE));
    camera.zoom += (goal.zoom - camera.zoom) * k;
    camera.updateProjectionMatrix();
    if (!following && controls.target.distanceTo(goal.target) < 0.02 && Math.abs(camera.zoom - goal.zoom) < 0.05 && currentDirection().distanceTo(goal.direction) < 0.002) goal = null;
  }
  controls.update();
  truckPin.visible = Boolean(followed);
  if (followed) {
    followed.body.getWorldPosition(truckPin.position); truckPin.position.y = 2.1 + 0.15 * Math.sin(time * 3);
    followed.body.getWorldPosition(brackets.position); brackets.rotation.y = followed.group.rotation.y;
  }
  if (pin.visible) pin.position.y = roofTop(selection.i) + 0.1 + 0.15 * Math.sin(time * 3);
  const attr = packetGeo.attributes.position;
  packetState.forEach((p, k) => {
    p.t += p.v * moveDt;
    if (p.t >= 1) { p.t = 0; p.li = randomLink(packetPool); }
    const pts = curves[p.li]; if (!pts) return;
    const f = p.t * (pts.length - 1), a = Math.floor(f), b = Math.min(a + 1, pts.length - 1);
    attr.setXYZ(k, ...pts[a].clone().lerp(pts[b], f - a).toArray());
  });
  attr.needsUpdate = true;
  // 分區在畫面上太小時隱藏標籤，避免拉遠時標籤互相重疊
  for (const zone of zones) zone.label.visible = zone.w * camera.zoom > 150 || (selection?.kind === "region" && selection.r === zone.r);
  renderer.render(scene, camera); labelRenderer.render(scene, camera);
  updateTag();
  requestAnimationFrame(tick);
}
tick();
</script>
<script>
  // importmap 模組載入失敗時（例如離線）顯示提示
  setTimeout(() => { if (!document.querySelector("#scene canvas")) document.getElementById("error").style.display = "flex"; }, 6000);
</script>
</body>
</html>
"""
