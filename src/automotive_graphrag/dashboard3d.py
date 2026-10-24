"""3D 戰略地圖儀表板：把專案的 GraphRAG 索引輸出呈現為可互動的 three.js 場景。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .projects import Project

THREE_VERSION = "0.160.0"
DASHBOARD_HEIGHT_PX = 760
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
    }


def render_dashboard_html(data: dict[str, Any]) -> str:
    """產生自足的 HTML 頁面（three.js 由 CDN 載入）。"""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        DASHBOARD_TEMPLATE
        .replace("__THREE_VERSION__", THREE_VERSION)
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
        f'style="width:100%;height:{DASHBOARD_HEIGHT_PX}px;border:0;border-radius:10px;background:#070b12" '
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
        '<div style="padding:48px 24px;text-align:center;border:1px dashed #94a3b8;'
        f'border-radius:10px;color:#64748b">{message}</div>'
    )


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GraphRAG 戰略地圖</title>
<style>
  :root {
    --bg: #070b12; --panel: rgba(12, 19, 31, 0.82); --line: rgba(120, 190, 255, 0.22);
    --text: #dbe7f5; --muted: #8296ad; --accent: #4fd1ff; --gold: #ffc857; --ok: #5ee6a0; --bad: #ff6b7a;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: var(--bg); color: var(--text);
    font: 13px/1.45 "Segoe UI", "Noto Sans TC", system-ui, sans-serif; }
  #scene { position: fixed; inset: 0; }
  .panel { position: absolute; background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
    backdrop-filter: blur(8px); box-shadow: 0 8px 28px rgba(0,0,0,.45); }
  .panel h3 { margin: 0; padding: 9px 12px; font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
    color: var(--accent); border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; }
  #topbar { top: 10px; left: 10px; right: 10px; display: flex; align-items: center; gap: 6px; padding: 6px 10px; flex-wrap: wrap; }
  #topbar .brand { font-weight: 700; margin-right: 10px; white-space: nowrap; }
  #topbar .brand small { color: var(--muted); font-weight: 400; margin-left: 6px; }
  .res { display: flex; align-items: baseline; gap: 6px; padding: 4px 10px; border-radius: 6px; background: rgba(255,255,255,.04); white-space: nowrap; }
  .res b { color: var(--gold); font-variant-numeric: tabular-nums; font-size: 15px; }
  .res span { color: var(--muted); font-size: 11px; }
  #search { margin-left: auto; background: rgba(0,0,0,.35); border: 1px solid var(--line); color: var(--text);
    border-radius: 6px; padding: 6px 10px; width: 210px; outline: none; }
  #search:focus { border-color: var(--accent); }
  #regions { top: var(--hud-top, 64px); left: 10px; width: 260px; bottom: 226px; display: flex; flex-direction: column; }
  #regionList, #inspectorBody, #logList { scrollbar-width: thin; scrollbar-color: rgba(79,209,255,.35) transparent; }
  #regionList { overflow-y: auto; flex: 1; padding: 4px; }
  .region { padding: 7px 9px; border-radius: 6px; cursor: pointer; display: grid; grid-template-columns: 10px 1fr auto; gap: 8px; align-items: center; }
  .region:hover, .region.active { background: rgba(79,209,255,.12); }
  .region .dot { width: 10px; height: 10px; border-radius: 2px; transform: rotate(45deg); }
  .region .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .region .meta { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  #minimap { left: 10px; bottom: 10px; width: 260px; height: 206px; padding: 0; overflow: hidden; }
  #minimap canvas { width: 100%; height: calc(100% - 33px); display: block; cursor: crosshair; }
  #inspector { top: var(--hud-top, 64px); right: 10px; width: 320px; max-height: calc(100% - var(--hud-top, 64px) - 316px); display: flex; flex-direction: column; }
  #inspectorBody { padding: 10px 12px 12px; overflow-y: auto; }
  #inspectorBody .title { font-size: 16px; font-weight: 700; margin-bottom: 4px; word-break: break-word; }
  .chip { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; margin-right: 4px; border: 1px solid currentColor; }
  .bars { display: grid; grid-template-columns: 64px 1fr 34px; gap: 4px 8px; align-items: center; margin: 10px 0; font-size: 11px; color: var(--muted); }
  .bar { height: 6px; background: rgba(255,255,255,.07); border-radius: 3px; overflow: hidden; }
  .bar i { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--gold)); }
  .desc { color: #b9c8da; white-space: pre-wrap; }
  .neighbors { margin-top: 10px; }
  .neighbors div { padding: 4px 6px; border-radius: 5px; cursor: pointer; display: flex; justify-content: space-between; gap: 8px; }
  .neighbors div:hover { background: rgba(79,209,255,.12); }
  .neighbors small { color: var(--muted); }
  .hint { color: var(--muted); }
  #legend { right: 10px; bottom: 226px; width: 320px; padding: 8px 10px; display: flex; flex-wrap: wrap; gap: 6px; }
  .legend-item { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; padding: 3px 8px; border-radius: 6px; background: rgba(255,255,255,.04); }
  .legend-item.off { opacity: .35; }
  .legend-item i { width: 10px; height: 10px; border-radius: 2px; }
  #log { right: 10px; bottom: 10px; width: 320px; height: 206px; display: flex; flex-direction: column; }
  #logList { overflow-y: auto; flex: 1; padding: 6px 10px; font-family: ui-monospace, "Cascadia Mono", monospace; font-size: 11.5px; }
  #logList div { padding: 3px 0; border-bottom: 1px dashed rgba(255,255,255,.06); }
  .tag { color: var(--accent); } .ok { color: var(--ok); } .bad { color: var(--bad); }
  #tooltip { position: absolute; pointer-events: none; padding: 5px 9px; border-radius: 6px; background: rgba(5,10,18,.92);
    border: 1px solid var(--line); display: none; white-space: nowrap; font-size: 12px; z-index: 5; }
  .label { color: #e9f3ff; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: rgba(5,10,18,.66);
    border: 1px solid rgba(255,255,255,.12); white-space: nowrap; max-width: 220px; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  #help { position: absolute; left: 50%; bottom: 12px; transform: translateX(-50%); color: var(--muted); font-size: 11px;
    background: rgba(5,10,18,.6); padding: 4px 12px; border-radius: 999px; pointer-events: none; }
  #error { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; text-align: center; padding: 24px; color: var(--muted); }
  @media (max-width: 980px) { #regions, #legend, #log { display: none; } #inspector { width: 260px; } }
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
<div id="topbar" class="panel">
  <div class="brand" id="brand"></div>
  <div id="resources" style="display:contents"></div>
  <input id="search" list="titles" placeholder="搜尋實體… (Enter)">
  <datalist id="titles"></datalist>
</div>
<div id="regions" class="panel"><h3><span>領地 · 社群</span><span id="regionCount"></span></h3><div id="regionList"></div></div>
<div id="minimap" class="panel"><h3><span>小地圖</span><span>點擊移動</span></h3><canvas></canvas></div>
<div id="inspector" class="panel"><h3><span id="inspectorKind">情報</span><span id="inspectorClose" style="cursor:pointer">✕</span></h3><div id="inspectorBody"></div></div>
<div id="legend" class="panel"></div>
<div id="log" class="panel"><h3><span>指令紀錄 · 問答</span><span id="logCount"></span></h3><div id="logList"></div></div>
<div id="tooltip"></div>
<div id="help">拖曳旋轉 · 右鍵平移 · 滾輪縮放 · WASD 移動 · 點擊建築查看情報</div>
<div id="error">無法載入 three.js（需要可連線至 cdn.jsdelivr.net）。</div>
<script type="application/json" id="data">__DASHBOARD_DATA__</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const DATA = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const TYPE_COLORS = ["#4fd1ff", "#ffc857", "#5ee6a0", "#ff7eb6", "#a78bfa", "#ff9f5a", "#7dd3fc", "#e2e8f0"];
const REGION_COLORS = ["#2563eb", "#0d9488", "#7c3aed", "#c2410c", "#be123c", "#15803d", "#a16207", "#0369a1", "#9333ea", "#b91c1c"];
const FRONTIER = -1;
const typeColor = {};
DATA.types.forEach((t, i) => { typeColor[t.name] = TYPE_COLORS[i % TYPE_COLORS.length]; });
const regionColor = (r) => r === FRONTIER ? "#475569" : REGION_COLORS[r % REGION_COLORS.length];

// ---------- HUD ----------
$("brand").innerHTML = `${esc(DATA.project.name)}<small>${esc(DATA.project.id)} · ${esc(DATA.project.status)}</small>`;
const S = DATA.stats;
const resources = [
  ["實體", S.entities], ["關係", S.relationships], ["社群", S.communities], ["文件頁", S.documents],
  ["Chunk", S.text_units], ["問答", `${S.queries_completed}/${S.queries}`], ["建圖", `${Math.round(S.index_runtime_s)}s`],
];
$("resources").innerHTML = resources.map(([k, v]) => `<div class="res"><b>${esc(v)}</b><span>${k}</span></div>`).join("");
$("titles").innerHTML = DATA.nodes.map((n) => `<option value="${esc(n.t)}">`).join("");

// ---------- 版面配置：社群 = 六角形領地，實體 = 建築 ----------
const groups = new Map();
DATA.nodes.forEach((n, i) => { if (!groups.has(n.r)) groups.set(n.r, []); groups.get(n.r).push(i); });
for (const members of groups.values()) members.sort((a, b) => DATA.nodes[b].deg - DATA.nodes[a].deg);
const SPACING = 2.4;
const plots = [];
const sortedRegions = [...groups.keys()].filter((r) => r !== FRONTIER).sort((a, b) => groups.get(b).length - groups.get(a).length);
for (const r of sortedRegions) {
  const radius = Math.sqrt(groups.get(r).length) * SPACING * 0.62 + 3;
  let angle = 0, dist = 0, x = 0, z = 0;
  for (let step = 0; step < 5000; step++) {
    x = Math.cos(angle) * dist; z = Math.sin(angle) * dist;
    if (plots.every((p) => Math.hypot(p.x - x, p.z - z) > p.radius + radius + 2.5)) break;
    angle += 0.45; dist += 0.35;
  }
  plots.push({ r, x, z, radius });
}
const coreRadius = plots.reduce((m, p) => Math.max(m, Math.hypot(p.x, p.z) + p.radius), 10);
const positions = new Array(DATA.nodes.length);
for (const p of plots) {
  groups.get(p.r).forEach((nodeIndex, k) => {        // 向日葵排列：重要建築在中心
    const a = k * 2.39996, d = Math.sqrt(k) * SPACING * 0.62;
    positions[nodeIndex] = new THREE.Vector3(p.x + Math.cos(a) * d, 0, p.z + Math.sin(a) * d);
  });
}
const frontier = groups.get(FRONTIER) || [];
const frontierInner = coreRadius + 6;
frontier.forEach((nodeIndex, k) => {                  // 邊境：未歸入社群的實體圍成外環
  const perRing = Math.max(24, Math.floor((2 * Math.PI * frontierInner) / 2.2));
  const ring = Math.floor(k / perRing), a = (k % perRing) / perRing * Math.PI * 2 + ring * 0.13;
  const d = frontierInner + ring * 2.2;
  positions[nodeIndex] = new THREE.Vector3(Math.cos(a) * d, 0, Math.sin(a) * d);
});
const worldRadius = frontier.length ? frontierInner + Math.ceil(frontier.length / Math.max(24, Math.floor((2 * Math.PI * frontierInner) / 2.2))) * 2.2 + 4 : coreRadius + 6;
const heightOf = (n) => 0.8 + Math.log2(n.deg + 1) * 1.6;

// ---------- 場景 ----------
const container = $("scene");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (err) {
  $("error").style.display = "flex"; $("error").textContent = "此瀏覽器不支援 WebGL，無法顯示 3D 儀表板。";
  throw err;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
container.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
Object.assign(labelRenderer.domElement.style, { position: "absolute", top: "0", pointerEvents: "none" });
container.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#070b12");
scene.fog = new THREE.Fog("#070b12", worldRadius * 1.6, worldRadius * 4);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.5, worldRadius * 10);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.44;
controls.minDistance = 8;
controls.maxDistance = worldRadius * 3.2;
camera.position.set(0, worldRadius * 1.9, worldRadius * 1.9);

scene.add(new THREE.HemisphereLight("#9cc9ff", "#0b1220", 0.9));
const sun = new THREE.DirectionalLight("#fff3dd", 1.6);
sun.position.set(worldRadius * 0.6, worldRadius * 1.2, worldRadius * 0.4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -worldRadius, right: worldRadius, top: worldRadius, bottom: -worldRadius, far: worldRadius * 4 });
scene.add(sun);

const ground = new THREE.Mesh(new THREE.CircleGeometry(worldRadius * 1.15, 96),
  new THREE.MeshStandardMaterial({ color: "#0c1420", roughness: 0.95 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
const grid = new THREE.PolarGridHelper(worldRadius * 1.15, 24, 12, 96, "#16324a", "#0f2133");
grid.position.y = 0.01; scene.add(grid);

const regionLabels = [];
for (const p of plots) {
  const color = new THREE.Color(regionColor(p.r));
  const pad = new THREE.Mesh(new THREE.CylinderGeometry(p.radius, p.radius + 0.4, 0.4, 6),
    new THREE.MeshStandardMaterial({ color: color.clone().multiplyScalar(0.35), roughness: 0.7, metalness: 0.2 }));
  pad.position.set(p.x, 0.2, p.z); pad.receiveShadow = true; scene.add(pad);
  const edge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.CylinderGeometry(p.radius, p.radius + 0.4, 0.4, 6)),
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85 }));
  edge.position.copy(pad.position); scene.add(edge);
  const div = document.createElement("div"); div.className = "label";
  const region = DATA.regions[p.r];
  div.textContent = region.t; div.style.borderColor = regionColor(p.r);
  const label = new CSS2DObject(div);
  label.position.set(p.x, 0.5, p.z + p.radius + 0.6); scene.add(label);
  regionLabels.push(label);
}
if (frontier.length) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(frontierInner - 1.2, worldRadius - 2.8, 96),
    new THREE.MeshStandardMaterial({ color: "#111c2b", roughness: 1, side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.03; ring.receiveShadow = true; scene.add(ring);
  const div = document.createElement("div"); div.className = "label"; div.textContent = `邊境 · 未歸入社群 (${frontier.length})`;
  const label = new CSS2DObject(div); label.position.set(0, 0.5, worldRadius - 1.5); scene.add(label);
}

// 建築（InstancedMesh：一次繪製全部實體）
const boxGeometry = new THREE.BoxGeometry(1, 1, 1); boxGeometry.translate(0, 0.5, 0);
const buildings = new THREE.InstancedMesh(boxGeometry,
  new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.35 }), DATA.nodes.length);
buildings.castShadow = true; buildings.receiveShadow = true;
const matrix = new THREE.Matrix4(), quaternion = new THREE.Quaternion(), tmpColor = new THREE.Color();
const visibleTypes = new Set(DATA.types.map((t) => t.name));
const baseY = (i) => DATA.nodes[i].r === FRONTIER ? 0.03 : 0.4;
function layoutBuildings() {
  DATA.nodes.forEach((n, i) => {
    const visible = visibleTypes.has(n.y);
    const w = n.r === FRONTIER ? 0.8 : 1.15;
    const pos = positions[i].clone(); pos.y = baseY(i);
    matrix.compose(pos, quaternion, visible ? new THREE.Vector3(w, heightOf(n), w) : new THREE.Vector3(0.0001, 0.0001, 0.0001));
    buildings.setMatrixAt(i, matrix);
    buildings.setColorAt(i, tmpColor.set(typeColor[n.y]));
  });
  buildings.instanceMatrix.needsUpdate = true; buildings.instanceColor.needsUpdate = true;
}
layoutBuildings();
scene.add(buildings);

// 關係：建築頂端之間的弧線
const roof = (i) => positions[i].clone().setY(baseY(i) + heightOf(DATA.nodes[i]));
function arcPoints(a, b, segments = 14) {
  const p0 = roof(a), p2 = roof(b);
  const mid = p0.clone().add(p2).multiplyScalar(0.5);
  mid.y += Math.min(14, 1.5 + p0.distanceTo(p2) * 0.22);
  return new THREE.QuadraticBezierCurve3(p0, mid, p2).getPoints(segments);
}
const curves = DATA.links.map(([s, t]) => arcPoints(s, t));
function linkGeometry(indices, colorFn) {
  const verts = [], colors = [];
  for (const li of indices) {
    const pts = curves[li], c = colorFn(li);
    for (let k = 0; k < pts.length - 1; k++) {
      verts.push(pts[k].x, pts[k].y, pts[k].z, pts[k + 1].x, pts[k + 1].y, pts[k + 1].z);
      const fade = 0.55 + 0.45 * Math.sin((k / (pts.length - 1)) * Math.PI);
      colors.push(c.r * fade, c.g * fade, c.b * fade, c.r * fade, c.g * fade, c.b * fade);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return g;
}
const allLinks = DATA.links.map((_, i) => i);
const linkMaterial = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending });
let linkLines = new THREE.LineSegments(linkGeometry(allLinks, (li) => new THREE.Color(regionColor(DATA.nodes[DATA.links[li][0]].r)).lerp(new THREE.Color("#4fd1ff"), 0.5)), linkMaterial);
scene.add(linkLines);
const highlight = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 1, depthWrite: false, blending: THREE.AdditiveBlending }));
scene.add(highlight);

// 資料封包：沿關係弧線移動的光點
const PACKETS = Math.min(140, DATA.links.length);
const packetGeo = new THREE.BufferGeometry();
packetGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(PACKETS * 3), 3));
const packets = new THREE.Points(packetGeo, new THREE.PointsMaterial({ color: "#ffe9a8", size: 0.55, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending }));
scene.add(packets);
const packetState = Array.from({ length: PACKETS }, () => ({ li: Math.floor(Math.random() * DATA.links.length), t: Math.random(), v: 0.15 + Math.random() * 0.25 }));
let packetFocus = null;

// 選取標記
const marker = new THREE.Mesh(new THREE.RingGeometry(0.95, 1.25, 48), new THREE.MeshBasicMaterial({ color: "#ffc857", transparent: true, side: THREE.DoubleSide }));
marker.rotation.x = -Math.PI / 2; marker.visible = false; scene.add(marker);
const beacon = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 30, 8), new THREE.MeshBasicMaterial({ color: "#ffc857", transparent: true, opacity: 0.35 }));
beacon.visible = false; scene.add(beacon);

// ---------- 鄰接表 ----------
const neighbors = DATA.nodes.map(() => []);
DATA.links.forEach(([s, t, w, d], li) => { neighbors[s].push({ o: t, w, d, li }); neighbors[t].push({ o: s, w, d, li }); });
neighbors.forEach((list) => list.sort((a, b) => b.w - a.w));
const maxDeg = Math.max(1, ...DATA.nodes.map((n) => n.deg));
const maxFreq = Math.max(1, ...DATA.nodes.map((n) => n.freq));

// ---------- 攝影機飛行 ----------
let flight = null;
function flyTo(target, distance) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.y < 0.45) { dir.y = 0.75; dir.normalize(); }
  flight = { t: 0, fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toTarget: target.clone(), toPos: target.clone().add(dir.multiplyScalar(distance)) };
}
const ease = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

// ---------- 情報面板 ----------
let selected = null;
function bar(label, value, max) {
  return `<span>${label}</span><div class="bar"><i style="width:${Math.max(3, (value / max) * 100)}%"></i></div><span>${value}</span>`;
}
function selectNode(i, fly = true) {
  selected = i;
  const n = DATA.nodes[i];
  const region = n.r === FRONTIER ? "邊境（未歸入社群）" : DATA.regions[n.r].t;
  $("inspectorKind").textContent = "單位情報";
  $("inspectorBody").innerHTML = `
    <div class="title">${esc(n.t)}</div>
    <span class="chip" style="color:${typeColor[n.y]}">${esc(n.y)}</span>
    <span class="chip" style="color:${regionColor(n.r)}">${esc(region)}</span>
    <div class="bars">${bar("連結度", n.deg, maxDeg)}${bar("出現次數", n.freq, maxFreq)}</div>
    <div class="desc">${esc(n.d) || '<span class="hint">無描述</span>'}</div>
    <div class="neighbors"><div style="cursor:default;color:var(--accent)">關聯單位 (${neighbors[i].length})</div>
      ${neighbors[i].slice(0, 40).map((x) => `<div data-node="${x.o}" title="${esc(x.d)}"><span>${esc(DATA.nodes[x.o].t)}</span><small>w ${x.w}</small></div>`).join("")}
    </div>`;
  const top = roof(i);
  marker.position.set(positions[i].x, baseY(i) + 0.02, positions[i].z); marker.visible = true;
  beacon.position.set(top.x, top.y + 15, top.z); beacon.visible = true;
  const linkIdx = neighbors[i].map((x) => x.li);
  highlight.geometry.dispose();
  highlight.geometry = linkGeometry(linkIdx, () => new THREE.Color("#ffc857"));
  linkMaterial.opacity = 0.1;
  packetFocus = linkIdx.length ? linkIdx : null;
  packetState.forEach((p) => { if (packetFocus) p.li = packetFocus[Math.floor(Math.random() * packetFocus.length)]; });
  if (fly) flyTo(positions[i].clone().setY(heightOf(n) * 0.5), 26);
  setActiveRegion(n.r);
}
function selectRegion(r) {
  const p = plots.find((x) => x.r === r); if (!p) return;
  clearSelection(false);
  const region = DATA.regions[r];
  const members = groups.get(r) || [];
  $("inspectorKind").textContent = "領地情報";
  $("inspectorBody").innerHTML = `
    <div class="title">${esc(region.t)}</div>
    <span class="chip" style="color:${regionColor(r)}">Community ${region.c}</span>
    ${region.rank != null ? `<span class="chip" style="color:var(--gold)">重要度 ${region.rank}</span>` : ""}
    <div class="bars">${bar("實體數", members.length, Math.max(...[...groups.values()].map((g) => g.length)))}</div>
    <div class="desc">${esc(region.s) || '<span class="hint">此社群尚無報告摘要</span>'}</div>
    <div class="neighbors"><div style="cursor:default;color:var(--accent)">主要建築</div>
      ${members.slice(0, 25).map((m) => `<div data-node="${m}"><span>${esc(DATA.nodes[m].t)}</span><small>deg ${DATA.nodes[m].deg}</small></div>`).join("")}
    </div>`;
  flyTo(new THREE.Vector3(p.x, 0, p.z), p.radius * 3.2 + 14);
  setActiveRegion(r);
}
function clearSelection(resetPanel = true) {
  selected = null; marker.visible = false; beacon.visible = false;
  highlight.geometry.dispose(); highlight.geometry = new THREE.BufferGeometry();
  linkMaterial.opacity = 0.32; packetFocus = null;
  if (resetPanel) overview();
}
function overview() {
  setActiveRegion(null);
  $("inspectorKind").textContent = "戰況總覽";
  const isolated = DATA.nodes.filter((n) => n.deg === 0).length;
  const hubs = DATA.nodes.map((n, i) => [n, i]).sort((a, b) => b[0].deg - a[0].deg).slice(0, 8);
  $("inspectorBody").innerHTML = `
    <div class="bars">
      ${bar("社群覆蓋", DATA.nodes.length - frontier.length, DATA.nodes.length)}
      ${bar("孤立實體", isolated, DATA.nodes.length)}
      ${bar("問答完成", S.queries_completed, Math.max(1, S.queries))}
    </div>
    <div class="neighbors"><div style="cursor:default;color:var(--accent)">樞紐實體 Top 8</div>
      ${hubs.map(([n, i]) => `<div data-node="${i}"><span>${esc(n.t)}</span><small>deg ${n.deg}</small></div>`).join("")}
    </div>
    <p class="hint">點擊地圖上的建築或左側領地以查看情報；建築高度代表連結度，顏色代表實體類型。</p>`;
}
$("inspectorBody").addEventListener("click", (e) => {
  const row = e.target.closest("[data-node]"); if (row) selectNode(Number(row.dataset.node));
});
$("inspectorClose").onclick = () => clearSelection();

// ---------- 領地列表 / 圖例 / 紀錄 ----------
$("regionCount").textContent = DATA.regions.length;
const regionOrder = DATA.regions.map((r, i) => i).sort((a, b) => (DATA.regions[b].rank ?? 0) - (DATA.regions[a].rank ?? 0));
$("regionList").innerHTML = regionOrder.map((r) => {
  const reg = DATA.regions[r];
  return `<div class="region" data-region="${r}"><i class="dot" style="background:${regionColor(r)}"></i>
    <span class="name" title="${esc(reg.t)}">${esc(reg.t)}</span><span class="meta">${reg.rank ?? "–"} ★ · ${reg.n}</span></div>`;
}).join("") || '<div class="hint" style="padding:10px">尚無社群資料</div>';
$("regionList").addEventListener("click", (e) => { const row = e.target.closest("[data-region]"); if (row) selectRegion(Number(row.dataset.region)); });
function setActiveRegion(r) {
  document.querySelectorAll(".region").forEach((el) => el.classList.toggle("active", Number(el.dataset.region) === r));
}
$("legend").innerHTML = DATA.types.map((t) => `<div class="legend-item" data-type="${esc(t.name)}"><i style="background:${typeColor[t.name]}"></i>${esc(t.name)} <span class="hint">${t.count}</span></div>`).join("");
$("legend").addEventListener("click", (e) => {
  const item = e.target.closest("[data-type]"); if (!item) return;
  const type = item.dataset.type;
  visibleTypes.has(type) ? visibleTypes.delete(type) : visibleTypes.add(type);
  item.classList.toggle("off", !visibleTypes.has(type));
  layoutBuildings();
});
const METHOD = { local: "LOCAL", global: "GLOBAL", drift: "DRIFT", basic: "BASIC", auto: "AUTO" };
$("logCount").textContent = S.queries ? `${S.queries} 筆` : "";
$("logList").innerHTML = DATA.events.map((ev) => {
  const ok = ev.s === "COMPLETED";
  const time = ev.at ? ev.at.slice(5, 16).replace("T", " ") : "";
  return `<div><span class="hint">${esc(time)}</span> <span class="tag">[${METHOD[ev.m] || esc(ev.m)}]</span> ${esc(ev.q)} <span class="${ok ? "ok" : "bad"}">${ok ? "✓" : esc(ev.s)}${ev.sec ? ` ${Math.round(ev.sec)}s` : ""}</span></div>`;
}).join("") || '<div class="hint">尚無問答紀錄。到「2 問答測試」提問後會顯示在這裡。</div>';

// ---------- 搜尋 ----------
$("search").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const q = e.target.value.trim().toLowerCase(); if (!q) return;
  const exact = DATA.nodes.findIndex((n) => n.t.toLowerCase() === q);
  const hit = exact >= 0 ? exact : DATA.nodes.findIndex((n) => n.t.toLowerCase().includes(q));
  if (hit >= 0) selectNode(hit);
});

// ---------- 滑鼠互動 ----------
const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
const tooltip = $("tooltip");
function pick(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(buildings)[0];
  return hit && visibleTypes.has(DATA.nodes[hit.instanceId].y) ? hit.instanceId : null;
}
let downAt = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5 || e.button !== 0) return;
  const i = pick(e);
  i == null ? clearSelection() : selectNode(i, false);
});
renderer.domElement.addEventListener("pointermove", (e) => {
  const i = pick(e);
  renderer.domElement.style.cursor = i == null ? "" : "pointer";
  if (i == null) { tooltip.style.display = "none"; return; }
  const n = DATA.nodes[i];
  tooltip.innerHTML = `<b>${esc(n.t)}</b> <span style="color:${typeColor[n.y]}">${esc(n.y)}</span> <span class="hint">deg ${n.deg}</span>`;
  tooltip.style.display = "block";
  tooltip.style.left = `${e.clientX + 14}px`; tooltip.style.top = `${e.clientY + 12}px`;
});

// WASD 平移
const keys = new Set();
addEventListener("keydown", (e) => { if (e.target.tagName !== "INPUT") keys.add(e.key.toLowerCase()); });
addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
function panWithKeys(dt) {
  const forward = new THREE.Vector3(); camera.getWorldDirection(forward); forward.y = 0; forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const move = new THREE.Vector3();
  if (keys.has("w") || keys.has("arrowup")) move.add(forward);
  if (keys.has("s") || keys.has("arrowdown")) move.sub(forward);
  if (keys.has("d") || keys.has("arrowright")) move.add(right);
  if (keys.has("a") || keys.has("arrowleft")) move.sub(right);
  if (!move.lengthSq()) return;
  move.normalize().multiplyScalar(dt * camera.position.distanceTo(controls.target) * 0.8);
  camera.position.add(move); controls.target.add(move); flight = null;
}

// ---------- 小地圖 ----------
const mini = document.querySelector("#minimap canvas"), mctx = mini.getContext("2d");
function sizeMinimap() { mini.width = mini.clientWidth * devicePixelRatio; mini.height = mini.clientHeight * devicePixelRatio; }
const toMini = (x, z) => {
  const s = Math.min(mini.width, mini.height) / (worldRadius * 2.3);
  return [mini.width / 2 + x * s, mini.height / 2 + z * s, s];
};
function drawMinimap() {
  mctx.clearRect(0, 0, mini.width, mini.height);
  const [cx, cy, s] = toMini(0, 0);
  mctx.strokeStyle = "rgba(79,209,255,.25)"; mctx.beginPath(); mctx.arc(cx, cy, worldRadius * s, 0, Math.PI * 2); mctx.stroke();
  for (const p of plots) {
    const [x, y] = toMini(p.x, p.z);
    mctx.fillStyle = regionColor(p.r) + "88"; mctx.beginPath(); mctx.arc(x, y, p.radius * s, 0, Math.PI * 2); mctx.fill();
  }
  mctx.fillStyle = "rgba(148,163,184,.6)";
  for (const i of frontier) { const [x, y] = toMini(positions[i].x, positions[i].z); mctx.fillRect(x - 0.7, y - 0.7, 1.4, 1.4); }
  if (selected != null) {
    const [x, y] = toMini(positions[selected].x, positions[selected].z);
    mctx.fillStyle = "#ffc857"; mctx.beginPath(); mctx.arc(x, y, 4 * devicePixelRatio, 0, Math.PI * 2); mctx.fill();
  }
  // 視野：攝影機位置與注視點
  const [tx, ty] = toMini(controls.target.x, controls.target.z), [px, py] = toMini(camera.position.x, camera.position.z);
  const viewR = Math.max(6, camera.position.distanceTo(controls.target) * 0.45) * s;
  mctx.strokeStyle = "#ffffff"; mctx.lineWidth = 1.2 * devicePixelRatio;
  mctx.strokeRect(tx - viewR, ty - viewR * 0.7, viewR * 2, viewR * 1.4);
  mctx.strokeStyle = "rgba(255,255,255,.35)"; mctx.beginPath(); mctx.moveTo(px, py); mctx.lineTo(tx, ty); mctx.stroke();
}
mini.addEventListener("click", (e) => {
  const rect = mini.getBoundingClientRect();
  const [cx, cy, s] = toMini(0, 0);
  const x = ((e.clientX - rect.left) * devicePixelRatio - cx) / s, z = ((e.clientY - rect.top) * devicePixelRatio - cy) / s;
  flyTo(new THREE.Vector3(x, 0, z), camera.position.distanceTo(controls.target));
});

// ---------- 迴圈 ----------
function resize() {
  document.documentElement.style.setProperty("--hud-top", `${$("topbar").offsetHeight + 18}px`);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); labelRenderer.setSize(innerWidth, innerHeight); sizeMinimap();
}
addEventListener("resize", resize); resize();
overview();
flyTo(new THREE.Vector3(0, 0, 0), worldRadius * 1.55);
const clock = new THREE.Clock();
function tick() {
  const dt = Math.min(clock.getDelta(), 0.05), time = clock.elapsedTime;
  if (flight) {
    flight.t = Math.min(1, flight.t + dt / 1.1);
    const k = ease(flight.t);
    camera.position.lerpVectors(flight.fromPos, flight.toPos, k);
    controls.target.lerpVectors(flight.fromTarget, flight.toTarget, k);
    if (flight.t >= 1) flight = null;
  }
  panWithKeys(dt);
  controls.update();
  const pos = packetGeo.attributes.position;
  packetState.forEach((p, k) => {
    p.t += p.v * dt;
    if (p.t >= 1) { p.t = 0; p.li = packetFocus ? packetFocus[Math.floor(Math.random() * packetFocus.length)] : Math.floor(Math.random() * DATA.links.length); }
    const pts = curves[p.li]; if (!pts) return;
    const f = p.t * (pts.length - 1), a = Math.floor(f), b = Math.min(a + 1, pts.length - 1);
    pos.setXYZ(k, ...pts[a].clone().lerp(pts[b], f - a).toArray());
  });
  pos.needsUpdate = true;
  if (marker.visible) { const s = 1 + 0.15 * Math.sin(time * 4); marker.scale.set(s, s, s); marker.material.opacity = 0.6 + 0.4 * Math.sin(time * 4); }
  const far = camera.position.distanceTo(controls.target) > worldRadius * 3;
  regionLabels.forEach((l) => { l.visible = !far; });
  renderer.render(scene, camera); labelRenderer.render(scene, camera);
  drawMinimap();
  requestAnimationFrame(tick);
}
if (!DATA.links.length) packets.visible = false;
tick();
</script>
<script>
  // importmap 模組載入失敗時（例如離線）顯示提示
  setTimeout(() => { if (!document.querySelector("#scene canvas")) document.getElementById("error").style.display = "flex"; }, 6000);
</script>
</body>
</html>
"""
