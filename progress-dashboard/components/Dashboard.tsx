"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";

import { STATUS_ORDER, priorityRank, statusRank } from "@/lib/layout";
import { plainText } from "@/lib/markdown";
import type { Commit, PlanItem, ProgressData, TestReport } from "@/lib/types";

import { Icon, type IconName } from "./icons";
import { commitColor, recentCommits, statusStyle, type CameraCommand, type Selection } from "./theme";

const World = dynamic(() => import("./World"), {
  ssr: false,
  loading: () => <div className="loading">載入工地地圖…</div>,
});

const REFRESH_MS = 20_000;

export default function Dashboard({ initial }: { initial: ProgressData }) {
  const [data, setData] = useState(initial);
  const [selection, setSelection] = useState<Selection>(null);
  const [statusFilter, setStatusFilter] = useState<Set<string>>(new Set());
  const [cameraCommand, setCameraCommand] = useState<CameraCommand | null>(null);
  const [testRunning, setTestRunning] = useState(false);
  const [testError, setTestError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const response = await fetch("/api/progress", { cache: "no-store" });
    if (!response.ok) return;
    const next = (await response.json()) as ProgressData & { testRunActive: boolean };
    setData(next);
    setTestRunning((running) => running || next.testRunActive);
  }, []);

  useEffect(() => {
    const timer = setInterval(() => { refresh().catch(() => undefined); }, REFRESH_MS);
    return () => clearInterval(timer);
  }, [refresh]);

  const runTests = async () => {
    setTestRunning(true);
    setTestError(null);
    try {
      const response = await fetch("/api/tests", { method: "POST" });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? response.statusText);
      setData((current) => ({ ...current, tests: body as TestReport }));
    } catch (error) {
      setTestError(String(error));
    } finally {
      setTestRunning(false);
    }
  };

  const camera = (kind: CameraCommand["kind"]) => setCameraCommand({ kind, nonce: Date.now() });

  return (
    <main className="shell">
      <div className="scene">
        <World data={data} selection={selection} statusFilter={statusFilter} cameraCommand={cameraCommand} onSelect={setSelection} />
      </div>
      <TopNav data={data} onSelect={setSelection} />
      <KpiRow data={data} testRunning={testRunning} onRunTests={runTests} />
      <div className="card map-controls">
        {([["plus", "zoomIn", "放大"], ["minus", "zoomOut", "縮小"], ["rotateLeft", "rotateLeft", "向左旋轉"],
          ["rotateRight", "rotateRight", "向右旋轉"], ["home", "home", "回到全景"]] as [IconName, CameraCommand["kind"], string][])
          .map(([icon, kind, label]) => (
            <button key={kind} title={label} aria-label={label} onClick={() => { if (kind === "home") setSelection(null); camera(kind); }}>
              <Icon name={icon} />
            </button>
          ))}
      </div>
      <DetailCard data={data} selection={selection} onSelect={setSelection} />
      <Tracker data={data} selection={selection} />
      <TabsCard
        data={data}
        selection={selection}
        statusFilter={statusFilter}
        setStatusFilter={setStatusFilter}
        testRunning={testRunning}
        testError={testError}
        onRunTests={runTests}
        onSelect={setSelection}
      />
    </main>
  );
}

/* ---------------- 頂部導覽 ---------------- */

function TopNav({ data, onSelect }: { data: ProgressData; onSelect: (s: Selection) => void }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const plans = data.plan
      .filter((item) => `${item.id} ${item.title}`.toLowerCase().includes(q))
      .map((item) => ({ key: item.id, label: `${item.id} ${plainText(item.title)}`, hint: statusStyle(item.status).label, select: { kind: "plan", id: item.id } as Selection }));
    const modules = (data.tests?.modules ?? [])
      .filter((module) => module.module.toLowerCase().includes(q))
      .map((module) => ({ key: module.module, label: module.module, hint: "測試貨櫃", select: { kind: "module", module: module.module } as Selection }));
    return [...plans, ...modules].slice(0, 8);
  }, [data, query]);
  const failing = data.tests?.failed ?? 0;
  const initials = (data.git.user || "Dev").split(/\s+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase();
  return (
    <nav className="topnav">
      <button className="logo" onClick={() => onSelect(null)}>
        <span className="logo-mark"><Icon name="box" size={22} /></span>
        <span>DevSite</span>
      </button>
      <div className="search">
        <Icon name="search" size={18} />
        <input
          value={query}
          placeholder="搜尋改善項目、測試模組…"
          onChange={(event) => { setQuery(event.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && results[0]) { onSelect(results[0].select); setOpen(false); }
          }}
        />
        <kbd>/</kbd>
        {open && results.length > 0 && (
          <div className="search-results">
            {results.map((result) => (
              <button key={result.key} onMouseDown={() => { onSelect(result.select); setOpen(false); setQuery(""); }}>
                <span>{result.label}</span><small>{result.hint}</small>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="site-pill">
        <span className="site-badge"><Icon name="branch" size={14} /></span>
        <div>
          <b>{data.repoName}</b>
          <small>{data.git.branch} · {data.git.changes.length} 個未提交變更</small>
        </div>
      </div>
      <LiveClock />
      <button className="bell" title={failing ? `${failing} 項測試失敗` : "沒有警示"}>
        <Icon name="bell" size={20} />
        {failing > 0 && <i />}
      </button>
      <div className="user">
        <span className="avatar">{initials}</span>
        <div>
          <b>{data.git.user || "開發者"}</b>
          <small>專案開發者</small>
        </div>
      </div>
    </nav>
  );
}

function LiveClock() {
  const [now, setNow] = useState<string | null>(null);
  useEffect(() => {
    const tick = () => setNow(new Date().toLocaleTimeString("zh-TW", { hour: "2-digit", minute: "2-digit", hour12: false }));
    tick();
    const timer = setInterval(tick, 10_000);
    return () => clearInterval(timer);
  }, []);
  return <div className="live"><i />Live {now ?? "--:--"}</div>;
}

/* ---------------- KPI 卡片 ---------------- */

function KpiRow({ data, testRunning, onRunTests }: { data: ProgressData; testRunning: boolean; onRunTests: () => void }) {
  const done = data.plan.filter((item) => item.status === "完成").length;
  const active = data.plan.filter((item) => item.status === "進行中").length;
  const weekly = useMemo(() => {
    const weekAgo = Date.now() - 7 * 86_400_000;
    return data.git.commits.filter((commit) => Date.parse(commit.date) >= weekAgo).length;
  }, [data.git.commits]);
  const tests = data.tests;
  return (
    <div className="kpis">
      <Kpi icon="crane" label="計畫完成" value={`${done}`} unit={`/ ${data.plan.length}`}
        delta={active ? `${active} 施工中` : undefined} sub="docs/改善計畫.md" />
      <Kpi
        icon="check"
        label="測試通過"
        value={tests ? `${tests.passed}` : "—"}
        unit={tests ? `/ ${tests.total}` : ""}
        delta={tests ? (tests.failed ? `${tests.failed} 失敗` : "全數通過") : undefined}
        tone={tests?.failed ? "bad" : "good"}
        sub={tests ? <>上次 <Time iso={tests.ranAt} /> · {tests.durationS}s</> : "尚未執行"}
        action={
          <button className="mini-run" onClick={onRunTests} disabled={testRunning} title="執行 pytest">
            {testRunning ? <span className="spinner" /> : <Icon name="play" size={14} />}
          </button>
        }
      />
      <Kpi icon="truck" label="本週提交" value={`${weekly}`} unit={`/ ${data.git.commits.length}`}
        delta={data.git.commits[0] ? data.git.commits[0].type : undefined} sub={`最新 ${data.git.commits[0]?.hash ?? "—"}`} />
    </div>
  );
}

function Kpi({ icon, label, value, unit, delta, tone = "good", sub, action }: {
  icon: IconName; label: string; value: string; unit: string; delta?: string; tone?: "good" | "bad";
  sub: ReactNode; action?: ReactNode;
}) {
  return (
    <div className="card kpi">
      <span className="kpi-icon"><Icon name={icon} size={22} /></span>
      <div className="kpi-body">
        <span className="kpi-label">{label}</span>
        <div className="kpi-value">
          <b>{value}</b><span className="unit">{unit}</span>
          {delta && <span className={`delta ${tone}`}>{delta}</span>}
        </div>
        <span className="kpi-sub">{sub}</span>
      </div>
      {action}
    </div>
  );
}

/* ---------------- 右側詳細卡 ---------------- */

function DetailCard({ data, selection, onSelect }: { data: ProgressData; selection: Selection; onSelect: (s: Selection) => void }) {
  let body: ReactNode = <SiteOverview data={data} onSelect={onSelect} />;
  if (selection?.kind === "plan") {
    const item = data.plan.find((plan) => plan.id === selection.id);
    if (item) body = <PlanDetail item={item} />;
  } else if (selection?.kind === "module") {
    const module = data.tests?.modules.find((m) => m.module === selection.module);
    if (module) {
      const total = module.passed + module.failed + module.skipped;
      body = (
        <>
          <Header eyebrow="CONTAINER · 測試模組" title={module.module} subtitle={`tests/${module.module}.py`} icon="box" />
          <StatusRow pill={module.failed ? { label: "有失敗", fg: "#c43d43", bg: "#fde8e8" } : { label: "全數通過", fg: "#15803d", bg: "#e3f6ea" }}
            text={`${module.passed} 通過 · ${module.failed} 失敗 · ${module.skipped} 略過`} />
          <div className="stat-grid">
            <Stat label="通過率" value={`${module.passed} / ${total}`} ratio={module.passed / Math.max(1, total)} tone={module.failed ? "bad" : "good"} />
            <Stat label="失敗" value={`${module.failed}`} unit="項" />
          </div>
          <h4>失敗項目</h4>
          {module.failures.map((failure) => (
            <div key={failure.name} className="failure">
              <b>{failure.name}</b>
              <pre>{failure.message}</pre>
            </div>
          ))}
          {!module.failures.length && <p className="muted">此模組全部通過。</p>}
        </>
      );
    }
  } else if (selection?.kind === "commit") {
    const commit = data.git.commits.find((c) => c.hash === selection.hash);
    if (commit) body = <CommitDetail commit={commit} />;
  } else if (selection?.kind === "zone") {
    const items = data.plan.filter((item) => item.district === selection.name);
    body = (
      <>
        <Header eyebrow="ZONE · 分區" title={`${selection.name}區`} subtitle={`${items.length} 個改善項目`} icon="pin" />
        <ItemTable items={items} onSelect={onSelect} />
      </>
    );
  }
  return (
    <aside className="card detail">
      {selection && (
        <button className="detail-close" onClick={() => onSelect(null)} aria-label="關閉"><Icon name="close" size={16} /></button>
      )}
      <div className="detail-scroll">{body}</div>
    </aside>
  );
}

function SiteOverview({ data, onSelect }: { data: ProgressData; onSelect: (s: Selection) => void }) {
  const counts = countBy(data.plan.map((item) => item.status));
  const done = counts.get("完成") ?? 0;
  const tests = data.tests;
  const zones = useMemo(() => {
    const groups = new Map<string, PlanItem[]>();
    for (const item of data.plan) groups.set(item.district, [...(groups.get(item.district) ?? []), item]);
    return [...groups.entries()];
  }, [data.plan]);
  const weekly = recentCommits(data.git.commits).length;
  return (
    <>
      <Header eyebrow={`SITE · ${data.git.branch}`} title={data.repoName} subtitle={plainText(data.handoff?.title ?? "開發進度總覽")} icon="crane" />
      <StatusRow
        pill={done === data.plan.length ? { label: "全部完工", fg: "#15803d", bg: "#e3f6ea" } : { label: "施工中", fg: "#2c56e0", bg: "#e6edff" }}
        text={`${counts.get("進行中") ?? 0} 施工中 · ${counts.get("待確認") ?? 0} 待確認 · ${counts.get("待辦") ?? 0} 待開工`}
      />
      <div className="stat-grid">
        <Stat label="計畫完成" value={`${done}`} unit={`/ ${data.plan.length}`} ratio={done / Math.max(1, data.plan.length)} />
        <Stat label="測試通過" value={tests ? `${tests.passed}` : "—"} unit={tests ? `/ ${tests.total}` : "未執行"}
          ratio={tests ? tests.passed / Math.max(1, tests.total) : undefined} tone={tests?.failed ? "bad" : "good"} />
        <Stat label="道路上的卡車" value={`${weekly}`} unit="個近期提交" />
        <Stat label="未提交檔案" value={`${data.git.changes.length}`} unit="個" />
      </div>
      <h4>分區 <span className="muted">完工 / 總數</span></h4>
      <div className="inventory">
        {zones.map(([name, items]) => {
          const zoneDone = items.filter((item) => item.status === "完成").length;
          const busy = items.some((item) => item.status === "進行中");
          const style = zoneDone === items.length ? statusStyle("完成") : busy ? statusStyle("進行中") : statusStyle("待辦");
          return (
            <button key={name} className="inventory-row" onClick={() => onSelect({ kind: "zone", name })}>
              <span className="inv-icon" style={{ color: style.fg, background: style.bg }}><Icon name="box" size={16} /></span>
              <span className="inv-name">{name}區</span>
              <b>{zoneDone}/{items.length}</b>
              <span className="pill" style={{ color: style.fg, background: style.bg }}>{style.label}</span>
            </button>
          );
        })}
      </div>
      {data.git.changes.length > 0 && (
        <>
          <h4>未提交的變更 <span className="muted">{data.git.changes.length}</span></h4>
          <div className="changes">
            {data.git.changes.slice(0, 8).map((change) => (
              <div key={change.path}><span>{change.code}</span>{change.path}</div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function PlanDetail({ item }: { item: PlanItem }) {
  const style = statusStyle(item.status);
  const blocks = parseDetail(item.detail);
  return (
    <>
      <Header eyebrow={`LOT · ${item.district}區`} title={`${item.id} ${plainText(item.title)}`} subtitle={`類別：${item.category}`} icon="crane" />
      <StatusRow pill={{ label: style.label, fg: style.fg, bg: style.bg }} text={`優先 ${item.priority}${item.uncommitted ? " · 已完成未提交" : ""}`} />
      <div className="stat-grid">
        <Stat label="優先度" value={item.priority} unit={item.priority === "高" ? "優先處理" : ""} />
        <Stat label="提交" value={item.commit ?? (item.uncommitted ? "未提交" : "—")} mono={Boolean(item.commit)} />
      </div>
      {item.result && (
        <>
          <h4>結果</h4>
          <p className="prose">{plainText(item.result)}</p>
        </>
      )}
      {blocks.map((block, index) => (
        <div key={index} className="detail-block">
          {block.label && <h4>{block.label}</h4>}
          {block.lines.map((line, k) => <p key={k} className={`prose ${line.indent ? "indent" : ""}`}>{line.text}</p>)}
        </div>
      ))}
      {!item.result && !blocks.length && <p className="muted">此項目沒有細節段落。</p>}
    </>
  );
}

function CommitDetail({ commit }: { commit: Commit }) {
  return (
    <>
      <Header eyebrow={`TRUCK · ${commit.type}`} title={commit.subject} subtitle={`${commit.author} · ${commit.hash}`} icon="truck" />
      <StatusRow pill={{ label: commit.type, fg: commitColor(commit.type), bg: "#eef2ff" }} text={<Time iso={commit.date} />} />
      <div className="stat-grid">
        <Stat label="變更檔案" value={`${commit.files}`} unit="個" />
        <Stat label="行數" value={`+${commit.insertions}`} unit={`−${commit.deletions}`} />
      </div>
    </>
  );
}

function ItemTable({ items, onSelect }: { items: PlanItem[]; onSelect: (s: Selection) => void }) {
  return (
    <div className="table">
      {items.map((item) => {
        const style = statusStyle(item.status);
        return (
          <button key={item.id} className="table-row" onClick={() => onSelect({ kind: "plan", id: item.id })}>
            <span className="cell-id">{item.id}</span>
            <span className="cell-main">{plainText(item.title)}</span>
            <span className="pill" style={{ color: style.fg, background: style.bg }}>{style.label}</span>
            <Icon name="chevronRight" size={16} className="chev" />
          </button>
        );
      })}
    </div>
  );
}

function Header({ eyebrow, title, subtitle, icon }: { eyebrow: string; title: string; subtitle: string; icon: IconName }) {
  return (
    <div className="detail-header">
      <span className="detail-icon"><Icon name={icon} size={26} /></span>
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
        <span className="muted">{subtitle}</span>
      </div>
    </div>
  );
}

function StatusRow({ pill, text }: { pill: { label: string; fg: string; bg: string }; text: ReactNode }) {
  return (
    <div className="status-row">
      <span className="pill big" style={{ color: pill.fg, background: pill.bg }}>{pill.label}</span>
      <span className="muted">{text}</span>
    </div>
  );
}

function Stat({ label, value, unit, ratio, tone = "good", mono }: {
  label: string; value: string; unit?: string; ratio?: number; tone?: "good" | "bad"; mono?: boolean;
}) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <div><b className={mono ? "mono" : ""}>{value}</b> {unit && <span className="muted">{unit}</span>}</div>
      {ratio !== undefined && (
        <div className="track"><i className={tone} style={{ width: `${Math.max(3, Math.min(100, ratio * 100))}%` }} /></div>
      )}
    </div>
  );
}

/* ---------------- 左下：任務追蹤 ---------------- */

const STAGES: { label: string; icon: IconName }[] = [
  { label: "待開工", icon: "flag" },
  { label: "施工中", icon: "crane" },
  { label: "完工", icon: "box" },
  { label: "已提交", icon: "check" },
];

function stageOf(item: PlanItem): number {
  if (item.status === "完成") return item.uncommitted ? 2 : 3;
  if (item.status === "進行中") return 1;
  return 0;
}

function Tracker({ data, selection }: { data: ProgressData; selection: Selection }) {
  const item = selection?.kind === "plan" ? data.plan.find((plan) => plan.id === selection.id) : undefined;
  const stageCounts = [0, 0, 0, 0];
  for (const plan of data.plan) if (plan.status !== "不處理") stageCounts[stageOf(plan)] += 1;
  const current = item ? stageOf(item) : -1;
  const nextSteps = data.handoff?.sections.find((section) => section.heading.includes("下一步"))?.items ?? [];
  const [page, setPage] = useState(0);
  const step = nextSteps.length ? nextSteps[page % nextSteps.length] : null;
  return (
    <section className="card tracker">
      <div className="tracker-main">
        <div className="tracker-head">
          <span className="tracker-title"><Icon name="flag" size={18} /> 任務追蹤</span>
          <span className="muted">{item ? `${item.id} · ${item.district}區` : "全部改善項目"}</span>
        </div>
        <ol className="stepper">
          {STAGES.map((stage, index) => {
            const state = item ? (index < current || (index === current && current === 3) ? "done" : index === current ? "current" : "todo") : stageCounts[index] ? "done" : "todo";
            return (
              <li key={stage.label} className={state}>
                <span className="step-dot"><Icon name={state === "done" && item ? "check" : stage.icon} size={16} /></span>
                <b>{stage.label}</b>
                <small>{item ? (index === current ? "目前" : index < current ? "已完成" : "—") : `${stageCounts[index]} 項`}</small>
              </li>
            );
          })}
        </ol>
      </div>
      <div className="next-card">
        <div className="next-head">
          <span className="eyebrow">交接 · 建議下一步</span>
          {nextSteps.length > 1 && (
            <span className="pager">
              <button onClick={() => setPage((p) => (p - 1 + nextSteps.length) % nextSteps.length)} aria-label="上一則"><Icon name="chevronLeft" size={16} /></button>
              {page % nextSteps.length + 1}/{nextSteps.length}
              <button onClick={() => setPage((p) => (p + 1) % nextSteps.length)} aria-label="下一則"><Icon name="chevronRight" size={16} /></button>
            </span>
          )}
        </div>
        <p>{step ? plainText(step) : "docs/交接紀錄.md 沒有「建議下一步」。"}</p>
        <span className="pill" style={{ color: "#2c56e0", background: "#e6edff" }}>{plainText(data.handoff?.title ?? "").replace(/^目前狀態/, "").replace(/[（）()]/g, "") || "交接紀錄"}</span>
      </div>
    </section>
  );
}

/* ---------------- 右下：分頁表格 ---------------- */

type Tab = "plan" | "tests" | "commits" | "handoff";

function TabsCard({ data, statusFilter, setStatusFilter, testRunning, testError, onRunTests, onSelect }: {
  data: ProgressData; selection: Selection; statusFilter: Set<string>; setStatusFilter: (s: Set<string>) => void;
  testRunning: boolean; testError: string | null; onRunTests: () => void; onSelect: (s: Selection) => void;
}) {
  const [tab, setTab] = useState<Tab>("plan");
  const counts = countBy(data.plan.map((item) => item.status));
  const tests = data.tests;
  const okModules = tests?.modules.filter((module) => !module.failed).length ?? 0;
  const items = [...data.plan]
    .filter((item) => statusFilter.size === 0 || statusFilter.has(item.status))
    .sort((a, b) => statusRank(a.status) - statusRank(b.status) || priorityRank(a.priority) - priorityRank(b.priority));
  const toggle = (status: string) => {
    const next = new Set(statusFilter);
    if (next.has(status)) next.delete(status);
    else next.add(status);
    setStatusFilter(next);
  };
  const handoff = data.handoff?.sections ?? [];
  return (
    <section className="card tabs-card">
      <div className="tabs-head">
        <div className="segmented">
          <button className={tab === "plan" ? "on" : ""} onClick={() => setTab("plan")}>任務 <span>{data.plan.length}</span></button>
          <button className={tab === "tests" ? "on" : ""} onClick={() => setTab("tests")}>測試 <span>{tests ? `${okModules}/${tests.modules.length}` : "—"}</span></button>
          <button className={tab === "commits" ? "on" : ""} onClick={() => setTab("commits")}>提交 <span>{data.git.commits.length}</span></button>
          <button className={tab === "handoff" ? "on" : ""} onClick={() => setTab("handoff")}>交接</button>
        </div>
      </div>
      {tab === "plan" && (
        <>
          <div className="filters">
            {STATUS_ORDER.filter((status) => counts.has(status)).map((status) => {
              const style = statusStyle(status);
              const off = statusFilter.size > 0 && !statusFilter.has(status);
              return (
                <button key={status} className={`pill filter ${off ? "off" : ""}`} style={{ color: style.fg, background: style.bg }} onClick={() => toggle(status)}>
                  {style.label} {counts.get(status)}
                </button>
              );
            })}
          </div>
          <ItemTable items={items} onSelect={onSelect} />
        </>
      )}
      {tab === "tests" && (
        <>
          <div className="filters">
            <button className="run-button" onClick={onRunTests} disabled={testRunning}>
              {testRunning ? <span className="spinner" /> : <Icon name="play" size={14} />} {testRunning ? "測試執行中…" : "執行 pytest"}
            </button>
            {testError && <span className="error">{testError}</span>}
            {tests && !testError && <span className="muted">上次 <Time iso={tests.ranAt} /> · {tests.durationS}s</span>}
          </div>
          <div className="table">
            {(tests?.modules ?? []).slice().sort((a, b) => b.failed - a.failed).map((module) => {
              const total = module.passed + module.failed + module.skipped;
              return (
                <button key={module.module} className="table-row" onClick={() => onSelect({ kind: "module", module: module.module })}>
                  <span className="cell-main mono">{module.module}</span>
                  <span className={`pill ${module.failed ? "pill-bad" : "pill-good"}`}>{module.failed ? `${module.failed} 失敗` : "通過"}</span>
                  <span className="mini-track"><i className={module.failed ? "bad" : "good"} style={{ width: `${(module.passed / Math.max(1, total)) * 100}%` }} /></span>
                  <span className="cell-num">{module.passed}/{total}</span>
                </button>
              );
            })}
            {!tests && <p className="muted pad">尚未執行測試。執行後每個測試模組會成為貨櫃場上的一個貨櫃。</p>}
          </div>
        </>
      )}
      {tab === "commits" && (
        <div className="table">
          {data.git.commits.slice(0, 40).map((commit) => (
            <button key={commit.hash} className="table-row" onClick={() => onSelect({ kind: "commit", hash: commit.hash })}>
              <span className="cell-id mono">{commit.hash}</span>
              <span className="cell-main">{commit.subject}</span>
              <span className="pill" style={{ color: commitColor(commit.type), background: "#f1f4fb" }}>{commit.type}</span>
              <span className="cell-num">{commit.date.slice(5, 10)}</span>
            </button>
          ))}
        </div>
      )}
      {tab === "handoff" && (
        <div className="table handoff">
          {handoff.map((section) => (
            <div key={section.heading}>
              <h4>{section.heading}</h4>
              <ul>{section.items.map((entry, i) => <li key={i}>{plainText(entry)}</li>)}</ul>
            </div>
          ))}
          {!handoff.length && <p className="muted pad">找不到 docs/交接紀錄.md 的「目前狀態」段落。</p>}
        </div>
      )}
    </section>
  );
}

/* ---------------- 工具 ---------------- */

/** 把改善項目的細節段落拆成「標籤＋內容行」區塊，例如「- **現況**：…」。 */
function parseDetail(detail: string): { label: string | null; lines: { text: string; indent: boolean }[] }[] {
  const blocks: { label: string | null; lines: { text: string; indent: boolean }[] }[] = [];
  for (const raw of detail.split("\n")) {
    if (!raw.trim()) continue;
    const labelled = raw.match(/^-\s+\*\*(.+?)\*\*[：:]?\s*(.*)$/);
    if (labelled) {
      blocks.push({ label: labelled[1], lines: labelled[2] ? [{ text: plainText(labelled[2]), indent: false }] : [] });
      continue;
    }
    const indent = /^\s{2,}/.test(raw);
    const text = plainText(raw.trim().replace(/^[-*]\s+/, ""));
    if (!blocks.length) blocks.push({ label: null, lines: [] });
    blocks[blocks.length - 1].lines.push({ text, indent });
  }
  return blocks;
}

function countBy(values: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return counts;
}

/** 伺服器與瀏覽器時區可能不同：SSR 先輸出 ISO 片段，掛載後再換成瀏覽器當地時間，避免 hydration 不一致。 */
function Time({ iso }: { iso: string }) {
  const [text, setText] = useState(() => iso.slice(5, 16).replace("T", " "));
  useEffect(() => {
    const date = new Date(iso);
    if (!Number.isNaN(date.getTime())) {
      setText(date.toLocaleString("zh-TW", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }));
    }
  }, [iso]);
  return <>{text}</>;
}
