import type { Selection } from "@/components/theme";

import { plainText } from "./markdown";
import type { ProgressData } from "./types";

export type AlertTone = "bad" | "warn" | "good" | "info";

export interface Alert {
  id: string;
  tone: AlertTone;
  title: string;
  body: string;
  select: Selection;
}

const TONE_RANK: Record<AlertTone, number> = { bad: 0, warn: 1, info: 2, good: 3 };
const STATUS_LABEL: Record<string, string> = { 完成: "完工", 進行中: "施工中", 待辦: "待開工", 待確認: "待確認", 不處理: "不處理" };

/** 鈴鐺清單：目前需要注意的狀況（失敗測試、未提交、待確認），嚴重的排前面。 */
export function buildAlerts(data: ProgressData): Alert[] {
  const alerts: Alert[] = [];
  for (const module of data.tests?.modules ?? []) {
    if (!module.failed) continue;
    alerts.push({
      id: `test:${module.module}`,
      tone: "bad",
      title: `${module.module} 有 ${module.failed} 項測試失敗`,
      body: module.failures[0]?.name ?? "",
      select: { kind: "module", module: module.module },
    });
  }
  if (!data.tests) {
    alerts.push({ id: "test:none", tone: "info", title: "尚未執行測試", body: "按 KPI 卡上的執行鍵跑 pytest", select: null });
  }
  for (const item of data.plan) {
    const title = plainText(item.title);
    if (item.status === "完成" && item.uncommitted) {
      alerts.push({ id: `plan:${item.id}:uncommitted`, tone: "warn", title: `${item.id} 已完成但未提交`, body: title, select: { kind: "plan", id: item.id } });
    } else if (item.status === "待確認") {
      alerts.push({ id: `plan:${item.id}:confirm`, tone: "warn", title: `${item.id} 需要決定方向`, body: title, select: { kind: "plan", id: item.id } });
    }
  }
  if (data.git.changes.length) {
    alerts.push({
      id: "git:changes",
      tone: "info",
      title: `${data.git.changes.length} 個未提交變更`,
      body: data.git.changes.slice(0, 3).map((change) => change.path).join("、"),
      select: null,
    });
  }
  return alerts.sort((a, b) => TONE_RANK[a.tone] - TONE_RANK[b.tone]);
}

/** 自動更新時比較前後兩份資料，產生「新提交、狀態變更、測試結果」等動態，用於提示與鈴鐺的最近動態。 */
export function diffProgress(prev: ProgressData, next: ProgressData): Alert[] {
  const events: Alert[] = [];
  const known = new Set(prev.git.commits.map((commit) => commit.hash));
  for (const commit of next.git.commits.slice(0, 5)) {
    if (known.has(commit.hash)) continue;
    events.push({ id: `commit:${commit.hash}`, tone: "info", title: `新提交 ${commit.hash}`, body: commit.subject, select: { kind: "commit", hash: commit.hash } });
  }
  const before = new Map(prev.plan.map((item) => [item.id, item.status]));
  for (const item of next.plan) {
    const old = before.get(item.id);
    if (old === undefined || old === item.status) continue;
    events.push({
      id: `plan:${item.id}:${item.status}`,
      tone: item.status === "完成" ? "good" : "info",
      title: `${item.id} ${STATUS_LABEL[old] ?? old} → ${STATUS_LABEL[item.status] ?? item.status}`,
      body: plainText(item.title),
      select: { kind: "plan", id: item.id },
    });
  }
  if (next.tests && next.tests.ranAt !== prev.tests?.ranAt) {
    const { passed, failed, total } = next.tests;
    events.push({
      id: `tests:${next.tests.ranAt}`,
      tone: failed ? "bad" : "good",
      title: failed ? `測試完成：${failed} 項失敗` : "測試完成：全數通過",
      body: `${passed} / ${total} 通過 · ${next.tests.durationS}s`,
      select: null,
    });
  }
  return events;
}
