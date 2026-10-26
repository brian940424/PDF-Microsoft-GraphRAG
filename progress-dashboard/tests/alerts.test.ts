import { describe, expect, it } from "vitest";

import { buildAlerts, diffProgress } from "@/lib/alerts";
import type { Commit, PlanItem, ProgressData } from "@/lib/types";

const plan = (id: string, status: string, uncommitted = false): PlanItem => ({
  id, title: `項目 \`${id}\``, category: "功能", district: "功能", priority: "中", status,
  result: "", commit: null, uncommitted, detail: "",
});
const commit = (hash: string): Commit => ({
  hash, date: "2026-10-24T10:00:00+08:00", subject: `feat: ${hash}`, author: "dev", type: "feat", files: 1, insertions: 1, deletions: 0,
});
const data = (overrides: Partial<ProgressData> = {}): ProgressData => ({
  generatedAt: "2026-10-24T10:00:00Z",
  repoName: "repo",
  plan: [plan("F-01", "進行中"), plan("D-01", "待確認"), plan("F-02", "完成", true), plan("F-03", "完成")],
  handoff: null,
  git: { branch: "main", user: "dev", commits: [commit("aaa1111")], changes: [{ code: "M", path: "a.ts" }] },
  tests: {
    ranAt: "2026-10-24T09:00:00Z", durationS: 3, total: 5, passed: 4, failed: 1, skipped: 0,
    modules: [
      { module: "test_ok", passed: 3, failed: 0, skipped: 0, failures: [] },
      { module: "test_bad", passed: 1, failed: 1, skipped: 0, failures: [{ name: "test_x", message: "boom" }] },
    ],
  },
  ...overrides,
});

describe("buildAlerts", () => {
  it("lists failing modules first, then uncommitted and undecided items, then working changes", () => {
    const alerts = buildAlerts(data());
    expect(alerts.map((alert) => alert.id)).toEqual(["test:test_bad", "plan:D-01:confirm", "plan:F-02:uncommitted", "git:changes"]);
    expect(alerts[0]).toMatchObject({ tone: "bad", body: "test_x", select: { kind: "module", module: "test_bad" } });
    expect(alerts[1].body).toBe("項目 D-01");
  });

  it("suggests running tests when there is no report", () => {
    const alerts = buildAlerts(data({ tests: null, git: { branch: "main", user: "dev", commits: [], changes: [] } }));
    expect(alerts.map((alert) => alert.id)).toEqual(["plan:D-01:confirm", "plan:F-02:uncommitted", "test:none"]);
  });
});

describe("diffProgress", () => {
  it("reports new commits, status changes and a new test run", () => {
    const prev = data();
    const next = data({
      plan: [plan("F-01", "完成"), plan("D-01", "待確認"), plan("F-02", "完成", true), plan("F-03", "完成"), plan("F-04", "待辦")],
      git: { ...prev.git, commits: [commit("bbb2222"), commit("aaa1111")] },
      tests: { ...prev.tests!, ranAt: "2026-10-24T10:00:00Z", failed: 0, passed: 5 },
    });
    const events = diffProgress(prev, next);
    expect(events.map((event) => event.title)).toEqual(["新提交 bbb2222", "F-01 施工中 → 完工", "測試完成：全數通過"]);
    expect(events[1]).toMatchObject({ tone: "good", select: { kind: "plan", id: "F-01" } });
    expect(events[2]).toMatchObject({ tone: "good", body: "5 / 5 通過 · 3s" });
  });

  it("is empty when nothing changed", () => {
    expect(diffProgress(data(), data())).toEqual([]);
  });
});
