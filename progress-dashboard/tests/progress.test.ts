import { describe, expect, it } from "vitest";

import { commitType, parseLog, parseStatus } from "@/lib/git";
import { recentCommits } from "@/components/theme";
import { layoutSite } from "@/lib/layout";
import { parseHandoff, parsePlan, plainText } from "@/lib/markdown";
import { parseJUnit } from "@/lib/tests";

const PLAN = `# 改善計畫

## 狀態總覽

| 編號 | 項目 | 類別 | 優先 | 狀態 | 結果 |
|---|---|---|---|---|---|
| D-01 | 本機 \`.env\` 只讀取 API Key | 程式／文件 | 高 | 待確認 |  |
| F-02 | 新增自動檢索功能 | 功能 | 高 | 完成 | 做好了（未提交） |
| F-05 | 舊功能 | 功能 | 低 | 完成 | 已合併 \`abc1234\` |

## 文件與設定問題

### D-01 本機 \`.env\` 只讀取 API Key

- **現況**：只讀 API Key。
- **驗收**：設定一致。

## 功能
`;

const HANDOFF = `# 交接

## 目前狀態（main，2026-10-23）

### 已知問題

- 測試 16 項失敗

### 建議下一步

1. 修正 T-01
2. 提交

---

## 歷史
- 不應讀到
`;

describe("parsePlan", () => {
  it("reads table rows, districts, commits and detail sections", () => {
    const items = parsePlan(PLAN);
    expect(items.map((item) => item.id)).toEqual(["D-01", "F-02", "F-05"]);
    expect(items[0]).toMatchObject({ district: "程式", priority: "高", status: "待確認", uncommitted: false });
    expect(items[0].detail).toContain("**現況**：只讀 API Key。");
    expect(items[0].detail).not.toContain("## 功能");
    expect(items[1]).toMatchObject({ uncommitted: true, commit: null, detail: "" });
    expect(items[2].commit).toBe("abc1234");
  });
});

describe("parseHandoff", () => {
  it("reads only the current-status section", () => {
    const handoff = parseHandoff(HANDOFF)!;
    expect(handoff.title).toBe("目前狀態（main，2026-10-23）");
    expect(handoff.sections).toEqual([
      { heading: "已知問題", items: ["測試 16 項失敗"] },
      { heading: "建議下一步", items: ["修正 T-01", "提交"] },
    ]);
    expect(parseHandoff("# 無")).toBeNull();
  });
});

describe("git parsing", () => {
  it("parses log records with shortstat and conventional types", () => {
    const log = "\x1eaa32119\x1f2026-10-23T21:00:00+08:00\x1ffeat(ui): add map\x1fTai\n\n 3 files changed, 10 insertions(+), 2 deletions(-)\n"
      + "\x1ebb00000\x1f2026-10-22T10:00:00+08:00\x1fRefactor settings\x1fTai\n\n 1 file changed, 1 insertion(+)\n";
    const commits = parseLog(log);
    expect(commits[0]).toMatchObject({ hash: "aa32119", type: "feat", files: 3, insertions: 10, deletions: 2 });
    expect(commits[1]).toMatchObject({ type: "other", files: 1, insertions: 1, deletions: 0 });
    expect(commitType("fix!: breaking")).toBe("fix");
  });

  it("parses porcelain status", () => {
    expect(parseStatus(" M src/app.py\n?? progress-dashboard/\n")).toEqual([
      { code: "M", path: "src/app.py" },
      { code: "??", path: "progress-dashboard/" },
    ]);
  });
});

describe("parseJUnit", () => {
  it("groups test cases by module and keeps failure messages", () => {
    const xml = `<testsuites><testsuite>
      <testcase classname="tests.test_reviews.ReviewServiceTests" name="test_ok" time="0.1" />
      <testcase classname="tests.test_reviews.ReviewServiceTests" name="test_bad" time="0.1"><failure message="AssertionError: 1 &lt; 2">trace</failure></testcase>
      <testcase classname="tests.test_cases" name="test_skip"><skipped message="x" /></testcase>
    </testsuite></testsuites>`;
    const report = parseJUnit(xml, "2026-10-24T00:00:00Z", 1.234);
    expect(report).toMatchObject({ total: 3, passed: 1, failed: 1, skipped: 1, durationS: 1.2 });
    expect(report.modules.map((m) => m.module)).toEqual(["test_cases", "test_reviews"]);
    expect(report.modules[1].failures).toEqual([{ name: "test_bad", message: "AssertionError: 1 < 2" }]);
  });
});

describe("layoutSite", () => {
  const modules = Array.from({ length: 5 }, (_, i) => ({ module: `test_${i}`, passed: 1, failed: i % 2, skipped: 0, failures: [] }));

  it("places every plan item in non-overlapping zones and every test module in the yard", () => {
    const plan = parsePlan(PLAN);
    const layout = layoutSite(plan, modules);
    expect(layout.zones.map((zone) => zone.name)).toEqual(["程式", "功能"]);
    expect(layout.zones.flatMap((zone) => zone.lots).length).toBe(plan.length);
    expect(layout.yard?.slots).toHaveLength(5);
    const rects = [...layout.zones, layout.yard!];
    for (let i = 0; i < rects.length; i++) {
      for (let j = i + 1; j < rects.length; j++) {
        const a = rects[i], b = rects[j];
        const overlapX = Math.abs(a.x - b.x) < (a.w + b.w) / 2;
        const overlapZ = Math.abs(a.z - b.z) < (a.d + b.d) / 2;
        expect(overlapX && overlapZ).toBe(false);
      }
    }
    for (const rect of rects) {
      expect(Math.abs(rect.x) + rect.w / 2).toBeLessThan(layout.road.w / 2);
      expect(Math.abs(rect.z) + rect.d / 2).toBeLessThan(layout.road.d / 2);
    }
  });

  it("omits the yard when tests have not run", () => {
    expect(layoutSite(parsePlan(PLAN), []).yard).toBeNull();
  });
});

describe("recentCommits", () => {
  it("keeps between three and eight trucks", () => {
    const now = Date.parse("2026-10-24T00:00:00Z");
    const make = (n: number, daysAgo: number) =>
      Array.from({ length: n }, (_, i) => ({ hash: `h${daysAgo}${i}`, date: new Date(now - daysAgo * 86_400_000).toISOString(),
        subject: "x", author: "a", type: "feat", files: 1, insertions: 1, deletions: 0 }));
    expect(recentCommits(make(5, 30), now)).toHaveLength(3);
    expect(recentCommits(make(12, 1), now)).toHaveLength(8);
    expect(recentCommits([...make(4, 1), ...make(4, 30)], now)).toHaveLength(4);
  });
});

describe("plainText", () => {
  it("strips inline markdown", () => {
    expect(plainText("**做法**：見 [文件](a.md) 與 `x`")).toBe("做法：見 文件 與 x");
  });
});

