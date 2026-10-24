import type { Handoff, HandoffSection, PlanItem } from "./types";

const PLAN_ROW = /^\|\s*([A-Z]+-\d+)\s*\|/;
const COMMIT_HASH = /`?\b([0-9a-f]{7,40})\b`?/;

/** 解析 docs/改善計畫.md：狀態總覽表格＋各項目的細節段落。 */
export function parsePlan(markdown: string): PlanItem[] {
  const lines = markdown.split(/\r?\n/);
  const details = parseDetailSections(lines);
  const items: PlanItem[] = [];
  for (const line of lines) {
    if (!PLAN_ROW.test(line)) continue;
    const cells = line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
    const [id, title = "", category = "", priority = "", status = "", result = ""] = cells;
    items.push({
      id,
      title,
      category,
      district: category.split(/[／/]/)[0].trim() || "其他",
      priority,
      status,
      result,
      commit: result.match(COMMIT_HASH)?.[1] ?? null,
      uncommitted: result.includes("未提交"),
      detail: details.get(id) ?? "",
    });
  }
  return items;
}

function parseDetailSections(lines: string[]): Map<string, string> {
  const sections = new Map<string, string>();
  let current: string | null = null;
  let buffer: string[] = [];
  const flush = () => {
    if (current) sections.set(current, buffer.join("\n").trim());
  };
  for (const line of lines) {
    const heading = line.match(/^#{2,4}\s+([A-Z]+-\d+)\b/);
    if (heading || /^#{1,3}\s/.test(line)) {
      flush();
      current = heading ? heading[1] : null;
      buffer = [];
      continue;
    }
    if (current) buffer.push(line);
  }
  flush();
  return sections;
}

/** 解析 docs/交接紀錄.md 第一個「## 目前狀態」段落。 */
export function parseHandoff(markdown: string): Handoff | null {
  const lines = markdown.split(/\r?\n/);
  const start = lines.findIndex((line) => /^##\s+目前狀態/.test(line));
  if (start < 0) return null;
  const sections: HandoffSection[] = [];
  let section: HandoffSection | null = null;
  for (const line of lines.slice(start + 1)) {
    if (/^##\s/.test(line) || /^---\s*$/.test(line)) break;
    const heading = line.match(/^###\s+(.*)/);
    if (heading) {
      section = { heading: heading[1].trim(), items: [] };
      sections.push(section);
      continue;
    }
    const bullet = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)/);
    if (bullet && section) section.items.push(bullet[1].trim());
    else if (line.trim() && section) section.items.push(line.trim());
  }
  return { title: lines[start].replace(/^##\s+/, "").trim(), sections };
}

/** 去除行內 Markdown 標記，供純文字顯示。 */
export function plainText(markdown: string): string {
  return markdown
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
}
