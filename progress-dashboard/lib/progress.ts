import { readFile } from "node:fs/promises";
import path from "node:path";

import { readGit } from "./git";
import { parseHandoff, parsePlan } from "./markdown";
import { readCachedTests } from "./tests";
import type { ProgressData } from "./types";

/** 儀表板讀取的 repo 根目錄；預設為本資料夾的上一層。 */
export const REPO_ROOT = path.resolve(process.env.REPO_ROOT ?? path.join(process.cwd(), ".."));

export async function loadProgress(): Promise<ProgressData> {
  const read = (relative: string) => readFile(path.join(REPO_ROOT, relative), "utf-8").catch(() => "");
  const [planMarkdown, handoffMarkdown, git, tests] = await Promise.all([
    read("docs/改善計畫.md"),
    read("docs/交接紀錄.md"),
    readGit(REPO_ROOT),
    readCachedTests(),
  ]);
  return {
    generatedAt: new Date().toISOString(),
    repoName: path.basename(REPO_ROOT),
    plan: parsePlan(planMarkdown),
    handoff: parseHandoff(handoffMarkdown),
    git,
    tests,
  };
}
