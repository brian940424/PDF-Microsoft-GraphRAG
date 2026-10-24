import { execFile } from "node:child_process";
import { promisify } from "node:util";

import type { Commit, GitSummary, WorkingChange } from "./types";

const run = promisify(execFile);
const RECORD = "\x1e";
const FIELD = "\x1f";

export async function readGit(repoRoot: string, limit = 300): Promise<GitSummary> {
  const git = async (...args: string[]) =>
    (await run("git", ["-c", "core.quotePath=false", ...args], { cwd: repoRoot, maxBuffer: 32 * 1024 * 1024 })).stdout;
  const [branch, user, log, status] = await Promise.all([
    git("rev-parse", "--abbrev-ref", "HEAD").catch(() => "unknown"),
    git("config", "user.name").catch(() => ""),
    git("log", `-n${limit}`, "--shortstat", `--format=${RECORD}%h${FIELD}%aI${FIELD}%s${FIELD}%an`).catch(() => ""),
    git("status", "--porcelain", "--untracked-files=normal").catch(() => ""),
  ]);
  return { branch: branch.trim(), user: user.trim(), commits: parseLog(log), changes: parseStatus(status) };
}

export function parseLog(output: string): Commit[] {
  return output
    .split(RECORD)
    .map((record) => record.trim())
    .filter(Boolean)
    .map((record) => {
      const [header, ...rest] = record.split("\n");
      const [hash, date, subject = "", author = ""] = header.split(FIELD);
      const stat = rest.join(" ");
      const count = (pattern: RegExp) => Number(stat.match(pattern)?.[1] ?? 0);
      return {
        hash,
        date,
        subject,
        author,
        type: commitType(subject),
        files: count(/(\d+) files? changed/),
        insertions: count(/(\d+) insertions?\(\+\)/),
        deletions: count(/(\d+) deletions?\(-\)/),
      };
    });
}

/** Conventional Commits 類型；不符合格式者歸為 other。 */
export function commitType(subject: string): string {
  const match = subject.match(/^(\w+)(?:\([^)]*\))?!?:/);
  return match ? match[1].toLowerCase() : "other";
}

export function parseStatus(output: string): WorkingChange[] {
  return output
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => ({ code: line.slice(0, 2).trim() || "?", path: line.slice(3).replace(/^"|"$/g, "") }));
}
