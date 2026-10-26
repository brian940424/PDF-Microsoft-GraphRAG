import { execFile } from "node:child_process";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import type { TestModule, TestReport } from "./types";

const CACHE_FILE = path.join(process.cwd(), ".cache", "tests.json");
let running: Promise<TestReport> | null = null;

export async function readCachedTests(): Promise<TestReport | null> {
  try {
    return JSON.parse(await readFile(CACHE_FILE, "utf-8")) as TestReport;
  } catch {
    return null;
  }
}

export function isTestRunActive(): boolean {
  return running !== null;
}

/** 以專案的 .venv 執行 pytest，解析 JUnit XML 並快取結果；同時只允許一個執行中的測試。 */
export function runTests(repoRoot: string): Promise<TestReport> {
  running ??= (async () => {
    const xmlPath = path.join(tmpdir(), `graphrag-progress-${process.pid}.xml`);
    const python = path.join(repoRoot, ".venv", "bin", "python");
    const started = Date.now();
    try {
      await new Promise<void>((resolve) => {
        // pytest 有失敗時 exit code 非 0，仍以 XML 為準
        execFile(python, ["-m", "pytest", "-q", "-p", "no:cacheprovider", `--junitxml=${xmlPath}`, "tests"],
          { cwd: repoRoot, timeout: 15 * 60 * 1000, maxBuffer: 64 * 1024 * 1024 }, () => resolve());
      });
      const xml = await readFile(xmlPath, "utf-8");
      const report = parseJUnit(xml, new Date().toISOString(), (Date.now() - started) / 1000);
      await mkdir(path.dirname(CACHE_FILE), { recursive: true });
      await writeFile(CACHE_FILE, JSON.stringify(report, null, 2));
      return report;
    } finally {
      await rm(xmlPath, { force: true });
      running = null;
    }
  })();
  return running;
}

const TESTCASE = /<testcase\b([^>]*?)(?:\/>|>([\s\S]*?)<\/testcase>)/g;
const ATTRIBUTE = /(\w+)="([^"]*)"/g;

export function parseJUnit(xml: string, ranAt: string, durationS: number): TestReport {
  const modules = new Map<string, TestModule>();
  for (const [, rawAttributes, body = ""] of xml.matchAll(TESTCASE)) {
    const attributes = Object.fromEntries([...rawAttributes.matchAll(ATTRIBUTE)].map(([, key, value]) => [key, decode(value)]));
    const classParts = (attributes.classname ?? "").split(".");
    const moduleName = classParts.find((part) => part.startsWith("test_")) ?? classParts[0] ?? "unknown";
    const entry = modules.get(moduleName) ?? { module: moduleName, passed: 0, failed: 0, skipped: 0, failures: [] };
    const failure = body.match(/<(failure|error)\b(?:[^>]*?\smessage="([^"]*)")?[^>]*>/);
    if (failure) {
      entry.failed += 1;
      entry.failures.push({ name: attributes.name ?? "", message: decode(failure[2] ?? failure[1]).slice(0, 400) });
    } else if (/<skipped\b/.test(body)) {
      entry.skipped += 1;
    } else {
      entry.passed += 1;
    }
    modules.set(moduleName, entry);
  }
  const list = [...modules.values()].sort((a, b) => a.module.localeCompare(b.module));
  const sum = (key: "passed" | "failed" | "skipped") => list.reduce((total, module) => total + module[key], 0);
  return {
    ranAt,
    durationS: Math.round(durationS * 10) / 10,
    total: sum("passed") + sum("failed") + sum("skipped"),
    passed: sum("passed"),
    failed: sum("failed"),
    skipped: sum("skipped"),
    modules: list,
  };
}

function decode(value: string): string {
  return value
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#10;/g, "\n")
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&");
}
