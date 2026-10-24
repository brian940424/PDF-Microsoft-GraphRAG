export type PlanStatus = "完成" | "進行中" | "待辦" | "待確認" | "不處理" | string;

export interface PlanItem {
  id: string;
  title: string;
  category: string;
  /** 類別欄的第一段（例如「程式／文件」→「程式」），作為地圖上的區域 */
  district: string;
  priority: string;
  status: PlanStatus;
  result: string;
  commit: string | null;
  uncommitted: boolean;
  detail: string;
}

export interface HandoffSection {
  heading: string;
  items: string[];
}

export interface Handoff {
  title: string;
  sections: HandoffSection[];
}

export interface Commit {
  hash: string;
  date: string;
  subject: string;
  author: string;
  type: string;
  files: number;
  insertions: number;
  deletions: number;
}

export interface WorkingChange {
  code: string;
  path: string;
}

export interface GitSummary {
  branch: string;
  user: string;
  commits: Commit[];
  changes: WorkingChange[];
}

export interface TestFailure {
  name: string;
  message: string;
}

export interface TestModule {
  module: string;
  passed: number;
  failed: number;
  skipped: number;
  failures: TestFailure[];
}

export interface TestReport {
  ranAt: string;
  durationS: number;
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  modules: TestModule[];
}

export interface ProgressData {
  generatedAt: string;
  repoName: string;
  plan: PlanItem[];
  handoff: Handoff | null;
  git: GitSummary;
  tests: TestReport | null;
}
