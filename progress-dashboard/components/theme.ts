import type { Commit } from "@/lib/types";

export type Selection =
  | { kind: "plan"; id: string }
  | { kind: "module"; module: string }
  | { kind: "commit"; hash: string }
  | { kind: "zone"; name: string }
  | null;

export type CameraCommand = { kind: "zoomIn" | "zoomOut" | "rotateLeft" | "rotateRight" | "home"; nonce: number };

/** 狀態徽章配色：文字色／底色 */
export const STATUS_STYLE: Record<string, { fg: string; bg: string; label: string }> = {
  完成: { fg: "#15803d", bg: "#e3f6ea", label: "完工" },
  進行中: { fg: "#2c56e0", bg: "#e6edff", label: "施工中" },
  待確認: { fg: "#b45309", bg: "#fff3d6", label: "待確認" },
  待辦: { fg: "#5b6476", bg: "#eef1f6", label: "待開工" },
  不處理: { fg: "#8a93a4", bg: "#f1f3f7", label: "不處理" },
};

export const statusStyle = (status: string) => STATUS_STYLE[status] ?? STATUS_STYLE["待辦"];

export const COMMIT_COLORS: Record<string, string> = {
  feat: "#3563f0",
  fix: "#ec5a5f",
  docs: "#8b5cf6",
  ci: "#f6a821",
  refactor: "#1fb5a8",
  test: "#ff8a3d",
  chore: "#94a3b8",
  other: "#64748b",
};

export const commitColor = (type: string) => COMMIT_COLORS[type] ?? COMMIT_COLORS.other;

/** 道路上的卡車：最近 7 天的提交（至少 3 台、至多 8 台）。 */
export function recentCommits(commits: Commit[], now = Date.now()): Commit[] {
  const weekAgo = now - 7 * 86_400_000;
  const recent = commits.filter((commit) => Date.parse(commit.date) >= weekAgo);
  return (recent.length >= 3 ? recent : commits.slice(0, 3)).slice(0, 8);
}
