import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { fileURLToPath } from "node:url";

export const WORLD_SOURCE_PATH = fileURLToPath(new URL("../lib/world.ts", import.meta.url));
export const WORLD_MODULE_PATH = fileURLToPath(new URL("../../src/automotive_graphrag/dashboard_world.py", import.meta.url));

/** 產生 dashboard_world.py 的內容：剝除型別後去掉 export，內嵌到儀表板的 module script。 */
export function exportedWorldModule() {
  const js = stripTypeScriptTypes(readFileSync(WORLD_SOURCE_PATH, "utf8"), { mode: "transform" })
    .replace(/^export /gm, "")
    .split("\n").map((line) => line.trimEnd()).join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  if (js.includes('"""')) throw new Error('world.ts must not contain """');
  return `"""由 progress-dashboard/lib/world.ts 自動產生（cd progress-dashboard && npm run export-world），請勿手動修改。"""\n\nWORLD_JS = r"""\n${js}\n"""\n`;
}
