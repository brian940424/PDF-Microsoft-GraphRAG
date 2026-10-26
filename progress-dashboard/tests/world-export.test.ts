import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

// @ts-expect-error 純 JS 腳本模組，沒有型別宣告
import { WORLD_MODULE_PATH, exportedWorldModule } from "../scripts/world-module.mjs";

describe("world export", () => {
  it("keeps the Python dashboard's copy of lib/world.ts up to date", () => {
    expect(readFileSync(WORLD_MODULE_PATH, "utf8"), "修改 lib/world.ts 後請執行 npm run export-world").toBe(exportedWorldModule());
  });
});
