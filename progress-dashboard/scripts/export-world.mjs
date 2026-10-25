// 把 lib/world.ts 轉成純 JS，寫入 Python 套件，讓「5 3D 儀表板」與本儀表板共用同一份世界模擬。
// 用法：npm run export-world（修改 lib/world.ts 後執行；tests/world-export.test.ts 會檢查是否同步）
import { writeFileSync } from "node:fs";

import { exportedWorldModule, WORLD_MODULE_PATH } from "./world-module.mjs";

writeFileSync(WORLD_MODULE_PATH, exportedWorldModule());
console.log(`wrote ${WORLD_MODULE_PATH}`);
