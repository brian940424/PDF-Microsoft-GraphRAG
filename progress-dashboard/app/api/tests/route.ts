import { NextResponse } from "next/server";

import { REPO_ROOT } from "@/lib/progress";
import { runTests } from "@/lib/tests";

export const dynamic = "force-dynamic";
export const maxDuration = 900;

/** 執行完整 pytest；可能需要數十秒。 */
export async function POST() {
  try {
    return NextResponse.json(await runTests(REPO_ROOT));
  } catch (error) {
    return NextResponse.json({ error: String(error) }, { status: 500 });
  }
}
