import { NextResponse } from "next/server";

import { loadProgress } from "@/lib/progress";
import { isTestRunActive } from "@/lib/tests";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json({ ...(await loadProgress()), testRunActive: isTestRunActive() });
}
