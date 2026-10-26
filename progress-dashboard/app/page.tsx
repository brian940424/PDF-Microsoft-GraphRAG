import Dashboard from "@/components/Dashboard";
import { loadProgress } from "@/lib/progress";

export const dynamic = "force-dynamic";

export default async function Page() {
  return <Dashboard initial={await loadProgress()} />;
}
