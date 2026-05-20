import { Badge } from "@/components/ui/badge";
import type { RunStatus } from "@/lib/api-types";

const VARIANT_BY_STATUS: Record<RunStatus, "default" | "primary" | "success" | "warning" | "destructive" | "outline"> = {
  queued: "outline",
  running: "primary",
  complete: "success",
  stopped: "warning",
  failed: "destructive",
};

const LABELS: Record<RunStatus, string> = {
  queued: "Queued",
  running: "Running",
  complete: "Complete",
  stopped: "Stopped",
  failed: "Failed",
};

export function RunStatusBadge({ status }: { status: RunStatus }) {
  return (
    <Badge variant={VARIANT_BY_STATUS[status]} className="font-mono">
      {status === "running" && (
        <span className="inline-block h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
      )}
      {LABELS[status]}
    </Badge>
  );
}
