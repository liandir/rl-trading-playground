"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQueries } from "@tanstack/react-query";
import { ArrowLeft, Loader2, RotateCcw } from "lucide-react";
import { api } from "@/lib/api";
import type { RunEvent, RunRecord } from "@/lib/api-types";
import { StreamingChart, type Series } from "@/components/charts/StreamingChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/empty";
import { PageHeader } from "@/components/shell/PageHeader";
import { formatNumber } from "@/lib/format";

const PALETTE = ["#6366f1", "#22c55e", "#f97316", "#ec4899", "#06b6d4", "#a855f7", "#facc15"];

export default function ComparePageWrapper() {
  return (
    <Suspense
      fallback={
        <div className="flex h-48 items-center justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <ComparePage />
    </Suspense>
  );
}

function ComparePage() {
  const [resetSignal, setResetSignal] = useState(0);
  const params = useSearchParams();
  const ids = useMemo(
    () =>
      (params.get("ids") ?? "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    [params]
  );

  const runQueries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["run", id],
      queryFn: () => api.getRun(id),
    })),
  });
  const eventQueries = useQueries({
    queries: ids.map((id) => ({
      queryKey: ["run", id, "events"],
      queryFn: () => api.runEvents(id),
    })),
  });

  const runs = runQueries.map((q) => q.data).filter(Boolean) as RunRecord[];
  const eventsByRun = eventQueries.map((q) => q.data ?? []) as RunEvent[][];
  const loading = runQueries.some((q) => q.isLoading) || eventQueries.some((q) => q.isLoading);
  const allFailed = runQueries.length > 0 && runQueries.every((q) => q.isError);

  if (!ids.length) {
    return (
      <>
        <PageHeader title="Compare runs" />
        <EmptyState
          title="No runs to compare"
          description="Pick two or more rows on the Runs page, then click Compare."
          action={
            <Link href="/runs" className="text-sm text-primary hover:underline">
              Back to Runs
            </Link>
          }
        />
      </>
    );
  }

  if (loading && runs.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (allFailed) {
    return (
      <>
        <PageHeader title="Compare runs" />
        <EmptyState title="Could not load runs" description="None of the requested runs could be fetched." />
      </>
    );
  }

  return (
    <>
      <PageHeader
        title={`Compare ${ids.length} runs`}
        description={
          <span className="inline-flex items-center gap-2 text-sm">
            <Link
              href="/runs"
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-3 w-3" /> Back to runs
            </Link>
          </span>
        }
        actions={
          <Button variant="ghost" onClick={() => history.back()}>
            Cancel
          </Button>
        }
      />

      <Card>
        <CardHeader>
          <CardTitle>Selected runs</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {runs.map((run, i) => (
              <Badge
                key={run.id}
                variant="outline"
                className="border-l-4"
                style={{ borderLeftColor: PALETTE[i % PALETTE.length] }}
              >
                <span className="font-mono text-[11px]">{run.id.slice(0, 8)}</span>
                <span className="ml-1">{run.name}</span>
                <span className="ml-2 text-muted-foreground">
                  {run.spec?.agent.agent_type}/{run.spec?.agent.network_preset}
                </span>
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="mt-6 mb-3 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Overlay
        </h2>
        <Button variant="outline" size="sm" onClick={() => setResetSignal((n) => n + 1)}>
          <RotateCcw className="h-3.5 w-3.5" />
          Reset all views
        </Button>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <MetricPanel
          title="Average reward"
          runs={runs}
          events={eventsByRun}
          pickValue={(e) => Number(e.avg_reward ?? NaN)}
          yLabel="reward"
          resetSignal={resetSignal}
        />
        <MetricPanel
          title="Average portfolio"
          runs={runs}
          events={eventsByRun}
          pickValue={(e) => Number(e.avg_portfolio ?? NaN)}
          yLabel="portfolio value"
          resetSignal={resetSignal}
        />
        <MetricPanel
          title="Total loss"
          runs={runs}
          events={eventsByRun}
          pickValue={(e) => Number(e.metrics.total_loss ?? NaN)}
          yLabel="loss"
          resetSignal={resetSignal}
          updateOnly
        />
        <MetricPanel
          title="Entropy"
          runs={runs}
          events={eventsByRun}
          pickValue={(e) => Number(e.metrics.entropy ?? NaN)}
          yLabel="entropy (nats)"
          resetSignal={resetSignal}
          updateOnly
        />
      </div>

      <h2 className="mt-8 mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Summary
      </h2>
      <Card>
        <CardContent className="overflow-x-auto pt-5">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-3 py-2 text-left">Run</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-right">Avg reward</th>
                <th className="px-3 py-2 text-right">Final portfolio</th>
                <th className="px-3 py-2 text-right">Steps</th>
                <th className="px-3 py-2 text-right">Updates</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {runs.map((run, i) => {
                const s = run.summary;
                return (
                  <tr key={run.id}>
                    <td className="px-3 py-2">
                      <span
                        className="mr-2 inline-block h-2 w-2 rounded-full"
                        style={{ backgroundColor: PALETTE[i % PALETTE.length] }}
                      />
                      <Link href={`/runs/${run.id}`} className="font-medium hover:text-primary">
                        {run.name}
                      </Link>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs">{run.status}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {formatNumber(numericFromSummary(s, "avg_reward"))}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {formatNumber(numericFromSummary(s, "final_portfolio"))}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {formatNumber(numericFromSummary(s, "total_steps") ?? numericFromSummary(s, "steps"), 0)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {formatNumber(numericFromSummary(s, "total_updates"), 0)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>
    </>
  );
}

function numericFromSummary(summary: Record<string, unknown>, key: string): number | null {
  const value = summary[key];
  if (typeof value === "number") return value;
  return null;
}

function MetricPanel({
  title,
  runs,
  events,
  pickValue,
  yLabel,
  resetSignal,
  updateOnly,
}: {
  title: string;
  runs: RunRecord[];
  events: RunEvent[][];
  pickValue: (e: RunEvent) => number;
  yLabel?: string;
  resetSignal?: number;
  updateOnly?: boolean;
}) {
  const series: Series[] = runs.map((run, i) => {
    const evts = events[i] ?? [];
    const filtered = updateOnly ? evts.filter((e) => e.kind === "update") : evts;
    return {
      name: `${run.name} (${run.id.slice(0, 6)})`,
      values: filtered.map(pickValue),
      color: PALETTE[i % PALETTE.length],
    };
  });
  return (
    <div className="surface relative p-4">
      <h3 className="text-sm font-semibold mb-3">{title}</h3>
      <StreamingChart
        series={series}
        xLabel="event index"
        yLabel={yLabel}
        resetSignal={resetSignal}
        height={220}
      />
    </div>
  );
}
