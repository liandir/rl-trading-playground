"use client";

import { use } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { Loader2, Square, ArrowLeft } from "lucide-react";
import { api } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageHeader } from "@/components/shell/PageHeader";
import { RunStatusBadge } from "@/components/runs/RunStatusBadge";
import { MetricsCharts } from "@/components/runs/MetricsCharts";
import { EventLog } from "@/components/runs/EventLog";
import { formatDuration, formatNumber, formatPercent } from "@/lib/format";

export default function RunDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const qc = useQueryClient();
  const { data: run } = useQuery({
    queryKey: ["run", id],
    queryFn: () => api.getRun(id),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "running" || status === "queued" ? 1_500 : false;
    },
  });
  const { events, state } = useEventStream(id);
  const stop = useMutation({
    mutationFn: () => api.stopRun(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["run", id] }),
  });

  if (!run) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  const latestUpdate = events.filter((e) => e.kind === "update").at(-1);
  const progress = latestUpdate?.percent ?? 0;
  const isActive = run.status === "running" || run.status === "queued";

  return (
    <>
      <PageHeader
        title={run.name}
        description={
          <span className="inline-flex items-center gap-2 text-sm">
            <Link href="/runs" className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-3 w-3" /> All runs
            </Link>
            <span className="text-muted-foreground">•</span>
            <span className="font-mono text-xs">{run.id}</span>
            <span className="text-muted-foreground">•</span>
            <span className="text-muted-foreground">{run.kind}</span>
          </span>
        }
        actions={
          <div className="flex items-center gap-2">
            <RunStatusBadge status={run.status} />
            {isActive && (
              <Button
                variant="destructive"
                size="sm"
                disabled={stop.isPending}
                onClick={() => stop.mutate()}
              >
                <Square className="h-3.5 w-3.5" />
                Stop
              </Button>
            )}
          </div>
        }
      />

      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
              Progress
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold tabular-nums">{formatPercent(progress, 1)}</p>
            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-all"
                style={{ width: `${Math.min(100, progress)}%` }}
              />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
              Step
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold tabular-nums">
              {formatNumber(latestUpdate?.step ?? 0, 0)}
            </p>
            <p className="text-xs text-muted-foreground mt-1">episode {latestUpdate?.episode ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
              Avg reward
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold tabular-nums">
              {formatNumber(latestUpdate?.avg_reward ?? null)}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
              Duration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-semibold tabular-nums">
              {formatDuration(run.started_at, run.ended_at)}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              {isActive ? "running" : run.ended_at ? "ended" : "—"}
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="metrics" className="mt-6">
        <TabsList>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="events">Events ({events.length})</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>
        <TabsContent value="metrics">
          <MetricsCharts events={events} />
          <p className="mt-2 text-xs text-muted-foreground">
            WebSocket: <span className="font-mono">{state}</span>
          </p>
        </TabsContent>
        <TabsContent value="events">
          <EventLog events={events} />
        </TabsContent>
        <TabsContent value="config">
          <pre className="surface overflow-auto p-4 text-xs font-mono leading-relaxed">
            {JSON.stringify(run.spec, null, 2)}
          </pre>
        </TabsContent>
      </Tabs>
    </>
  );
}
