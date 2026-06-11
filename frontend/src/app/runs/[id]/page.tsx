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
import { ValidationView } from "@/components/runs/ValidationView";
import { CheckpointList } from "@/components/runs/CheckpointList";
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
  const checkpoints = useQuery({
    queryKey: ["run-checkpoints", id],
    queryFn: () => api.listCheckpoints({ run_id: id }),
    refetchInterval: 5_000,
  });
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

  const isValidation = run.kind === "validation";
  // warm_up events carry step/percent too, so progress moves during warm-up.
  const latestUpdate = events
    .filter((e) =>
      isValidation ? e.kind === "validation_step" : e.kind === "update" || e.kind === "warm_up"
    )
    .at(-1);
  const progress = latestUpdate?.percent ?? 0;
  const isActive = run.status === "running" || run.status === "queued";

  // Current-episode header data (training only).
  const currentEpisode = !isValidation
    ? events.reduce((max, e) => (e.episode > max ? e.episode : max), 0)
    : 0;
  const currentEpEvents = !isValidation
    ? events.filter((e) => e.episode === currentEpisode)
    : [];
  const currentEpEndEvent = !isValidation
    ? events.find((e) => e.kind === "episode_end" && e.episode === currentEpisode)
    : undefined;
  const currentEpLastUpdate = !isValidation
    ? [...currentEpEvents].reverse().find((e) => e.kind === "update")
    : undefined;
  // Mean past-episode length used to estimate current-episode progress %.
  const pastLengths = !isValidation
    ? events
        .filter((e) => e.kind === "episode_end" && e.episode !== currentEpisode)
        .map((e) => Number(e.metrics.episode_length ?? e.step))
        .filter((n) => Number.isFinite(n) && n > 0)
    : [];
  const avgPastLength =
    pastLengths.length > 0
      ? pastLengths.reduce((a, b) => a + b, 0) / pastLengths.length
      : null;
  const currentEpStep = currentEpEvents.reduce((max, e) => (e.step > max ? e.step : max), 0);
  const currentEpProgressPct = currentEpEndEvent
    ? 100
    : avgPastLength
    ? Math.min(100, (currentEpStep / avgPastLength) * 100)
    : 0;

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

      {!isValidation && currentEpisode > 0 && (
        <div className="mt-4 grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
                Current episode
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold tabular-nums">{currentEpisode}</p>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary/80 transition-all"
                  style={{ width: `${currentEpProgressPct}%` }}
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                step {formatNumber(currentEpStep, 0)}
                {avgPastLength && !currentEpEndEvent
                  ? ` of ~${formatNumber(avgPastLength, 0)}`
                  : ""}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
                Episode avg reward
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold tabular-nums">
                {formatNumber(
                  currentEpEndEvent?.avg_reward ?? currentEpLastUpdate?.avg_reward ?? null
                )}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                {currentEpEndEvent ? "final" : isActive ? "live" : "—"}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
                Latest total loss
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold tabular-nums">
                {formatNumber(
                  currentEpLastUpdate
                    ? Number(currentEpLastUpdate.metrics.total_loss ?? NaN)
                    : null
                )}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
                Latest entropy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold tabular-nums">
                {formatNumber(
                  currentEpLastUpdate
                    ? Number(currentEpLastUpdate.metrics.entropy ?? NaN)
                    : null
                )}
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue={isValidation ? "results" : "metrics"} className="mt-6">
        <TabsList>
          {isValidation ? (
            <TabsTrigger value="results">Results</TabsTrigger>
          ) : (
            <TabsTrigger value="metrics">Metrics</TabsTrigger>
          )}
          <TabsTrigger value="events">Events ({events.length})</TabsTrigger>
          <TabsTrigger value="checkpoints">
            Checkpoints {checkpoints.data ? `(${checkpoints.data.length})` : ""}
          </TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>
        {isValidation ? (
          <TabsContent value="results">
            <ValidationView runId={run.id} />
            <p className="mt-2 text-xs text-muted-foreground">
              WebSocket: <span className="font-mono">{state}</span>
            </p>
          </TabsContent>
        ) : (
          <TabsContent value="metrics">
            <MetricsCharts events={events} />
            <p className="mt-2 text-xs text-muted-foreground">
              WebSocket: <span className="font-mono">{state}</span>
            </p>
          </TabsContent>
        )}
        <TabsContent value="events">
          <EventLog events={events} />
        </TabsContent>
        <TabsContent value="checkpoints">
          <CheckpointList
            checkpoints={checkpoints.data ?? []}
            emptyHint="Checkpoints appear here once an episode finishes and save_checkpoint is on."
          />
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
