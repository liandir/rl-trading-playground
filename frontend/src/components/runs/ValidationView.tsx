"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, RotateCcw } from "lucide-react";
import { api } from "@/lib/api";
import type { ValidationArtifact, ValidationSeries } from "@/lib/api-types";
import { StreamingChart, type Series } from "@/components/charts/StreamingChart";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty";
import { formatNumber, formatPercent } from "@/lib/format";

export function ValidationView({ runId }: { runId: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["run-artifact", runId, "validation"],
    queryFn: () => api.runArtifact(runId, "validation"),
    retry: 1,
    // The artifact 404s until the rollout finishes — keep polling so the
    // results appear without a manual refresh.
    refetchInterval: (query) => (query.state.data ? false : 5_000),
  });

  if (isLoading) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }
  if (error || !data) {
    return (
      <EmptyState
        title="No validation artifact yet"
        description="It appears once the validation rollout finishes. If the run failed, check the events tab."
      />
    );
  }
  return (
    <div className="space-y-6">
      <MetricsSummary metrics={data.metrics} />
      <ValidationCharts series={data.series} />
    </div>
  );
}

function MetricsSummary({ metrics }: { metrics: ValidationArtifact["metrics"] }) {
  const headline: { key: string; value: number | null }[] = [
    { key: "Final value", value: numeric(metrics.final_value) },
    { key: "Total return", value: numeric(metrics.total_return_pct) },
    { key: "Max drawdown", value: numeric(metrics.max_drawdown_pct) },
    {
      key: "Trades",
      value: asInt(metrics.long_count) + asInt(metrics.short_count) + asInt(metrics.close_count),
    },
    { key: "Total reward", value: numeric(metrics.total_reward) },
    { key: "Steps", value: asInt(metrics.steps) },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      {headline.map(({ key, value }) => (
        <Card key={key} className="px-4 py-3">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{key}</p>
          <p className="mt-1 text-lg font-semibold tabular-nums">
            {key === "Total return" || key === "Max drawdown"
              ? formatPercent(value, 2)
              : key === "Final value"
              ? `$${formatNumber(value, 2)}`
              : value !== null
              ? formatNumber(value, 2)
              : "—"}
          </p>
        </Card>
      ))}
    </div>
  );
}

function ValidationCharts({ series }: { series: ValidationSeries }) {
  const [resetSignal, setResetSignal] = useState(0);

  const valueSeries: Series[] = [
    { name: "Portfolio", values: series.portfolio_value },
    { name: "Cash", values: series.cash },
  ];

  const priceSeries: Series[] = [
    { name: "Portfolio", values: series.normalized_value },
    ...series.asset_names.map<Series>((name, i) => ({
      name,
      values: series.normalized_prices.map((row) => row[i]),
    })),
  ];

  const drawdownSeries: Series[] = [
    { name: "Drawdown %", values: series.drawdown.map((d) => d * 100) },
  ];

  const rewardSeries: Series[] = [
    { name: "Step reward", values: series.rewards },
    { name: "Cumulative", values: series.cumulative_reward },
  ];

  const actionSeries: Series[] = [
    { name: "Longs", values: series.cumulative_longs },
    { name: "Shorts", values: series.cumulative_shorts },
    { name: "Closes", values: series.cumulative_closes },
    { name: "Invalid", values: series.cumulative_invalid },
  ];

  return (
    <div className="space-y-4">
      <div>
        <Button variant="outline" size="sm" onClick={() => setResetSignal((n) => n + 1)}>
          <RotateCcw className="h-3.5 w-3.5" />
          Reset all views
        </Button>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Portfolio value & cash">
          <StreamingChart
            series={valueSeries}
            xLabel="step"
            yLabel="value ($)"
            resetSignal={resetSignal}
            height={240}
          />
        </Panel>
        <Panel title="Normalized prices vs portfolio">
          <StreamingChart
            series={priceSeries}
            xLabel="step"
            yLabel="normalized"
            resetSignal={resetSignal}
            height={240}
          />
        </Panel>
        <Panel title="Drawdown">
          <StreamingChart
            series={drawdownSeries}
            xLabel="step"
            yLabel="drawdown (%)"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>
        <Panel title="Rewards">
          <StreamingChart
            series={rewardSeries}
            xLabel="step"
            yLabel="reward"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>
        <Panel title="Cumulative actions" wide>
          <StreamingChart
            series={actionSeries}
            xLabel="step"
            yLabel="count"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>
      </div>
    </div>
  );
}

function Panel({
  title,
  children,
  wide,
}: {
  title: string;
  children: React.ReactNode;
  wide?: boolean;
}) {
  return (
    <div className={"surface relative p-4 " + (wide ? "lg:col-span-2" : "")}>
      <h3 className="text-sm font-semibold mb-3">{title}</h3>
      {children}
    </div>
  );
}

function numeric(value: unknown): number | null {
  if (typeof value === "number") return value;
  if (typeof value === "string" && value !== "") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

function asInt(value: unknown): number {
  const n = numeric(value);
  return n === null ? 0 : Math.trunc(n);
}
