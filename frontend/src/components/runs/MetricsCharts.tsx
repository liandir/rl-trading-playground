"use client";

import { useMemo } from "react";
import type { RunEvent } from "@/lib/api-types";
import { StreamingChart, type Series } from "@/components/charts/StreamingChart";

export function MetricsCharts({ events }: { events: RunEvent[] }) {
  const updateEvents = useMemo(() => events.filter((e) => e.kind === "update"), [events]);
  const portfolioEvents = useMemo(
    () => events.filter((e) => e.avg_portfolio !== null && e.avg_portfolio !== undefined),
    [events]
  );

  const rewardSeries: Series[] = [
    {
      name: "Avg reward",
      values: portfolioEvents.map((e) => Number(e.avg_reward ?? NaN)),
    },
  ];
  const portfolioSeries: Series[] = [
    {
      name: "Avg portfolio",
      values: portfolioEvents.map((e) => Number(e.avg_portfolio ?? NaN)),
    },
  ];
  const lossSeries: Series[] = [
    { name: "Total loss", values: updateEvents.map((e) => Number(e.metrics.total_loss ?? NaN)) },
    { name: "Value loss", values: updateEvents.map((e) => Number(e.metrics.value_loss ?? NaN)) },
  ];
  const entropySeries: Series[] = [
    { name: "Entropy", values: updateEvents.map((e) => Number(e.metrics.entropy ?? NaN)) },
  ];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="surface p-4">
        <h3 className="text-sm font-semibold mb-3">Reward</h3>
        <StreamingChart series={rewardSeries} height={220} />
      </div>
      <div className="surface p-4">
        <h3 className="text-sm font-semibold mb-3">Portfolio value</h3>
        <StreamingChart series={portfolioSeries} height={220} />
      </div>
      <div className="surface p-4">
        <h3 className="text-sm font-semibold mb-3">Losses</h3>
        <StreamingChart series={lossSeries} height={220} />
      </div>
      <div className="surface p-4">
        <h3 className="text-sm font-semibold mb-3">Entropy</h3>
        <StreamingChart series={entropySeries} height={220} />
      </div>
    </div>
  );
}
