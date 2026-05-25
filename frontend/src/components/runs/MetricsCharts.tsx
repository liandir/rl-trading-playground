"use client";

import { useMemo, useState } from "react";
import { RotateCcw, ArrowLeft } from "lucide-react";
import type { RunEvent } from "@/lib/api-types";
import { StreamingChart, type Series, type EpisodeBoundary } from "@/components/charts/StreamingChart";
import { Select } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { formatNumber } from "@/lib/format";
import { cn } from "@/lib/cn";

type Selection = "all" | "latest" | { kind: "episode"; n: number };

function resolveSelection(sel: Selection, latest: number | null): number | null {
  if (sel === "all") return null;
  if (sel === "latest") return latest;
  return sel.n;
}

export function MetricsCharts({ events }: { events: RunEvent[] }) {
  const [selection, setSelection] = useState<Selection>("all");
  const [resetSignal, setResetSignal] = useState(0);

  const episodeNumbers = useMemo(() => {
    const set = new Set<number>();
    for (const e of events) if (e.episode > 0) set.add(e.episode);
    return Array.from(set).sort((a, b) => a - b);
  }, [events]);
  const latestEpisode = episodeNumbers.length > 0 ? episodeNumbers[episodeNumbers.length - 1] : null;

  // Per-episode finished lengths from episode_end events.
  const episodeLengths = useMemo(() => {
    const map = new Map<number, number>();
    for (const e of events) {
      if (e.kind === "episode_end") {
        const len = e.metrics.episode_length ?? e.step;
        if (Number.isFinite(len) && len > 0) map.set(e.episode, len);
      }
    }
    return map;
  }, [events]);

  // Cumulative step offset for each episode (sum of all prior finished episode lengths).
  const episodeOffsets = useMemo(() => {
    const offsets = new Map<number, number>();
    let acc = 0;
    for (const ep of episodeNumbers) {
      offsets.set(ep, acc);
      acc += episodeLengths.get(ep) ?? 0;
    }
    return offsets;
  }, [episodeNumbers, episodeLengths]);

  const cumulativeStep = (e: RunEvent): number =>
    (episodeOffsets.get(e.episode) ?? 0) + e.step;

  // Filter events according to the active selection.
  const resolvedEp = resolveSelection(selection, latestEpisode);
  const scoped = useMemo(() => {
    if (resolvedEp === null) return events;
    return events.filter((e) => e.episode === resolvedEp);
  }, [events, resolvedEp]);

  // Episode boundaries (start step on the cumulative axis) for the "all" view.
  const boundaries: EpisodeBoundary[] | undefined = useMemo(() => {
    if (resolvedEp !== null) return undefined;
    return episodeNumbers.map((ep) => ({ episode: ep, step: episodeOffsets.get(ep) ?? 0 }));
  }, [resolvedEp, episodeNumbers, episodeOffsets]);

  const updateEvents = useMemo(() => scoped.filter((e) => e.kind === "update"), [scoped]);
  const episodeEndEvents = useMemo(
    () => scoped.filter((e) => e.kind === "episode_end" && e.metrics.total_reward !== undefined),
    [scoped]
  );
  const portfolioEvents = useMemo(
    () => scoped.filter((e) => e.avg_portfolio !== null && e.avg_portfolio !== undefined),
    [scoped]
  );

  const xOfEvent = (e: RunEvent): number =>
    resolvedEp === null ? cumulativeStep(e) : e.step;

  const xLabel = resolvedEp === null ? "episode" : `step (episode ${resolvedEp})`;

  // ----- Series -----
  const rewardSeries: Series[] = [
    {
      name: "Avg reward",
      values: updateEvents.map((e) => Number(e.avg_reward ?? NaN)),
    },
  ];
  const rewardX = updateEvents.map(xOfEvent);

  const totalRewardX = episodeEndEvents.map((e) => e.episode);
  const totalRewardSeries: Series[] = [
    {
      name: "Total reward",
      values: episodeEndEvents.map((e) => Number(e.metrics.total_reward ?? NaN)),
    },
  ];

  const portfolioSeries: Series[] = [
    {
      name: "Avg portfolio",
      values: portfolioEvents.map((e) => Number(e.avg_portfolio ?? NaN)),
    },
  ];
  const portfolioX = portfolioEvents.map(xOfEvent);

  const lossSeries: Series[] = [
    { name: "Total loss", values: updateEvents.map((e) => Number(e.metrics.total_loss ?? NaN)) },
    { name: "Value loss", values: updateEvents.map((e) => Number(e.metrics.value_loss ?? NaN)) },
    {
      name: "Policy objective",
      values: updateEvents.map((e) => Number(e.metrics.policy_objective ?? NaN)),
    },
  ];

  const entropySeries: Series[] = [
    { name: "Entropy", values: updateEvents.map((e) => Number(e.metrics.entropy ?? NaN)) },
  ];

  const explainedVarSeries: Series[] = [
    {
      name: "Explained variance",
      values: updateEvents.map((e) => Number(e.metrics.explained_variance ?? NaN)),
    },
  ];

  const confidenceSeries: Series[] = [
    {
      name: "Chosen action prob",
      values: updateEvents.map((e) => Number(e.metrics.chosen_action_prob_mean ?? NaN)),
    },
    {
      name: "Policy confidence",
      values: updateEvents.map((e) => Number(e.metrics.policy_confidence_mean ?? NaN)),
    },
  ];

  const holdFracSeries: Series[] = [
    {
      name: "Hold fraction",
      values: updateEvents.map((e) => Number(e.metrics.hold_frac ?? NaN)),
    },
  ];

  const advantageSeries: Series[] = [
    { name: "Return mean", values: updateEvents.map((e) => Number(e.metrics.return_mean ?? NaN)) },
    { name: "|Adv| mean", values: updateEvents.map((e) => Number(e.metrics.adv_abs_mean ?? NaN)) },
  ];

  const hasModelLoss = updateEvents.some((e) => e.metrics.model_loss !== undefined);
  const modelLossSeries: Series[] = hasModelLoss
    ? [
        { name: "Model loss", values: updateEvents.map((e) => Number(e.metrics.model_loss ?? NaN)) },
        {
          name: "Latent loss",
          values: updateEvents.map((e) => Number(e.metrics.model_latent_loss ?? NaN)),
        },
        {
          name: "Reward loss",
          values: updateEvents.map((e) => Number(e.metrics.model_reward_loss ?? NaN)),
        },
      ]
    : [];

  // Summary card stats for the selected episode.
  const summaryEnd = resolvedEp !== null
    ? events.find((e) => e.kind === "episode_end" && e.episode === resolvedEp)
    : null;
  const summaryLastUpdate = resolvedEp !== null
    ? [...events].reverse().find((e) => e.kind === "update" && e.episode === resolvedEp)
    : null;

  const scoped_ = resolvedEp !== null;

  return (
    <div className={cn(scoped_ && "rounded-lg border-l-4 border-l-primary/70 pl-3")}>
      <Toolbar
        selection={selection}
        setSelection={setSelection}
        episodeNumbers={episodeNumbers}
        latestEpisode={latestEpisode}
        onResetAll={() => setResetSignal((n) => n + 1)}
      />

      {scoped_ && (
        <div className="mt-3 flex items-center justify-between rounded-md border border-primary/40 bg-primary/5 px-3 py-2">
          <span className="text-sm">
            Viewing <span className="font-semibold">Episode {resolvedEp}</span> only — other episodes
            are hidden.
          </span>
          <Button size="sm" variant="ghost" onClick={() => setSelection("all")}>
            <ArrowLeft className="h-3.5 w-3.5" />
            Back to global view
          </Button>
        </div>
      )}

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Panel title="Reward (per-step mean)">
          <StreamingChart
            series={rewardSeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="reward"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        {scoped_ ? (
          <Panel title={`Episode ${resolvedEp} summary`}>
            <EpisodeSummary
              episode={resolvedEp!}
              totalReward={summaryEnd ? Number(summaryEnd.metrics.total_reward ?? NaN) : null}
              avgReward={summaryEnd?.avg_reward ?? summaryLastUpdate?.avg_reward ?? null}
              episodeLength={
                summaryEnd ? Number(summaryEnd.metrics.episode_length ?? summaryEnd.step) : null
              }
              finalPortfolio={summaryEnd?.avg_portfolio ?? summaryLastUpdate?.avg_portfolio ?? null}
              lastLoss={summaryLastUpdate ? Number(summaryLastUpdate.metrics.total_loss ?? NaN) : null}
              lastEntropy={
                summaryLastUpdate ? Number(summaryLastUpdate.metrics.entropy ?? NaN) : null
              }
            />
          </Panel>
        ) : (
          <Panel title="Total reward per episode">
            <StreamingChart
              series={totalRewardSeries}
              xValues={totalRewardX}
              xLabel="episode"
              yLabel="total reward"
              resetSignal={resetSignal}
              height={220}
            />
          </Panel>
        )}

        <Panel title="Portfolio value">
          <StreamingChart
            series={portfolioSeries}
            xValues={portfolioX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="portfolio value"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        <Panel title="Losses">
          <StreamingChart
            series={lossSeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="loss"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        <Panel title="Entropy">
          <StreamingChart
            series={entropySeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="entropy (nats)"
            yPrecision={3}
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        <Panel title="Explained variance">
          <StreamingChart
            series={explainedVarSeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="expl. var"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        <Panel title="Policy confidence">
          <StreamingChart
            series={confidenceSeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="probability"
            yPrecision={3}
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        <Panel title="Hold action fraction">
          <StreamingChart
            series={holdFracSeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="fraction"
            yPrecision={3}
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        <Panel title="Return & advantage">
          <StreamingChart
            series={advantageSeries}
            xValues={rewardX}
            episodeBoundaries={boundaries}
            xLabel={xLabel}
            yLabel="value"
            resetSignal={resetSignal}
            height={220}
          />
        </Panel>

        {hasModelLoss && (
          <Panel title="Model loss">
            <StreamingChart
              series={modelLossSeries}
              xValues={rewardX}
              episodeBoundaries={boundaries}
              xLabel={xLabel}
              yLabel="loss"
              resetSignal={resetSignal}
              height={220}
            />
          </Panel>
        )}
      </div>
    </div>
  );
}

function Toolbar({
  selection,
  setSelection,
  episodeNumbers,
  latestEpisode,
  onResetAll,
}: {
  selection: Selection;
  setSelection: (s: Selection) => void;
  episodeNumbers: number[];
  latestEpisode: number | null;
  onResetAll: () => void;
}) {
  const value =
    selection === "all" ? "all" : selection === "latest" ? "latest" : `ep:${selection.n}`;

  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="flex items-center gap-2">
        <label className="text-xs uppercase tracking-wide text-muted-foreground">Episode</label>
        <Select
          value={value}
          onChange={(e) => {
            const v = e.target.value;
            if (v === "all") setSelection("all");
            else if (v === "latest") setSelection("latest");
            else if (v.startsWith("ep:")) setSelection({ kind: "episode", n: Number(v.slice(3)) });
          }}
          className="w-44"
        >
          <option value="all">All</option>
          <option value="latest">
            Latest{latestEpisode !== null ? ` (Episode ${latestEpisode})` : ""}
          </option>
          {episodeNumbers.map((n) => (
            <option key={n} value={`ep:${n}`}>
              Episode {n}
            </option>
          ))}
        </Select>
      </div>
      <Button variant="outline" size="sm" onClick={onResetAll}>
        <RotateCcw className="h-3.5 w-3.5" />
        Reset all views
      </Button>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="surface relative p-4">
      <h3 className="text-sm font-semibold mb-3">{title}</h3>
      {children}
    </div>
  );
}

function EpisodeSummary({
  episode,
  totalReward,
  avgReward,
  episodeLength,
  finalPortfolio,
  lastLoss,
  lastEntropy,
}: {
  episode: number;
  totalReward: number | null;
  avgReward: number | null;
  episodeLength: number | null;
  finalPortfolio: number | null;
  lastLoss: number | null;
  lastEntropy: number | null;
}) {
  const items: { label: string; value: string }[] = [
    {
      label: "Total reward",
      value: totalReward === null || !Number.isFinite(totalReward) ? "—" : formatNumber(totalReward, 3),
    },
    {
      label: "Avg reward",
      value: avgReward === null || !Number.isFinite(avgReward) ? "—" : formatNumber(avgReward, 4),
    },
    {
      label: "Steps",
      value: episodeLength === null || !Number.isFinite(episodeLength) ? "—" : formatNumber(episodeLength, 0),
    },
    {
      label: "Final portfolio",
      value: finalPortfolio === null || !Number.isFinite(finalPortfolio) ? "—" : formatNumber(finalPortfolio, 2),
    },
    {
      label: "Latest total loss",
      value: lastLoss === null || !Number.isFinite(lastLoss) ? "—" : formatNumber(lastLoss, 4),
    },
    {
      label: "Latest entropy",
      value: lastEntropy === null || !Number.isFinite(lastEntropy) ? "—" : formatNumber(lastEntropy, 4),
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
      {items.map(({ label, value }) => (
        <Card key={label} className="px-3 py-2">
          <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
          <p className="mt-1 text-base font-semibold tabular-nums">{value}</p>
        </Card>
      ))}
      <div className="col-span-2 md:col-span-3 text-[11px] text-muted-foreground">
        Showing Episode {episode}.
      </div>
    </div>
  );
}
