"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import type { AlignedData, Options } from "uplot";
import { RotateCcw } from "lucide-react";
import { cn } from "@/lib/cn";

export type Series = {
  name: string;
  values: number[];
  color?: string;
};

export type EpisodeBoundary = {
  step: number;
  episode: number;
};

const PALETTE = [
  "#6366f1", // indigo
  "#22c55e", // green
  "#f97316", // orange
  "#ec4899", // pink
  "#06b6d4", // cyan
  "#a855f7", // violet
  "#facc15", // yellow
];

function readPalette(): string[] {
  if (typeof window === "undefined") return PALETTE;
  const cs = getComputedStyle(document.documentElement);
  const primary = cs.getPropertyValue("--primary").trim();
  if (!primary) return PALETTE;
  return PALETTE;
}

export function StreamingChart({
  title,
  series,
  height = 240,
  xLabel,
  yLabel,
  xValues,
  episodeBoundaries,
  resetSignal,
  yPrecision,
  className,
}: {
  title?: string;
  series: Series[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xValues?: number[];
  episodeBoundaries?: EpisodeBoundary[];
  resetSignal?: number;
  yPrecision?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const userZoomedRef = useRef(false);
  const boundariesRef = useRef<EpisodeBoundary[] | undefined>(episodeBoundaries);
  const [isZoomed, setIsZoomed] = useState(false);

  const maxLen = series.reduce((m, s) => Math.max(m, s.values.length), 0);
  const data: AlignedData = useMemo(() => {
    const xs =
      xValues && xValues.length === maxLen
        ? xValues
        : Array.from({ length: maxLen }, (_, i) => i);
    const cols: number[][] = series.map((s) => {
      if (s.values.length === maxLen) return s.values;
      const padded = new Array<number>(maxLen).fill(NaN);
      for (let i = 0; i < s.values.length; i++) padded[i] = s.values[i];
      return padded;
    });
    return [xs, ...cols] as AlignedData;
  }, [series, maxLen, xValues]);

  // Keep latest boundaries available to the persistent uPlot instance.
  useEffect(() => {
    boundariesRef.current = episodeBoundaries;
    // Recalc axes so the splits/values callbacks pick up new boundary list.
    plotRef.current?.redraw(false, true);
  }, [episodeBoundaries]);

  const seriesKey = series.map((s) => s.name).join("|");
  const hasBoundaries = !!episodeBoundaries;

  // (Re)build the plot when the series shape or label-strategy changes.
  useEffect(() => {
    if (!ref.current) return;
    const palette = readPalette();
    const plotEl = ref.current;

    const xAxis: NonNullable<Options["axes"]>[number] = {
      stroke: "rgba(120,120,140,0.7)",
      grid: { stroke: "rgba(120,120,140,0.12)", width: 1 },
      ticks: { stroke: "rgba(120,120,140,0.2)" },
      label: xLabel,
    };

    if (hasBoundaries) {
      xAxis.splits = (
        u: uPlot,
        _axisIdx: number,
        scaleMin: number,
        scaleMax: number
      ) => {
        const bounds = boundariesRef.current ?? [];
        if (bounds.length === 0) return [];
        const visible = bounds
          .map((b) => b.step)
          .filter((s) => s >= scaleMin && s <= scaleMax);
        if (visible.length === 0) return [];
        // Decimate so labels don't collide. ~60 CSS px reserved per "Ep N" label.
        const plotPx = u.width || 600;
        const maxLabels = Math.max(2, Math.floor(plotPx / 60));
        if (visible.length <= maxLabels) return visible;
        const stride = Math.ceil(visible.length / maxLabels);
        const out: number[] = [];
        for (let i = 0; i < visible.length; i += stride) out.push(visible[i]);
        return out;
      };
      xAxis.values = (_u: uPlot, splits: number[]) => {
        const bounds = boundariesRef.current ?? [];
        const byStep = new Map(bounds.map((b) => [b.step, b.episode]));
        return splits.map((s) => {
          const ep = byStep.get(s);
          return ep === undefined ? "" : `Ep ${ep}`;
        });
      };
    }

    const opts: Options = {
      title,
      width: plotEl.clientWidth,
      height,
      legend: { show: true, live: true },
      scales: { x: { time: false } },
      cursor: {
        drag: { x: true, y: true, setScale: true, uni: 10 },
        x: true,
        y: true,
        points: { show: true, size: 6 },
      },
      axes: [
        xAxis,
        {
          stroke: "rgba(120,120,140,0.7)",
          grid: { stroke: "rgba(120,120,140,0.12)", width: 1 },
          ticks: { stroke: "rgba(120,120,140,0.2)" },
          label: yLabel,
          ...(yPrecision !== undefined && {
            values: (_u: uPlot, splits: number[]) =>
              splits.map((v) => (v == null ? "" : v.toFixed(yPrecision))),
          }),
        },
      ],
      series: [
        {},
        ...series.map((s, i) => ({
          label: s.name,
          stroke: s.color ?? palette[i % palette.length],
          width: 1.5,
          spanGaps: false,
        })),
      ],
      hooks: {
        // Draw faint vertical guides at each episode boundary.
        // uPlot iterates each hook key with .forEach(), so every value must be an array.
        draw: [
          (u: uPlot) => {
            const bounds = boundariesRef.current ?? [];
            if (bounds.length === 0) return;
            const ctx = u.ctx;
            const { left, top, width, height: h } = u.bbox;
            ctx.save();
            ctx.beginPath();
            ctx.rect(left, top, width, h);
            ctx.clip();
            ctx.strokeStyle = "rgba(120,120,140,0.22)";
            ctx.lineWidth = 1;
            for (const b of bounds) {
              const xPos = u.valToPos(b.step, "x", true);
              if (xPos < left || xPos > left + width) continue;
              ctx.beginPath();
              ctx.moveTo(xPos, top);
              ctx.lineTo(xPos, top + h);
              ctx.stroke();
            }
            ctx.restore();
          },
        ],
        // Mark zoomed only when the user finishes a drag-to-zoom selection.
        setSelect: [
          (u: uPlot) => {
            const sel = u.select;
            if (sel && sel.width > 0 && sel.height > 0) {
              userZoomedRef.current = true;
              setIsZoomed(true);
            }
          },
        ],
        // Preserve the current view when the user toggles series visibility
        // from the legend. uPlot auto-rescales to fit the now-visible series
        // by default; capture the pre-toggle scales and restore them after
        // uPlot's internal rescale has run.
        setSeries: [
          (u: uPlot, seriesIdx: number | null, opts: { show?: boolean }) => {
            if (seriesIdx === null || opts.show === undefined) return;
            const xMin = u.scales.x.min;
            const xMax = u.scales.x.max;
            const yMin = u.scales.y.min;
            const yMax = u.scales.y.max;
            if (xMin == null || xMax == null || yMin == null || yMax == null) return;
            queueMicrotask(() => {
              if (!plotRef.current) return;
              plotRef.current.setScale("x", { min: xMin, max: xMax });
              plotRef.current.setScale("y", { min: yMin, max: yMax });
            });
          },
        ],
      },
    };

    plotRef.current?.destroy();
    plotRef.current = new uPlot(opts, data, plotEl);
    userZoomedRef.current = false;
    setIsZoomed(false);

    const resize = () => {
      if (plotEl && plotRef.current) {
        plotRef.current.setSize({ width: plotEl.clientWidth, height });
      }
    };
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seriesKey, height, title, xLabel, yLabel, hasBoundaries, yPrecision]);

  // Push incremental data into the existing plot. Auto-rescale while not zoomed,
  // freeze the view once the user has zoomed.
  useEffect(() => {
    if (!plotRef.current) return;
    plotRef.current.setData(data, !userZoomedRef.current);
  }, [data]);

  // Reset on external signal.
  useEffect(() => {
    if (resetSignal === undefined) return;
    const u = plotRef.current;
    if (!u) return;
    userZoomedRef.current = false;
    setIsZoomed(false);
    u.setData(data, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetSignal]);

  const handleReset = () => {
    const u = plotRef.current;
    if (!u) return;
    userZoomedRef.current = false;
    setIsZoomed(false);
    u.setData(data, true);
  };

  return (
    <>
      {isZoomed && (
        <button
          type="button"
          onClick={handleReset}
          className={cn(
            "absolute right-2 top-2 z-10 inline-flex items-center gap-1 rounded-md",
            "border border-border bg-background/80 px-2 py-1 text-[11px] text-muted-foreground",
            "shadow-sm backdrop-blur hover:text-foreground hover:bg-background"
          )}
          aria-label="Reset zoom"
        >
          <RotateCcw className="h-3 w-3" />
          Reset
        </button>
      )}
      <div ref={ref} className={cn("w-full", className)} style={{ minHeight: height }} />
    </>
  );
}
