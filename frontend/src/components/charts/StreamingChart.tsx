"use client";

import { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import type { AlignedData, Options } from "uplot";
import { cn } from "@/lib/cn";

export type Series = {
  name: string;
  values: number[];
  color?: string;
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
  // Allow swapping palette per-theme by reading CSS variables, but fall back to defaults.
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
  yLabel,
  className,
}: {
  title?: string;
  series: Series[];
  height?: number;
  yLabel?: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);

  const maxLen = series.reduce((m, s) => Math.max(m, s.values.length), 0);
  const data: AlignedData = useMemo(() => {
    const xs = Array.from({ length: maxLen }, (_, i) => i);
    const cols: number[][] = series.map((s) => {
      if (s.values.length === maxLen) return s.values;
      const padded = new Array<number>(maxLen).fill(NaN);
      for (let i = 0; i < s.values.length; i++) padded[i] = s.values[i];
      return padded;
    });
    return [xs, ...cols] as AlignedData;
  }, [series, maxLen]);

  // (Re)build the plot when the series shape changes.
  useEffect(() => {
    if (!ref.current) return;
    const palette = readPalette();
    const opts: Options = {
      title,
      width: ref.current.clientWidth,
      height,
      legend: { show: true, live: true },
      scales: { x: { time: false } },
      cursor: {
        // Drag-to-zoom on both axes; double-click resets.
        drag: { x: true, y: true, setScale: true, uni: 10 },
        // Crosshair stays visible so the user knows where they are.
        x: true,
        y: true,
        points: { show: true, size: 6 },
      },
      axes: [
        {
          stroke: "rgba(120,120,140,0.7)",
          grid: { stroke: "rgba(120,120,140,0.12)", width: 1 },
          ticks: { stroke: "rgba(120,120,140,0.2)" },
        },
        {
          stroke: "rgba(120,120,140,0.7)",
          grid: { stroke: "rgba(120,120,140,0.12)", width: 1 },
          ticks: { stroke: "rgba(120,120,140,0.2)" },
          label: yLabel,
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
    };
    plotRef.current?.destroy();
    plotRef.current = new uPlot(opts, data, ref.current);
    const resize = () => {
      if (ref.current && plotRef.current) {
        plotRef.current.setSize({ width: ref.current.clientWidth, height });
      }
    };
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series.map((s) => s.name).join("|"), height, title, yLabel]);

  // Push incremental data into the existing plot.
  useEffect(() => {
    if (!plotRef.current) return;
    plotRef.current.setData(data);
  }, [data]);

  return <div ref={ref} className={cn("w-full", className)} style={{ minHeight: height }} />;
}
