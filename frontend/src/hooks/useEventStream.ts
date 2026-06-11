"use client";

import { useEffect, useRef, useState } from "react";
import type { RunEvent } from "@/lib/api-types";
import { wsUrl } from "@/lib/api";

export type StreamState = "connecting" | "open" | "closed" | "error";

const EVENT_KINDS = new Set<RunEvent["kind"]>([
  "run_started",
  "episode_start",
  "warm_up",
  "update",
  "episode_end",
  "checkpoint_saved",
  "validation_step",
  "run_stopped",
  "run_finished",
  "run_failed",
]);

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function nullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function normalizeEvent(value: unknown): RunEvent | null {
  const raw = asRecord(value);
  const kind = raw.kind;
  if (typeof kind !== "string" || !EVENT_KINDS.has(kind as RunEvent["kind"])) {
    return null;
  }

  const metrics: Record<string, number> = {};
  for (const [key, metricValue] of Object.entries(asRecord(raw.metrics))) {
    if (typeof metricValue === "number" && Number.isFinite(metricValue)) {
      metrics[key] = metricValue;
    }
  }

  return {
    t: finiteNumber(raw.t, Date.now() / 1000),
    kind: kind as RunEvent["kind"],
    episode: Math.max(0, Math.trunc(finiteNumber(raw.episode, 0))),
    step: Math.max(0, Math.trunc(finiteNumber(raw.step, 0))),
    percent: finiteNumber(raw.percent, 0),
    avg_reward: nullableNumber(raw.avg_reward),
    avg_portfolio: nullableNumber(raw.avg_portfolio),
    sim_elapsed: typeof raw.sim_elapsed === "string" ? raw.sim_elapsed : "",
    metrics,
    message: typeof raw.message === "string" ? raw.message : "",
    extra: asRecord(raw.extra),
  };
}

function eventKey(event: RunEvent): string {
  return `${event.t}|${event.kind}|${event.episode}|${event.step}|${event.message}`;
}

export function useEventStream(runId: string | null): {
  events: RunEvent[];
  state: StreamState;
} {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<StreamState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const seenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!runId) return;
    let active = true;
    seenRef.current.clear();
    setEvents([]);
    setState("connecting");
    const ws = new WebSocket(wsUrl(`/ws/runs/${runId}`));
    wsRef.current = ws;

    ws.onopen = () => active && setState("open");
    ws.onclose = () => active && setState("closed");
    ws.onerror = () => active && setState("error");
    ws.onmessage = (msg) => {
      if (!active) return;
      try {
        const event = normalizeEvent(JSON.parse(msg.data));
        if (!event) return;
        const key = eventKey(event);
        setEvents((prev) => {
          if (seenRef.current.has(key)) return prev;
          seenRef.current.add(key);
          return [...prev, event];
        });
      } catch {
        // ignore malformed payloads
      }
    };
    return () => {
      active = false;
      ws.close();
      wsRef.current = null;
    };
  }, [runId]);

  return { events, state };
}
