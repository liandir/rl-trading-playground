"use client";

import { useEffect, useRef, useState } from "react";
import type { RunEvent } from "@/lib/api-types";
import { wsUrl } from "@/lib/api";

export type StreamState = "connecting" | "open" | "closed" | "error";

export function useEventStream(runId: string | null): {
  events: RunEvent[];
  state: StreamState;
} {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<StreamState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!runId) return;
    setEvents([]);
    setState("connecting");
    const ws = new WebSocket(wsUrl(`/ws/runs/${runId}`));
    wsRef.current = ws;

    ws.onopen = () => setState("open");
    ws.onclose = () => setState("closed");
    ws.onerror = () => setState("error");
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as RunEvent;
        setEvents((prev) => [...prev, event]);
      } catch {
        // ignore malformed payloads
      }
    };
    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [runId]);

  return { events, state };
}
