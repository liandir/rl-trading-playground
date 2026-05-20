"use client";

import { useEffect, useRef } from "react";
import type { RunEvent } from "@/lib/api-types";
import { cn } from "@/lib/cn";

const KIND_COLOR: Record<RunEvent["kind"], string> = {
  run_started: "text-primary",
  episode_start: "text-foreground",
  warm_up: "text-muted-foreground",
  update: "text-foreground",
  episode_end: "text-foreground",
  checkpoint_saved: "text-success",
  validation_step: "text-foreground",
  run_stopped: "text-amber-500",
  run_finished: "text-success",
  run_failed: "text-destructive",
};

export function EventLog({ events }: { events: RunEvent[] }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [events.length]);

  return (
    <div
      ref={ref}
      className="surface max-h-[480px] overflow-auto p-3 font-mono text-xs leading-relaxed"
    >
      {events.length === 0 ? (
        <p className="text-muted-foreground">Waiting for events…</p>
      ) : (
        <ul className="space-y-0.5">
          {events.map((e, i) => (
            <li key={i} className="flex gap-3">
              <span className="shrink-0 text-muted-foreground tabular-nums">
                {new Date(e.t * 1000).toLocaleTimeString()}
              </span>
              <span className={cn("shrink-0 w-32", KIND_COLOR[e.kind])}>{e.kind}</span>
              <span className="text-muted-foreground tabular-nums">
                ep {e.episode} step {e.step}
              </span>
              {e.message && <span className="truncate">{e.message}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
