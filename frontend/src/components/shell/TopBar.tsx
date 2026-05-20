"use client";

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CircleDot } from "lucide-react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { ThemeToggle } from "./ThemeToggle";

export function TopBar() {
  const [now, setNow] = useState<string>("");
  useEffect(() => {
    const tick = () => setNow(new Date().toLocaleTimeString());
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  const { data: runs } = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 2_000,
  });
  const running = runs?.filter((r) => r.status === "running" || r.status === "queued").length ?? 0;

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 5_000,
    retry: 1,
  });

  return (
    <header className="flex h-14 items-center justify-between border-b border-border bg-card/60 px-6">
      <div className="flex items-center gap-3">
        <Badge variant={health.isError ? "destructive" : "success"} className="font-mono">
          <CircleDot className="h-3 w-3" />
          {health.isError ? "API offline" : "API ready"}
        </Badge>
        {running > 0 && (
          <Badge variant="primary">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
            {running} active run{running > 1 ? "s" : ""}
          </Badge>
        )}
      </div>
      <div className="flex items-center gap-3">
        <span className="font-mono text-xs text-muted-foreground tabular-nums">{now}</span>
        <ThemeToggle />
      </div>
    </header>
  );
}
