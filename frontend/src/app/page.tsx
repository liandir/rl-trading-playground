"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { Activity, Bot, Database, Plus, Server } from "lucide-react";
import { api } from "@/lib/api";
import { PageHeader } from "@/components/shell/PageHeader";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { buttonVariants } from "@/components/ui/button";
import { RunStatusBadge } from "@/components/runs/RunStatusBadge";
import { formatRelativeTime } from "@/lib/format";

export default function Dashboard() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 3_000 });
  const agents = useQuery({ queryKey: ["agents"], queryFn: api.listAgents });
  const envs = useQuery({ queryKey: ["envs"], queryFn: api.listEnvs });
  const sources = useQuery({ queryKey: ["data-sources"], queryFn: api.listDataSources });

  const allRuns = runs.data ?? [];
  const activeRuns = allRuns.filter((r) => r.status === "running" || r.status === "queued");
  const completedRuns = allRuns.filter((r) => r.status === "complete").length;
  const failedRuns = allRuns.filter((r) => r.status === "failed").length;

  const stats = [
    { label: "Active runs", value: activeRuns.length, icon: Activity, accent: "text-primary" },
    { label: "Agents", value: agents.data?.length ?? 0, icon: Bot, accent: "text-foreground" },
    { label: "Environments", value: envs.data?.length ?? 0, icon: Server, accent: "text-foreground" },
    { label: "Data sources", value: sources.data?.length ?? 0, icon: Database, accent: "text-foreground" },
  ];

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Quick overview of active runs, models, and environments."
        actions={
          <Link href="/runs/new" className={buttonVariants() + " inline-flex items-center gap-2"}>
            <Plus className="h-4 w-4" /> New run
          </Link>
        }
      />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {stats.map((s) => (
          <Card key={s.label}>
            <CardContent className="flex items-center gap-3 pt-5">
              <div className={"rounded-md p-2 bg-muted " + s.accent}>
                <s.icon className="h-5 w-5" />
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-muted-foreground">{s.label}</p>
                <p className="text-2xl font-semibold tabular-nums">{s.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Recent runs</CardTitle>
            <CardDescription>Last 10 training and validation runs.</CardDescription>
          </CardHeader>
          <CardContent>
            {allRuns.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No runs yet — launch one from the <Link className="text-primary" href="/runs/new">New run</Link> page.
              </p>
            ) : (
              <ul className="divide-y divide-border">
                {allRuns.slice(0, 10).map((run) => (
                  <li key={run.id} className="py-3">
                    <Link
                      href={`/runs/${run.id}`}
                      className="flex items-center justify-between gap-4 hover:text-foreground"
                    >
                      <div className="flex min-w-0 items-center gap-3">
                        <RunStatusBadge status={run.status} />
                        <div className="min-w-0">
                          <p className="truncate text-sm font-medium">{run.name}</p>
                          <p className="truncate text-xs text-muted-foreground">
                            {run.kind} • {run.spec?.agent.agent_type ?? "—"} / {run.spec?.agent.network_preset ?? "—"}
                          </p>
                        </div>
                      </div>
                      <span className="shrink-0 text-xs text-muted-foreground">
                        {formatRelativeTime(run.started_at)}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Run health</CardTitle>
            <CardDescription>Lifetime status counts.</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3 text-sm">
              <li className="flex items-center justify-between">
                <span className="text-muted-foreground">Complete</span>
                <span className="font-mono tabular-nums">{completedRuns}</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-muted-foreground">Failed</span>
                <span className="font-mono tabular-nums text-destructive">{failedRuns}</span>
              </li>
              <li className="flex items-center justify-between">
                <span className="text-muted-foreground">Active</span>
                <span className="font-mono tabular-nums text-primary">{activeRuns.length}</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
