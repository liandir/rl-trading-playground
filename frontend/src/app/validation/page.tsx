"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/card";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { EmptyState } from "@/components/ui/empty";
import { PageHeader } from "@/components/shell/PageHeader";
import { RunStatusBadge } from "@/components/runs/RunStatusBadge";
import { formatNumber, formatRelativeTime } from "@/lib/format";
import { buttonVariants } from "@/components/ui/button";
import { Plus } from "lucide-react";

export default function ValidationPage() {
  const { data } = useQuery({
    queryKey: ["runs", "validation"],
    queryFn: api.listRuns,
    refetchInterval: 3_000,
  });
  const runs = (data ?? []).filter((r) => r.kind === "validation");

  return (
    <>
      <PageHeader
        title="Validation"
        description="Backtests on held-out data. Each row is a deterministic or exploratory rollout."
        actions={
          <Link href="/runs/new" className={buttonVariants() + " inline-flex items-center gap-2"}>
            <Plus className="h-4 w-4" /> New validation
          </Link>
        }
      />

      {runs.length === 0 ? (
        <EmptyState
          title="No validation runs yet"
          description="Launch one to evaluate a checkpoint on the validation slice."
        />
      ) : (
        <Card>
          <Table>
            <THead>
              <TR>
                <TH>Name</TH>
                <TH>Status</TH>
                <TH>Steps</TH>
                <TH>Final value</TH>
                <TH>Total return</TH>
                <TH>Started</TH>
              </TR>
            </THead>
            <TBody>
              {runs.map((run) => (
                <TR key={run.id}>
                  <TD>
                    <Link href={`/runs/${run.id}`} className="font-medium hover:text-primary">
                      {run.name}
                    </Link>
                  </TD>
                  <TD>
                    <RunStatusBadge status={run.status} />
                  </TD>
                  <TD className="tabular-nums">{formatNumber(Number(run.summary.steps ?? 0), 0)}</TD>
                  <TD className="tabular-nums">
                    {run.summary.final_value !== undefined
                      ? `$${formatNumber(Number(run.summary.final_value), 2)}`
                      : "—"}
                  </TD>
                  <TD className="tabular-nums">
                    {run.summary.total_return_pct !== undefined
                      ? `${formatNumber(Number(run.summary.total_return_pct), 2)}%`
                      : "—"}
                  </TD>
                  <TD className="text-xs text-muted-foreground">{formatRelativeTime(run.started_at)}</TD>
                </TR>
              ))}
            </TBody>
          </Table>
        </Card>
      )}
    </>
  );
}
