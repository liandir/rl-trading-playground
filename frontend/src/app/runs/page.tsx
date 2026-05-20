"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { api } from "@/lib/api";
import { buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { PageHeader } from "@/components/shell/PageHeader";
import { RunStatusBadge } from "@/components/runs/RunStatusBadge";
import { EmptyState } from "@/components/ui/empty";
import { formatRelativeTime } from "@/lib/format";

export default function RunsPage() {
  const { data: runs } = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 2_000,
  });

  return (
    <>
      <PageHeader
        title="Runs"
        description="Every training and validation run."
        actions={
          <Link href="/runs/new" className={buttonVariants() + " inline-flex items-center gap-2"}>
            <Plus className="h-4 w-4" /> New run
          </Link>
        }
      />

      {!runs || runs.length === 0 ? (
        <EmptyState
          title="No runs yet"
          description="Configure an environment and agent, then launch your first run."
          action={
            <Link href="/runs/new" className={buttonVariants()}>
              Launch a run
            </Link>
          }
        />
      ) : (
        <Card>
          <Table>
            <THead>
              <TR>
                <TH>Name</TH>
                <TH>Kind</TH>
                <TH>Agent</TH>
                <TH>Status</TH>
                <TH>Started</TH>
              </TR>
            </THead>
            <TBody>
              {runs.map((run) => (
                <TR key={run.id} className="cursor-pointer" onClick={() => (window.location.href = `/runs/${run.id}`)}>
                  <TD>
                    <Link href={`/runs/${run.id}`} className="font-medium hover:text-primary">
                      {run.name}
                    </Link>
                  </TD>
                  <TD className="text-muted-foreground">{run.kind}</TD>
                  <TD className="font-mono text-xs">
                    {run.spec?.agent.agent_type ?? "—"} / {run.spec?.agent.network_preset ?? "—"}
                  </TD>
                  <TD>
                    <RunStatusBadge status={run.status} />
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
