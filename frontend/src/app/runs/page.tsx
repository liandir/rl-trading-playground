"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { GitCompareArrows, Plus } from "lucide-react";
import { api } from "@/lib/api";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { PageHeader } from "@/components/shell/PageHeader";
import { RunStatusBadge } from "@/components/runs/RunStatusBadge";
import { EmptyState } from "@/components/ui/empty";
import { formatRelativeTime } from "@/lib/format";

export default function RunsPage() {
  const router = useRouter();
  const { data: runs } = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 2_000,
  });

  const [selected, setSelected] = useState<Set<string>>(new Set());

  const allIds = useMemo(() => (runs ?? []).map((r) => r.id), [runs]);
  const allSelected = selected.size > 0 && selected.size === allIds.length;
  const partial = selected.size > 0 && !allSelected;

  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleAll = () =>
    setSelected((prev) => {
      if (prev.size === allIds.length) return new Set();
      return new Set(allIds);
    });

  const compareHref = `/runs/compare?ids=${Array.from(selected).join(",")}`;

  return (
    <>
      <PageHeader
        title="Runs"
        description={
          selected.size > 1
            ? `${selected.size} runs selected.`
            : selected.size === 1
            ? "1 run selected — select more to compare."
            : "Every training and validation run."
        }
        actions={
          <div className="flex items-center gap-2">
            {selected.size > 1 && (
              <Button
                variant="secondary"
                onClick={() => router.push(compareHref)}
                className="inline-flex items-center gap-2"
              >
                <GitCompareArrows className="h-4 w-4" /> Compare ({selected.size})
              </Button>
            )}
            <Link href="/runs/new" className={buttonVariants() + " inline-flex items-center gap-2"}>
              <Plus className="h-4 w-4" /> New run
            </Link>
          </div>
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
                <TH className="w-8">
                  <Checkbox
                    checked={allSelected}
                    onChange={toggleAll}
                    aria-label={allSelected ? "Deselect all" : "Select all"}
                    ref={(el) => {
                      if (el) el.indeterminate = partial;
                    }}
                  />
                </TH>
                <TH>Name</TH>
                <TH>Kind</TH>
                <TH>Agent</TH>
                <TH>Status</TH>
                <TH>Started</TH>
              </TR>
            </THead>
            <TBody>
              {runs.map((run) => {
                const isSelected = selected.has(run.id);
                return (
                  <TR
                    key={run.id}
                    data-selected={isSelected}
                    onClick={(e) => {
                      if ((e.target as HTMLElement).closest("[data-row-action]")) return;
                      router.push(`/runs/${run.id}`);
                    }}
                    className="cursor-pointer"
                  >
                    <TD onClick={(e) => e.stopPropagation()}>
                      <div data-row-action>
                        <Checkbox
                          checked={isSelected}
                          onChange={() => toggle(run.id)}
                          aria-label={`Select ${run.name}`}
                        />
                      </div>
                    </TD>
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
                    <TD className="text-xs text-muted-foreground">
                      {formatRelativeTime(run.started_at)}
                    </TD>
                  </TR>
                );
              })}
            </TBody>
          </Table>
        </Card>
      )}
    </>
  );
}
