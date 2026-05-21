"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FileUp, Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { PageHeader } from "@/components/shell/PageHeader";
import { formatRelativeTime } from "@/lib/format";

export default function AgentsPage() {
  const qc = useQueryClient();
  const { data: agents } = useQuery({ queryKey: ["agents"], queryFn: api.listAgents });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteAgent(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });

  return (
    <>
      <PageHeader
        title="Agents"
        description="Saved agent presets and their last checkpoint."
        actions={
          <div className="flex items-center gap-2">
            <Link
              href="/agents/import"
              className={buttonVariants({ variant: "secondary" }) + " inline-flex items-center gap-2"}
            >
              <FileUp className="h-4 w-4" /> Import
            </Link>
            <Link href="/agents/new" className={buttonVariants() + " inline-flex items-center gap-2"}>
              <Plus className="h-4 w-4" /> New agent
            </Link>
          </div>
        }
      />

      {!agents || agents.length === 0 ? (
        <EmptyState
          title="No agents yet"
          description="Create one to load checkpoints, run validation, or kick off training."
          action={
            <Link href="/agents/new" className={buttonVariants()}>
              Create agent
            </Link>
          }
        />
      ) : (
        <Card>
          <Table>
            <THead>
              <TR>
                <TH>Name</TH>
                <TH>Type</TH>
                <TH>Network</TH>
                <TH>Checkpoint</TH>
                <TH>Created</TH>
                <TH />
              </TR>
            </THead>
            <TBody>
              {agents.map((a) => (
                <TR key={a.id}>
                  <TD>
                    <Link href={`/agents/${a.id}`} className="font-medium hover:text-primary">
                      {a.name}
                    </Link>
                  </TD>
                  <TD>
                    <Badge variant="primary" className="font-mono">
                      {a.config.agent_type}
                    </Badge>
                  </TD>
                  <TD className="font-mono text-xs text-muted-foreground">{a.config.network_preset}</TD>
                  <TD className="text-xs">
                    {a.checkpoint_path ? (
                      <span className="font-mono text-success">stored</span>
                    ) : (
                      <span className="text-muted-foreground">none</span>
                    )}
                  </TD>
                  <TD className="text-xs text-muted-foreground">{formatRelativeTime(a.created_at)}</TD>
                  <TD>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        if (confirm(`Delete agent "${a.name}"?`)) remove.mutate(a.id);
                      }}
                      aria-label="Delete agent"
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </TD>
                </TR>
              ))}
            </TBody>
          </Table>
        </Card>
      )}
    </>
  );
}
