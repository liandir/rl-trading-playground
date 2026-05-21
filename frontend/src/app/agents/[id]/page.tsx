"use client";

import { use } from "react";
import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/shell/PageHeader";
import { CheckpointList } from "@/components/runs/CheckpointList";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { RunStatusBadge } from "@/components/runs/RunStatusBadge";
import { formatNumber, formatRelativeTime } from "@/lib/format";

export default function AgentDetail({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const { id } = use(params);
  const qc = useQueryClient();
  const agent = useQuery({ queryKey: ["agent", id], queryFn: () => api.getAgent(id) });
  const checkpoints = useQuery({
    queryKey: ["checkpoints", { agent_id: id }],
    queryFn: () => api.listCheckpoints({ agent_id: id }),
  });
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.listRuns });
  const remove = useMutation({
    mutationFn: () => api.deleteAgent(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agents"] });
      router.push("/agents");
    },
  });

  if (agent.isLoading || !agent.data) {
    return (
      <div className="flex h-48 items-center justify-center text-muted-foreground">
        <Loader2 className="h-5 w-5 animate-spin" />
      </div>
    );
  }

  const a = agent.data;
  const cpList = checkpoints.data ?? [];
  // Surface runs that produced (or were spawned from) this agent.
  const relatedRuns = (runs.data ?? []).filter(
    (r) => r.agent_id === a.id || r.parent_run_id === a.parent_run_id
  );

  return (
    <>
      <PageHeader
        title={a.name}
        description={
          <span className="inline-flex items-center gap-2 text-sm">
            <Link
              href="/agents"
              className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="h-3 w-3" /> All agents
            </Link>
            <span className="text-muted-foreground">•</span>
            <Badge variant="primary" className="font-mono">
              {a.config.agent_type}
            </Badge>
            <Badge variant="outline" className="font-mono">
              {a.config.network_preset}
            </Badge>
          </span>
        }
        actions={
          <Button
            variant="destructive"
            size="sm"
            onClick={() => {
              if (confirm(`Delete agent "${a.name}"?`)) remove.mutate();
            }}
            disabled={remove.isPending}
          >
            <Trash2 className="h-3.5 w-3.5" />
            Delete
          </Button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Active checkpoint</CardTitle>
            <CardDescription>
              The runner loads this file when a run references this agent.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {a.checkpoint_path ? (
              <>
                <p className="font-mono text-xs text-muted-foreground break-all">
                  {a.checkpoint_path}
                </p>
                {a.parent_run_id && (
                  <p className="text-xs">
                    Source run:{" "}
                    <Link
                      className="font-mono text-primary hover:underline"
                      href={`/runs/${a.parent_run_id}`}
                    >
                      {a.parent_run_id}
                    </Link>
                  </p>
                )}
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                No checkpoint attached yet. Train a run targeting this agent, or attach an existing
                checkpoint below.
              </p>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Hyperparameters</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-3">
              {[
                ["gamma", a.config.gamma],
                ["ent_coef", a.config.ent_coef],
                ["vf_coef", a.config.vf_coef],
                ["eps_clip", a.config.eps_clip],
                ["advantage", a.config.advantage_type],
                ["gae_lambda", a.config.gae_lambda],
                ["device", a.config.device],
                ["normalize_adv", String(a.config.normalize_advantages)],
                ["created", formatRelativeTime(a.created_at)],
              ].map(([key, val]) => (
                <div key={key} className="flex justify-between gap-2">
                  <dt className="text-muted-foreground">{key}</dt>
                  <dd className="font-mono tabular-nums text-foreground">
                    {typeof val === "number" ? formatNumber(val) : String(val)}
                  </dd>
                </div>
              ))}
            </dl>
          </CardContent>
        </Card>
      </div>

      <h2 className="mt-8 mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Checkpoints recorded against this agent
      </h2>
      <CheckpointList
        checkpoints={cpList}
        showAttach={false}
        emptyHint="Train a run that references this agent and a checkpoint will appear here."
      />

      <h2 className="mt-8 mb-3 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Related runs
      </h2>
      {relatedRuns.length === 0 ? (
        <p className="text-sm text-muted-foreground">No runs linked to this agent yet.</p>
      ) : (
        <Card>
          <Table>
            <THead>
              <TR>
                <TH>Name</TH>
                <TH>Kind</TH>
                <TH>Status</TH>
                <TH>Started</TH>
              </TR>
            </THead>
            <TBody>
              {relatedRuns.map((run) => (
                <TR key={run.id}>
                  <TD>
                    <Link href={`/runs/${run.id}`} className="font-medium hover:text-primary">
                      {run.name}
                    </Link>
                  </TD>
                  <TD className="text-muted-foreground">{run.kind}</TD>
                  <TD>
                    <RunStatusBadge status={run.status} />
                  </TD>
                  <TD className="text-xs text-muted-foreground">
                    {formatRelativeTime(run.started_at)}
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
