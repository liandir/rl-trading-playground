"use client";

import Link from "next/link";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import { Button, buttonVariants } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty";
import { Table, TBody, TD, TH, THead, TR } from "@/components/ui/table";
import { PageHeader } from "@/components/shell/PageHeader";
import { formatRelativeTime, formatNumber } from "@/lib/format";

export default function EnvsPage() {
  const qc = useQueryClient();
  const { data: envs } = useQuery({ queryKey: ["envs"], queryFn: api.listEnvs });
  const remove = useMutation({
    mutationFn: (id: string) => api.deleteEnv(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["envs"] }),
  });

  return (
    <>
      <PageHeader
        title="Environments"
        description="Saved environment presets used by training and validation runs."
        actions={
          <Link href="/envs/new" className={buttonVariants() + " inline-flex items-center gap-2"}>
            <Plus className="h-4 w-4" /> New environment
          </Link>
        }
      />

      {!envs || envs.length === 0 ? (
        <EmptyState
          title="No environments saved"
          description="Save a preset to reuse it across runs and compare apples-to-apples."
          action={
            <Link href="/envs/new" className={buttonVariants()}>
              Create environment
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
                <TH>Initial cash</TH>
                <TH>Max leverage</TH>
                <TH>Created</TH>
                <TH />
              </TR>
            </THead>
            <TBody>
              {envs.map((env) => (
                <TR key={env.id}>
                  <TD className="font-medium">{env.name}</TD>
                  <TD className="font-mono text-xs text-muted-foreground">{env.config.env_type}</TD>
                  <TD className="tabular-nums">${formatNumber(env.config.cash, 2)}</TD>
                  <TD className="tabular-nums">{env.config.max_leverage.toFixed(1)}×</TD>
                  <TD className="text-xs text-muted-foreground">{formatRelativeTime(env.created_at)}</TD>
                  <TD>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        if (confirm(`Delete env "${env.name}"?`)) remove.mutate(env.id);
                      }}
                      aria-label="Delete env"
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
