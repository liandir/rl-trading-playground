"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Field } from "@/components/forms/Field";
import { PageHeader } from "@/components/shell/PageHeader";
import type { EnvironmentConfig } from "@/lib/api-types";

const DEFAULTS: EnvironmentConfig = {
  env_type: "longshort_hierarchical_leverage",
  cash: 1000,
  tau_minutes: [30, 240, 1440, 10080],
  bankruptcy_threshold: 10,
  min_open_dollars: 2,
  transaction_eps: 0.01,
  use_dollar_volume: true,
  size_buckets: [0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
  close_fee: 1,
  open_fee: 1,
  tax_rate: 0.26,
  reward_mode: "log",
  val_coeff: 20,
  roi_coeff: 100,
  done_reward_penalty: 100,
  max_leverage: 10,
  maintenance_margin_ratio: null,
  dtype: "float32",
  device: "cpu",
  eps: 1e-8,
};

export default function NewEnvPage() {
  const router = useRouter();
  const [name, setName] = useState("default");
  const [config, setConfig] = useState<EnvironmentConfig>(DEFAULTS);
  const create = useMutation({
    mutationFn: () => api.createEnv(name, config),
    onSuccess: () => router.push("/envs"),
  });

  const set = <K extends keyof EnvironmentConfig>(key: K, value: EnvironmentConfig[K]) =>
    setConfig((prev) => ({ ...prev, [key]: value }));

  return (
    <>
      <PageHeader title="New environment" description="Tune the trading environment used by runs." />
      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>Only long/short hierarchical leverage is supported today.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <Field label="Name" className="md:col-span-2">
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          <Field label="Initial cash ($)">
            <Input
              type="number"
              step="100"
              value={config.cash}
              onChange={(e) => set("cash", Number(e.target.value))}
            />
          </Field>
          <Field label="Max leverage">
            <Input
              type="number"
              step="0.5"
              value={config.max_leverage}
              onChange={(e) => set("max_leverage", Number(e.target.value))}
            />
          </Field>
          <Field label="Reward mode">
            <Input value={config.reward_mode} onChange={(e) => set("reward_mode", e.target.value)} />
          </Field>
          <Field label="Open fee (bps)">
            <Input
              type="number"
              step="0.5"
              value={config.open_fee}
              onChange={(e) => set("open_fee", Number(e.target.value))}
            />
          </Field>
          <Field label="Close fee (bps)">
            <Input
              type="number"
              step="0.5"
              value={config.close_fee}
              onChange={(e) => set("close_fee", Number(e.target.value))}
            />
          </Field>
          <Field label="Tax rate">
            <Input
              type="number"
              step="0.01"
              value={config.tax_rate}
              onChange={(e) => set("tax_rate", Number(e.target.value))}
            />
          </Field>
          <Field label="Bankruptcy threshold ($)">
            <Input
              type="number"
              step="1"
              value={config.bankruptcy_threshold}
              onChange={(e) => set("bankruptcy_threshold", Number(e.target.value))}
            />
          </Field>
          <Field label="Min open ($)">
            <Input
              type="number"
              step="0.5"
              value={config.min_open_dollars}
              onChange={(e) => set("min_open_dollars", Number(e.target.value))}
            />
          </Field>
          <Field label="Size buckets" className="md:col-span-2" hint="Comma-separated fractions of available cash.">
            <Input
              value={config.size_buckets.join(",")}
              onChange={(e) =>
                set(
                  "size_buckets",
                  e.target.value
                    .split(",")
                    .map((v) => Number(v.trim()))
                    .filter((v) => !Number.isNaN(v))
                )
              }
            />
          </Field>
          <Field label="Tau windows (minutes)" className="md:col-span-2" hint="Time windows used in feature extraction.">
            <Input
              value={config.tau_minutes.join(",")}
              onChange={(e) =>
                set(
                  "tau_minutes",
                  e.target.value
                    .split(",")
                    .map((v) => Number(v.trim()))
                    .filter((v) => !Number.isNaN(v))
                )
              }
            />
          </Field>
        </CardContent>
        <CardFooter className="justify-end">
          <Button variant="ghost" onClick={() => router.back()}>
            Cancel
          </Button>
          <Button disabled={create.isPending || !name} onClick={() => create.mutate()}>
            {create.isPending ? "Creating…" : "Create environment"}
          </Button>
        </CardFooter>
      </Card>
      {create.error && (
        <p className="mt-3 text-sm text-destructive">{(create.error as Error).message}</p>
      )}
    </>
  );
}
