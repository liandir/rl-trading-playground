"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useRegistry } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Field } from "@/components/forms/Field";
import { PageHeader } from "@/components/shell/PageHeader";
import type { AgentConfig, AgentType, NetworkPreset } from "@/lib/api-types";

const DEFAULTS: AgentConfig = {
  agent_type: "ppo",
  network_preset: "attention_memory",
  gamma: 0.9999,
  vf_coef: 0.5,
  ent_coef: 0.05,
  eps_clip: 0.2,
  advantage_type: "mc",
  gae_lambda: 0.95,
  normalize_advantages: true,
  aux_coef: 0.05,
  context_length: 64,
  device: "cpu",
  network_config: null,
};

export default function NewAgentPage() {
  const router = useRouter();
  const { data: registry } = useRegistry();
  const [name, setName] = useState("baseline");
  const [config, setConfig] = useState<AgentConfig>(DEFAULTS);
  const create = useMutation({
    mutationFn: () => api.createAgent(name, config),
    onSuccess: () => router.push("/agents"),
  });

  const set = <K extends keyof AgentConfig>(key: K, value: AgentConfig[K]) =>
    setConfig((prev) => ({ ...prev, [key]: value }));

  const isPPO = config.agent_type.startsWith("ppo");

  return (
    <>
      <PageHeader title="New agent" description="Define the agent, network preset, and core hyperparameters." />
      <Card>
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
          <CardDescription>
            Defaults follow the existing presets. Tweak only what you need — overrides land in network_config.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <Field label="Name" className="md:col-span-2">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="baseline" />
          </Field>
          <Field label="Agent type">
            <Select
              value={config.agent_type}
              onChange={(e) => set("agent_type", e.target.value as AgentType)}
            >
              {registry?.agent_types.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.name} — {t.description}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Network preset">
            <Select
              value={config.network_preset}
              onChange={(e) => set("network_preset", e.target.value as NetworkPreset)}
            >
              {registry?.network_presets.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name} — {p.network_type}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Gamma">
            <Input
              type="number"
              step="0.0001"
              value={config.gamma}
              onChange={(e) => set("gamma", Number(e.target.value))}
            />
          </Field>
          <Field label="Entropy coef">
            <Input
              type="number"
              step="0.01"
              value={config.ent_coef}
              onChange={(e) => set("ent_coef", Number(e.target.value))}
            />
          </Field>
          {isPPO && (
            <Field label="PPO clip">
              <Input
                type="number"
                step="0.05"
                value={config.eps_clip}
                onChange={(e) => set("eps_clip", Number(e.target.value))}
              />
            </Field>
          )}
          <Field label="Value coef">
            <Input
              type="number"
              step="0.05"
              value={config.vf_coef}
              onChange={(e) => set("vf_coef", Number(e.target.value))}
            />
          </Field>
          <Field label="Advantage type">
            <Select
              value={config.advantage_type}
              onChange={(e) => set("advantage_type", e.target.value as AgentConfig["advantage_type"])}
            >
              <option value="mc">Monte Carlo</option>
              <option value="td0">TD(0)</option>
              <option value="gae">GAE</option>
            </Select>
          </Field>
          <Field label="GAE lambda">
            <Input
              type="number"
              step="0.01"
              value={config.gae_lambda}
              onChange={(e) => set("gae_lambda", Number(e.target.value))}
            />
          </Field>
          <Field label="Device">
            <Select value={config.device} onChange={(e) => set("device", e.target.value)}>
              <option value="cpu">cpu</option>
              <option value="cuda">cuda</option>
            </Select>
          </Field>
        </CardContent>
        <CardFooter className="justify-end">
          <Button variant="ghost" onClick={() => router.back()}>
            Cancel
          </Button>
          <Button disabled={create.isPending || !name} onClick={() => create.mutate()}>
            {create.isPending ? "Creating…" : "Create agent"}
          </Button>
        </CardFooter>
      </Card>
      {create.error && (
        <p className="mt-3 text-sm text-destructive">{(create.error as Error).message}</p>
      )}
    </>
  );
}
