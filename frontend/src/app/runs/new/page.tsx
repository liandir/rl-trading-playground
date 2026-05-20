"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { Play } from "lucide-react";
import { api } from "@/lib/api";
import { useRegistry } from "@/hooks/useRegistry";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Field } from "@/components/forms/Field";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PageHeader } from "@/components/shell/PageHeader";
import type {
  AgentConfig,
  DataConfig,
  EnvironmentConfig,
  RunSpec,
  TrainingConfig,
  ValidationConfig,
} from "@/lib/api-types";

const AGENT_DEFAULTS: AgentConfig = {
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

const ENV_DEFAULTS: EnvironmentConfig = {
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

const DATA_DEFAULT: DataConfig = {
  source: "synthetic",
  prepared_path: "data/historical_data0.ptt",
  raw_base_path: "data/Kraken_OHLCVT",
  interval: 5,
  train_split: 0.9,
};

const TRAINING_DEFAULT: TrainingConfig = {
  n_episodes: 3,
  batch_size: 8,
  max_steps: 50_000,
  warm_up: 10_000,
  update_interval: 128,
  n_updates: 4,
  burn_in_updates: 1,
  lr: 1e-5,
  optim: "AdamW",
  init_optimizer: false,
  max_grad_norm: null,
  save_checkpoint: true,
  seed: null,
};

const VALIDATION_DEFAULT: ValidationConfig = {
  start_index: null,
  length: 10_000,
  explore: false,
  seed: null,
};

export default function NewRunPage() {
  const router = useRouter();
  const { data: registry } = useRegistry();
  const agents = useQuery({ queryKey: ["agents"], queryFn: api.listAgents });
  const envs = useQuery({ queryKey: ["envs"], queryFn: api.listEnvs });
  const sources = useQuery({ queryKey: ["data-sources"], queryFn: api.listDataSources });

  const [name, setName] = useState("");
  const [kind, setKind] = useState<"training" | "validation">("training");

  const [agentMode, setAgentMode] = useState<"new" | "existing">("new");
  const [agentId, setAgentId] = useState<string>("");
  const [agentConfig, setAgentConfig] = useState<AgentConfig>(AGENT_DEFAULTS);

  const [envMode, setEnvMode] = useState<"new" | "existing">("new");
  const [envId, setEnvId] = useState<string>("");
  const [envConfig, setEnvConfig] = useState<EnvironmentConfig>(ENV_DEFAULTS);

  const [dataConfig, setDataConfig] = useState<DataConfig>(DATA_DEFAULT);
  const [training, setTraining] = useState<TrainingConfig>(TRAINING_DEFAULT);
  const [validation, setValidation] = useState<ValidationConfig>(VALIDATION_DEFAULT);

  const resolvedAgent = useMemo(() => {
    if (agentMode === "existing") {
      const found = agents.data?.find((a) => a.id === agentId);
      return found?.config ?? agentConfig;
    }
    return agentConfig;
  }, [agentMode, agentId, agents.data, agentConfig]);

  const resolvedEnv = useMemo(() => {
    if (envMode === "existing") {
      const found = envs.data?.find((e) => e.id === envId);
      return found?.config ?? envConfig;
    }
    return envConfig;
  }, [envMode, envId, envs.data, envConfig]);

  const launch = useMutation({
    mutationFn: async () => {
      const spec: RunSpec = {
        kind,
        data: dataConfig,
        env: resolvedEnv,
        agent: resolvedAgent,
        training: kind === "training" ? training : null,
        validation: kind === "validation" ? validation : null,
        agent_id: agentMode === "existing" ? agentId || null : null,
        env_id: envMode === "existing" ? envId || null : null,
      };
      return api.createRun(spec, name || undefined);
    },
    onSuccess: (run) => router.push(`/runs/${run.id}`),
  });

  return (
    <>
      <PageHeader
        title="New run"
        description="Pick a data source, environment, and agent — then launch training or validation."
      />

      <Card>
        <CardHeader>
          <CardTitle>Identification</CardTitle>
          <CardDescription>Give the run a memorable name. Leave blank for an auto-generated one.</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          <Field label="Name" className="md:col-span-2">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="ppo tiny synth 1" />
          </Field>
          <Field label="Kind">
            <Select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
              <option value="training">training</option>
              <option value="validation">validation</option>
            </Select>
          </Field>
        </CardContent>
      </Card>

      <Tabs defaultValue="agent" className="mt-6">
        <TabsList>
          <TabsTrigger value="agent">Agent</TabsTrigger>
          <TabsTrigger value="env">Environment</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
          <TabsTrigger value="run">Run options</TabsTrigger>
        </TabsList>

        <TabsContent value="agent">
          <Card>
            <CardContent className="grid gap-4 md:grid-cols-2 pt-5">
              <Field label="Source">
                <Select value={agentMode} onChange={(e) => setAgentMode(e.target.value as "new" | "existing")}>
                  <option value="new">Inline (define here)</option>
                  <option value="existing">Existing agent</option>
                </Select>
              </Field>
              {agentMode === "existing" ? (
                <Field label="Choose agent">
                  <Select value={agentId} onChange={(e) => setAgentId(e.target.value)}>
                    <option value="">— select —</option>
                    {agents.data?.map((a) => (
                      <option key={a.id} value={a.id}>
                        {a.name} ({a.config.agent_type}/{a.config.network_preset})
                      </option>
                    ))}
                  </Select>
                </Field>
              ) : (
                <>
                  <Field label="Agent type">
                    <Select
                      value={agentConfig.agent_type}
                      onChange={(e) =>
                        setAgentConfig({ ...agentConfig, agent_type: e.target.value as AgentConfig["agent_type"] })
                      }
                    >
                      {registry?.agent_types.map((t) => (
                        <option key={t.name} value={t.name}>
                          {t.name}
                        </option>
                      ))}
                    </Select>
                  </Field>
                  <Field label="Network preset">
                    <Select
                      value={agentConfig.network_preset}
                      onChange={(e) =>
                        setAgentConfig({
                          ...agentConfig,
                          network_preset: e.target.value as AgentConfig["network_preset"],
                        })
                      }
                    >
                      {registry?.network_presets.map((p) => (
                        <option key={p.name} value={p.name}>
                          {p.name}
                        </option>
                      ))}
                    </Select>
                  </Field>
                  <Field label="Entropy coef">
                    <Input
                      type="number"
                      step="0.01"
                      value={agentConfig.ent_coef}
                      onChange={(e) => setAgentConfig({ ...agentConfig, ent_coef: Number(e.target.value) })}
                    />
                  </Field>
                  <Field label="Gamma">
                    <Input
                      type="number"
                      step="0.0001"
                      value={agentConfig.gamma}
                      onChange={(e) => setAgentConfig({ ...agentConfig, gamma: Number(e.target.value) })}
                    />
                  </Field>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="env">
          <Card>
            <CardContent className="grid gap-4 md:grid-cols-2 pt-5">
              <Field label="Source">
                <Select value={envMode} onChange={(e) => setEnvMode(e.target.value as "new" | "existing")}>
                  <option value="new">Inline (define here)</option>
                  <option value="existing">Existing environment</option>
                </Select>
              </Field>
              {envMode === "existing" ? (
                <Field label="Choose environment">
                  <Select value={envId} onChange={(e) => setEnvId(e.target.value)}>
                    <option value="">— select —</option>
                    {envs.data?.map((e) => (
                      <option key={e.id} value={e.id}>
                        {e.name}
                      </option>
                    ))}
                  </Select>
                </Field>
              ) : (
                <>
                  <Field label="Initial cash">
                    <Input
                      type="number"
                      step="100"
                      value={envConfig.cash}
                      onChange={(e) => setEnvConfig({ ...envConfig, cash: Number(e.target.value) })}
                    />
                  </Field>
                  <Field label="Max leverage">
                    <Input
                      type="number"
                      step="0.5"
                      value={envConfig.max_leverage}
                      onChange={(e) => setEnvConfig({ ...envConfig, max_leverage: Number(e.target.value) })}
                    />
                  </Field>
                  <Field label="Reward mode">
                    <Input
                      value={envConfig.reward_mode}
                      onChange={(e) => setEnvConfig({ ...envConfig, reward_mode: e.target.value })}
                    />
                  </Field>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="data">
          <Card>
            <CardContent className="grid gap-4 md:grid-cols-2 pt-5">
              <Field label="Use saved preset">
                <Select
                  onChange={(e) => {
                    const found = sources.data?.find((s) => s.id === e.target.value);
                    if (found) setDataConfig(found.config);
                  }}
                  defaultValue=""
                >
                  <option value="">— inline —</option>
                  {sources.data?.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name} ({s.config.source})
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Source">
                <Select
                  value={dataConfig.source}
                  onChange={(e) => setDataConfig({ ...dataConfig, source: e.target.value as DataConfig["source"] })}
                >
                  {registry?.data_sources.map((s) => (
                    <option key={s} value={s}>
                      {s}
                    </option>
                  ))}
                </Select>
              </Field>
              <Field label="Prepared path" className="md:col-span-2">
                <Input
                  value={dataConfig.prepared_path}
                  onChange={(e) => setDataConfig({ ...dataConfig, prepared_path: e.target.value })}
                  disabled={dataConfig.source === "synthetic" || dataConfig.source === "kraken_csv"}
                />
              </Field>
              <Field label="Raw CSV folder" className="md:col-span-2">
                <Input
                  value={dataConfig.raw_base_path}
                  onChange={(e) => setDataConfig({ ...dataConfig, raw_base_path: e.target.value })}
                  disabled={dataConfig.source === "synthetic" || dataConfig.source === "prepared"}
                />
              </Field>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="run">
          {kind === "training" ? (
            <Card>
              <CardContent className="grid gap-4 md:grid-cols-3 pt-5">
                <Field label="Episodes">
                  <Input
                    type="number"
                    value={training.n_episodes}
                    onChange={(e) => setTraining({ ...training, n_episodes: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Batch size">
                  <Input
                    type="number"
                    value={training.batch_size}
                    onChange={(e) => setTraining({ ...training, batch_size: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Max steps">
                  <Input
                    type="number"
                    value={training.max_steps}
                    onChange={(e) => setTraining({ ...training, max_steps: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Warm-up">
                  <Input
                    type="number"
                    value={training.warm_up}
                    onChange={(e) => setTraining({ ...training, warm_up: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Update interval">
                  <Input
                    type="number"
                    value={training.update_interval}
                    onChange={(e) => setTraining({ ...training, update_interval: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Learning rate">
                  <Input
                    type="number"
                    step="0.000001"
                    value={training.lr}
                    onChange={(e) => setTraining({ ...training, lr: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Optimizer">
                  <Select
                    value={training.optim}
                    onChange={(e) => setTraining({ ...training, optim: e.target.value as TrainingConfig["optim"] })}
                  >
                    <option value="AdamW">AdamW</option>
                    <option value="Adam">Adam</option>
                  </Select>
                </Field>
                <Field label="Save checkpoint">
                  <Select
                    value={training.save_checkpoint ? "yes" : "no"}
                    onChange={(e) => setTraining({ ...training, save_checkpoint: e.target.value === "yes" })}
                  >
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                  </Select>
                </Field>
                <Field label="Seed">
                  <Input
                    type="number"
                    value={training.seed ?? ""}
                    onChange={(e) =>
                      setTraining({ ...training, seed: e.target.value === "" ? null : Number(e.target.value) })
                    }
                  />
                </Field>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="grid gap-4 md:grid-cols-3 pt-5">
                <Field label="Length (steps)">
                  <Input
                    type="number"
                    value={validation.length}
                    onChange={(e) => setValidation({ ...validation, length: Number(e.target.value) })}
                  />
                </Field>
                <Field label="Start index">
                  <Input
                    type="number"
                    value={validation.start_index ?? ""}
                    onChange={(e) =>
                      setValidation({
                        ...validation,
                        start_index: e.target.value === "" ? null : Number(e.target.value),
                      })
                    }
                  />
                </Field>
                <Field label="Explore">
                  <Select
                    value={validation.explore ? "yes" : "no"}
                    onChange={(e) => setValidation({ ...validation, explore: e.target.value === "yes" })}
                  >
                    <option value="no">Deterministic</option>
                    <option value="yes">Exploratory</option>
                  </Select>
                </Field>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <div className="mt-6 flex items-center justify-end gap-2 border-t border-border pt-4">
        <Link href="/runs" className="text-sm text-muted-foreground hover:text-foreground">
          Cancel
        </Link>
        <Button onClick={() => launch.mutate()} disabled={launch.isPending}>
          <Play className="h-4 w-4" />
          {launch.isPending ? "Launching…" : "Launch run"}
        </Button>
      </div>
      {launch.error && (
        <p className="mt-3 text-sm text-destructive">{(launch.error as Error).message}</p>
      )}
    </>
  );
}
