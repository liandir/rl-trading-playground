"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useMutation } from "@tanstack/react-query";
import { CircleCheck, FileUp, FolderOpen, Info } from "lucide-react";
import { api } from "@/lib/api";
import { useRegistry } from "@/hooks/useRegistry";
import type { AgentConfig, FileEntry, InspectResponse } from "@/lib/api-types";
import { PageHeader } from "@/components/shell/PageHeader";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Field } from "@/components/forms/Field";
import { FileBrowser } from "@/components/files/FileBrowser";

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

export default function ImportAgentPage() {
  const router = useRouter();
  const { data: registry } = useRegistry();

  const [pickerOpen, setPickerOpen] = useState(false);
  const [sourcePath, setSourcePath] = useState<string>("");
  const [inspectResult, setInspectResult] = useState<InspectResponse | null>(null);
  const [name, setName] = useState<string>("");
  const [config, setConfig] = useState<AgentConfig>(DEFAULTS);

  const inspect = useMutation({
    mutationFn: (path: string) => api.inspectCheckpoint(path),
    onSuccess: (r) => {
      setInspectResult(r);
      if (r.suggested) {
        setConfig(r.suggested);
      }
    },
  });

  const importMut = useMutation({
    mutationFn: () => api.importAgent(name, sourcePath, config),
    onSuccess: (rec) => router.push(`/agents/${rec.id}`),
  });

  const onPicked = (entry: FileEntry) => {
    setSourcePath(entry.path);
    if (!name) {
      // Pre-fill name from the file stem so the user gets a reasonable default.
      const base = entry.name.replace(/\.[^/.]+$/, "");
      setName(base);
    }
    inspect.mutate(entry.path);
  };

  const set = <K extends keyof AgentConfig>(key: K, value: AgentConfig[K]) =>
    setConfig((prev) => ({ ...prev, [key]: value }));

  const isPPO = config.agent_type.startsWith("ppo");

  return (
    <>
      <PageHeader
        title="Import agent"
        description="Bring an existing .ptm checkpoint into the studio. Hyperparameters are editable; the original config is preserved alongside."
      />

      <Card>
        <CardHeader>
          <CardTitle>Checkpoint file</CardTitle>
          <CardDescription>
            Pick a .ptm/.pt/.pth file. If a matching .meta.json sidecar is found, the form pre-fills.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={() => setPickerOpen(true)} className="inline-flex items-center gap-2">
              <FolderOpen className="h-4 w-4" />
              Browse…
            </Button>
            <Input
              value={sourcePath}
              onChange={(e) => setSourcePath(e.target.value)}
              onBlur={() => sourcePath && inspect.mutate(sourcePath)}
              placeholder="or paste an absolute path to a .ptm file"
              className="flex-1 min-w-0 font-mono"
            />
          </div>

          {inspect.isPending && (
            <p className="text-xs text-muted-foreground">Inspecting…</p>
          )}
          {inspect.error && (
            <p className="text-sm text-destructive">{(inspect.error as Error).message}</p>
          )}
          {inspectResult && (
            <div className="surface-muted flex items-start gap-2 p-3 text-xs">
              {inspectResult.has_sidecar ? (
                <CircleCheck className="h-4 w-4 shrink-0 text-success" />
              ) : (
                <Info className="h-4 w-4 shrink-0 text-amber-500" />
              )}
              <div className="space-y-1">
                <p>
                  {inspectResult.has_sidecar ? (
                    <>Sidecar found — pre-filled <Badge variant="primary" className="ml-1 font-mono">{inspectResult.suggested?.agent_type}</Badge> / <Badge variant="outline" className="font-mono">{inspectResult.suggested?.network_preset}</Badge>.</>
                  ) : (
                    inspectResult.note || "No sidecar — pick the agent type and network preset by hand."
                  )}
                </p>
                {inspectResult.parent_run_id && (
                  <p className="text-muted-foreground">
                    Parent run: <span className="font-mono">{inspectResult.parent_run_id}</span>
                  </p>
                )}
                {!inspectResult.has_sidecar && inspectResult.network_keys.length > 0 && (
                  <details className="text-muted-foreground">
                    <summary className="cursor-pointer">First state_dict keys</summary>
                    <pre className="mt-1 font-mono whitespace-pre-wrap break-all">
                      {inspectResult.network_keys.slice(0, 16).join("\n")}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Agent configuration</CardTitle>
          <CardDescription>
            Edit only what you need to change. The original config (if any) is kept under the new agent&apos;s sidecar
            as <span className="font-mono">imported_from</span>.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          <Field label="Agent name" className="md:col-span-2">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="imported-1" />
          </Field>
          <Field label="Agent type">
            <Select
              value={config.agent_type}
              onChange={(e) => set("agent_type", e.target.value as AgentConfig["agent_type"])}
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
              onChange={(e) => set("network_preset", e.target.value as AgentConfig["network_preset"])}
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
          <Field label="Value coef">
            <Input
              type="number"
              step="0.05"
              value={config.vf_coef}
              onChange={(e) => set("vf_coef", Number(e.target.value))}
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
          <Button
            disabled={!name || !sourcePath || importMut.isPending}
            onClick={() => importMut.mutate()}
            className="inline-flex items-center gap-2"
          >
            <FileUp className="h-4 w-4" />
            {importMut.isPending ? "Importing…" : "Import agent"}
          </Button>
        </CardFooter>
      </Card>
      {importMut.error && (
        <p className="mt-3 text-sm text-destructive">{(importMut.error as Error).message}</p>
      )}

      <FileBrowser open={pickerOpen} onOpenChange={setPickerOpen} ext=".ptm" onSelect={onPicked} />
    </>
  );
}
