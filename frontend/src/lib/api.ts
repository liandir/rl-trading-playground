import type {
  AgentConfig,
  AgentRecord,
  CheckpointRecord,
  DataConfig,
  DataPreview,
  DataSourceRecord,
  EnvironmentConfig,
  EnvironmentRecord,
  RegistryResponse,
  RunEvent,
  RunRecord,
  RunSpec,
} from "./api-types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? (typeof window !== "undefined" ? "" : "http://localhost:8000");

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown, message: string) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let body: unknown;
    try {
      body = await res.json();
    } catch {
      body = await res.text();
    }
    const msg = typeof body === "object" && body && "detail" in body ? String((body as { detail: unknown }).detail) : res.statusText;
    throw new ApiError(res.status, body, msg);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; time: string }>("/healthz"),
  registry: () => request<RegistryResponse>("/registry"),

  // agents
  listAgents: () => request<AgentRecord[]>("/agents"),
  getAgent: (id: string) => request<AgentRecord>(`/agents/${id}`),
  createAgent: (name: string, config: AgentConfig) =>
    request<AgentRecord>("/agents", { method: "POST", body: JSON.stringify({ name, config }) }),
  deleteAgent: (id: string) => request<void>(`/agents/${id}`, { method: "DELETE" }),

  // envs
  listEnvs: () => request<EnvironmentRecord[]>("/envs"),
  getEnv: (id: string) => request<EnvironmentRecord>(`/envs/${id}`),
  createEnv: (name: string, config: EnvironmentConfig) =>
    request<EnvironmentRecord>("/envs", { method: "POST", body: JSON.stringify({ name, config }) }),
  deleteEnv: (id: string) => request<void>(`/envs/${id}`, { method: "DELETE" }),

  // data
  listDataSources: () => request<DataSourceRecord[]>("/data/sources"),
  createDataSource: (name: string, config: DataConfig) =>
    request<DataSourceRecord>("/data/sources", { method: "POST", body: JSON.stringify({ name, config }) }),
  previewData: (config: DataConfig) =>
    request<DataPreview>("/data/preview", { method: "POST", body: JSON.stringify(config) }),

  // runs
  listRuns: () => request<RunRecord[]>("/runs"),
  getRun: (id: string) => request<RunRecord>(`/runs/${id}`),
  createRun: (spec: RunSpec, name?: string) =>
    request<RunRecord>("/runs", { method: "POST", body: JSON.stringify({ name, spec }) }),
  stopRun: (id: string) => request<RunRecord>(`/runs/${id}/stop`, { method: "POST" }),
  runEvents: (id: string) => request<RunEvent[]>(`/runs/${id}/events`),
  runCheckpoints: (id: string) => request<CheckpointRecord[]>(`/runs/${id}/checkpoints`),

  // checkpoints
  listCheckpoints: (params: { run_id?: string; agent_id?: string } = {}) => {
    const q = new URLSearchParams(params as Record<string, string>).toString();
    return request<CheckpointRecord[]>(`/checkpoints${q ? `?${q}` : ""}`);
  },
};

export function wsUrl(path: string): string {
  if (typeof window === "undefined") return "";
  const apiBase = API_BASE || `${window.location.protocol}//${window.location.host}`;
  return apiBase.replace(/^http/, "ws") + path;
}
