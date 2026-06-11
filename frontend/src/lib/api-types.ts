// Hand-written mirror of `src/api/schemas/*` until `pnpm gen:types` is run.

export type RunKind = "training" | "validation";
export type RunStatus = "queued" | "running" | "complete" | "stopped" | "failed";

export type AgentType =
  | "aac"
  | "aaq"
  | "ppo"
  | "aac_aux"
  | "aaq_aux"
  | "ppo_aux"
  | "aac_model"
  | "spatiotemporal_aac";

export type NetworkPreset =
  | "attention_memory"
  | "flat_per_asset"
  | "auxiliary_per_asset"
  | "flat_model"
  | "attention_model"
  | "spatiotemporal_auxiliary"
  | "tiny";

export type DataSource = "auto" | "prepared" | "kraken_csv" | "synthetic";

export interface DataConfig {
  source: DataSource;
  prepared_path: string;
  raw_base_path: string;
  interval: number;
  train_split: number;
}

export interface EnvironmentConfig {
  env_type: "longshort_hierarchical_leverage";
  cash: number;
  tau_minutes: number[];
  bankruptcy_threshold: number;
  min_open_dollars: number;
  transaction_eps: number;
  use_dollar_volume: boolean;
  size_buckets: number[];
  close_fee: number;
  open_fee: number;
  tax_rate: number;
  reward_mode: string;
  val_coeff: number;
  roi_coeff: number;
  done_reward_penalty: number;
  max_leverage: number;
  maintenance_margin_ratio: number | null;
  dtype: "float32" | "float64";
  device: string;
  eps: number;
}

export interface AgentConfig {
  agent_type: AgentType;
  network_preset: NetworkPreset;
  gamma: number;
  vf_coef: number;
  ent_coef: number;
  eps_clip: number;
  advantage_type: "mc" | "td0" | "gae";
  gae_lambda: number;
  normalize_advantages: boolean;
  aux_coef: number;
  context_length: number;
  device: string;
  network_config: Record<string, unknown> | null;
}

export interface TrainingConfig {
  n_episodes: number;
  batch_size: number;
  max_steps: number;
  warm_up: number;
  update_interval: number;
  n_updates: number;
  burn_in_updates: number;
  lr: number;
  optim: "AdamW" | "Adam";
  init_optimizer: boolean;
  max_grad_norm: number | null;
  save_checkpoint: boolean;
  seed: number | null;
}

export interface ValidationConfig {
  start_index: number | null;
  length: number;
  explore: boolean;
  seed: number | null;
}

export interface AgentRecord {
  id: string;
  name: string;
  config: AgentConfig;
  checkpoint_path: string | null;
  parent_run_id: string | null;
  created_at: string;
}

export interface EnvironmentRecord {
  id: string;
  name: string;
  config: EnvironmentConfig;
  created_at: string;
}

export interface DataSourceRecord {
  id: string;
  name: string;
  config: DataConfig;
  created_at: string;
}

export interface RunSpec {
  kind: RunKind;
  data: DataConfig;
  env: EnvironmentConfig;
  agent: AgentConfig;
  training?: TrainingConfig | null;
  validation?: ValidationConfig | null;
  agent_id?: string | null;
  env_id?: string | null;
  parent_run_id?: string | null;
}

export interface RunRecord {
  id: string;
  kind: RunKind;
  status: RunStatus;
  name: string;
  started_at: string;
  ended_at: string | null;
  agent_id: string | null;
  env_id: string | null;
  parent_run_id: string | null;
  pid: number | null;
  exit_code: number | null;
  notes: string;
  spec: RunSpec | null;
  summary: Record<string, number | string | boolean>;
}

export type EventKind =
  | "run_started"
  | "episode_start"
  | "warm_up"
  | "update"
  | "episode_end"
  | "checkpoint_saved"
  | "validation_step"
  | "run_stopped"
  | "run_finished"
  | "run_failed";

export interface RunEvent {
  t: number;
  kind: EventKind;
  episode: number;
  step: number;
  percent: number;
  avg_reward: number | null;
  avg_portfolio: number | null;
  sim_elapsed: string;
  metrics: Record<string, number>;
  message: string;
  extra: Record<string, unknown>;
}

export type DeploymentStatus = "queued" | "running" | "complete" | "stopped" | "failed";
export type DeploymentMode = "paper";

export interface DeploymentSpec {
  agent_id: string;
  env: EnvironmentConfig;
  pairs: string[];
  asset_names?: string[] | null;
  interval_minutes: number;
  mode: DeploymentMode;
  override_interval_mismatch: boolean;
}

export interface DeploymentRecord {
  id: string;
  status: DeploymentStatus;
  name: string;
  started_at: string;
  ended_at: string | null;
  agent_id: string;
  pid: number | null;
  exit_code: number | null;
  notes: string;
  spec: DeploymentSpec | null;
  summary: Record<string, number | string | boolean>;
}

export interface RegistryAgentType {
  name: AgentType;
  family: string;
  supports_aux: boolean;
  description: string;
}

export interface RegistryNetworkPreset {
  name: NetworkPreset;
  network_type: string;
  description: string;
}

export interface RegistryEnvironmentType {
  name: EnvironmentConfig["env_type"];
  description: string;
}

export interface RegistryResponse {
  agent_types: RegistryAgentType[];
  network_presets: RegistryNetworkPreset[];
  environment_types: RegistryEnvironmentType[];
  data_sources: DataSource[];
  network_schemas: Record<string, Record<string, unknown>>;
}

export interface CheckpointRecord {
  id: string;
  run_id: string | null;
  agent_id: string | null;
  step: number;
  metric_name: string | null;
  metric_value: number | null;
  tag: string | null;
  path: string;
  created_at: string;
}

export interface DataPreview {
  n_steps: number;
  n_assets: number;
  asset_names: string[];
  first_time: number | null;
  last_time: number | null;
}

export interface ValidationSeries {
  timestamps: string[];
  asset_names: string[];
  portfolio_value: number[];
  cash: number[];
  prices: number[][];
  normalized_prices: number[][];
  normalized_value: number[];
  rewards: number[];
  cumulative_reward: number[];
  drawdown: number[];
  portfolio_fraction: number[][];
  cash_fraction: number[];
  committed: number[];
  cumulative_longs: number[];
  cumulative_shorts: number[];
  cumulative_closes: number[];
  cumulative_invalid: number[];
  cumulative_realized_pnl: number[];
  cumulative_realized_cost: number[];
  actions: unknown[];
}

export interface ValidationArtifact {
  metrics: Record<string, number | string | boolean>;
  series: ValidationSeries;
}

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  modified: string | null;
}

export interface BrowseResponse {
  root: string;
  path: string;
  parent: string | null;
  entries: FileEntry[];
}

export interface InspectResponse {
  has_sidecar: boolean;
  source_path: string;
  sidecar_path: string | null;
  suggested: AgentConfig | null;
  parent_run_id: string | null;
  saved_at: string | null;
  network_keys: string[];
  note: string;
}
