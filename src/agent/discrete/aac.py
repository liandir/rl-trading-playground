import torch
from torch import distributions

from src.agent.discrete.network import build_network


def apply_action_mask(logits: torch.Tensor, valid: torch.BoolTensor) -> torch.Tensor:
    """Set logits of invalid actions to -inf before any sampling or loss computation."""
    return logits.masked_fill(~valid, float("-inf"))


def _fmt_sim_elapsed(elapsed_seconds: float) -> str:
    s = int(elapsed_seconds)
    d = s // 86400
    h = (s % 86400) // 3600
    return f"{d} day{'s' if d != 1 else ''} {h} h"


def _sample_start_indices(total_points: int, max_steps: int, batch_size: int = 1) -> torch.Tensor:
    """
    Sample valid reset indices for rollouts of ``max_steps`` environment steps.

    Reset consumes one data point and each environment step consumes one future
    point, so a full rollout requires at least ``max_steps + 1`` aligned market
    observations. Valid starts therefore lie in ``[0, total_points - max_steps)``.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}.")
    if total_points <= max_steps:
        raise ValueError(
            f"Need at least max_steps + 1 data points to sample a rollout of {max_steps} steps, "
            f"got total_points={total_points}."
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return torch.randint(total_points - max_steps, (batch_size,))


# ---------------------------------------------------------------------------
# Advantage helpers
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_t: torch.Tensor,
    v_tp1: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (Schulman et al., 2016).

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t^GAE = sum_{k>=0} (gamma * lambda)^k * delta_{t+k}

    Returns (advantages, returns) where returns = advantages + V(s_t),
    used as targets for the value loss.

    rewards, dones, v_t, v_tp1 : (T, ...) — leading time dimension first.
    """
    deltas = rewards + gamma * (1.0 - dones) * v_tp1 - v_t

    shape    = deltas.shape
    T        = shape[0]
    gae      = torch.zeros(shape[1:], device=deltas.device, dtype=deltas.dtype)
    adv_list = []

    for t in reversed(range(T)):
        gae = deltas[t] + gamma * lam * (1.0 - dones[t]) * gae
        adv_list.append(gae)

    advantages = torch.stack(list(reversed(adv_list)), dim=0)  # (T, ...)
    returns    = advantages + v_t

    return advantages, returns


def _compute_mc(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_t: torch.Tensor,
    v_tp1_last: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Discounted Monte Carlo returns with bootstrap from V(s_{T+1}) at truncation.

        G_t = r_t + gamma * (1 - done_t) * G_{t+1}
        G_T ← bootstrapped from V(s_{T+1}) (zeroed if the final step is terminal)
        A_t = G_t - V(s_t)

    Equivalent to GAE with lambda=1.
    """
    T = rewards.shape[0]
    returns = torch.empty_like(rewards)
    g = v_tp1_last
    for t in reversed(range(T)):
        g = rewards[t] + gamma * (1.0 - dones[t]) * g
        returns[t] = g
    advantages = returns - v_t
    return advantages, returns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

UPDATE_METRIC_NAMES = (
    "policy_objective",
    "value_loss",
    "entropy",
    "total_loss",
    "return_mean",
    "value_mean",
    "adv_abs_mean",
    "adv_pos_frac",
    "chosen_action_prob_mean",
    "policy_confidence_mean",
    "explained_variance",
)


def _empty_update_metrics() -> dict[str, list[float]]:
    return {name: [0.0] for name in UPDATE_METRIC_NAMES}


def _init_update_metrics() -> dict[str, list[float]]:
    return {name: [] for name in UPDATE_METRIC_NAMES}


def _explained_variance(targets: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    target_var = targets.var(unbiased=False)
    if float(target_var.detach().cpu().item()) <= 1e-12:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)
    return 1.0 - residuals.var(unbiased=False) / (target_var + 1e-12)


def _append_update_metrics(
    metric_store: dict[str, list[float]],
    *,
    logits: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    raw_advantages: torch.Tensor,
    policy_objective: torch.Tensor,
    value_loss: torch.Tensor,
    entropy: torch.Tensor,
    loss: torch.Tensor,
):
    probs                    = torch.softmax(logits.detach(), dim=-1)
    values_detached          = values.detach()
    returns_detached         = returns.detach()
    raw_advantages_detached  = raw_advantages.detach()
    residuals                = returns_detached - values_detached

    metrics = {
        "policy_objective":        policy_objective.detach(),
        "value_loss":              value_loss.detach(),
        "entropy":                 entropy.detach(),
        "total_loss":              loss.detach(),
        "return_mean":             returns_detached.mean(),
        "value_mean":              values_detached.mean(),
        "adv_abs_mean":            raw_advantages_detached.abs().mean(),
        "adv_pos_frac":            (raw_advantages_detached > 0).to(dtype=returns_detached.dtype).mean(),
        "chosen_action_prob_mean": log_probs.detach().exp().mean(),
        "policy_confidence_mean":  probs.max(dim=-1).values.mean(),
        "explained_variance":      _explained_variance(returns_detached, residuals),
    }
    for name, value in metrics.items():
        metric_store[name].append(float(value.cpu().item()))


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.dones       = []
        self.valid_masks = []

    def store(self, state, action, reward, done, valid_mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states  = torch.stack(self.states).to(device=device, dtype=dtype)
        actions = torch.stack(self.actions).to(device=device, dtype=dtype)
        rewards = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones   = torch.stack(self.dones).to(device=device, dtype=dtype)
        valid_masks = None
        if self.valid_masks and self.valid_masks[0] is not None:
            valid_masks = torch.stack(self.valid_masks).to(device=device)
        return states, actions, rewards, dones, valid_masks


# ---------------------------------------------------------------------------
# AACAgent — feedforward
# ---------------------------------------------------------------------------

class AACAgent:
    """
    Advantage Actor-Critic (feedforward).

    The network shares a feedforward trunk and splits into:
      - an actor head that parameterises a categorical policy  pi(a|s)
      - a value head that predicts a scalar baseline  V(s)

    Returns are computed by TD(0) bootstrapping with the current value function:
        G_t = r_t + gamma * (1 - done_t) * V(s_{t+1})

    Advantage subtracts the baseline:
        A_t = G_t - V(s_t)

    The joint loss is:
        L = -E[log pi(a_t|s_t) * A_t.detach()]
            + vf_coef * 0.5 * E[(G_t.detach() - V(s_t))^2]
            - ent_coef * H(pi(.|s_t))

    There is no target network: the value bootstrap V(s_{t+1}) is computed
    once before the update loop and treated as a fixed target thereafter.
    """

    def __init__(
        self,
        network: dict,
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        advantage_type: str = "td0",
        gae_lambda: float = 0.95,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if advantage_type not in ("td0", "gae", "mc"):
            raise ValueError(f"advantage_type must be 'td0', 'gae', or 'mc', got '{advantage_type}'.")
        self.gamma                = gamma
        self.vf_coef              = vf_coef
        self.ent_coef             = ent_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type       = advantage_type
        self.gae_lambda           = gae_lambda
        self.dtype                = dtype
        self.device               = torch.device(device)

        self.net = build_network(network).to(device=self.device, dtype=self.dtype)
        self.state_dim  = getattr(self.net, "state_dim", None)
        self.action_dim = self.net.action_dim

        self.buffer = RolloutBuffer()

    def infer_logits(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits

    def act(
        self,
        state: torch.Tensor,
        valid_mask: torch.BoolTensor | None = None,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(grad_enabled):
            state  = state.to(dtype=self.dtype, device=self.device)
            logits = self.infer_logits(state)
            if valid_mask is not None:
                logits = apply_action_mask(logits, valid_mask.to(self.device))
            if explore:
                action = distributions.Categorical(logits=logits).sample().squeeze()
            else:
                action = logits.argmax(dim=-1).squeeze()
        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, done, valid_mask=None):
        self.buffer.store(state, action, reward, done, valid_mask=valid_mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def compute_advantages(
        self,
        last_next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor | None]:
        states, actions, rewards, dones, valid_masks = self.buffer.to_tensors(self.device, self.dtype)
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones   = dones.squeeze(-1)   if dones.dim()   > 1 else dones

        with torch.no_grad():
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, values_all = self.net(all_states)   # (T+1, [B,])

            v_t   = values_all[:-1]
            v_tp1 = values_all[1:]

            if self.advantage_type == "gae":
                raw_adv, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            elif self.advantage_type == "mc":
                raw_adv, returns = _compute_mc(rewards, dones, v_t, v_tp1[-1], self.gamma)
            else:  # td0
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t

            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, raw_adv, returns, states, actions, valid_masks

    def update(
        self,
        last_next_state: torch.Tensor,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, valid_masks = self.compute_advantages(last_next_state)

        logits, values = self.net(states)
        if valid_masks is not None:
            logits = apply_action_mask(logits, valid_masks)
        dist      = distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())

        raw_entropy = dist.entropy()  # (T, [B,])
        if valid_masks is not None:
            k = valid_masks.sum(dim=-1).float()  # (T, [B,])
            log_k = torch.log(k.clamp(min=2))    # clamp so log >= log(2) when K>=2; K=1 → entropy=0 anyway
            entropy = (raw_entropy / log_k).mean()
        else:
            entropy = raw_entropy.mean()

        adv              = advantages.detach()
        policy_objective = (log_probs * adv).mean()
        value_loss       = 0.5 * (returns.detach() - values).pow(2).mean()
        loss             = -policy_objective + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
        self.optim.step()

        _append_update_metrics(
            metrics,
            logits=logits,
            log_probs=log_probs,
            values=values,
            returns=returns,
            raw_advantages=raw_adv,
            policy_objective=policy_objective,
            value_loss=value_loss,
            entropy=entropy,
            loss=loss,
        )

        return metrics

    def save(self, path: str):
        torch.save({"network": self.net.state_dict()}, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        total_reward = []
        total_loss   = []
        total_info   = []

        for episode in range(1, n_episodes + 1):
            start  = int(_sample_start_indices(len(data), max_steps).item())
            sim_t0 = float(data[start]["time"])
            state  = env.reset(data[start])
            self.buffer.clear()

            if store_results:
                episode_reward = []
                episode_info   = []
                episode_loss   = []

            step = 1
            while True:
                valid_mask = env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    action = self.act(state.to_tensor(), valid_mask=valid_mask, explore=True, grad_enabled=False)
                else:
                    action = None

                next_state, reward, done, info = env.step(action, data[start + step])

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done,   dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        valid_mask=valid_mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal      = done or (step >= max_steps)

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_state.to_tensor(),
                            max_grad_norm=max_grad_norm,
                        )
                        self.buffer.clear()
                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[start + step]['time']) - sim_t0)}] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break
                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[start + step]['time']) - sim_t0)}] - warm up in progress...", end="\r")

                state = next_state
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[start + step]['time']) - sim_t0)}] - reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_vec(
        self,
        vec_env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        N        = vec_env.n_envs
        dtype    = vec_env.dtype
        data_len = len(data)

        total_reward = []
        total_loss   = []
        total_info   = []

        for episode in range(1, n_episodes + 1):
            starts = _sample_start_indices(data_len, max_steps, batch_size=N).tolist()
            sim_t0 = float(data[starts[0]]["time"])
            obs, _ = vec_env.reset([data[starts[i]] for i in range(N)])
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(N)]
                ep_loss   = []
                ep_info   = [[] for _ in range(N)]

            step = 1
            while step <= max_steps:
                valid_mask = vec_env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    actions = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                else:
                    actions = None

                if store_results:
                    next_obs, _, rewards, dones, infos = vec_env.step_with_info(
                        actions, [data[starts[i] + step] for i in range(N)]
                    )
                    for i in range(N):
                        ep_reward[i].append(float(rewards[i]))
                        ep_info[i].append(infos[i])
                else:
                    next_obs, _, rewards, dones = vec_env.step(
                        actions, [data[starts[i] + step] for i in range(N)]
                    )

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
                        valid_mask=valid_mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_obs,
                            max_grad_norm=max_grad_norm,
                        )
                        self.buffer.clear()
                        if store_results:
                            ep_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[starts[0] + step]['time']) - sim_t0)}] - avg. reward: {rewards.mean():.5f}"
                        if store_results:
                            msg += f" - avg. portfolio: {sum(infos[i]['V'] for i in range(N)) / N:.2f}"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[starts[0] + step]['time']) - sim_t0)}] - warm up in progress...", end="\r")

                obs  = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)
                total_info.append(ep_info)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[starts[0] + step]['time']) - sim_t0)}] - avg. reward: {sum(all_ep_rewards) / max(1, len(all_ep_rewards)):.5f}"
            if store_results:
                msg += f" - avg. portfolio: {sum(infos[i]['V'] for i in range(N)) / N:.2f}"
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_bat(
        self,
        bat_env,
        close, high, low, volume, times, open_,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        B     = bat_env.B
        dtype = bat_env.dtype
        T     = close.shape[0]

        total_loss   = []
        total_reward = []

        for episode in range(1, n_episodes + 1):
            starts = _sample_start_indices(T, max_steps, batch_size=B)
            sim_t0 = float(times[starts[0]])
            obs    = bat_env.reset(
                close[starts], high[starts], low[starts], volume[starts], times[starts],
                open_=open_[starts],
            )
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(B)]
                ep_loss   = []

            step = 1
            while step <= max_steps:
                si = starts + step
                valid_mask = bat_env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    actions      = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions      = None
                    step_actions = torch.zeros(B, dtype=torch.long)

                next_obs, rewards, dones = bat_env.step(
                    step_actions, close[si], high[si], low[si], volume[si], times[si], open_[si],
                )

                if store_results and actions is not None:
                    for i in range(B):
                        ep_reward[i].append(float(rewards[i]))

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
                        valid_mask=valid_mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_obs,
                            max_grad_norm=max_grad_norm,
                        )
                        self.buffer.clear()
                        if store_results:
                            ep_loss.append(loss_dict)

                        msg = (
                            f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}]"
                            f" - avg. reward: {rewards.mean():.5f}"
                            f" - avg. portfolio: {bat_env.V.mean():.2f}"
                        )
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}] - warm up in progress...", end="\r")

                obs  = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}] - avg. reward: {sum(all_ep_rewards) / max(1, len(all_ep_rewards)):.5f} - avg. portfolio: {bat_env.V.mean():.2f}"
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward


# ---------------------------------------------------------------------------
# RecurrentAACAgent — recurrent
# ---------------------------------------------------------------------------

class RecurrentAACAgent:
    """
    Recurrent Advantage Actor-Critic.

    Identical to AACAgent but uses a RecurrentActionValueNetwork so the policy
    and value estimate are conditioned on the full history via a learned hidden
    state.  The update replays each buffered segment from a saved hidden state
    h0, exactly as in RecurrentVAQAgent.

    `burn_in_updates` skips the first N gradient updates after warm-up so the
    recurrent state can move away from its reset state before training begins.
    """

    def __init__(
        self,
        network: dict,
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        advantage_type: str = "td0",
        gae_lambda: float = 0.95,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if advantage_type not in ("td0", "gae", "mc"):
            raise ValueError(f"advantage_type must be 'td0', 'gae', or 'mc', got '{advantage_type}'.")
        self.gamma                = gamma
        self.vf_coef              = vf_coef
        self.ent_coef             = ent_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type       = advantage_type
        self.gae_lambda           = gae_lambda
        self.dtype                = dtype
        self.device               = torch.device(device)

        self.net = build_network(network).to(device=self.device, dtype=self.dtype)
        self.state_dim  = getattr(self.net, "state_dim", None)
        self.action_dim = self.net.action_dim

        self.buffer = RolloutBuffer()
        self.net.reset(1)

    def infer_logits(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits

    def infer_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        return self.net.forward_seq(state_seq)

    def act(
        self,
        state: torch.Tensor,
        valid_mask: torch.BoolTensor | None = None,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        with torch.set_grad_enabled(grad_enabled):
            state  = state.to(dtype=self.dtype, device=self.device)
            logits = self.infer_logits(state)
            if valid_mask is not None:
                logits = apply_action_mask(logits, valid_mask.to(self.device))
            if explore:
                action = distributions.Categorical(logits=logits).sample().squeeze()
            else:
                action = logits.argmax(dim=-1).squeeze()
        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, done, valid_mask=None):
        self.buffer.store(state, action, reward, done, valid_mask=valid_mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def compute_advantages(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor | None]:
        states, actions, rewards, dones, valid_masks = self.buffer.to_tensors(self.device, self.dtype)
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones   = dones.squeeze(-1)   if dones.dim()   > 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, values_all = self.net.forward_seq(all_states)  # (T+1, [B,])

            v_t   = values_all[:-1]
            v_tp1 = values_all[1:]

            if self.advantage_type == "gae":
                raw_adv, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            elif self.advantage_type == "mc":
                raw_adv, returns = _compute_mc(rewards, dones, v_t, v_tp1[-1], self.gamma)
            else:  # td0
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t

            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, raw_adv, returns, states, actions, valid_masks

    def update(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, valid_masks = self.compute_advantages(last_next_state, h0=h0)

        logits, values = self.infer_from_seq(states, h0=h0)
        if valid_masks is not None:
            logits = apply_action_mask(logits, valid_masks)
        dist      = distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())

        raw_entropy = dist.entropy()  # (T, [B,])
        if valid_masks is not None:
            k = valid_masks.sum(dim=-1).float()  # (T, [B,])
            log_k = torch.log(k.clamp(min=2))
            entropy = (raw_entropy / log_k).mean()
        else:
            entropy = raw_entropy.mean()

        adv              = advantages.detach()
        policy_objective = (log_probs * adv).mean()
        value_loss       = 0.5 * (returns.detach() - values).pow(2).mean()
        loss             = -policy_objective + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
        self.optim.step()

        _append_update_metrics(
            metrics,
            logits=logits,
            log_probs=log_probs,
            values=values,
            returns=returns,
            raw_advantages=raw_adv,
            policy_objective=policy_objective,
            value_loss=value_loss,
            entropy=entropy,
            loss=loss,
        )

        return metrics

    def save(self, path: str):
        torch.save({"network": self.net.state_dict()}, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        burn_in_updates=0,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        total_reward = []
        total_loss   = []
        total_info   = []

        for episode in range(1, n_episodes + 1):
            start   = int(_sample_start_indices(len(data), max_steps).item())
            sim_t0  = float(data[start]["time"])
            state   = env.reset(data[start])
            burn_in = burn_in_updates
            self.net.reset()
            self.buffer.clear()
            h0 = self.net.get_states(clone=True, detach=True)

            if store_results:
                episode_reward = []
                episode_info   = []
                episode_loss   = []

            step = 1
            while True:
                valid_mask = env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    action = self.act(state.to_tensor(), valid_mask=valid_mask, explore=True, grad_enabled=False)
                else:
                    action = None

                next_state, reward, done, info = env.step(action, data[start + step])

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done,   dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        valid_mask=valid_mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal      = done or (step >= max_steps)

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        h_t = self.net.get_states(clone=True, detach=True)

                        if burn_in > 0:
                            burn_in -= 1
                        else:
                            loss_dict = self.update(
                                last_next_state=next_state.to_tensor(),
                                h0=h0,
                                max_grad_norm=max_grad_norm,
                            )
                            if store_results:
                                episode_loss.append(loss_dict)

                            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[start + step]['time']) - sim_t0)}] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        self.net.set_states(h0 := h_t, clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break
                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[start + step]['time']) - sim_t0)}] - warm up in progress...", end="\r")

                state = next_state
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[start + step]['time']) - sim_t0)}] - reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_vec(
        self,
        vec_env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        burn_in_updates=0,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        N        = vec_env.n_envs
        dtype    = vec_env.dtype
        data_len = len(data)

        total_reward = []
        total_loss   = []
        total_info   = []

        for episode in range(1, n_episodes + 1):
            self.net.reset(N)
            h0      = self.net.get_states(clone=True, detach=True)
            burn_in = burn_in_updates

            starts = _sample_start_indices(data_len, max_steps, batch_size=N).tolist()
            sim_t0 = float(data[starts[0]]["time"])
            obs, _ = vec_env.reset([data[starts[i]] for i in range(N)])
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(N)]
                ep_loss   = []
                ep_info   = [[] for _ in range(N)]

            step = 1
            while step <= max_steps:
                valid_mask = vec_env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    actions = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                else:
                    actions = None

                if store_results:
                    next_obs, _, rewards, dones, infos = vec_env.step_with_info(
                        actions, [data[starts[i] + step] for i in range(N)]
                    )
                    for i in range(N):
                        ep_reward[i].append(float(rewards[i]))
                        ep_info[i].append(infos[i])
                else:
                    next_obs, _, rewards, dones = vec_env.step(
                        actions, [data[starts[i] + step] for i in range(N)]
                    )

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
                        valid_mask=valid_mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        h_t = self.net.get_states(clone=True, detach=True)

                        if burn_in > 0:
                            burn_in -= 1
                        else:
                            loss_dict = self.update(
                                last_next_state=next_obs,
                                h0=h0,
                                max_grad_norm=max_grad_norm,
                            )
                            if store_results:
                                ep_loss.append(loss_dict)

                            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[starts[0] + step]['time']) - sim_t0)}] - avg. reward: {rewards.mean():.5f}"
                            if store_results:
                                msg += f" - avg. portfolio: {sum(infos[i]['V'] for i in range(N)) / N:.2f}"
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        self.net.set_states(h0 := h_t, clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[starts[0] + step]['time']) - sim_t0)}] - warm up in progress...", end="\r")

                obs  = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)
                total_info.append(ep_info)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(data[starts[0] + step]['time']) - sim_t0)}] - avg. reward: {sum(all_ep_rewards) / max(1, len(all_ep_rewards)):.5f}"
            if store_results:
                msg += f" - avg. portfolio: {sum(infos[i]['V'] for i in range(N)) / N:.2f}"
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_bat(
        self,
        bat_env,
        close, high, low, volume, times, open_,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        burn_in_updates=1,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """
        Train on a BatchedMultiCurrencyEnv using pre-stacked tensor data.

        close, high, low, volume, open_ : (T, N) float tensors
        times                           : (T,)  float64 tensor of Unix timestamps

        Each episode picks B independent random start offsets. All B
        environments step together — market data indexed as data[starts + step].
        The buffer stores (B, ...) tensors per step; a single update() call
        averages gradients over both the time and batch dimensions.

        `burn_in_updates` skips the first N gradient updates after warm-up so
        the shared recurrent state can stabilise before training begins.

        Returns (total_loss, total_reward).
        """
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        B     = bat_env.B
        dtype = bat_env.dtype
        T     = close.shape[0]

        total_loss   = []
        total_reward = []

        for episode in range(1, n_episodes + 1):
            self.net.reset(B)
            h0      = self.net.get_states(clone=True, detach=True)
            burn_in = burn_in_updates

            starts = _sample_start_indices(T, max_steps, batch_size=B)
            sim_t0 = float(times[starts[0]])
            obs    = bat_env.reset(
                close[starts], high[starts], low[starts], volume[starts], times[starts],
                open_=open_[starts],
            )
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(B)]
                ep_loss   = []

            step = 1
            while step <= max_steps:
                si = starts + step
                valid_mask = bat_env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    actions      = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions      = None
                    step_actions = torch.zeros(B, dtype=torch.long)

                next_obs, rewards, dones = bat_env.step(
                    step_actions, close[si], high[si], low[si], volume[si], times[si], open_[si],
                )

                if store_results and actions is not None:
                    for i in range(B):
                        ep_reward[i].append(float(rewards[i]))

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
                        valid_mask=valid_mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        h_t = self.net.get_states(clone=True, detach=True)

                        if burn_in > 0:
                            burn_in -= 1
                        else:
                            loss_dict = self.update(
                                last_next_state=next_obs,
                                h0=h0,
                                max_grad_norm=max_grad_norm,
                            )
                            if store_results:
                                ep_loss.append(loss_dict)

                            msg = (
                                f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}]"
                                f" - avg. reward: {rewards.mean():.5f}"
                                f" - avg. portfolio: {bat_env.V.mean():.2f}"
                            )
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        self.net.set_states(h0 := h_t, clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}] - warm up in progress...", end="\r")

                obs  = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = (
                f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}]"
                f" - avg. reward: {sum(all_ep_rewards) / max(1, len(all_ep_rewards)):.5f}"
                f" - avg. portfolio: {bat_env.V.mean():.2f}"
            )
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward
