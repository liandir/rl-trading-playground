import torch
from torch import distributions

from src.agent.discrete.aac import (
    apply_action_mask,
    _compute_gae,
    _explained_variance,
    _fmt_sim_elapsed,
    _sample_start_indices,
)
from src.agent.hybrid.network import (
    HybridActionValueNetwork,
    RecurrentHybridActionValueNetwork,
)


# ---------------------------------------------------------------------------
# Hybrid policy — discrete (masked categorical) + continuous (Beta) heads
# ---------------------------------------------------------------------------
#
# Action space:
#   a_d ∈ {hold} ∪ {buy_i}_{i=1..N} ∪ {sell_i}_{i=1..N}         |A_d| = 2N + 1
#   a_c ∈ (0, 1)                                                  (amount fraction)
#
# Joint policy factorises as
#   π(a_d, a_c | s) = π_d(a_d | s) · π_c(a_c | s) · 1[a_d ≠ hold]
#
# Both heads share a trunk φ(s) and are conditionally independent given s.
# The continuous action is irrelevant when a_d = hold — its log-prob is
# masked out and does not enter the objective.
#
# Hold is assumed to be index 0 of the discrete action space.
# ---------------------------------------------------------------------------


HOLD_INDEX = 0
AC_EPS     = 1e-4


# ---------------------------------------------------------------------------
# Hybrid policy helpers (sampling, log-prob, entropy)
# ---------------------------------------------------------------------------

def _build_hybrid_dist(
    logits: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    valid_mask: torch.BoolTensor | None = None,
) -> tuple[distributions.Categorical, distributions.Beta]:
    if valid_mask is not None:
        logits = apply_action_mask(logits, valid_mask)
    cat  = distributions.Categorical(logits=logits)
    beta = distributions.Beta(alpha, beta)
    return cat, beta


def _hybrid_log_prob(
    cat: distributions.Categorical,
    beta_dist: distributions.Beta,
    action_d: torch.Tensor,
    action_c: torch.Tensor,
) -> torch.Tensor:
    """log π(a_d, a_c | s) = log π_d(a_d|s) + 1[a_d != hold] · log π_c(a_c|s)."""
    logp_d     = cat.log_prob(action_d.long())
    ac_clamped = action_c.clamp(AC_EPS, 1.0 - AC_EPS)
    logp_c     = beta_dist.log_prob(ac_clamped)
    nonhold    = (action_d != HOLD_INDEX).to(dtype=logp_d.dtype)
    return logp_d + nonhold * logp_c


def _hybrid_entropy_components(
    cat: distributions.Categorical,
    beta_dist: distributions.Beta,
    valid_mask: torch.BoolTensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    H(π) = ~H_d(π_d) + P[a_d != hold] · H(π_c)

    ~H_d is the raw categorical entropy normalised by log(K(s)), where K(s) is
    the number of valid actions (∥m(s)∥₀). K(s) is clamped at 2 so log(K) ≥
    log(2); when K(s) = 1 the raw entropy is already 0 and the ratio collapses.

    The continuous term uses Beta differential entropy, whose maximum on [0, 1]
    is 0 and whose value is usually negative. Detaching P[a_d != hold] prevents
    that negative term from creating a spurious discrete-policy incentive to
    collapse into hold. The continuous head still receives entropy gradients.
    """
    raw_entropy_d = cat.entropy()
    if valid_mask is not None:
        k      = valid_mask.sum(dim=-1).to(dtype=raw_entropy_d.dtype)
        log_k  = torch.log(k.clamp(min=2))
        ent_d  = raw_entropy_d / log_k
    else:
        ent_d  = raw_entropy_d

    probs_d   = cat.probs
    nonhold_p = 1.0 - probs_d[..., HOLD_INDEX]
    ent_c     = beta_dist.entropy()
    ent_c_weighted = nonhold_p.detach() * ent_c
    entropy = (ent_d + ent_c_weighted).mean()
    return entropy, ent_d.mean(), ent_c.mean(), ent_c_weighted.mean()


def _hybrid_entropy(
    cat: distributions.Categorical,
    beta_dist: distributions.Beta,
    valid_mask: torch.BoolTensor | None,
) -> torch.Tensor:
    return _hybrid_entropy_components(cat, beta_dist, valid_mask)[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

UPDATE_METRIC_NAMES = (
    "policy_objective",
    "value_loss",
    "entropy",
    "entropy_discrete",
    "entropy_continuous",
    "entropy_continuous_weighted",
    "total_loss",
    "return_mean",
    "value_mean",
    "adv_abs_mean",
    "adv_pos_frac",
    "chosen_action_prob_mean",
    "policy_confidence_mean",
    "explained_variance",
    "alpha_mean",
    "beta_mean",
    "nonhold_prob_mean",
)


def _empty_update_metrics() -> dict[str, list[float]]:
    return {name: [0.0] for name in UPDATE_METRIC_NAMES}


def _init_update_metrics() -> dict[str, list[float]]:
    return {name: [] for name in UPDATE_METRIC_NAMES}


def _append_update_metrics(
    metric_store: dict[str, list[float]],
    *,
    logits: torch.Tensor,
    log_probs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    raw_advantages: torch.Tensor,
    policy_objective: torch.Tensor,
    value_loss: torch.Tensor,
    entropy: torch.Tensor,
    entropy_discrete: torch.Tensor,
    entropy_continuous: torch.Tensor,
    entropy_continuous_weighted: torch.Tensor,
    loss: torch.Tensor,
):
    probs                   = torch.softmax(logits.detach(), dim=-1)
    values_detached         = values.detach()
    returns_detached        = returns.detach()
    raw_advantages_detached = raw_advantages.detach()
    residuals               = returns_detached - values_detached

    metrics = {
        "policy_objective":        policy_objective.detach(),
        "value_loss":              value_loss.detach(),
        "entropy":                 entropy.detach(),
        "entropy_discrete":        entropy_discrete.detach(),
        "entropy_continuous":      entropy_continuous.detach(),
        "entropy_continuous_weighted": entropy_continuous_weighted.detach(),
        "total_loss":              loss.detach(),
        "return_mean":             returns_detached.mean(),
        "value_mean":              values_detached.mean(),
        "adv_abs_mean":            raw_advantages_detached.abs().mean(),
        "adv_pos_frac":            (raw_advantages_detached > 0).to(dtype=returns_detached.dtype).mean(),
        "chosen_action_prob_mean": log_probs.detach().exp().mean(),
        "policy_confidence_mean":  probs.max(dim=-1).values.mean(),
        "explained_variance":      _explained_variance(returns_detached, residuals),
        "alpha_mean":              alpha.detach().mean(),
        "beta_mean":               beta.detach().mean(),
        "nonhold_prob_mean":       (1.0 - probs[..., HOLD_INDEX]).mean(),
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
        self.states       = []
        self.actions_d    = []
        self.actions_c    = []
        self.rewards      = []
        self.dones        = []
        self.valid_masks  = []

    def store(self, state, action_d, action_c, reward, done, valid_mask=None):
        self.states     .append(state)
        self.actions_d  .append(action_d)
        self.actions_c  .append(action_c)
        self.rewards    .append(reward)
        self.dones      .append(done)
        self.valid_masks.append(valid_mask)

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states    = torch.stack(self.states   ).to(device=device, dtype=dtype)
        actions_d = torch.stack(self.actions_d).to(device=device)
        actions_c = torch.stack(self.actions_c).to(device=device, dtype=dtype)
        rewards   = torch.stack(self.rewards  ).to(device=device, dtype=dtype)
        dones     = torch.stack(self.dones    ).to(device=device, dtype=dtype)
        valid_masks = None
        if self.valid_masks and self.valid_masks[0] is not None:
            valid_masks = torch.stack(self.valid_masks).to(device=device)
        return states, actions_d, actions_c, rewards, dones, valid_masks


# ---------------------------------------------------------------------------
# HybridAACAgent — feedforward
# ---------------------------------------------------------------------------

class AACAgent:
    """
    Hybrid Advantage Actor-Critic (feedforward).

    Joint policy factorises as
        π(a_d, a_c | s) = π_d(a_d | s) · π_c(a_c | s) · 1[a_d ≠ hold]

    Discrete head: masked categorical over {hold, buy_1..N, sell_1..N}.
    Continuous head: Beta(α(s), β(s)) over the amount fraction a_c ∈ (0, 1).
    Value head: scalar baseline V(s), action-independent.

    Loss:
        L = -E[log π(a_d, a_c | s) · A]  +  vf_coef · E[(G - V)²]  -  ent_coef · H(π)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        advantage_type: str = "td0",
        gae_lambda: float = 0.95,
        hidden_dims: list[int] = None,
        hidden_dims_actor: list[int] = None,
        hidden_dims_value: list[int] = None,
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if advantage_type not in ("td0", "gae"):
            raise ValueError(f"advantage_type must be 'td0' or 'gae', got '{advantage_type}'.")
        self.gamma                = gamma
        self.vf_coef              = vf_coef
        self.ent_coef             = ent_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type       = advantage_type
        self.gae_lambda           = gae_lambda
        self.dtype                = dtype
        self.device               = torch.device(device)
        self.state_dim            = state_dim
        self.action_dim           = action_dim

        self.net = HybridActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        self.buffer = RolloutBuffer()

    def infer_heads(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, alpha, beta, _ = self.net(state)
        return logits, alpha, beta

    def act(
        self,
        state: torch.Tensor,
        valid_mask: torch.BoolTensor | None = None,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits, alpha, beta = self.infer_heads(state)
            if valid_mask is not None:
                logits = apply_action_mask(logits, valid_mask.to(self.device))
            cat  = distributions.Categorical(logits=logits)
            dist = distributions.Beta(alpha, beta)
            if explore:
                action_d = cat.sample().squeeze()
                action_c = dist.sample().clamp(AC_EPS, 1.0 - AC_EPS).squeeze()
            else:
                action_d = logits.argmax(dim=-1).squeeze()
                action_c = (alpha / (alpha + beta)).clamp(AC_EPS, 1.0 - AC_EPS).squeeze()
        if grad_enabled:
            return action_d, action_c
        return action_d.cpu(), action_c.cpu()

    def store(self, state, action_d, action_c, reward, done, valid_mask=None):
        self.buffer.store(state, action_d, action_c, reward, done, valid_mask=valid_mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def compute_advantages(
        self,
        last_next_state: torch.Tensor,
    ):
        states, actions_d, actions_c, rewards, dones, valid_masks = self.buffer.to_tensors(self.device, self.dtype)
        actions_d = actions_d.squeeze(-1) if actions_d.dim() > 1 else actions_d
        actions_c = actions_c.squeeze(-1) if actions_c.dim() > 1 else actions_c
        rewards   = rewards  .squeeze(-1) if rewards  .dim() > 1 else rewards
        dones     = dones    .squeeze(-1) if dones    .dim() > 1 else dones

        with torch.no_grad():
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, _, _, values_all = self.net(all_states)       # (T+1, [B,])

            v_t   = values_all[:-1]
            v_tp1 = values_all[1:]

            if self.advantage_type == "gae":
                raw_adv, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            else:  # td0
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t

            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, raw_adv, returns, states, actions_d, actions_c, valid_masks

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
        advantages, raw_adv, returns, states, actions_d, actions_c, valid_masks = self.compute_advantages(last_next_state)

        logits, alpha, beta, values = self.net(states)
        cat, beta_dist = _build_hybrid_dist(logits, alpha, beta, valid_mask=valid_masks)

        log_probs = _hybrid_log_prob(cat, beta_dist, actions_d, actions_c)
        entropy, entropy_d, entropy_c, entropy_c_weighted = _hybrid_entropy_components(
            cat, beta_dist, valid_masks
        )

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
            alpha=alpha,
            beta=beta,
            values=values,
            returns=returns,
            raw_advantages=raw_adv,
            policy_objective=policy_objective,
            value_loss=value_loss,
            entropy=entropy,
            entropy_discrete=entropy_d,
            entropy_continuous=entropy_c,
            entropy_continuous_weighted=entropy_c_weighted,
            loss=loss,
        )

        return metrics

    def save(self, path: str):
        torch.save({"network": self.net.state_dict()}, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)

    # -------------------------------------------------------------------------
    # Training loops — env.step(...) receives the tuple (action_d, action_c)
    # -------------------------------------------------------------------------

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
                    action_d, action_c = self.act(state.to_tensor(), valid_mask=valid_mask, explore=True, grad_enabled=False)
                    action = (action_d, action_c)
                else:
                    action = None

                next_state, reward, done, info = env.step(action, data[start + step])

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    env_dtype = env.dtype if hasattr(env, "dtype") else torch.float32
                    self.store(
                        state.to_tensor().detach(),
                        action_d.detach(),
                        action_c.detach().to(dtype=env_dtype),
                        torch.tensor(reward, dtype=env_dtype),
                        torch.tensor(done,   dtype=env_dtype),
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
                    actions_d, actions_c = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    actions = (actions_d, actions_c)
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
                        actions_d.detach(),
                        actions_c.detach().to(dtype=dtype),
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
                close[starts], high[starts], low[starts], volume[starts], times[starts], open_=open_[starts],
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
                    actions_d, actions_c = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    step_actions = (actions_d, actions_c)
                else:
                    actions_d    = None
                    actions_c    = None
                    step_actions = (torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=dtype))

                next_obs, rewards, dones = bat_env.step(
                    step_actions, close[si], high[si], low[si], volume[si], times[si], open_[si],
                )

                if store_results and actions_d is not None:
                    for i in range(B):
                        ep_reward[i].append(float(rewards[i]))

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions_d.detach(),
                        actions_c.detach().to(dtype=dtype),
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
# RecurrentHybridAACAgent — recurrent
# ---------------------------------------------------------------------------

class RecurrentAACAgent:
    """
    Recurrent Hybrid Advantage Actor-Critic.

    Identical to AACAgent but uses a RecurrentHybridActionValueNetwork so the
    policy and value estimate are conditioned on the full history via a learned
    hidden state. The update replays each buffered segment from a saved hidden
    state h0, matching the pattern used by the discrete recurrent AAC agent.

    `burn_in_updates` skips the first N gradient updates after warm-up so the
    recurrent state can move away from its reset state before training begins.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        advantage_type: str = "td0",
        gae_lambda: float = 0.95,
        hidden_dims: list[int] = None,
        hidden_dims_actor: list[int] = None,
        hidden_dims_value: list[int] = None,
        activation: callable = torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if advantage_type not in ("td0", "gae"):
            raise ValueError(f"advantage_type must be 'td0' or 'gae', got '{advantage_type}'.")
        self.gamma                = gamma
        self.vf_coef              = vf_coef
        self.ent_coef             = ent_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type       = advantage_type
        self.gae_lambda           = gae_lambda
        self.dtype                = dtype
        self.device               = torch.device(device)
        self.state_dim            = state_dim
        self.action_dim           = action_dim

        self.net = RecurrentHybridActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
        ).to(device=self.device, dtype=self.dtype)

        self.buffer = RolloutBuffer()
        self.net.reset(1)

    def infer_heads(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, alpha, beta, _ = self.net(state)
        return logits, alpha, beta

    def infer_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ):
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        return self.net.forward_seq(state_seq)

    def act(
        self,
        state: torch.Tensor,
        valid_mask: torch.BoolTensor | None = None,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits, alpha, beta = self.infer_heads(state)
            if valid_mask is not None:
                logits = apply_action_mask(logits, valid_mask.to(self.device))
            cat  = distributions.Categorical(logits=logits)
            dist = distributions.Beta(alpha, beta)
            if explore:
                action_d = cat.sample().squeeze()
                action_c = dist.sample().clamp(AC_EPS, 1.0 - AC_EPS).squeeze()
            else:
                action_d = logits.argmax(dim=-1).squeeze()
                action_c = (alpha / (alpha + beta)).clamp(AC_EPS, 1.0 - AC_EPS).squeeze()
        if grad_enabled:
            return action_d, action_c
        return action_d.cpu(), action_c.cpu()

    def store(self, state, action_d, action_c, reward, done, valid_mask=None):
        self.buffer.store(state, action_d, action_c, reward, done, valid_mask=valid_mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def compute_advantages(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ):
        states, actions_d, actions_c, rewards, dones, valid_masks = self.buffer.to_tensors(self.device, self.dtype)
        actions_d = actions_d.squeeze(-1) if actions_d.dim() > 1 else actions_d
        actions_c = actions_c.squeeze(-1) if actions_c.dim() > 1 else actions_c
        rewards   = rewards  .squeeze(-1) if rewards  .dim() > 1 else rewards
        dones     = dones    .squeeze(-1) if dones    .dim() > 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, _, _, values_all = self.net.forward_seq(all_states)

            v_t   = values_all[:-1]
            v_tp1 = values_all[1:]

            if self.advantage_type == "gae":
                raw_adv, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            else:  # td0
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t

            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, raw_adv, returns, states, actions_d, actions_c, valid_masks

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
        advantages, raw_adv, returns, states, actions_d, actions_c, valid_masks = self.compute_advantages(last_next_state, h0=h0)

        logits, alpha, beta, values = self.infer_from_seq(states, h0=h0)
        cat, beta_dist = _build_hybrid_dist(logits, alpha, beta, valid_mask=valid_masks)

        log_probs = _hybrid_log_prob(cat, beta_dist, actions_d, actions_c)
        entropy, entropy_d, entropy_c, entropy_c_weighted = _hybrid_entropy_components(
            cat, beta_dist, valid_masks
        )

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
            alpha=alpha,
            beta=beta,
            values=values,
            returns=returns,
            raw_advantages=raw_adv,
            policy_objective=policy_objective,
            value_loss=value_loss,
            entropy=entropy,
            entropy_discrete=entropy_d,
            entropy_continuous=entropy_c,
            entropy_continuous_weighted=entropy_c_weighted,
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
                    action_d, action_c = self.act(state.to_tensor(), valid_mask=valid_mask, explore=True, grad_enabled=False)
                    action = (action_d, action_c)
                else:
                    action = None

                next_state, reward, done, info = env.step(action, data[start + step])

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    env_dtype = env.dtype if hasattr(env, "dtype") else torch.float32
                    self.store(
                        state.to_tensor().detach(),
                        action_d.detach(),
                        action_c.detach().to(dtype=env_dtype),
                        torch.tensor(reward, dtype=env_dtype),
                        torch.tensor(done,   dtype=env_dtype),
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
                    actions_d, actions_c = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    actions = (actions_d, actions_c)
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
                        actions_d.detach(),
                        actions_c.detach().to(dtype=dtype),
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
                close[starts], high[starts], low[starts], volume[starts], times[starts], open_=open_[starts],
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
                    actions_d, actions_c = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    step_actions = (actions_d, actions_c)
                else:
                    actions_d    = None
                    actions_c    = None
                    step_actions = (torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=dtype))

                next_obs, rewards, dones = bat_env.step(
                    step_actions, close[si], high[si], low[si], volume[si], times[si], open_[si],
                )

                if store_results and actions_d is not None:
                    for i in range(B):
                        ep_reward[i].append(float(rewards[i]))

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions_d.detach(),
                        actions_c.detach().to(dtype=dtype),
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
