import torch
from torch import distributions

from src.agent.network import build_network
from src.agent.utils import (
    apply_action_mask,
    _compute_gae,
    _explained_variance,
    _fmt_sim_elapsed,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

UPDATE_METRIC_NAMES = (
    "actor_loss",
    "critic_loss",
    "entropy",
    "total_loss",
    "ratio_mean",
    "ratio_max",
    "clip_frac",
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


def _append_update_metrics(
    metric_store: dict[str, list[float]],
    *,
    logits: torch.Tensor,
    new_log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    raw_advantages: torch.Tensor,
    ratios: torch.Tensor,
    eps_clip: float,
    actor_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    entropy: torch.Tensor,
    loss: torch.Tensor,
):
    probs          = torch.softmax(logits.detach(), dim=-1)
    ratios_d       = ratios.detach()
    values_d       = values.detach()
    returns_d      = returns.detach()
    raw_adv_d      = raw_advantages.detach()
    residuals      = returns_d - values_d

    metrics = {
        "actor_loss":              actor_loss.detach(),
        "critic_loss":             critic_loss.detach(),
        "entropy":                 entropy.detach(),
        "total_loss":              loss.detach(),
        "ratio_mean":              ratios_d.mean(),
        "ratio_max":               ratios_d.max(),
        "clip_frac":               ((ratios_d - 1.0).abs() > eps_clip).to(dtype=returns_d.dtype).mean(),
        "adv_abs_mean":            raw_adv_d.abs().mean(),
        "adv_pos_frac":            (raw_adv_d > 0).to(dtype=returns_d.dtype).mean(),
        "chosen_action_prob_mean": new_log_probs.detach().exp().mean(),
        "policy_confidence_mean":  probs.max(dim=-1).values.mean(),
        "explained_variance":      _explained_variance(returns_d, residuals),
    }
    for name, value in metrics.items():
        metric_store[name].append(float(value.cpu().item()))


# ---------------------------------------------------------------------------
# Rollout buffer (stores old log_probs for the PPO importance ratio)
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states      = []
        self.actions     = []
        self.log_probs   = []
        self.rewards     = []
        self.dones       = []
        self.valid_masks = []

    def store(self, state, action, log_prob, reward, done, valid_mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states    = torch.stack(self.states).to(device=device, dtype=dtype)
        actions   = torch.stack(self.actions).to(device=device, dtype=dtype)
        log_probs = torch.stack(self.log_probs).to(device=device, dtype=dtype)
        rewards   = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones     = torch.stack(self.dones).to(device=device, dtype=dtype)
        valid_masks = None
        if self.valid_masks and self.valid_masks[0] is not None:
            valid_masks = torch.stack(self.valid_masks).to(device=device)
        return states, actions, log_probs, rewards, dones, valid_masks


# ---------------------------------------------------------------------------
# PPOAgent — feedforward
class PPOAgent:
    """
    Proximal Policy Optimization.

    Identical to PPOAgent but the shared trunk is recurrent; each update
    replays the buffered segment from a saved hidden state h0 — matching the
    pattern used by sequence-capable AAC agents.

    `burn_in_updates` skips the first N gradient updates after warm-up so the
    recurrent state can move away from its reset state before training begins.
    """

    def __init__(
        self,
        network: dict | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        gamma: float = 0.999,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        advantage_type: str = "gae",
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
        self.eps_clip             = eps_clip
        self.vf_coef              = vf_coef
        self.ent_coef             = ent_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type       = advantage_type
        self.gae_lambda           = gae_lambda
        self.dtype                = dtype
        self.device               = torch.device(device)
        if network is None:
            if state_dim is None or action_dim is None:
                raise ValueError("Pass either network or both state_dim and action_dim.")
            network = {
                "type": "recurrent_action_value",
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dims": hidden_dims,
                "hidden_dims_actor": hidden_dims_actor,
                "hidden_dims_value": hidden_dims_value,
                "activation": activation,
                "recurrent_type": recurrent_type,
                "recurrent_kwargs": recurrent_kwargs,
            }
        self.net = build_network(network).to(device=self.device, dtype=self.dtype)
        self.state_dim = getattr(self.net, "state_dim", state_dim)
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad_enabled):
            state  = state.to(dtype=self.dtype, device=self.device)
            logits = self.infer_logits(state)
            if valid_mask is not None:
                logits = apply_action_mask(logits, valid_mask.to(self.device))
            dist = distributions.Categorical(logits=logits)
            if explore:
                action = dist.sample().squeeze()
            else:
                action = logits.argmax(dim=-1).squeeze()
            log_prob = dist.log_prob(action)
        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def store(self, state, action, log_prob, reward, done, valid_mask=None):
        self.buffer.store(state, action, log_prob, reward, done, valid_mask=valid_mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def compute_advantages(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor | None]:
        states, actions, old_log_probs, rewards, dones, valid_masks = self.buffer.to_tensors(self.device, self.dtype)
        actions       = actions.squeeze(-1)       if actions.dim()       > 1 else actions
        rewards       = rewards.squeeze(-1)       if rewards.dim()       > 1 else rewards
        dones         = dones.squeeze(-1)         if dones.dim()         > 1 else dones
        old_log_probs = old_log_probs.squeeze(-1) if old_log_probs.dim() > 1 else old_log_probs

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, values_all = self.net.forward_seq(all_states)

            v_t   = values_all[:-1]
            v_tp1 = values_all[1:]

            if self.advantage_type == "gae":
                raw_adv, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            else:  # td0
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t

            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)

        return advantages, raw_adv, returns, states, actions, old_log_probs, valid_masks

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 4,
        h0: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, old_log_probs, valid_masks = self.compute_advantages(last_next_state, h0=h0)

        for _ in range(k_epochs):
            logits, values = self.infer_from_seq(states, h0=h0)
            if valid_masks is not None:
                logits = apply_action_mask(logits, valid_masks)
            dist          = distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions.long())

            raw_entropy = dist.entropy()
            if valid_masks is not None:
                k = valid_masks.sum(dim=-1).float()
                log_k = torch.log(k.clamp(min=2))
                entropy = (raw_entropy / log_k).mean()
            else:
                entropy = raw_entropy.mean()

            adv    = advantages.detach()
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            surr1  = ratios * adv
            surr2  = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
            actor_loss  = -torch.min(surr1, surr2).mean()

            critic_loss = 0.5 * (returns.detach() - values).pow(2).mean()
            loss        = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
            self.optim.step()

            _append_update_metrics(
                metrics,
                logits=logits,
                new_log_probs=new_log_probs,
                values=values,
                returns=returns,
                raw_advantages=raw_adv,
                ratios=ratios,
                eps_clip=self.eps_clip,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
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
        n_updates=4,
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
            start   = torch.randint(len(data) - max_steps, size=[1]).item()
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
                    action, log_prob = self.act(state.to_tensor(), valid_mask=valid_mask, explore=True, grad_enabled=False)
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
                        log_prob.detach(),
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
                                k_epochs=n_updates,
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

    def train_on_historical_bat(
        self,
        bat_env,
        open_, close, high, low, volume, times,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=4,
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
        averages gradients over both the time and batch dimensions, running
        k_epochs of PPO clipped-surrogate updates per call.

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

            starts = torch.randint(T - max_steps, (B,))
            sim_t0 = float(times[starts[0]])
            obs    = bat_env.reset(
                open_[starts], close[starts], high[starts], low[starts], volume[starts], times[starts],
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
                    actions, log_probs = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions      = None
                    step_actions = torch.zeros(B, dtype=torch.long)

                next_obs, rewards, dones = bat_env.step(
                    step_actions, open_[si], close[si], high[si], low[si], volume[si], times[si],
                )

                if store_results and actions is not None:
                    for i in range(B):
                        ep_reward[i].append(float(rewards[i]))

                terminal = bool(dones.any()) or step >= max_steps

                if step > warm_up:
                    self.buffer.store(
                        obs.detach(),
                        actions.detach(),
                        log_probs.detach(),
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
                                k_epochs=n_updates,
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
