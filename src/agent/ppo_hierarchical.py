import torch

from src.agent.aac_hierarchical import (
    HierarchicalRolloutBuffer,
    _HierarchicalPolicyMixin,
)
from src.agent.network import build_network
from src.agent.ppo import (
    _append_update_metrics,
    _empty_update_metrics,
    _init_update_metrics,
)
from src.agent.utils import _compute_gae, _compute_mc, _fmt_sim_elapsed, _sample_start_indices


class HierarchicalPPORolloutBuffer(HierarchicalRolloutBuffer):
    """Hierarchical rollout buffer with old log-probs for PPO ratios."""

    def clear(self):
        super().clear()
        self.log_probs = []

    def store(self, state, action, log_prob, reward, done, mask=None):
        super().store(state, action, reward, done, mask=mask)
        self.log_probs.append(log_prob)

    def to_tensors(self, device, dtype):
        states, actions, rewards, dones, masks = super().to_tensors(device, dtype)
        log_probs = torch.stack(self.log_probs).to(device=device, dtype=dtype)
        return states, actions, log_probs, rewards, dones, masks


class HierarchicalPPOAgent(_HierarchicalPolicyMixin):
    """Hierarchical PPO for feedforward or recurrent networks."""

    def __init__(
        self,
        network: dict,
        n_assets: int,
        n_buckets: int,
        gamma: float = 0.999,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        advantage_type: str = "gae",
        gae_lambda: float = 0.95,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if advantage_type not in ("td0", "gae", "mc"):
            raise ValueError(f"advantage_type must be 'td0', 'gae', or 'mc', got '{advantage_type}'.")
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type = advantage_type
        self.gae_lambda = gae_lambda
        self.dtype = dtype
        self.device = torch.device(device)

        self.n_assets = int(n_assets)
        self.K = int(n_buckets)
        self.primary_dim = 1 + 3 * self.n_assets

        self.net = build_network(network).to(device=self.device, dtype=self.dtype)
        expected = self.primary_dim + 2 * self.K
        if int(self.net.action_dim) != expected:
            raise ValueError(f"Network action_dim must equal {expected}, got {self.net.action_dim}.")
        self.state_dim = getattr(self.net, "state_dim", None)
        self.action_dim = self.net.action_dim
        self.buffer = HierarchicalPPORolloutBuffer()
        self.net.reset(1)

    def act(self, state: torch.Tensor, mask: dict | None = None, explore: bool = False, grad_enabled: bool = False):
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits, _ = self.net(state)
            mask_dev = {k: v.to(self.device) for k, v in mask.items()} if mask is not None else None
            action = self._sample_hierarchical(logits, mask_dev, explore)
            log_prob, _, _ = self._policy_terms(logits, action, mask_dev)
        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def store(self, state, action, log_prob, reward, done, mask=None):
        self.buffer.store(state, action, log_prob, reward, done, mask=mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def infer_from_seq(self, state_seq, h0=None):
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        return self.net.forward_seq(state_seq)

    def compute_advantages(self, last_next_state: torch.Tensor, h0=None):
        states, actions, old_log_probs, rewards, dones, masks = self.buffer.to_tensors(self.device, self.dtype)
        rewards = rewards.squeeze(-1) if rewards.dim() > actions.dim() - 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > actions.dim() - 1 else dones
        old_log_probs = old_log_probs.squeeze(-1) if old_log_probs.dim() > rewards.dim() else old_log_probs

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, values_all = self.net.forward_seq(all_states)
            v_t = values_all[:-1]
            v_tp1 = values_all[1:]
            if self.advantage_type == "gae":
                raw_adv, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            elif self.advantage_type == "mc":
                raw_adv, returns = _compute_mc(rewards, dones, v_t, v_tp1[-1], self.gamma)
            else:
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t
            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)
        return advantages, raw_adv, returns, states, actions, old_log_probs, masks

    def update(self, last_next_state: torch.Tensor, k_epochs: int = 4, h0=None, max_grad_norm: float | None = None):
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, old_log_probs, masks = self.compute_advantages(last_next_state, h0=h0)

        for _ in range(k_epochs):
            logits, values = self.infer_from_seq(states, h0=h0)
            new_log_probs, entropy, primary_logits_m = self._policy_terms(logits, actions, masks)
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            adv = advantages.detach()
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns.detach() - values).pow(2).mean()
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
            self.optim.step()

            _append_update_metrics(
                metrics,
                logits=primary_logits_m,
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
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        B = bat_env.B
        dtype = bat_env.dtype
        T = close.shape[0]
        total_loss = []
        total_reward = []

        for episode in range(1, n_episodes + 1):
            self.net.reset(B)
            h0 = self.net.get_states(clone=True, detach=True)
            burn_in = burn_in_updates

            starts = _sample_start_indices(T, max_steps, batch_size=B)
            sim_t0 = float(times[starts[0]])
            obs = bat_env.reset(
                open_[starts], close[starts], high[starts], low[starts], volume[starts], times[starts],
            )
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(B)]
                ep_loss = []

            step = 1
            while step <= max_steps:
                si = starts + step
                mask = bat_env.valid_action_mask() if step > warm_up else None
                if step > warm_up:
                    actions, log_probs = self.act(obs, mask=mask, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions = None
                    log_probs = None
                    step_actions = (torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long))

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
                        mask=mask,
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        h_t = self.net.get_states(clone=True, detach=True)
                        if burn_in > 0:
                            burn_in -= 1
                        else:
                            loss_dict = self.update(
                                next_obs,
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
                    if step % update_interval == 1:
                        print(
                            f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}] - warm up in progress...",
                            end="\r",
                        )

                obs = next_obs
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
                for key in ep_loss[0].keys():
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward
