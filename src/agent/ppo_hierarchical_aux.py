import torch

from src.agent.aac_hierarchical_aux import AuxiliaryHierarchicalRolloutBuffer
from src.agent.ppo_hierarchical import HierarchicalPPOAgent
from src.agent.ppo import _append_update_metrics, _empty_update_metrics, _init_update_metrics
from src.agent.utils import (
    _aux_loss,
    _build_aux_target,
    _compute_gae,
    _compute_mc,
    _fmt_sim_elapsed,
    _sample_start_indices,
    _unpack_policy_value_aux,
)


class AuxiliaryHierarchicalPPORolloutBuffer(AuxiliaryHierarchicalRolloutBuffer):
    """Auxiliary hierarchical PPO buffer with old log-probs."""

    def clear(self):
        super().clear()
        self.log_probs = []

    def store(self, state, action, log_prob, reward, done, mask=None, aux_target=None):
        super().store(state, action, reward, done, mask=mask, aux_target=aux_target)
        self.log_probs.append(log_prob)

    def to_tensors(self, device, dtype):
        states, actions, rewards, dones, masks, aux_targets = super().to_tensors(device, dtype)
        log_probs = torch.stack(self.log_probs).to(device=device, dtype=dtype)
        return states, actions, log_probs, rewards, dones, masks, aux_targets


class HierarchicalAuxPPOAgent(HierarchicalPPOAgent):
    """Sequence-capable hierarchical PPO agent with auxiliary prediction heads."""

    def __init__(
        self,
        *args,
        aux_coef: float = 0.05,
        aux_loss_coefs: dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_coef = aux_coef
        self.aux_loss_coefs = dict(aux_loss_coefs or {"reward": 1.0, "asset_ret": 0.5, "asset_vol": 0.25})
        self.aux_horizons = tuple(getattr(self.net, "aux_horizons", (1, 5, 20)))
        self.buffer = AuxiliaryHierarchicalPPORolloutBuffer()

    def act(self, state: torch.Tensor, mask: dict | None = None, explore: bool = False, grad_enabled: bool = False):
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits, _, _ = _unpack_policy_value_aux(self.net(state))
            mask_dev = {k: v.to(self.device) for k, v in mask.items()} if mask is not None else None
            action = self._sample_hierarchical(logits, mask_dev, explore)
            log_prob, _, _ = self._policy_terms(logits, action, mask_dev)
        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def infer_from_seq(self, state_seq, h0=None):
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        return self.net.forward_seq(state_seq)

    def store(self, state, action, log_prob, reward, done, mask=None, aux_target=None):
        self.buffer.store(state, action, log_prob, reward, done, mask=mask, aux_target=aux_target)

    def compute_advantages(self, last_next_state: torch.Tensor, h0=None):
        states, actions, old_log_probs, rewards, dones, masks, aux_targets = self.buffer.to_tensors(self.device, self.dtype)
        rewards = rewards.squeeze(-1) if rewards.dim() > actions.dim() - 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > actions.dim() - 1 else dones
        old_log_probs = old_log_probs.squeeze(-1) if old_log_probs.dim() > rewards.dim() else old_log_probs

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            _, values_all, _ = _unpack_policy_value_aux(self.net.forward_seq(all_states))
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
        return advantages, raw_adv, returns, states, actions, old_log_probs, rewards, masks, aux_targets

    def update(self, last_next_state: torch.Tensor, k_epochs: int = 4, h0=None, max_grad_norm: float | None = None):
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, old_log_probs, rewards, masks, aux_targets = self.compute_advantages(
            last_next_state, h0=h0
        )

        for _ in range(k_epochs):
            logits, values, aux = _unpack_policy_value_aux(self.infer_from_seq(states, h0=h0))
            new_log_probs, entropy, primary_logits_m = self._policy_terms(logits, actions, masks)
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            adv = advantages.detach()
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns.detach() - values).pow(2).mean()
            aux_loss, aux_losses = _aux_loss(aux, aux_targets, rewards=rewards, coefs=self.aux_loss_coefs)
            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy + self.aux_coef * aux_loss

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
            for name, value in aux_losses.items():
                metrics.setdefault(name, []).append(float(value.detach().cpu().item()))
        return metrics

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
        close_src = close.to(device=self.device, dtype=self.dtype)
        horizon_pad = max(self.aux_horizons)
        total_loss = []
        total_reward = []

        for episode in range(1, n_episodes + 1):
            self.net.reset(B)
            h0 = self.net.get_states(clone=True, detach=True)
            burn_in = burn_in_updates

            starts = _sample_start_indices(T, max_steps + horizon_pad, batch_size=B)
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
                    aux_target = _build_aux_target(
                        close_src,
                        starts.to(self.device) + step - 1,
                        self.aux_horizons,
                        reward=rewards.to(device=self.device, dtype=self.dtype),
                    )
                    self.buffer.store(
                        obs.detach(),
                        actions.detach(),
                        log_probs.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
                        mask=mask,
                        aux_target={k: v.detach().cpu() for k, v in aux_target.items()},
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
