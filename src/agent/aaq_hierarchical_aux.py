import torch

from src.agent.utils import (
    _append_update_metrics,
    _aux_loss,
    _compute_gae,
    _compute_mc,
    _empty_update_metrics,
    _init_update_metrics,
    _unpack_policy_value_aux,
    _check_same_shape,
    _hierarchical_policy_value_from_q,
    _hierarchical_q_taken,
    _next_masks,
)
from src.agent.aac_hierarchical_aux import HierarchicalAuxAACAgent as _HierarchicalAuxAACAgent


class HierarchicalAuxAAQAgent(_HierarchicalAuxAACAgent):
    """
    Auxiliary hierarchical actor-Q critic.

    The network must return ``(logits, q_values, aux)``. ``q_values`` uses the
    same factored action layout as the logits, and the policy-gradient baseline
    is the policy-weighted Q expectation over valid actions.

    ``advantage_type`` controls the Q-target construction. The policy
    advantage is always ``Q(s, a) - E_pi[Q(s, a)]``.
    """

    def compute_advantages(self, last_next_state: torch.Tensor, h0=None, last_next_mask: dict | None = None):
        states, actions, rewards, dones, masks, aux_targets = self.buffer.to_tensors(self.device, self.dtype)
        rewards = rewards.squeeze(-1) if rewards.dim() > actions.dim() - 1 else rewards
        dones   = dones.squeeze(-1)   if dones.dim()   > actions.dim() - 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            logits_all, q_all, _ = _unpack_policy_value_aux(self.net.forward_seq(all_states))

            v_t = _hierarchical_policy_value_from_q(
                logits_all[:-1], q_all[:-1], masks, self.n_assets, self.K, self.primary_dim
            )
            v_tp1 = _hierarchical_policy_value_from_q(
                logits_all[1:], q_all[1:], _next_masks(masks, last_next_mask), self.n_assets, self.K, self.primary_dim
            )
            q_sa = _hierarchical_q_taken(q_all[:-1], actions, self.n_assets, self.K, self.primary_dim)
            _check_same_shape("v_t", v_t, "rewards", rewards)
            _check_same_shape("v_tp1", v_tp1, "rewards", rewards)
            _check_same_shape("q_sa", q_sa, "rewards", rewards)

            if self.advantage_type == "gae":
                _, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            elif self.advantage_type == "mc":
                _, returns = _compute_mc(rewards, dones, v_t, v_tp1[-1], self.gamma)
            else:
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
            _check_same_shape("returns", returns, "rewards", rewards)

            raw_adv = q_sa - v_t
            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)

        return advantages, raw_adv, returns, states, actions, rewards, masks, aux_targets

    def update(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
        last_next_mask: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, rewards, masks, aux_targets = self.compute_advantages(
            last_next_state, h0=h0, last_next_mask=last_next_mask
        )

        logits, q_values, aux = _unpack_policy_value_aux(self.infer_from_seq(states, h0=h0))
        log_probs, entropy, primary_logits_m = self._policy_terms(logits, actions, masks)
        values = _hierarchical_policy_value_from_q(logits, q_values, masks, self.n_assets, self.K, self.primary_dim)
        q_sa = _hierarchical_q_taken(q_values, actions, self.n_assets, self.K, self.primary_dim)
        _check_same_shape("values", values, "rewards", rewards)
        _check_same_shape("q_sa", q_sa, "rewards", rewards)
        _check_same_shape("log_probs", log_probs, "rewards", rewards)
        _check_same_shape("returns", returns, "rewards", rewards)

        adv = advantages.detach()
        policy_objective = (log_probs * adv).mean()
        q_loss = 0.5 * (returns.detach() - q_sa).pow(2).mean()
        aux_loss, aux_losses = _aux_loss(aux, aux_targets, rewards=rewards, coefs=self.aux_loss_coefs)
        loss = -policy_objective + self.vf_coef * q_loss - self.ent_coef * entropy + self.aux_coef * aux_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
        self.optim.step()

        _append_update_metrics(
            metrics,
            logits=primary_logits_m,
            log_probs=log_probs,
            values=values,
            returns=returns,
            raw_advantages=raw_adv,
            policy_objective=policy_objective,
            value_loss=q_loss,
            entropy=entropy,
            loss=loss,
        )
        metrics.setdefault("q_loss", []).append(float(q_loss.detach().cpu().item()))

        for name, value in aux_losses.items():
            metrics.setdefault(name, []).append(float(value.detach().cpu().item()))

        return metrics


