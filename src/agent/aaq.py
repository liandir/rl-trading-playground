"""Aaq utilities for reinforcement-learning agents and training utilities."""
import torch
from torch import distributions

from src.agent.aac import (
    AACAgent as _AACAgent,
)
from src.agent.utils import (
    _append_update_metrics,
    _compute_gae,
    _compute_mc,
    _empty_update_metrics,
    _init_update_metrics,
    _next_valid_masks,
    _policy_value_from_q,
    _q_taken,
    apply_action_mask,
)


class AAQAgent(_AACAgent):
    """Sequence-capable advantage actor-Q critic."""

    def compute_advantages(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor | None]:
        """Compute the advantages.

        Args:
            last_next_state (torch.Tensor): The last next state value.
            h0 (dict | None): The h0 value. Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor | None]: The computed or requested result.
        """
        states, actions, rewards, dones, valid_masks = self.buffer.to_tensors(self.device, self.dtype)
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones   = dones.squeeze(-1)   if dones.dim()   > 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            all_states = torch.cat([states, last_next_state[None]], dim=0)
            logits_all, q_all = self.net.forward_seq(all_states)

            logits_t = logits_all[:-1]
            q_t = q_all[:-1]
            logits_tp1 = logits_all[1:]
            q_tp1 = q_all[1:]

            v_t = _policy_value_from_q(logits_t, q_t, valid_masks)
            v_tp1 = _policy_value_from_q(logits_tp1, q_tp1, _next_valid_masks(valid_masks, self.action_dim))
            q_sa = _q_taken(q_t, actions)

            if self.advantage_type == "gae":
                _, returns = _compute_gae(rewards, dones, v_t, v_tp1, self.gamma, self.gae_lambda)
            elif self.advantage_type == "mc":
                _, returns = _compute_mc(rewards, dones, v_t, v_tp1[-1], self.gamma)
            else:
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1

            raw_adv = q_sa - v_t
            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)

        return advantages, raw_adv, returns, states, actions, valid_masks

    def update(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        """Apply one update step.

        Args:
            last_next_state (torch.Tensor): The last next state value.
            h0 (dict | None): The h0 value. Defaults to ``None``.
            max_grad_norm (float | None): The max grad norm value. Defaults to ``None``.

        Returns:
            dict[str, list[float]]: The computed or requested result.
        """
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, valid_masks = self.compute_advantages(last_next_state, h0=h0)

        logits, q_values = self.infer_from_seq(states, h0=h0)
        if valid_masks is not None:
            logits = apply_action_mask(logits, valid_masks)
        dist = distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())

        raw_entropy = dist.entropy()
        if valid_masks is not None:
            k = valid_masks.sum(dim=-1).float()
            log_k = torch.log(k.clamp(min=2))
            entropy = (raw_entropy / log_k).mean()
        else:
            entropy = raw_entropy.mean()

        values = _policy_value_from_q(logits, q_values)
        q_sa = _q_taken(q_values, actions)

        adv = advantages.detach()
        policy_objective = (log_probs * adv).mean()
        value_loss = 0.5 * (returns.detach() - q_sa).pow(2).mean()
        loss = -policy_objective + self.vf_coef * value_loss - self.ent_coef * entropy

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
