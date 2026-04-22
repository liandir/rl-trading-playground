import math

import torch
from torch import distributions

from src.agent.discrete.aac import (
    AACAgent,
    RecurrentAACAgent,
    _explained_variance,
    apply_action_mask,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

UPDATE_METRIC_NAMES = (
    "policy_objective",
    "value_loss",
    "entropy",
    "total_loss",
    "kl",
    "ratio_mean",
    "ratio_max",
    "expected_improvement",
    "actual_improvement",
    "line_search_steps",
    "accepted",
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
    total_loss: torch.Tensor,
    kl: torch.Tensor,
    ratios: torch.Tensor,
    expected_improvement: float,
    actual_improvement: float,
    line_search_steps: int,
    accepted: bool,
):
    probs           = torch.softmax(logits.detach(), dim=-1)
    values_d        = values.detach()
    returns_d       = returns.detach()
    raw_adv_d       = raw_advantages.detach()
    ratios_d        = ratios.detach()
    residuals       = returns_d - values_d

    metrics = {
        "policy_objective":        policy_objective.detach(),
        "value_loss":              value_loss.detach(),
        "entropy":                 entropy.detach(),
        "total_loss":              total_loss.detach(),
        "kl":                      kl.detach(),
        "ratio_mean":              ratios_d.mean(),
        "ratio_max":               ratios_d.max(),
        "expected_improvement":    torch.as_tensor(expected_improvement, device=values.device, dtype=values.dtype),
        "actual_improvement":      torch.as_tensor(actual_improvement, device=values.device, dtype=values.dtype),
        "line_search_steps":       torch.as_tensor(float(line_search_steps), device=values.device, dtype=values.dtype),
        "accepted":                torch.as_tensor(1.0 if accepted else 0.0, device=values.device, dtype=values.dtype),
        "return_mean":             returns_d.mean(),
        "value_mean":              values_d.mean(),
        "adv_abs_mean":            raw_adv_d.abs().mean(),
        "adv_pos_frac":            (raw_adv_d > 0).to(dtype=returns_d.dtype).mean(),
        "chosen_action_prob_mean": log_probs.detach().exp().mean(),
        "policy_confidence_mean":  probs.max(dim=-1).values.mean(),
        "explained_variance":      _explained_variance(returns_d, residuals),
    }
    for name, value in metrics.items():
        metric_store[name].append(float(value.cpu().item()))


# ---------------------------------------------------------------------------
# Flat-parameter helpers
# ---------------------------------------------------------------------------

def _flat_params(params: list[torch.nn.Parameter]) -> torch.Tensor:
    if not params:
        return torch.empty(0)
    return torch.cat([p.detach().reshape(-1) for p in params])


def _set_flat_params(params: list[torch.nn.Parameter], flat: torch.Tensor):
    offset = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


def _flat_grad(
    scalar: torch.Tensor,
    params: list[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        scalar,
        params,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    flat = []
    for p, g in zip(params, grads):
        flat.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(flat)


def _conjugate_gradients(
    Avp,
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float,
) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        Avp_p = Avp(p)
        denom = torch.dot(p, Avp_p)
        if torch.abs(denom).item() <= 1e-20:
            break
        alpha = rdotr / denom
        x = x + alpha * p
        r = r - alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr.item() < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr

    return x


def _masked_log_softmax(logits: torch.Tensor, valid_masks: torch.BoolTensor | None) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    if valid_masks is not None:
        log_probs = log_probs.masked_fill(~valid_masks, 0.0)
    return log_probs


def _masked_softmax(logits: torch.Tensor, valid_masks: torch.BoolTensor | None) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    if valid_masks is not None:
        probs = probs.masked_fill(~valid_masks, 0.0)
    return probs


def _categorical_kl(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    valid_masks: torch.BoolTensor | None,
) -> torch.Tensor:
    old_log_probs = _masked_log_softmax(old_logits, valid_masks)
    new_log_probs = _masked_log_softmax(new_logits, valid_masks)
    old_probs = _masked_softmax(old_logits, valid_masks)
    return (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()


def _normalized_entropy(
    dist: distributions.Categorical,
    valid_masks: torch.BoolTensor | None,
) -> torch.Tensor:
    raw_entropy = dist.entropy()
    if valid_masks is None:
        return raw_entropy.mean()

    k = valid_masks.sum(dim=-1).float()
    log_k = torch.log(k.clamp(min=2))
    return (raw_entropy / log_k).mean()


# ---------------------------------------------------------------------------
# TRPO mixin
# ---------------------------------------------------------------------------

class _TRPOMixin:
    def _configure_trpo(
        self,
        *,
        max_kl: float,
        damping: float,
        cg_iters: int,
        cg_residual_tol: float,
        backtrack_iters: int,
        backtrack_coeff: float,
        accept_ratio: float,
    ):
        if max_kl <= 0:
            raise ValueError("max_kl must be positive.")
        if damping < 0:
            raise ValueError("damping must be non-negative.")
        if cg_iters <= 0:
            raise ValueError("cg_iters must be positive.")
        if not (0.0 < backtrack_coeff < 1.0):
            raise ValueError("backtrack_coeff must be in (0, 1).")

        self.max_kl = float(max_kl)
        self.damping = float(damping)
        self.cg_iters = int(cg_iters)
        self.cg_residual_tol = float(cg_residual_tol)
        self.backtrack_iters = int(backtrack_iters)
        self.backtrack_coeff = float(backtrack_coeff)
        self.accept_ratio = float(accept_ratio)

    def _policy_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        params.extend(self.net.layers.parameters())
        params.extend(self.net.actor.parameters())
        return list(params)

    def _value_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.net.v_head.parameters())

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self._value_parameters(), lr=lr)

    def _rollout_forward(
        self,
        states: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_trpo_recurrent", False):
            return self.infer_from_seq(states, h0=h0)
        return self.net(states)

    def _compute_advantages_for_update(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None,
    ):
        if getattr(self, "_trpo_recurrent", False):
            return self.compute_advantages(last_next_state, h0=h0)
        return self.compute_advantages(last_next_state)

    def _evaluate_policy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        valid_masks: torch.BoolTensor | None,
        h0: dict | None,
    ) -> dict[str, torch.Tensor]:
        logits, values = self._rollout_forward(states, h0=h0)
        if valid_masks is not None:
            logits = apply_action_mask(logits, valid_masks)
        dist = distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())
        ratios = torch.exp(log_probs - old_log_probs)
        entropy = _normalized_entropy(dist, valid_masks)
        policy_objective = (ratios * advantages.detach()).mean()
        objective = policy_objective + self.ent_coef * entropy
        return {
            "logits": logits,
            "values": values,
            "log_probs": log_probs,
            "ratios": ratios,
            "entropy": entropy,
            "policy_objective": policy_objective,
            "objective": objective,
        }

    def _make_fisher_vector_product(
        self,
        states: torch.Tensor,
        old_logits: torch.Tensor,
        valid_masks: torch.BoolTensor | None,
        h0: dict | None,
        policy_params: list[torch.nn.Parameter],
    ):
        def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
            new_logits, _ = self._rollout_forward(states, h0=h0)
            if valid_masks is not None:
                new_logits = apply_action_mask(new_logits, valid_masks)
            kl = _categorical_kl(old_logits, new_logits, valid_masks)
            flat_grad_kl = _flat_grad(
                kl,
                policy_params,
                retain_graph=True,
                create_graph=True,
            )
            grad_vector_product = torch.dot(flat_grad_kl, vector)
            hvp = _flat_grad(
                grad_vector_product,
                policy_params,
                retain_graph=False,
                create_graph=False,
            )
            return hvp + self.damping * vector

        return fisher_vector_product

    def _trpo_policy_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_logits: torch.Tensor,
        old_log_probs: torch.Tensor,
        valid_masks: torch.BoolTensor | None,
        h0: dict | None,
    ) -> tuple[float, float, int, bool]:
        policy_params = self._policy_parameters()
        if not policy_params:
            return 0.0, 0.0, 0, False

        old_params = _flat_params(policy_params)
        old_eval = self._evaluate_policy(
            states,
            actions,
            advantages,
            old_log_probs,
            valid_masks,
            h0,
        )
        old_objective = old_eval["objective"].detach()
        policy_grad = _flat_grad(
            old_eval["objective"],
            policy_params,
            retain_graph=False,
            create_graph=False,
        ).detach()

        if policy_grad.norm().item() <= 1e-12:
            return 0.0, 0.0, 0, False

        fvp = self._make_fisher_vector_product(
            states,
            old_logits,
            valid_masks,
            h0,
            policy_params,
        )
        step_dir = _conjugate_gradients(
            fvp,
            policy_grad,
            nsteps=self.cg_iters,
            residual_tol=self.cg_residual_tol,
        )
        fvp_step = fvp(step_dir)
        shs = 0.5 * torch.dot(step_dir, fvp_step)
        if not torch.isfinite(shs) or shs.item() <= 0.0:
            _set_flat_params(policy_params, old_params)
            return 0.0, 0.0, 0, False

        full_step = step_dir * math.sqrt(self.max_kl / (shs.item() + 1e-12))
        expected_improvement = torch.dot(policy_grad, full_step).item()
        if expected_improvement <= 0.0:
            _set_flat_params(policy_params, old_params)
            return expected_improvement, 0.0, 0, False

        actual_improvement = 0.0
        accepted = False
        line_search_steps = 0

        for i in range(self.backtrack_iters):
            step_frac = self.backtrack_coeff ** i
            line_search_steps = i + 1
            _set_flat_params(policy_params, old_params + step_frac * full_step)

            with torch.no_grad():
                new_eval = self._evaluate_policy(
                    states,
                    actions,
                    advantages,
                    old_log_probs,
                    valid_masks,
                    h0,
                )
                new_objective = new_eval["objective"]
                actual_improvement = (new_objective - old_objective).item()

                new_logits = new_eval["logits"]
                kl = _categorical_kl(old_logits, new_logits, valid_masks)
                expected = expected_improvement * step_frac
                improve_ratio = actual_improvement / max(expected, 1e-12)

            if (
                torch.isfinite(kl)
                and actual_improvement > 0.0
                and improve_ratio >= self.accept_ratio
                and kl.item() <= self.max_kl
            ):
                accepted = True
                break

        if not accepted:
            _set_flat_params(policy_params, old_params)
            actual_improvement = 0.0

        return expected_improvement, actual_improvement, line_search_steps, accepted

    def _fit_value(
        self,
        states: torch.Tensor,
        returns: torch.Tensor,
        h0: dict | None,
        k_epochs: int,
        max_grad_norm: float | None,
    ):
        value_params = self._value_parameters()
        n_epochs = max(0, int(k_epochs))

        for _ in range(n_epochs):
            _, values = self._rollout_forward(states, h0=h0)
            value_loss = 0.5 * (returns.detach() - values).pow(2).mean()
            loss = self.vf_coef * value_loss

            self.net.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm)
            self.optim.step()

        self.net.zero_grad(set_to_none=True)

    def _trpo_update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int,
        h0: dict | None,
        max_grad_norm: float | None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, valid_masks = self._compute_advantages_for_update(
            last_next_state,
            h0,
        )

        with torch.no_grad():
            old_logits, _ = self._rollout_forward(states, h0=h0)
            if valid_masks is not None:
                old_logits = apply_action_mask(old_logits, valid_masks)
            old_dist = distributions.Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions.long()).detach()
            old_logits = old_logits.detach()

        expected_improvement, actual_improvement, line_search_steps, accepted = self._trpo_policy_step(
            states,
            actions,
            advantages,
            old_logits,
            old_log_probs,
            valid_masks,
            h0,
        )

        self._fit_value(
            states,
            returns,
            h0,
            k_epochs=k_epochs,
            max_grad_norm=max_grad_norm,
        )

        with torch.no_grad():
            eval_out = self._evaluate_policy(
                states,
                actions,
                advantages,
                old_log_probs,
                valid_masks,
                h0,
            )
            value_loss = 0.5 * (returns.detach() - eval_out["values"]).pow(2).mean()
            kl = _categorical_kl(old_logits, eval_out["logits"], valid_masks)
            total_loss = (
                -eval_out["policy_objective"]
                + self.vf_coef * value_loss
                - self.ent_coef * eval_out["entropy"]
            )

        _append_update_metrics(
            metrics,
            logits=eval_out["logits"],
            log_probs=eval_out["log_probs"],
            values=eval_out["values"],
            returns=returns,
            raw_advantages=raw_adv,
            policy_objective=eval_out["policy_objective"],
            value_loss=value_loss,
            entropy=eval_out["entropy"],
            total_loss=total_loss,
            kl=kl,
            ratios=eval_out["ratios"],
            expected_improvement=expected_improvement,
            actual_improvement=actual_improvement,
            line_search_steps=line_search_steps,
            accepted=accepted,
        )

        return metrics


# ---------------------------------------------------------------------------
# TRPOAgent - feedforward
# ---------------------------------------------------------------------------

class TRPOAgent(_TRPOMixin, AACAgent):
    """
    Trust Region Policy Optimization for discrete actions.

    The public training API matches AACAgent: act() returns only the action,
    store() accepts state/action/reward/done, and the historical training
    helpers are inherited unchanged. The policy update is the original TRPO
    natural-gradient step with a mean-KL trust-region constraint and
    backtracking line search. The value head is fit with the optimizer created
    by init_optimizer().
    """

    _trpo_recurrent = False

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
        max_kl: float = 1e-2,
        damping: float = 1e-2,
        cg_iters: int = 10,
        cg_residual_tol: float = 1e-10,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        accept_ratio: float = 0.1,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            normalize_advantages=normalize_advantages,
            advantage_type=advantage_type,
            gae_lambda=gae_lambda,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
            dtype=dtype,
            device=device,
        )
        self._configure_trpo(
            max_kl=max_kl,
            damping=damping,
            cg_iters=cg_iters,
            cg_residual_tol=cg_residual_tol,
            backtrack_iters=backtrack_iters,
            backtrack_coeff=backtrack_coeff,
            accept_ratio=accept_ratio,
        )

    def infer_logits(self, state: torch.Tensor) -> torch.Tensor:
        shape = state.shape
        z = state.reshape(-1, self.state_dim)
        for layer in self.net.layers:
            z = self.net.activation(layer(z))
        return self.net.actor(z).view(*shape[:-1], self.action_dim)

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 1,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        return self._trpo_update(
            last_next_state=last_next_state,
            k_epochs=k_epochs,
            h0=None,
            max_grad_norm=max_grad_norm,
        )


# ---------------------------------------------------------------------------
# RecurrentTRPOAgent - recurrent
# ---------------------------------------------------------------------------

class RecurrentTRPOAgent(_TRPOMixin, RecurrentAACAgent):
    """
    Recurrent Trust Region Policy Optimization.

    This mirrors RecurrentAACAgent's API and replay-from-h0 update pattern,
    while using the same TRPO constrained policy step as TRPOAgent.
    """

    _trpo_recurrent = True

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
        max_kl: float = 1e-2,
        damping: float = 1e-2,
        cg_iters: int = 10,
        cg_residual_tol: float = 1e-10,
        backtrack_iters: int = 10,
        backtrack_coeff: float = 0.8,
        accept_ratio: float = 0.1,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            normalize_advantages=normalize_advantages,
            advantage_type=advantage_type,
            gae_lambda=gae_lambda,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
            dtype=dtype,
            device=device,
        )
        self._configure_trpo(
            max_kl=max_kl,
            damping=damping,
            cg_iters=cg_iters,
            cg_residual_tol=cg_residual_tol,
            backtrack_iters=backtrack_iters,
            backtrack_coeff=backtrack_coeff,
            accept_ratio=accept_ratio,
        )

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 1,
        h0: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        return self._trpo_update(
            last_next_state=last_next_state,
            k_epochs=k_epochs,
            h0=h0,
            max_grad_norm=max_grad_norm,
        )


__all__ = [
    "TRPOAgent",
    "RecurrentTRPOAgent",
]
