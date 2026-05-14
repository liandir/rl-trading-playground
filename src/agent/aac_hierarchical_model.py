import torch
import torch.nn.functional as F
from torch import distributions

from src.agent.utils import (
    _append_update_metrics,
    _compute_gae,
    _compute_mc,
    _empty_update_metrics,
    _fmt_sim_elapsed,
    _gather_per_asset_mask,
    _init_update_metrics,
    _resolve_historical_source,
    _sample_start_indices,
    _split_logits,
    apply_action_mask,
)
from src.agent.network import build_network


# ---------------------------------------------------------------------------
# Hierarchical rollout buffer
# ---------------------------------------------------------------------------

class HierarchicalRolloutBuffer:
    """
    Buffer for hybrid (a_d, a_q) actions and dict-shaped per-step masks
    {"primary": ..., "buy": ..., "sell": ...}.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.states  = []
        self.actions = []   # each: (..., 2) long tensor stacking [a_d, a_q]
        self.rewards = []
        self.dones   = []
        self.masks_primary = []
        self.masks_buy = []
        self.masks_sell = []
        self._has_masks = None

    def store(self, state, action, reward, done, mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if self._has_masks is None:
            self._has_masks = mask is not None
        if self._has_masks and mask is None:
            raise ValueError("Mask presence is inconsistent across stored steps.")
        if mask is not None:
            self.masks_primary.append(mask["primary"])
            self.masks_buy.append(mask["buy"])
            self.masks_sell.append(mask["sell"])

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states  = torch.stack(self.states).to(device=device, dtype=dtype)
        actions = torch.stack(self.actions).to(device=device, dtype=torch.long)
        rewards = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones   = torch.stack(self.dones).to(device=device, dtype=dtype)
        masks = None
        if self._has_masks:
            masks = {
                "primary": torch.stack(self.masks_primary).to(device=device),
                "buy":     torch.stack(self.masks_buy).to(device=device),
                "sell":    torch.stack(self.masks_sell).to(device=device),
            }
        return states, actions, rewards, dones, masks


# ---------------------------------------------------------------------------
# Mixin: hierarchical policy math (sampling, log-prob, routed entropy)
# ---------------------------------------------------------------------------

class _HierarchicalPolicyMixin:
    """
    Shared policy logic. Concrete agents must define:
        self.n_assets, self.K, self.primary_dim
    """

    def _sample_hierarchical(
        self,
        logits: torch.Tensor,
        masks: dict | None,
        explore: bool,
    ) -> torch.Tensor:
        """
        logits : (..., 1+3N+2K)
        masks  : dict with 'primary' (..., 1+3N), 'buy' / 'sell' (..., N, K), or None
        returns actions (..., 2) long: [a_d, a_q]
        """
        N = self.n_assets
        K = self.K
        pdim = self.primary_dim

        logits_d, logits_buy, logits_sell = _split_logits(logits, pdim, K)

        if masks is not None:
            logits_d = apply_action_mask(logits_d, masks["primary"])

        if explore:
            a_d = distributions.Categorical(logits=logits_d).sample()
        else:
            a_d = logits_d.argmax(dim=-1)

        is_buy  = (a_d >= 1) & (a_d <= N)
        is_sell = (a_d >= N + 1) & (a_d <= 2 * N)
        asset_buy  = (a_d - 1).clamp(0, N - 1)
        asset_sell = (a_d - 1 - N).clamp(0, N - 1)

        if masks is not None:
            buy_m_sel  = _gather_per_asset_mask(masks["buy"],  asset_buy)
            sell_m_sel = _gather_per_asset_mask(masks["sell"], asset_sell)
        else:
            shape = (*a_d.shape, K)
            buy_m_sel  = torch.ones(shape, dtype=torch.bool, device=logits.device)
            sell_m_sel = buy_m_sel

        bucket_logits = torch.where(is_sell.unsqueeze(-1), logits_sell, logits_buy)
        bucket_mask   = torch.where(is_sell.unsqueeze(-1), sell_m_sel,  buy_m_sel)

        not_bucket = ~(is_buy | is_sell)
        if not_bucket.any():
            all_true = torch.ones_like(bucket_mask)
            bucket_mask = torch.where(not_bucket.unsqueeze(-1), all_true, bucket_mask)

        bucket_logits_masked = apply_action_mask(bucket_logits, bucket_mask)

        if explore:
            a_q = distributions.Categorical(logits=bucket_logits_masked).sample()
        else:
            a_q = bucket_logits_masked.argmax(dim=-1)

        a_q = torch.where(is_buy | is_sell, a_q, torch.zeros_like(a_q))

        return torch.stack([a_d, a_q], dim=-1)

    def _policy_terms(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        masks: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute joint log-prob and mask-aware joint entropy normalised to [0, 1].

        logits  : (T, [B,] 1+3N+2K)
        actions : (T, [B,] 2) long  [a_d, a_q]
        masks   : dict of stacked masks or None

        The normalised entropy uses the chain rule
            H(a_d, a_q) = H(a_d) + E_{a_d}[H(a_q | a_d)]
        with the conditional expectation expanded per asset (bucket masks are
        per-asset), and divides by the log of the joint support size under the
        active masks, which is the max joint entropy achievable by any factored
        hierarchical policy.

        Returns (log_pi, entropy_routed_mean, primary_logits_masked)
        """
        N = self.n_assets
        K = self.K
        pdim = self.primary_dim

        logits_d, logits_buy, logits_sell = _split_logits(logits, pdim, K)

        a_d = actions[..., 0].long()
        a_q = actions[..., 1].long()

        if masks is not None:
            logits_d_m = apply_action_mask(logits_d, masks["primary"])
        else:
            logits_d_m = logits_d

        dist_d   = distributions.Categorical(logits=logits_d_m)
        log_pi_d = dist_d.log_prob(a_d)
        H_d_raw  = dist_d.entropy()

        is_buy  = (a_d >= 1) & (a_d <= N)
        is_sell = (a_d >= N + 1) & (a_d <= 2 * N)
        asset_buy  = (a_d - 1).clamp(0, N - 1)
        asset_sell = (a_d - 1 - N).clamp(0, N - 1)

        # --- Log-prob: use the per-asset bucket mask of the sampled action ---
        if masks is not None:
            buy_m_sel  = _gather_per_asset_mask(masks["buy"],  asset_buy)
            sell_m_sel = _gather_per_asset_mask(masks["sell"], asset_sell)
            buy_m_safe  = buy_m_sel  | (~is_buy.unsqueeze(-1))
            sell_m_safe = sell_m_sel | (~is_sell.unsqueeze(-1))
            logits_buy_sel  = apply_action_mask(logits_buy,  buy_m_safe)
            logits_sell_sel = apply_action_mask(logits_sell, sell_m_safe)
        else:
            logits_buy_sel  = logits_buy
            logits_sell_sel = logits_sell

        log_pi_buy  = distributions.Categorical(logits=logits_buy_sel).log_prob(a_q)
        log_pi_sell = distributions.Categorical(logits=logits_sell_sel).log_prob(a_q)

        log_pi = log_pi_d + is_buy.to(log_pi_d.dtype) * log_pi_buy + is_sell.to(log_pi_d.dtype) * log_pi_sell

        # --- Joint entropy: chain rule, expectation taken over all directions ---
        pi_d      = torch.softmax(logits_d_m, dim=-1)
        pi_d_buy  = pi_d[..., 1:1 + N]
        pi_d_sell = pi_d[..., 1 + N:1 + 2 * N]

        # Bucket logits are shared across assets, but masks are per-asset.
        logits_buy_b  = logits_buy.unsqueeze(-2).expand(*logits_buy.shape[:-1],  N, K)
        logits_sell_b = logits_sell.unsqueeze(-2).expand(*logits_sell.shape[:-1], N, K)

        if masks is not None:
            # Rows with no valid buckets would give all -inf logits and NaN entropy.
            # Those rows have pi_d(buy_i) = 0 (primary masked), so H_buy_i is weighted out;
            # substitute an all-true row to keep entropy finite.
            buy_any  = masks["buy"].any(dim=-1, keepdim=True)
            sell_any = masks["sell"].any(dim=-1, keepdim=True)
            buy_mask_safe  = masks["buy"]  | (~buy_any)
            sell_mask_safe = masks["sell"] | (~sell_any)
            logits_buy_b  = apply_action_mask(logits_buy_b,  buy_mask_safe)
            logits_sell_b = apply_action_mask(logits_sell_b, sell_mask_safe)

        H_buy_per_asset  = distributions.Categorical(logits=logits_buy_b).entropy()   # (..., N)
        H_sell_per_asset = distributions.Categorical(logits=logits_sell_b).entropy()  # (..., N)

        H_cond  = (pi_d_buy * H_buy_per_asset).sum(-1) + (pi_d_sell * H_sell_per_asset).sum(-1)
        H_joint = H_d_raw + H_cond

        # --- Max joint entropy under the active masks (joint support size) ---
        if masks is not None:
            p_bool   = masks["primary"].to(logits.dtype)
            n_hold   = p_bool[..., 0] + p_bool[..., 1 + 2 * N:1 + 3 * N].sum(-1)
            buy_valid  = p_bool[..., 1:1 + N]
            sell_valid = p_bool[..., 1 + N:1 + 2 * N]

            k_buy_i  = masks["buy"].sum(-1).to(logits.dtype)
            k_sell_i = masks["sell"].sum(-1).to(logits.dtype)

            buy_support  = (buy_valid  * k_buy_i).sum(-1)
            sell_support = (sell_valid * k_sell_i).sum(-1)

            n_joint = n_hold + buy_support + sell_support
        else:
            n_joint = torch.full_like(H_joint, float((1 + N) + 2 * N * K))

        log_joint_max  = torch.log(n_joint.clamp(min=2.0))
        entropy_routed = (H_joint / log_joint_max).mean()

        return log_pi, entropy_routed, logits_d_m


# ---------------------------------------------------------------------------
# HierarchicalModelAACAgent (feedforward)
class HierarchicalModelAACAgent(_HierarchicalPolicyMixin):
    """
    Sequence-capable hierarchical AAC agent with inference-only latent rollouts.

    The model-aware action path expects the network to provide:
        encode_latent(state) -> latent
        policy_value_from_latent(latent) -> (logits, value)
        model_step(latent, action) -> dict/tuple containing next latent, reward, done

    The imagined rollouts are deliberately not used by update(); they only score
    candidate actions at inference time.
    """

    def __init__(
        self,
        network: dict,
        n_assets: int,
        n_buckets: int,
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        model_coef: float = 0.1,
        normalize_advantages: bool = True,
        advantage_type: str = "td0",
        gae_lambda: float = 0.95,
        imagine_length: int = 0,
        n_imagined_trajectories: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        if advantage_type not in ("td0", "gae", "mc"):
            raise ValueError(f"advantage_type must be 'td0', 'gae', or 'mc', got '{advantage_type}'.")
        self.gamma                = gamma
        self.vf_coef              = vf_coef
        self.ent_coef             = ent_coef
        self.model_coef           = model_coef
        self.normalize_advantages = normalize_advantages
        self.advantage_type       = advantage_type
        self.gae_lambda           = gae_lambda
        self.imagine_length       = int(imagine_length)
        self.n_imagined_trajectories = int(n_imagined_trajectories)
        self.dtype                = dtype
        self.device               = torch.device(device)

        self.n_assets    = int(n_assets)
        self.K           = int(n_buckets)
        self.primary_dim = 1 + 3 * self.n_assets

        self.net = build_network(network).to(device=self.device, dtype=self.dtype)
        expected = self.primary_dim + 2 * self.K
        if int(self.net.action_dim) != expected:
            raise ValueError(
                f"Network action_dim must equal 1 + 3*N + 2*K = {expected}, got {self.net.action_dim}."
            )
        self.state_dim  = getattr(self.net, "state_dim", None)
        self.action_dim = self.net.action_dim

        self.buffer = HierarchicalRolloutBuffer()
        self.net.reset(1)

    # ------------------------------------------------------------------

    def infer_from_seq(self, state_seq, h0=None):
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        return self.net.forward_seq(state_seq)

    def _net_states(self):
        if not hasattr(self.net, "get_states"):
            return None
        return self.net.get_states(clone=True, detach=True)

    def _set_net_states(self, states):
        if states is not None and hasattr(self.net, "set_states"):
            self.net.set_states(states, clone=True, detach=True, strict=False)

    def _encode_latent(self, state: torch.Tensor) -> torch.Tensor:
        if hasattr(self.net, "encode_latent"):
            return self.net.encode_latent(state)
        raise AttributeError("Model-based inference requires net.encode_latent(state).")

    def _policy_value_from_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.net, "policy_value_from_latent"):
            return self.net.policy_value_from_latent(latent)
        raise AttributeError("Model-based inference requires net.policy_value_from_latent(latent).")

    def _model_step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not hasattr(self.net, "model_step"):
            raise AttributeError("Model-based inference requires net.model_step(latent, action).")

        out = self.net.model_step(latent, action)
        if isinstance(out, dict):
            next_latent = out.get("latent", out.get("next_latent"))
            reward = out.get("reward", None)
            done = out.get("done", None)
        else:
            if torch.is_tensor(out):
                next_latent = out
                reward = None
                done = None
            elif len(out) == 1:
                next_latent = out[0]
                reward = None
                done = None
            elif len(out) == 2:
                next_latent, reward = out
                done = None
            elif len(out) == 3:
                next_latent, reward, done = out
            else:
                raise ValueError("model_step must return next_latent, (next_latent, reward[, done]), or a matching dict.")

        if next_latent is None:
            raise ValueError("model_step output must include next_latent/latent.")

        if reward is None:
            reward = torch.zeros(next_latent.shape[:-1], dtype=self.dtype, device=self.device)
        else:
            reward = reward.to(device=self.device, dtype=self.dtype)
        if done is None:
            done = torch.zeros_like(reward, dtype=self.dtype, device=self.device)
        else:
            done = done.to(device=self.device, dtype=self.dtype)
        return next_latent, reward, done

    def imagine_rollouts(
        self,
        state: torch.Tensor,
        mask: dict | None = None,
        *,
        length: int | None = None,
        n_trajectories: int | None = None,
        explore_first_action: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Roll out candidate trajectories through the latent model for inference.

        Returns a dict with first_actions, returns, rewards, dones and actions.
        No gradients are tracked and the live recurrent state is restored after
        scoring candidates.
        """
        length = self.imagine_length if length is None else int(length)
        n_trajectories = self.n_imagined_trajectories if n_trajectories is None else int(n_trajectories)
        if length <= 0:
            raise ValueError(f"imagine_length must be positive, got {length}.")
        if n_trajectories <= 0:
            raise ValueError(f"n_imagined_trajectories must be positive, got {n_trajectories}.")

        state = state.to(dtype=self.dtype, device=self.device)
        mask_dev = None if mask is None else {k: v.to(self.device) for k, v in mask.items()}
        h_live = self._net_states()

        with torch.no_grad():
            latent0 = self._encode_latent(state)
            logits0, _ = self._policy_value_from_latent(latent0)
            if logits0.ndim != 1:
                raise ValueError(
                    "imagine_rollouts currently expects one unbatched state. "
                    f"Got policy logits with shape {tuple(logits0.shape)}."
                )
            latent = latent0.unsqueeze(0).expand(n_trajectories, *latent0.shape).contiguous()

            first_actions = None
            actions = []
            rewards = []
            dones = []
            alive = torch.ones(n_trajectories, device=self.device, dtype=self.dtype)
            returns = torch.zeros(n_trajectories, device=self.device, dtype=self.dtype)
            discount = torch.ones_like(returns)

            logits = logits0.unsqueeze(0).expand(n_trajectories, *logits0.shape).contiguous()
            if mask_dev is not None:
                mask_t = {
                    key: value.unsqueeze(0).expand(n_trajectories, *value.shape)
                    for key, value in mask_dev.items()
                }
            else:
                mask_t = None

            for step in range(length):
                action = self._sample_hierarchical(
                    logits,
                    mask_t if step == 0 else None,
                    explore=explore_first_action or step > 0,
                )
                if step == 0:
                    first_actions = action
                actions.append(action)

                latent, reward, done = self._model_step(latent, action)
                reward = reward.reshape(n_trajectories)
                done = done.reshape(n_trajectories).clamp(0.0, 1.0)
                returns = returns + discount * alive * reward
                rewards.append(reward)
                dones.append(done)

                alive = alive * (1.0 - done)
                discount = discount * self.gamma
                logits, values = self._policy_value_from_latent(latent)

                if bool((alive <= 0).all().item()):
                    break

            returns = returns + discount * alive * values.reshape(n_trajectories)

        self._set_net_states(h_live)
        return {
            "first_actions": first_actions,
            "returns": returns,
            "actions": torch.stack(actions, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "dones": torch.stack(dones, dim=0),
        }

    def act_imagined(
        self,
        state: torch.Tensor,
        mask: dict | None = None,
        *,
        explore_first_action: bool = True,
        advance_state: bool = True,
    ) -> torch.Tensor:
        rollouts = self.imagine_rollouts(
            state,
            mask=mask,
            explore_first_action=explore_first_action,
        )
        best = rollouts["returns"].argmax(dim=0)
        action = rollouts["first_actions"][best]
        if advance_state:
            with torch.no_grad():
                state_dev = state.to(dtype=self.dtype, device=self.device)
                self.net(state_dev)
        return action.cpu()

    def act(
        self,
        state: torch.Tensor,
        mask: dict | None = None,
        explore: bool = False,
        grad_enabled: bool = False,
        use_imagination: bool = False,
    ) -> torch.Tensor:
        if use_imagination:
            if grad_enabled:
                raise ValueError("Imagined action selection is inference-only; use grad_enabled=False.")
            if explore:
                raise ValueError("Imagined action selection already samples candidates; use explore=False.")
            return self.act_imagined(state, mask=mask)

        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits, _ = self.net(state)
            mask_dev = None
            if mask is not None:
                mask_dev = {k: v.to(self.device) for k, v in mask.items()}
            action = self._sample_hierarchical(logits, mask_dev, explore)
        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, done, mask=None):
        self.buffer.store(state, action, reward, done, mask=mask)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def compute_advantages(self, last_next_state: torch.Tensor, h0=None):
        states, actions, rewards, dones, masks = self.buffer.to_tensors(self.device, self.dtype)
        rewards = rewards.squeeze(-1) if rewards.dim() > actions.dim() - 1 else rewards
        dones   = dones.squeeze(-1)   if dones.dim()   > actions.dim() - 1 else dones

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
            elif self.advantage_type == "mc":
                raw_adv, returns = _compute_mc(rewards, dones, v_t, v_tp1[-1], self.gamma)
            else:
                returns = rewards + self.gamma * (1.0 - dones) * v_tp1
                raw_adv = returns - v_t

            advantages = raw_adv
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)

        return advantages, raw_adv, returns, states, actions, masks

    def model_prediction_loss(self, states: torch.Tensor, actions: torch.Tensor, last_next_state: torch.Tensor, h0=None):
        if not all(hasattr(self.net, name) for name in ("encode", "model_seq")):
            return torch.zeros((), dtype=self.dtype, device=self.device)

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        all_states = torch.cat([states, last_next_state[None]], dim=0)
        z_all = self.net.encode(all_states).detach()
        z_t = z_all[:-1]
        z_tp1 = z_all[1:]

        if h0 is not None:
            self.net.set_states(h0, strict=False)
        z_pred = self.net.model_seq(z_t, actions)
        return F.mse_loss(z_pred, z_tp1)

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
        advantages, raw_adv, returns, states, actions, masks = self.compute_advantages(last_next_state, h0=h0)

        logits, values = self.infer_from_seq(states, h0=h0)
        log_probs, entropy, primary_logits_m = self._policy_terms(logits, actions, masks)

        adv              = advantages.detach()
        policy_objective = (log_probs * adv).mean()
        value_loss       = 0.5 * (returns.detach() - values).pow(2).mean()
        model_loss       = self.model_prediction_loss(states, actions, last_next_state, h0=h0)
        loss             = (
            -policy_objective
            + self.vf_coef * value_loss
            + self.model_coef * model_loss
            - self.ent_coef * entropy
        )

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
            value_loss=value_loss,
            entropy=entropy,
            loss=loss,
        )
        metrics.setdefault("model_loss", []).append(float(model_loss.detach().cpu().item()))

        return metrics

    def save(self, path: str):
        torch.save({"network": self.net.state_dict()}, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)

    # ------------------------------------------------------------------
    # Training loops
    # ------------------------------------------------------------------

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
        *,
        high=None,
        low=None,
        volume=None,
        times=None,
        open_=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        T, time_at, reset_env, step_env = _resolve_historical_source(
            data,
            high=high,
            low=low,
            volume=volume,
            times=times,
            open_=open_,
        )
        total_reward = []
        total_loss   = []
        total_info   = []

        for episode in range(1, n_episodes + 1):
            start   = int(_sample_start_indices(T, max_steps).item())
            sim_t0  = time_at(start)
            state   = reset_env(env, start)
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
                mask = env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    action = self.act(state.to_tensor(), mask=mask, explore=True, grad_enabled=False)
                else:
                    action = None

                next_state, reward, done, info = step_env(env, action, start + step)

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done,   dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        mask=mask,
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

                            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(time_at(start + step) - sim_t0)}] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        self.net.set_states(h0 := h_t, clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break
                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(time_at(start + step) - sim_t0)}] - warm up in progress...", end="\r")

                state = next_state
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(time_at(start + step) - sim_t0)}] - reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}"
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
                open_[starts], close[starts], high[starts], low[starts], volume[starts], times[starts],
            )
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(B)]
                ep_loss   = []

            step = 1
            while step <= max_steps:
                si = starts + step
                mask = bat_env.valid_action_mask() if step > warm_up else None

                if step > warm_up:
                    actions      = self.act(obs, mask=mask, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions      = None
                    step_actions = (
                        torch.zeros(B, dtype=torch.long),
                        torch.zeros(B, dtype=torch.long),
                    )

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


HierarchicalAACModelAgent = HierarchicalModelAACAgent
HierarchicalAACAgent = HierarchicalModelAACAgent
