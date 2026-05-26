"""Spatiotemporal aac utilities for reinforcement-learning agents and training utilities."""
import torch

from rl_trading_playground.agent.utils import (
    _append_update_metrics,
    _aux_loss,
    _build_aux_target,
    _close_from_historical,
    _compute_gae,
    _compute_mc,
    _empty_update_metrics,
    _fmt_sim_elapsed,
    _init_update_metrics,
    _sample_start_indices,
    _unpack_policy_value_aux,
)
from rl_trading_playground.agent.aac_hierarchical import (
    HierarchicalRolloutBuffer,
    _HierarchicalPolicyMixin,
    _resolve_historical_source,
)
from rl_trading_playground.agent.aac_hierarchical_aux import AuxiliaryHierarchicalRolloutBuffer
from rl_trading_playground.network import build_network, SpatiotemporalAuxiliaryPerAssetActionValueNetwork


class SpatiotemporalHierarchicalAACAgent(_HierarchicalPolicyMixin):
    """
    Hierarchical AAC agent tailored for SpatiotemporalAuxiliaryPerAssetActionValueNetwork.

    Maintains a rolling ``context_length``-observation buffer that is prepended
    to every ``forward_seq`` call during training. This gives causal temporal
    attention at rollout position t=0 a proper historical view rather than
    starting from an empty context.

    Inference (``act``) delegates to ``net.forward``, which manages its own
    internal ``window_size`` rolling buffer independently.
    """

    def __init__(
        self,
        network: dict,
        n_assets: int,
        n_buckets: int,
        context_length: int = 64,
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        aux_coef: float = 0.05,
        aux_loss_coefs: dict[str, float] | None = None,
        normalize_advantages: bool = True,
        advantage_type: str = "td0",
        gae_lambda: float = 0.95,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize the instance.

        Args:
            network (dict): The network value.
            n_assets (int): The n assets value.
            n_buckets (int): The n buckets value.
            context_length (int): The context length value. Defaults to ``64``.
            gamma (float): The gamma value. Defaults to ``0.999``.
            vf_coef (float): The vf coef value. Defaults to ``0.5``.
            ent_coef (float): The ent coef value. Defaults to ``0.01``.
            aux_coef (float): The aux coef value. Defaults to ``0.05``.
            aux_loss_coefs (dict[str, float] | None): The aux loss coefs value. Defaults to ``None``.
            normalize_advantages (bool): The normalize advantages value. Defaults to ``True``.
            advantage_type (str): The advantage type value. Defaults to ``'td0'``.
            gae_lambda (float): The gae lambda value. Defaults to ``0.95``.
            dtype (torch.dtype): The dtype value. Defaults to ``torch.float32``.
            device (str): The device value. Defaults to ``'cpu'``.

        Returns:
            None: This function does not return a value.
        """
        if advantage_type not in ("td0", "gae", "mc"):
            raise ValueError(f"advantage_type must be 'td0', 'gae', or 'mc', got '{advantage_type}'.")

        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.aux_coef = aux_coef
        self.aux_loss_coefs = dict(aux_loss_coefs or {"reward": 1.0, "asset_ret": 0.5, "asset_vol": 0.25})
        self.normalize_advantages = normalize_advantages
        self.advantage_type = advantage_type
        self.gae_lambda = gae_lambda
        self.dtype = dtype
        self.device = torch.device(device)
        self.context_length = int(context_length)

        self.n_assets = int(n_assets)
        self.K = int(n_buckets)
        self.primary_dim = 1 + 3 * self.n_assets

        self.net = build_network(network).to(device=self.device, dtype=self.dtype)
        if not isinstance(self.net, SpatiotemporalAuxiliaryPerAssetActionValueNetwork):
            raise TypeError(
                f"SpatiotemporalHierarchicalAACAgent requires a "
                f"SpatiotemporalAuxiliaryPerAssetActionValueNetwork, got {type(self.net).__name__}."
            )
        expected = self.primary_dim + 2 * self.K
        if int(self.net.action_dim) != expected:
            raise ValueError(
                f"Network action_dim must equal 1 + 3*N + 2*K = {expected}, got {self.net.action_dim}."
            )
        self.aux_horizons = tuple(getattr(self.net, "aux_horizons", network.get("aux_horizons", (1, 5, 20))))
        self.state_dim = getattr(self.net, "state_dim", None)
        self.action_dim = self.net.action_dim

        self.buffer = AuxiliaryHierarchicalRolloutBuffer()
        self._ctx_buf: torch.Tensor | None = None
        self.net.reset(1)

    # ------------------------------------------------------------------
    # Context buffer helpers
    # ------------------------------------------------------------------

    def _init_ctx_buf(self, ref_state: torch.Tensor) -> None:
        """Allocate a zero-filled context buffer matching ref_state's shape.

        Args:
            ref_state (torch.Tensor): The ref state value.

        Returns:
            None: This function does not return a value.
        """
        shape = (self.context_length, *ref_state.shape)
        self._ctx_buf = torch.zeros(shape, device=self.device, dtype=self.dtype)

    def _push_ctx(self, state: torch.Tensor) -> None:
        """Shift the context buffer left by one and append state at the end.

        Args:
            state (torch.Tensor): The state value.

        Returns:
            None: This function does not return a value.
        """
        state = state.to(device=self.device, dtype=self.dtype)
        if self._ctx_buf is None or self._ctx_buf.shape[1:] != state.shape:
            self._init_ctx_buf(state)
        self._ctx_buf = torch.cat([self._ctx_buf[1:], state.unsqueeze(0).detach()], dim=0)

    def _snapshot_ctx(self) -> torch.Tensor | None:
        """Return a detached clone of the current context buffer, or None.

        Returns:
            torch.Tensor | None: The computed or requested result.
        """
        if self._ctx_buf is None:
            return None
        return self._ctx_buf.clone().detach()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, batch_size: int = 1) -> None:
        """Reset internal state for a new episode or stream.

        Args:
            batch_size (int): The batch size value. Defaults to ``1``.

        Returns:
            None: This function does not return a value.
        """
        self.net.reset(batch_size)
        self._ctx_buf = None

    def act(self, state: torch.Tensor, mask: dict | None = None, explore: bool = False, grad_enabled: bool = False):
        """Act for SpatiotemporalHierarchicalAACAgent.

        Args:
            state (torch.Tensor): The state value.
            mask (dict | None): The mask value. Defaults to ``None``.
            explore (bool): The explore value. Defaults to ``False``.
            grad_enabled (bool): The grad enabled value. Defaults to ``False``.

        Returns:
            Any: The computed or requested result.
        """
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits, _, _ = _unpack_policy_value_aux(self.net(state))
            mask_dev = None if mask is None else {k: v.to(self.device) for k, v in mask.items()}
            action = self._sample_hierarchical(logits, mask_dev, explore)
        return action if grad_enabled else action.cpu()

    def store(self, state, action, reward, done, mask=None, aux_target=None) -> None:
        """Store one transition or payload in the buffer.

        Args:
            state (Any): The state value.
            action (Any): The action value.
            reward (Any): The reward value.
            done (Any): The done value.
            mask (Any): The mask value. Defaults to ``None``.
            aux_target (Any): The aux target value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
        state_t = state if isinstance(state, torch.Tensor) else torch.as_tensor(state)
        self._push_ctx(state_t.detach())
        self.buffer.store(state, action, reward, done, mask=mask, aux_target=aux_target)

    def init_optimizer(self, lr, optim: str = "AdamW") -> None:
        """Init optimizer for SpatiotemporalHierarchicalAACAgent.

        Args:
            lr (Any): The lr value.
            optim (str): The optim value. Defaults to ``'AdamW'``.

        Returns:
            None: This function does not return a value.
        """
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # forward_seq with context prefix
    # ------------------------------------------------------------------

    def infer_from_seq(self, state_seq, h0=None, ctx0=None):
        """Run forward_seq with an optional context prefix.

        ctx0 : (context_length, [B,] state_dim) prepended before state_seq.
               Outputs are sliced to remove the context prefix before returning.

        Args:
            state_seq (Any): The state seq value.
            h0 (Any): The h0 value. Defaults to ``None``.
            ctx0 (Any): The ctx0 value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
        """
        C = 0
        if ctx0 is not None and ctx0.shape[0] > 0:
            full_seq = torch.cat([ctx0.to(device=self.device, dtype=self.dtype), state_seq], dim=0)
            C = ctx0.shape[0]
        else:
            full_seq = state_seq

        if h0 is not None:
            self.net.set_states(h0, strict=False)

        logits, values, aux = self.net.forward_seq(full_seq)

        if C > 0:
            logits = logits[C:]
            values = values[C:]
            aux = {k: v[C:] for k, v in aux.items()}

        return logits, values, aux

    def compute_advantages(self, last_next_state: torch.Tensor, h0=None, ctx0=None):
        """Compute the advantages.

        Args:
            last_next_state (torch.Tensor): The last next state value.
            h0 (Any): The h0 value. Defaults to ``None``.
            ctx0 (Any): The ctx0 value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
        """
        states, actions, rewards, dones, masks, aux_targets = self.buffer.to_tensors(self.device, self.dtype)
        rewards = rewards.squeeze(-1) if rewards.dim() > actions.dim() - 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > actions.dim() - 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        C = 0
        if ctx0 is not None and ctx0.shape[0] > 0:
            C = ctx0.shape[0]
            ctx_dev = ctx0.to(device=self.device, dtype=self.dtype)
            all_states = torch.cat([ctx_dev, states, last_next_state[None]], dim=0)
        else:
            all_states = torch.cat([states, last_next_state[None]], dim=0)

        with torch.no_grad():
            if h0 is not None:
                self.net.set_states(h0, strict=False)
            _, values_all, _ = self.net.forward_seq(all_states)

        values_all = values_all[C:]
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

        return advantages, raw_adv, returns, states, actions, rewards, masks, aux_targets

    def update(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
        ctx0: torch.Tensor | None = None,
        max_grad_norm: float | None = None,
    ):
        """Apply one update step.

        Args:
            last_next_state (torch.Tensor): The last next state value.
            h0 (dict | None): The h0 value. Defaults to ``None``.
            ctx0 (torch.Tensor | None): The ctx0 value. Defaults to ``None``.
            max_grad_norm (float | None): The max grad norm value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
        """
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()
        advantages, raw_adv, returns, states, actions, rewards, masks, aux_targets = self.compute_advantages(
            last_next_state, h0=h0, ctx0=ctx0,
        )

        logits, values, aux = self.infer_from_seq(states, h0=h0, ctx0=ctx0)
        log_probs, entropy, primary_logits_m = self._policy_terms(logits, actions, masks)

        adv = advantages.detach()
        policy_objective = (log_probs * adv).mean()
        value_loss = 0.5 * (returns.detach() - values).pow(2).mean()
        aux_loss, aux_losses = _aux_loss(aux, aux_targets, rewards=rewards, coefs=self.aux_loss_coefs)
        loss = -policy_objective + self.vf_coef * value_loss - self.ent_coef * entropy + self.aux_coef * aux_loss

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
        for name, value in aux_losses.items():
            metrics.setdefault(name, []).append(float(value.detach().cpu().item()))
        return metrics

    def save(self, path: str) -> None:
        """Save for SpatiotemporalHierarchicalAACAgent.

        Args:
            path (str): The path value.

        Returns:
            None: This function does not return a value.
        """
        torch.save({"network": self.net.state_dict()}, path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load for SpatiotemporalHierarchicalAACAgent.

        Args:
            path (str): The path value.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
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
        max_steps: int = 2000,
        warm_up: int = 0,
        update_interval: int = 100,
        burn_in_updates: int = 0,
        lr: float = 3e-4,
        optim: str = "AdamW",
        init_optimizer: bool = False,
        store_results: bool = True,
        max_grad_norm=None,
        *,
        open_=None,
        high=None,
        low=None,
        volume=None,
        times=None,
    ):
        """Train on historical for SpatiotemporalHierarchicalAACAgent.

        Args:
            env (Any): The env value.
            data (Any): The data value.
            n_episodes (Any): The n episodes value.
            max_steps (int): The max steps value. Defaults to ``2000``.
            warm_up (int): The warm up value. Defaults to ``0``.
            update_interval (int): The update interval value. Defaults to ``100``.
            burn_in_updates (int): The burn in updates value. Defaults to ``0``.
            lr (float): The lr value. Defaults to ``0.0003``.
            optim (str): The optim value. Defaults to ``'AdamW'``.
            init_optimizer (bool): The init optimizer value. Defaults to ``False``.
            store_results (bool): The store results value. Defaults to ``True``.
            max_grad_norm (Any): The max grad norm value. Defaults to ``None``.
            open_ (Any): The open value. Defaults to ``None``.
            high (Any): The high value. Defaults to ``None``.
            low (Any): The low value. Defaults to ``None``.
            volume (Any): The volume value. Defaults to ``None``.
            times (Any): The times value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
        """
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        T, time_at, reset_env, step_env = _resolve_historical_source(
            data, high=high, low=low, volume=volume, times=times, open_=open_,
        )
        close_src = _close_from_historical(data).to(device=self.device, dtype=self.dtype)
        horizon_pad = max(self.aux_horizons)
        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            start = int(_sample_start_indices(T, max_steps + horizon_pad).item())
            sim_t0 = time_at(start)
            state = reset_env(env, start)
            burn_in = burn_in_updates
            self.reset(1)
            self.buffer.clear()
            h0 = self.net.get_states(clone=True, detach=True)
            ctx0 = None  # no context at episode start

            if store_results:
                episode_reward = []
                episode_info = []
                episode_loss = []

            step = 1
            while True:
                mask = env.valid_action_mask() if step > warm_up else None
                action = self.act(state.to_tensor(), mask=mask, explore=True, grad_enabled=False) if step > warm_up else None
                next_state, reward, done, info = step_env(env, action, start + step)

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    current_idx = start + step - 1
                    aux_target = _build_aux_target(
                        close_src,
                        current_idx,
                        self.aux_horizons,
                        reward=torch.as_tensor(reward, dtype=self.dtype, device=self.device),
                    )
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        mask=mask,
                        aux_target={k: v.detach().cpu() for k, v in aux_target.items()},
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = done or (step >= max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        h_t = self.net.get_states(clone=True, detach=True)
                        if burn_in > 0:
                            burn_in -= 1
                        else:
                            loss_dict = self.update(
                                next_state.to_tensor(), h0=h0, ctx0=ctx0, max_grad_norm=max_grad_norm,
                            )
                            if store_results:
                                episode_loss.append(loss_dict)
                            msg = (
                                f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(time_at(start + step) - sim_t0)}]"
                                f" - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                            )
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")
                        self.net.set_states(h0 := h_t, clone=True, detach=True)
                        ctx0 = self._snapshot_ctx()
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

            msg = (
                f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(time_at(start + step) - sim_t0)}]"
                f" - reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}"
            )
            if store_results and len(episode_loss) > 0:
                for key in episode_loss[0].keys():
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_bat(
        self,
        bat_env,
        open_, close, high, low, volume, times,
        n_episodes,
        max_steps: int = 2000,
        warm_up: int = 0,
        update_interval: int = 100,
        burn_in_updates: int = 1,
        lr: float = 3e-4,
        optim: str = "AdamW",
        init_optimizer: bool = False,
        store_results: bool = True,
        max_grad_norm=None,
    ):
        """Train on historical bat for SpatiotemporalHierarchicalAACAgent.

        Args:
            bat_env (Any): The bat env value.
            open_ (Any): The open value.
            close (Any): The close value.
            high (Any): The high value.
            low (Any): The low value.
            volume (Any): The volume value.
            times (Any): The times value.
            n_episodes (Any): The n episodes value.
            max_steps (int): The max steps value. Defaults to ``2000``.
            warm_up (int): The warm up value. Defaults to ``0``.
            update_interval (int): The update interval value. Defaults to ``100``.
            burn_in_updates (int): The burn in updates value. Defaults to ``1``.
            lr (float): The lr value. Defaults to ``0.0003``.
            optim (str): The optim value. Defaults to ``'AdamW'``.
            init_optimizer (bool): The init optimizer value. Defaults to ``False``.
            store_results (bool): The store results value. Defaults to ``True``.
            max_grad_norm (Any): The max grad norm value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
        """
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
            self.reset(B)
            h0 = self.net.get_states(clone=True, detach=True)
            ctx0 = None  # no context at episode start
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
                    actions = self.act(obs, mask=mask, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions = None
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
                    self.store(
                        obs.detach(),
                        actions.detach(),
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
                                next_obs, h0=h0, ctx0=ctx0, max_grad_norm=max_grad_norm,
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
                        ctx0 = self._snapshot_ctx()
                        self.buffer.clear()
                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}% - {_fmt_sim_elapsed(float(times[starts[0] + step]) - sim_t0)}] - warm up in progress...", end="\r")

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
