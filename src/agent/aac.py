"""Aac utilities for reinforcement-learning agents and training utilities."""
import torch
from torch import distributions

from src.network import build_network
from src.agent.utils import (
    UPDATE_METRIC_NAMES,
    _append_update_metrics,
    _compute_gae,
    _compute_mc,
    _empty_update_metrics,
    _explained_variance,
    _fmt_sim_elapsed,
    _init_update_metrics,
    _sample_start_indices,
    apply_action_mask,
)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """RolloutBuffer buffer for reinforcement-learning agents and training utilities."""
    def __init__(self):
        """Initialize the instance.

        Returns:
            None: This function does not return a value.
        """
        self.clear()

    def clear(self):
        """Clear buffered state.

        Returns:
            None: This function does not return a value.
        """
        self.states      = []
        self.actions     = []
        self.rewards     = []
        self.dones       = []
        self.valid_masks = []

    def store(self, state, action, reward, done, valid_mask=None):
        """Store one transition or payload in the buffer.

        Args:
            state (Any): The state value.
            action (Any): The action value.
            reward (Any): The reward value.
            done (Any): The done value.
            valid_mask (Any): The valid mask value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_masks.append(valid_mask)

    def __len__(self):
        """Return the number of contained items.

        Returns:
            Any: The computed or requested result.
        """
        return len(self.states)

    def to_tensors(self, device, dtype):
        """Convert buffered values to tensors.

        Args:
            device (Any): The device value.
            dtype (Any): The dtype value.

        Returns:
            Any: The computed or requested result.
        """
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
class AACAgent:
    """
    Advantage Actor-Critic.

    Uses a sequence-capable network so the policy and value estimate can be
    conditioned on the full history via a learned hidden state. The update
    replays each buffered segment from a saved hidden state h0.

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
        """Initialize the instance.

        Args:
            network (dict): The network value.
            gamma (float): The gamma value. Defaults to ``0.999``.
            vf_coef (float): The vf coef value. Defaults to ``0.5``.
            ent_coef (float): The ent coef value. Defaults to ``0.01``.
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
        """Infer logits for AACAgent.

        Args:
            state (torch.Tensor): The state value.

        Returns:
            torch.Tensor: The computed or requested result.
        """
        logits, _ = self.net(state)
        return logits

    def infer_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer from seq for AACAgent.

        Args:
            state_seq (torch.Tensor): The state seq value.
            h0 (dict | None): The h0 value. Defaults to ``None``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed or requested result.
        """
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
        """Act for AACAgent.

        Args:
            state (torch.Tensor): The state value.
            valid_mask (torch.BoolTensor | None): The valid mask value. Defaults to ``None``.
            explore (bool): The explore value. Defaults to ``False``.
            grad_enabled (bool): The grad enabled value. Defaults to ``False``.

        Returns:
            torch.Tensor: The computed or requested result.
        """
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
        """Store one transition or payload in the buffer.

        Args:
            state (Any): The state value.
            action (Any): The action value.
            reward (Any): The reward value.
            done (Any): The done value.
            valid_mask (Any): The valid mask value. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
        self.buffer.store(state, action, reward, done, valid_mask=valid_mask)

    def init_optimizer(self, lr, optim="AdamW"):
        """Init optimizer for AACAgent.

        Args:
            lr (Any): The lr value.
            optim (Any): The optim value. Defaults to ``'AdamW'``.

        Returns:
            None: This function does not return a value.
        """
        opt_cls    = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

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
        """Save for AACAgent.

        Args:
            path (str): The path value.

        Returns:
            None: This function does not return a value.
        """
        torch.save({"network": self.net.state_dict()}, path)

    def load(self, path: str, strict: bool = True):
        """Load for AACAgent.

        Args:
            path (str): The path value.
            strict (bool): The strict value. Defaults to ``True``.

        Returns:
            None: This function does not return a value.
        """
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
        """Train on historical for AACAgent.

        Args:
            env (Any): The env value.
            data (Any): The data value.
            n_episodes (Any): The n episodes value.
            max_steps (Any): The max steps value. Defaults to ``2000``.
            warm_up (Any): The warm up value. Defaults to ``0``.
            update_interval (Any): The update interval value. Defaults to ``100``.
            burn_in_updates (Any): The burn in updates value. Defaults to ``0``.
            lr (Any): The lr value. Defaults to ``0.0003``.
            optim (Any): The optim value. Defaults to ``'AdamW'``.
            init_optimizer (Any): The init optimizer value. Defaults to ``False``.
            store_results (Any): The store results value. Defaults to ``True``.
            max_grad_norm (Any): The max grad norm value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
        """
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
        """Train on a BatchedMultiCurrencyEnv using pre-stacked tensor data.

        close, high, low, volume, open_ : (T, N) float tensors
        times                           : (T,)  float64 tensor of Unix timestamps

        Each episode picks B independent random start offsets. All B
        environments step together — market data indexed as data[starts + step].
        The buffer stores (B, ...) tensors per step; a single update() call
        averages gradients over both the time and batch dimensions.

        `burn_in_updates` skips the first N gradient updates after warm-up so
        the shared recurrent state can stabilise before training begins.

        Returns (total_loss, total_reward).

        Args:
            bat_env (Any): The bat env value.
            open_ (Any): The open value.
            close (Any): The close value.
            high (Any): The high value.
            low (Any): The low value.
            volume (Any): The volume value.
            times (Any): The times value.
            n_episodes (Any): The n episodes value.
            max_steps (Any): The max steps value. Defaults to ``2000``.
            warm_up (Any): The warm up value. Defaults to ``0``.
            update_interval (Any): The update interval value. Defaults to ``100``.
            burn_in_updates (Any): The burn in updates value. Defaults to ``1``.
            lr (Any): The lr value. Defaults to ``0.0003``.
            optim (Any): The optim value. Defaults to ``'AdamW'``.
            init_optimizer (Any): The init optimizer value. Defaults to ``False``.
            store_results (Any): The store results value. Defaults to ``True``.
            max_grad_norm (Any): The max grad norm value. Defaults to ``None``.

        Returns:
            Any: The computed or requested result.
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
                    actions      = self.act(obs, valid_mask=valid_mask, explore=True, grad_enabled=False)
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
