import copy
import torch
from torch import distributions

from src.agent.discrete.network import (
    ActionQNetwork,
    RecurrentActionQNetwork,
)


def _expected_q(logits: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return (probs * q_values).sum(dim=-1)


def _selected_q(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    actions_long = actions.long()
    return q_values.gather(-1, actions_long.unsqueeze(-1)).squeeze(-1)



UPDATE_METRIC_NAMES = (
    "policy_objective",
    "critic_loss",
    "entropy",
    "total_loss",
    "td_error_mean",
    "td_error_abs_mean",
    "td_error_std",
    "q_taken_mean",
    "target_mean",
    "v_pi_mean",
    "q_max_mean",
    "q_std_mean",
    "chosen_action_prob_mean",
    "policy_confidence_mean",
    "explained_variance",
    "adv_abs_mean",
    "adv_pos_frac",
)


def _empty_update_metrics() -> dict[str, list[float]]:
    return {name: [0.0] for name in UPDATE_METRIC_NAMES}


def _init_update_metrics() -> dict[str, list[float]]:
    return {name: [] for name in UPDATE_METRIC_NAMES}


def _explained_variance(targets: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    target_var = targets.var(unbiased=False)
    if float(target_var.detach().cpu().item()) <= 1e-12:
        return torch.zeros((), device=targets.device, dtype=targets.dtype)
    return 1.0 - residuals.var(unbiased=False) / (target_var + 1e-12)


def _append_update_metrics(
    metric_store: dict[str, list[float]],
    *,
    logits: torch.Tensor,
    log_probs: torch.Tensor,
    q_values_pred: torch.Tensor,
    q_taken_pred: torch.Tensor,
    targets: torch.Tensor,
    raw_advantages: torch.Tensor,
    policy_objective: torch.Tensor,
    critic_loss: torch.Tensor,
    entropy: torch.Tensor,
    loss: torch.Tensor,
):
    probs = torch.softmax(logits.detach(), dim=-1)
    q_values_detached = q_values_pred.detach()
    q_taken_detached = q_taken_pred.detach()
    targets_detached = targets.detach()
    raw_advantages_detached = raw_advantages.detach()

    v_pi_pred = (probs * q_values_detached).sum(dim=-1)
    q_max_pred = q_values_detached.max(dim=-1).values
    q_std_pred = q_values_detached.std(dim=-1, unbiased=False)
    td = targets_detached - q_taken_detached

    metrics = {
        "policy_objective": policy_objective.detach(),
        "critic_loss": critic_loss.detach(),
        "entropy": entropy.detach(),
        "total_loss": loss.detach(),
        "td_error_mean": td.mean(),
        "td_error_abs_mean": td.abs().mean(),
        "td_error_std": td.std(unbiased=False),
        "q_taken_mean": q_taken_detached.mean(),
        "target_mean": targets_detached.mean(),
        "v_pi_mean": v_pi_pred.mean(),
        "q_max_mean": q_max_pred.mean(),
        "q_std_mean": q_std_pred.mean(),
        "chosen_action_prob_mean": log_probs.detach().exp().mean(),
        "policy_confidence_mean": probs.max(dim=-1).values.mean(),
        "explained_variance": _explained_variance(targets_detached, td),
        "adv_abs_mean": raw_advantages_detached.abs().mean(),
        "adv_pos_frac": (raw_advantages_detached > 0).to(dtype=targets_detached.dtype).mean(),
    }
    for name, value in metrics.items():
        metric_store[name].append(float(value.cpu().item()))


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states = torch.stack(self.states).to(device=device, dtype=dtype)
        actions = torch.stack(self.actions).to(device=device, dtype=dtype)
        rewards = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones = torch.stack(self.dones).to(device=device, dtype=dtype)
        return states, actions, rewards, dones


class VAQAgent:
    """
    On-policy discrete actor-critic with a Q critic.

    The network shares a feedforward trunk and splits into:
      - an actor head that parameterizes a categorical policy
      - a critic head that predicts Q(s, .) for every discrete action

    Training uses TD(0) targets with the policy expectation as the bootstrap:
      - y_t = r_t + gamma * (1 - done_t) * V_pi(s_{t+1})
      - V_pi(s) = sum_a pi(a | s) Q(s, a)
      - A_t = Q(s_t, a_t) - V_pi(s_t)

    `update()` returns per-epoch metric lists for the actor objective, critic
    loss, entropy, TD-error statistics, Q/value scale, policy confidence,
    explained variance, and raw-advantage diagnostics.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # basic properties
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        hidden_dims_q: list[int] = [256],
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantages = normalize_advantages
        self.dtype = dtype
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = ActionQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_q=hidden_dims_q,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        self.net_t = copy.deepcopy(self.net)
        self.net_t.requires_grad_(False)

        self.buffer = RolloutBuffer()

    def infer_logits(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits

    def infer_logits_from_seq(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits

    def infer_dist_from_seq(
        self,
        state: torch.Tensor,
    ) -> tuple[distributions.Categorical, torch.Tensor]:
        logits, q_values = self.net(state)
        return distributions.Categorical(logits=logits), q_values

    def act(
        self,
        state: torch.Tensor,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        """Sample or greedily select a discrete action and return only the action tensor."""
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits = self.infer_logits(state)
            if explore:
                action = distributions.Categorical(logits=logits).sample().squeeze()
            else:
                action = logits.argmax(dim=-1).squeeze()

        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, done):
        self.buffer.store(state, action, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def update_net_t(self, tau: float = 0.005):
        """Polyak-average net_t toward net: θ_t ← (1-τ)θ_t + τθ."""
        with torch.no_grad():
            for p, p_t in zip(self.net.parameters(), self.net_t.parameters()):
                p_t.lerp_(p, tau)

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 1,
        max_grad_norm: float | None = None,
        target_tau: float = 0.005,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()

        advantages, raw_advantages, targets, states, actions = self.compute_td0_advantages(
            last_next_state=last_next_state,
        )

        for _ in range(k_epochs):
            dist, q_values_pred = self.infer_dist_from_seq(states)
            actions_long = actions.long()

            log_probs = dist.log_prob(actions_long)
            entropy = dist.entropy().mean()
            q_taken_pred = _selected_q(q_values_pred, actions_long)

            adv = advantages.detach()
            policy_objective = (log_probs * adv).mean()

            td = targets.detach() - q_taken_pred
            critic_loss = 0.5 * td.pow(2).mean()

            loss = -policy_objective + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

            self.optim.step()

            _append_update_metrics(
                metrics,
                logits=dist.logits,
                log_probs=log_probs,
                q_values_pred=q_values_pred,
                q_taken_pred=q_taken_pred,
                targets=targets,
                raw_advantages=raw_advantages,
                policy_objective=policy_objective,
                critic_loss=critic_loss,
                entropy=entropy,
                loss=loss,
            )

        self.update_net_t(target_tau)
        return metrics

    def compute_td0_advantages(
        self,
        last_next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)

        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > 1 else dones

        with torch.no_grad():
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            logits, q_values = self.net_t(torch.concat([states, last_next_state[None]], dim=0))

            logits_t = logits[:-1]
            logits_tp1 = logits[1:]
            q_t = q_values[:-1]
            q_tp1 = q_values[1:]

            q_taken = _selected_q(q_t, actions)
            v_t = _expected_q(logits_t, q_t)
            v_tp1 = _expected_q(logits_tp1, q_tp1)

            targets = rewards + self.gamma * (1.0 - dones) * v_tp1
            raw_advantages = q_taken - v_t
            advantages = raw_advantages

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, raw_advantages, targets, states, actions

    def save(self, path: str):
        torch.save({
            "network": self.net.state_dict(),
        }, path)

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
        n_updates=1,
        lr=3e-4,
        target_tau=0.005,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """Train on contiguous historical rollouts and collect episode-level metric histories."""
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])

            if store_results:
                episode_reward = []
                episode_info = []
                episode_loss = []

            self.buffer.clear()

            step = 1
            while True:
                if step > warm_up:
                    action = self.act(state.to_tensor(), explore=True, grad_enabled=False)
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
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = done or (step >= max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_state.to_tensor(),
                            k_epochs=n_updates,
                            max_grad_norm=max_grad_norm,
                            target_tau=target_tau,
                        )
                        self.buffer.clear()

                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break

                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                state = next_state
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - total reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}"
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
        n_updates=1,
        lr=3e-4,
        target_tau=0.005,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """
        Train on N parallel historical rollouts using VecMultiCurrencyEnv.

        At each step all N environments contribute one transition, so the buffer
        stores (N, ...) tensors per time-step.  A single update() call processes
        the whole (T, N, ...) batch, giving a gradient estimate that averages
        over both time and environments.  The episode ends as soon as any
        environment reaches done or max_steps is hit.

        Returns (total_loss, total_reward, total_info) where:
          total_loss[ep]   : list of loss_dicts, one per rollout update
          total_reward[ep] : list[list[float]], one inner list per env
          total_info[ep]   : list[list[dict]],  one inner list per env
        """
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        N = vec_env.n_envs
        dtype = vec_env.dtype
        data_len = len(data)

        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            starts = [torch.randint(data_len - max_steps, size=[1]).item() for _ in range(N)]
            obs, _ = vec_env.reset([data[starts[i]] for i in range(N)])
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(N)]
                ep_loss = []
                ep_info = [[] for _ in range(N)]

            step = 1
            while step <= max_steps:
                if step > warm_up:
                    actions = self.act(obs, explore=True, grad_enabled=False)
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
                        actions.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
                    )

                    rollout_ready = len(self.buffer) >= update_interval

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_obs,
                            k_epochs=n_updates,
                            max_grad_norm=max_grad_norm,
                            target_tau=target_tau,
                        )
                        self.buffer.clear()

                        if store_results:
                            ep_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - mean reward: {rewards.mean():.5f}"
                        if store_results:
                            msg += f" - portfolio: {sum(d['V'] for d in infos) / N:.2f}"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                obs = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)
                total_info.append(ep_info)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = f"episode {episode} - total reward: {sum(all_ep_rewards):.5f} ({N} envs)"
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_bat(
        self,
        bat_env,
        close, high, low, volume, times,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=1,
        lr=3e-4,
        target_tau=0.005,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """
        Train on a BatchedMultiCurrencyEnv using pre-stacked tensor data.

        close, high, low, volume : (T, N) float tensors
        times                    : (T,)  float64 tensor of Unix timestamps

        Each episode picks B independent random start offsets. All B
        environments step together — market data indexed as data[starts + step].
        The buffer stores (B, ...) tensors per step; a single update() call
        averages gradients over both the time and batch dimensions.

        Returns (total_loss, total_reward).
        """
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        B = bat_env.B
        dtype = bat_env.dtype
        T = close.shape[0]

        total_loss = []
        total_reward = []

        for episode in range(1, n_episodes + 1):
            starts = torch.randint(T - max_steps, (B,))
            obs = bat_env.reset(
                close[starts], high[starts], low[starts], volume[starts], times[starts],
            )
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(B)]
                ep_loss = []

            step = 1
            while step <= max_steps:
                si = starts + step

                if step > warm_up:
                    actions = self.act(obs, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions = None
                    step_actions = torch.zeros(B, dtype=torch.long)

                next_obs, rewards, dones = bat_env.step(
                    step_actions, close[si], high[si], low[si], volume[si], times[si],
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
                    )

                    rollout_ready = len(self.buffer) >= update_interval

                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_obs,
                            k_epochs=n_updates,
                            max_grad_norm=max_grad_norm,
                            target_tau=target_tau,
                        )
                        self.buffer.clear()

                        if store_results:
                            ep_loss.append(loss_dict)

                        msg = (
                            f"episode {episode} [{100*step/max_steps:.1f}%]"
                            f" - mean reward: {rewards.mean():.5f}"
                            f" - mean V: {bat_env.V.mean():.2f}"
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
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                obs = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = f"episode {episode} - total reward: {sum(all_ep_rewards):.5f} ({B} envs)"
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward


class RecurrentVAQAgent:
    """
    On-policy recurrent discrete actor-critic with a Q critic.

    The recurrent network shares a recurrent trunk and splits into:
      - an actor head that parameterizes a categorical policy
      - a critic head that predicts Q(s, .) for every discrete action

    Training replays each buffered segment from a saved recurrent state `h0` and
    uses the same TD(0) target and advantage construction as `VAQAgent`:
      - y_t = r_t + gamma * (1 - done_t) * V_pi(s_{t+1})
      - V_pi(s) = sum_a pi(a | s) Q(s, a)
      - A_t = Q(s_t, a_t) - V_pi(s_t)

    `update()` returns the same diagnostic metric set as the feedforward agent.
    In `train_on_historical()`, `burn_in_updates` skips the first N optimizer
    updates after warm-up so the recurrent state can move away from the reset
    state before training begins.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # vaq properties
        gamma: float = 0.999,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        hidden_dims_q: list[int] = [256],
        activation: callable = torch.relu,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantages = normalize_advantages
        self.dtype = dtype
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = RecurrentActionQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_q=hidden_dims_q,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
        ).to(device=self.device, dtype=self.dtype)

        self.net_t = copy.deepcopy(self.net)
        self.net_t.requires_grad_(False)

        self.buffer = RolloutBuffer()
        self.net.reset(1)
        self.net_t.reset(1)

    def infer_logits(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(state)
        return logits

    def infer_logits_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ) -> torch.Tensor:
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        logits_seq, _ = self.net.forward_seq(state_seq)
        return logits_seq

    def infer_dist(
        self,
        state: torch.Tensor,
    ) -> distributions.Categorical:
        return distributions.Categorical(logits=self.infer_logits(state))

    def infer_dist_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[distributions.Categorical, torch.Tensor]:
        """
        Full-sequence inference using the CuDNN-accelerated forward_seq path.

        state_seq: (T, state_dim)

        Returns:
          dist  : Categorical over (T, action_dim) logits
          q_seq : (T, action_dim) Q-values
        """
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        logits_seq, q_seq = self.net.forward_seq(state_seq)   # (T, A), (T, A)

        return distributions.Categorical(logits=logits_seq), q_seq

    def act(
        self,
        state: torch.Tensor,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> torch.Tensor:
        """Sample or greedily select a discrete action and return only the action tensor."""
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            logits = self.infer_logits(state)
            if explore:
                action = distributions.Categorical(logits=logits).sample().squeeze()
            else:
                action = logits.argmax(dim=-1).squeeze()

        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, done):
        self.buffer.store(state, action, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def update_net_t(self, tau: float = 0.005):
        """Polyak-average net_t toward net: θ_t ← (1-τ)θ_t + τθ."""
        with torch.no_grad():
            for p, p_t in zip(self.net.parameters(), self.net_t.parameters()):
                p_t.lerp_(p, tau)

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 1,
        h0: dict | None = None,
        max_grad_norm: float | None = None,
        target_tau: float = 0.005,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()

        advantages, raw_advantages, targets, states, actions = self.compute_td0_advantages(
            last_next_state=last_next_state,
            h0=h0,
        )

        for _ in range(k_epochs):
            dists, q_values_pred = self.infer_dist_from_seq(states, h0=h0)
            actions_long = actions.long()

            log_probs = dists.log_prob(actions_long)
            entropy = dists.entropy().mean()
            q_taken_pred = _selected_q(q_values_pred, actions_long)

            adv = advantages.detach()
            policy_objective = (log_probs * adv).mean()

            td = targets.detach() - q_taken_pred
            critic_loss = 0.5 * td.pow(2).mean()

            loss = -policy_objective + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

            self.optim.step()

            _append_update_metrics(
                metrics,
                logits=dists.logits,
                log_probs=log_probs,
                q_values_pred=q_values_pred,
                q_taken_pred=q_taken_pred,
                targets=targets,
                raw_advantages=raw_advantages,
                policy_objective=policy_objective,
                critic_loss=critic_loss,
                entropy=entropy,
                loss=loss,
            )

        self.update_net_t(target_tau)
        return metrics

    def compute_td0_advantages(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            if h0 is not None:
                self.net_t.set_states(h0, strict=False)
            logits_seq, q_values = self.net_t.forward_seq(
                torch.concat([states, last_next_state[None]], dim=0)
            )

            logits_t = logits_seq[:-1]
            logits_tp1 = logits_seq[1:]
            q_t = q_values[:-1]
            q_tp1 = q_values[1:]

            q_taken = _selected_q(q_t, actions)
            v_t     = _expected_q(logits_t, q_t)
            v_tp1   = _expected_q(logits_tp1, q_tp1)

            targets = rewards + self.gamma * (1.0 - dones) * v_tp1
            raw_advantages = q_taken - v_t
            advantages = raw_advantages

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, raw_advantages, targets, states, actions

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=1,
        burn_in_updates=0,
        lr=3e-4,
        target_tau=0.005,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """
        Train on contiguous historical rollouts with recurrent state carry-over.

        `burn_in_updates` skips the first N ready-to-train rollout updates in
        each episode while still advancing the true recurrent state.
        """
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])
            burn_in = burn_in_updates
            self.net.reset()
            self.buffer.clear()
            h0 = self.net.get_states(clone=True, detach=True)

            if store_results:
                episode_reward = []
                episode_loss = []
                episode_info = []

            step = 1
            while True:
                if step > warm_up:
                    action = self.act(state.to_tensor(), explore=True, grad_enabled=False)
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
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = done or (step >= max_steps)

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
                                target_tau=target_tau
                            )
                            if store_results:
                                episode_loss.append(loss_dict)

                            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        self.net.set_states(h0 := h_t, clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break

                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                state = next_state
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - total reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}"
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
        n_updates=1,
        burn_in_updates=0,
        lr=3e-4,
        target_tau=0.005,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """
        Train on N parallel historical rollouts using VecMultiCurrencyEnv.

        At each step all N environments contribute one transition, so the buffer
        stores (N, ...) tensors per time-step.  forward_seq then processes the
        full (T, N, ...) batch via the fused RNN kernel in one call, and a single
        update() averages the gradient estimate over both time and environments.
        The episode ends as soon as any environment reaches done or max_steps.

        `burn_in_updates` skips the first N gradient updates after warm-up so
        the shared hidden state can move away from the zero-initialised state.

        Returns (total_loss, total_reward, total_info) where:
          total_loss[ep]   : list of loss_dicts, one per rollout update
          total_reward[ep] : list[list[float]], one inner list per env
          total_info[ep]   : list[list[dict]],  one inner list per env
        """
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        N = vec_env.n_envs
        dtype = vec_env.dtype
        data_len = len(data)

        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            self.net.reset(N)
            h0 = self.net.get_states(clone=True, detach=True)
            burn_in = burn_in_updates

            starts = [torch.randint(data_len - max_steps, size=[1]).item() for _ in range(N)]
            obs, _ = vec_env.reset([data[starts[i]] for i in range(N)])
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(N)]
                ep_loss = []
                ep_info = [[] for _ in range(N)]

            step = 1
            while step <= max_steps:
                if step > warm_up:
                    actions = self.act(obs, explore=True, grad_enabled=False)
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
                        actions.detach(),
                        rewards.to(dtype=dtype),
                        dones.to(dtype=dtype),
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
                                target_tau=target_tau
                            )
                            if store_results:
                                ep_loss.append(loss_dict)

                            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - mean reward: {rewards.mean():.5f}"
                            if store_results:
                                msg += f" - mean portfolio: {sum(infos[i]['V'] for i in range(N)) / N:.2f}"
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        # Restore the live hidden state (update's forward_seq rewrites it)
                        self.net.set_states(h_t, clone=True, detach=True)
                        h0 = self.net.get_states(clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                obs = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)
                total_info.append(ep_info)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = f"episode {episode} - total reward: {sum(all_ep_rewards):.5f} ({N} envs)"
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_bat(
        self,
        bat_env,
        close, high, low, volume, times,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=1,
        burn_in_updates=0,
        lr=3e-4,
        target_tau=0.005,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        """
        Train on a BatchedMultiCurrencyEnv using pre-stacked tensor data.

        close, high, low, volume : (T, N) float tensors
        times                    : (T,)  float64 tensor of Unix timestamps

        Each episode picks B independent random start offsets. All B
        environments step together — market data indexed as data[starts + step].
        The buffer stores (B, ...) tensors per step; a single update() call
        averages gradients over both the time and batch dimensions.

        `burn_in_updates` skips the first N gradient updates after warm-up so
        the shared recurrent state can stabilise before training begins.

        Returns (total_loss, total_reward).
        """
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

            starts = torch.randint(T - max_steps, (B,))
            obs = bat_env.reset(
                close[starts], high[starts], low[starts], volume[starts], times[starts],
            )
            self.buffer.clear()

            if store_results:
                ep_reward = [[] for _ in range(B)]
                ep_loss = []

            step = 1
            while step <= max_steps:
                si = starts + step

                if step > warm_up:
                    actions = self.act(obs, explore=True, grad_enabled=False)
                    step_actions = actions
                else:
                    actions = None
                    step_actions = torch.zeros(B, dtype=torch.long)

                next_obs, rewards, dones = bat_env.step(
                    step_actions, close[si], high[si], low[si], volume[si], times[si],
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
                                target_tau=target_tau,
                            )
                            if store_results:
                                ep_loss.append(loss_dict)

                            msg = (
                                f"episode {episode} [{100*step/max_steps:.1f}%]"
                                f" - avg. reward: {rewards.mean():.5f}"
                                f" - avg. portfolio: {bat_env.V.mean():.2f}"
                            )
                            for key, val in loss_dict.items():
                                msg += f" - {key}: {sum(val) / len(val):.5f}"
                            print(msg, end="\r")

                        # Restore the live hidden state (update's forward_seq rewrites it)
                        self.net.set_states(h_t, clone=True, detach=True)
                        h0 = self.net.get_states(clone=True, detach=True)
                        self.buffer.clear()

                    if terminal:
                        break
                else:
                    if terminal:
                        break
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                obs = next_obs
                step += 1

            if store_results:
                total_loss.append(ep_loss)
                total_reward.append(ep_reward)

            all_ep_rewards = [r for env_r in ep_reward for r in env_r] if store_results else []
            msg = (
                f"episode {episode} [{100*step/max_steps:.1f}%]"
                f" - avg. reward: {sum(all_ep_rewards) / B:.5f}"
                f" - avg. portfolio: {bat_env.V.mean():.2f}"
            )
            if store_results and len(ep_loss) > 0:
                keys = ep_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in ep_loss) / len(ep_loss):.5f}"
            print(msg)

        return total_loss, total_reward

    def save(self, path: str):
        torch.save({
            "network": self.net.state_dict(),
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)
