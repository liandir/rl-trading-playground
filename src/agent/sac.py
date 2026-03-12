import copy
import math
from collections import deque
import random

import torch
from torch import distributions

from src.agent.buffer import Buffer
from src.network.recurrent import RecurrentNetwork, _build_recurrent_cell
from src.network.vanilla import VanillaNetwork


def _as_hidden_dims(hidden_dims) -> list[int]:
    return list(hidden_dims) if hidden_dims is not None else []


def _squeeze_single_batch(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0) if x.dim() > 1 and x.shape[0] == 1 else x


def _categorical_stats(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, log_probs, entropy


class SequenceReplayBuffer:
    """
    Episode-aware replay for recurrent agents.

    Samples contiguous windows that include up to `burn_in` context steps
    before a train segment of length `train_length`. Loss masking is used
    to ignore the context portion and any right-padding at the tail.
    """

    def __init__(self, memory_size: int):
        self.memory_size = int(memory_size)
        self.episodes = deque()
        self.num_transitions = 0
        self.current_observations = []
        self.current_actions = []
        self.current_rewards = []
        self.current_dones = []

    def __len__(self):
        return self.num_transitions + len(self.current_actions)

    def clear(self):
        self.episodes.clear()
        self.num_transitions = 0
        self.current_observations.clear()
        self.current_actions.clear()
        self.current_rewards.clear()
        self.current_dones.clear()

    @staticmethod
    def _to_cpu_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return torch.as_tensor(value)

    def store(self, state, action, next_state, reward, done):
        state_t = self._to_cpu_tensor(state)
        action_t = self._to_cpu_tensor(action).view(())
        next_state_t = self._to_cpu_tensor(next_state)
        reward_t = self._to_cpu_tensor(reward).view(())
        done_t = self._to_cpu_tensor(done).view(())

        if not self.current_observations:
            self.current_observations.append(state_t)

        self.current_actions.append(action_t)
        self.current_rewards.append(reward_t)
        self.current_dones.append(done_t)
        self.current_observations.append(next_state_t)

        if bool(done_t.item()):
            self.end_episode()

    def end_episode(self):
        if not self.current_actions:
            self.current_observations.clear()
            return

        episode = {
            "observations": torch.stack(self.current_observations),
            "actions": torch.stack(self.current_actions).long(),
            "rewards": torch.stack(self.current_rewards),
            "dones": torch.stack(self.current_dones),
        }
        self.episodes.append(episode)
        self.num_transitions += int(episode["actions"].shape[0])

        while self.num_transitions > self.memory_size and self.episodes:
            removed = self.episodes.popleft()
            self.num_transitions -= int(removed["actions"].shape[0])

        self.current_observations.clear()
        self.current_actions.clear()
        self.current_rewards.clear()
        self.current_dones.clear()

    def _episode_views(self) -> list[dict[str, torch.Tensor]]:
        views = list(self.episodes)
        if self.current_actions:
            views.append(
                {
                    "observations": torch.stack(self.current_observations),
                    "actions": torch.stack(self.current_actions).long(),
                    "rewards": torch.stack(self.current_rewards),
                    "dones": torch.stack(self.current_dones),
                }
            )
        return [episode for episode in views if int(episode["actions"].shape[0]) > 0]

    def sample_sequences(
        self,
        batch_size: int,
        burn_in: int,
        train_length: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episodes = self._episode_views()
        if not episodes:
            raise RuntimeError("Cannot sample from an empty sequence replay buffer.")

        total_steps = int(burn_in) + int(train_length)
        obs_shape = tuple(episodes[0]["observations"].shape[1:])

        observations = torch.zeros((batch_size, total_steps + 1, *obs_shape), device=device, dtype=dtype)
        actions = torch.zeros((batch_size, total_steps), device=device, dtype=torch.long)
        rewards = torch.zeros((batch_size, total_steps), device=device, dtype=dtype)
        dones = torch.zeros((batch_size, total_steps), device=device, dtype=dtype)
        loss_mask = torch.zeros((batch_size, total_steps), device=device, dtype=dtype)

        episode_indices = random.choices(
            range(len(episodes)),
            weights=[int(episode["actions"].shape[0]) for episode in episodes],
            k=batch_size,
        )

        for batch_idx, episode_idx in enumerate(episode_indices):
            episode = episodes[episode_idx]
            episode_len = int(episode["actions"].shape[0])
            train_start = random.randrange(episode_len)

            segment_start = max(0, train_start - int(burn_in))
            segment_end = min(episode_len, train_start + int(train_length))

            num_steps = segment_end - segment_start
            loss_offset = train_start - segment_start
            loss_steps = segment_end - train_start

            obs_segment = episode["observations"][segment_start : segment_end + 1].to(device=device, dtype=dtype)
            action_segment = episode["actions"][segment_start:segment_end].to(device=device, dtype=torch.long)
            reward_segment = episode["rewards"][segment_start:segment_end].to(device=device, dtype=dtype)
            done_segment = episode["dones"][segment_start:segment_end].to(device=device, dtype=dtype)

            observations[batch_idx, : obs_segment.shape[0]] = obs_segment
            actions[batch_idx, :num_steps] = action_segment
            rewards[batch_idx, :num_steps] = reward_segment
            dones[batch_idx, :num_steps] = done_segment
            loss_mask[batch_idx, loss_offset : loss_offset + loss_steps] = 1.0

        return observations, actions, rewards, dones, loss_mask

    def state_dict(self) -> dict:
        return {
            "episodes": list(self.episodes),
            "num_transitions": self.num_transitions,
            "current_observations": list(self.current_observations),
            "current_actions": list(self.current_actions),
            "current_rewards": list(self.current_rewards),
            "current_dones": list(self.current_dones),
            "memory_size": self.memory_size,
        }

    def load_state_dict(self, state_dict: dict):
        self.memory_size = int(state_dict.get("memory_size", self.memory_size))
        self.episodes = deque(state_dict.get("episodes", []))
        self.num_transitions = int(
            state_dict.get(
                "num_transitions",
                sum(int(episode["actions"].shape[0]) for episode in self.episodes),
            )
        )
        self.current_observations = list(state_dict.get("current_observations", []))
        self.current_actions = list(state_dict.get("current_actions", []))
        self.current_rewards = list(state_dict.get("current_rewards", []))
        self.current_dones = list(state_dict.get("current_dones", []))


class DiscreteSACNetwork(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        hidden_dims_actor: tuple[int, ...] | list[int] | None = None,
        hidden_dims_critic: tuple[int, ...] | list[int] | None = None,
        activation: callable = torch.relu,
    ):
        super().__init__()
        hidden_dims = _as_hidden_dims(hidden_dims)
        hidden_dims_actor = _as_hidden_dims(hidden_dims_actor)
        hidden_dims_critic = _as_hidden_dims(hidden_dims_critic)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(torch.nn.Linear(prev, hidden_dim))
            prev = hidden_dim

        self.actor = VanillaNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
        )
        self.q1 = VanillaNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_critic,
            activation=activation,
        )
        self.q2 = VanillaNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_critic,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = self.activation(layer(z))
        return z

    def pi(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.forward(x))

    def q(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward(x)
        return self.q1(z), self.q2(z)

    def piq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.forward(x)
        return self.actor(z), self.q1(z), self.q2(z)


class RecurrentDiscreteSACNetwork(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int] | None = None,
        hidden_dims_actor: tuple[int, ...] | list[int] | None = None,
        hidden_dims_critic: tuple[int, ...] | list[int] | None = None,
        activation: callable = torch.relu,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
    ):
        super().__init__()
        hidden_dims = _as_hidden_dims(hidden_dims)
        hidden_dims_actor = _as_hidden_dims(hidden_dims_actor)
        hidden_dims_critic = _as_hidden_dims(hidden_dims_critic)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = activation
        self.recurrent_type = recurrent_type
        self.recurrent_kwargs = dict(recurrent_kwargs or {})

        self.layers = torch.nn.ModuleList()
        prev = state_dim
        for hidden_dim in hidden_dims:
            self.layers.append(
                _build_recurrent_cell(prev, hidden_dim, activation, recurrent_type, self.recurrent_kwargs)
            )
            prev = hidden_dim

        self.actor = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_actor,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )
        self.q1 = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_critic,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )
        self.q2 = RecurrentNetwork(
            prev,
            action_dim,
            hidden_dims=hidden_dims_critic,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=self.recurrent_kwargs,
        )

    def reset(self, batch_size: int = 1):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for layer in self.layers:
            layer.reset(batch_size, device=device, dtype=dtype)
        self.actor.reset(batch_size, device=device, dtype=dtype)
        self.q1.reset(batch_size, device=device, dtype=dtype)
        self.q2.reset(batch_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.view(-1, self.state_dim)
        for layer in self.layers:
            z = layer(z)
        return z

    def pi(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.forward(x))

    def q(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.forward(x)
        return self.q1(z), self.q2(z)

    def piq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.forward(x)
        return self.actor(z), self.q1(z), self.q2(z)

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        trunk = [layer.get_state(clone=clone, detach=detach) for layer in self.layers]
        actor = self.actor.get_states(clone=clone, detach=detach)
        q1 = self.q1.get_states(clone=clone, detach=detach)
        q2 = self.q2.get_states(clone=clone, detach=detach)
        return {"trunk": trunk, "actor": actor, "q1": q1, "q2": q2}

    def set_states(self, states: dict | None, clone: bool = True, detach: bool = True, strict: bool = True):
        if states is None:
            return

        if strict:
            for key in ("trunk", "actor", "q1", "q2"):
                if key not in states:
                    raise KeyError(f"Missing key '{key}' in states snapshot.")

        trunk_states = states.get("trunk", [])
        actor_states = states.get("actor", [])
        q1_states = states.get("q1", [])
        q2_states = states.get("q2", [])

        if strict and len(trunk_states) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} trunk states, got {len(trunk_states)}.")

        for layer, state in zip(self.layers, trunk_states):
            layer.set_state(state, clone=clone, detach=detach)

        self.actor.set_states(actor_states, clone=clone, detach=detach, strict=strict)
        self.q1.set_states(q1_states, clone=clone, detach=detach, strict=strict)
        self.q2.set_states(q2_states, clone=clone, detach=detach, strict=strict)


class _BaseDiscreteSACAgent:
    def __init__(
        self,
        network: torch.nn.Module,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.999,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float | None = None,
        batch_size: int = 64,
        replay_capacity: int = 200_000,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.autotune_alpha = bool(autotune_alpha)
        self.dtype = dtype
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = network.to(device=self.device, dtype=self.dtype)
        self.target_net = copy.deepcopy(self.net).to(device=self.device, dtype=self.dtype)
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad_(False)

        self.buffer = Buffer(replay_capacity)

        if target_entropy is None:
            target_entropy = 0.98 * math.log(action_dim) if action_dim > 1 else 0.0
        self.target_entropy = float(target_entropy)

        if self.autotune_alpha:
            self.log_alpha = torch.nn.Parameter(
                torch.tensor(math.log(max(alpha, 1e-8)), device=self.device, dtype=self.dtype)
            )
            self.alpha_opt = None
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self._alpha_fixed = float(alpha)

        self.optim = None
        self.reset()

    @property
    def alpha(self) -> torch.Tensor:
        if self.autotune_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self._alpha_fixed, device=self.device, dtype=self.dtype)

    def reset(self, batch_size: int = 1):
        if hasattr(self.net, "reset"):
            self.net.reset(batch_size)
        if hasattr(self.target_net, "reset"):
            self.target_net.reset(batch_size)

    def _snapshot_rollout_state(self):
        if hasattr(self.net, "get_states"):
            return self.net.get_states(clone=True, detach=True)
        return None

    def _restore_rollout_state(self, states):
        if states is not None and hasattr(self.net, "set_states"):
            self.net.set_states(states, clone=True, detach=True, strict=False)

    def _reset_learning_state(self, batch_size: int):
        if hasattr(self.net, "reset"):
            self.net.reset(batch_size)

    def _reset_target_state(self, batch_size: int):
        if hasattr(self.target_net, "reset"):
            self.target_net.reset(batch_size)

    def infer_dist(self, state: torch.Tensor) -> distributions.Categorical:
        logits = _squeeze_single_batch(self.net.pi(state))
        return distributions.Categorical(logits=logits)

    def infer_q(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = self.net.q(state)
        return _squeeze_single_batch(q1), _squeeze_single_batch(q2)

    def infer_dist_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ) -> distributions.Categorical:
        if h0 is not None and hasattr(self.net, "set_states"):
            self.net.set_states(h0, strict=False)

        logits = []
        for state in state_seq:
            logits.append(_squeeze_single_batch(self.net.pi(state)))
        return distributions.Categorical(logits=torch.stack(logits, dim=0))

    def infer_q_from_seq(
        self,
        state_seq: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h0 is not None and hasattr(self.net, "set_states"):
            self.net.set_states(h0, strict=False)

        q1_values = []
        q2_values = []
        for state in state_seq:
            q1, q2 = self.net.q(state)
            q1_values.append(_squeeze_single_batch(q1))
            q2_values.append(_squeeze_single_batch(q2))

        return torch.stack(q1_values, dim=0), torch.stack(q2_values, dim=0)

    def act(self, state: torch.Tensor, explore: bool = True, grad_enabled: bool = False) -> torch.Tensor:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            dist = self.infer_dist(state)
            if explore:
                action = dist.sample().squeeze()
            else:
                action = torch.argmax(dist.logits, dim=-1).squeeze()

        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state_t = state.detach().cpu()
        else:
            state_t = torch.tensor(state)

        if isinstance(action, torch.Tensor):
            action_t = action.detach().view(()).cpu()
        else:
            action_t = torch.tensor(action).view(())

        if isinstance(next_state, torch.Tensor):
            next_state_t = next_state.detach().cpu()
        else:
            next_state_t = torch.tensor(next_state)

        if isinstance(reward, torch.Tensor):
            reward_t = reward.detach().view(()).cpu()
        else:
            reward_t = torch.tensor(reward).view(())

        if isinstance(done, torch.Tensor):
            done_t = done.detach().view(()).cpu()
        else:
            done_t = torch.tensor(done).view(())

        self.buffer.store(state_t, action_t, next_state_t, reward_t, done_t)

    def init_optimizer(self, lr, alpha_lr: float | None = None, optim: str = "AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

        if self.autotune_alpha:
            self.alpha_opt = opt_cls([self.log_alpha], lr=lr if alpha_lr is None else alpha_lr)

    @torch.no_grad()
    def soft_update(self):
        for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def update(
        self,
        batch_size: int | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, float]:
        if self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        batch_size = self.batch_size if batch_size is None else int(batch_size)
        if len(self.buffer) < batch_size:
            return {
                "policy_objective": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "alpha": float(self.alpha.detach().cpu().item()),
                "alpha_loss": 0.0,
                "q_mean": 0.0,
                "td_error": 0.0,
            }

        rollout_state = self._snapshot_rollout_state()

        states, actions, next_states, rewards, dones = self.buffer.sample(batch_size, self.device, self.dtype)
        actions = actions.long().view(-1)
        rewards = rewards.view(-1)
        dones = dones.view(-1)

        batch_len = states.shape[0]
        alpha = self.alpha.detach()

        with torch.no_grad():
            self._reset_learning_state(batch_len)
            next_logits = self.net.pi(next_states)
            next_probs, next_log_probs, _ = _categorical_stats(next_logits)

            self._reset_target_state(batch_len)
            next_q1, next_q2 = self.target_net.q(next_states)
            next_q = torch.minimum(next_q1, next_q2)
            next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
            targets = rewards + self.gamma * (1.0 - dones) * next_v

        self._reset_learning_state(batch_len)
        logits, q1_all, q2_all = self.net.piq(states)
        probs, log_probs, entropy = _categorical_stats(logits)

        q1_pred = q1_all.gather(1, actions[:, None]).squeeze(-1)
        q2_pred = q2_all.gather(1, actions[:, None]).squeeze(-1)

        q1_error = targets - q1_pred
        q2_error = targets - q2_pred
        critic_loss = 0.5 * (q1_error.pow(2).mean() + q2_error.pow(2).mean())

        min_q = torch.minimum(q1_all, q2_all).detach()
        policy_objective = (probs * (min_q - alpha * log_probs)).sum(dim=-1).mean()
        entropy_mean = entropy.mean()

        loss = -policy_objective + critic_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

        self.optim.step()

        alpha_loss_value = 0.0
        if self.autotune_alpha and self.alpha_opt is not None:
            alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.detach().cpu().item())

        self.soft_update()
        self._restore_rollout_state(rollout_state)

        td_error = 0.5 * (q1_error.abs() + q2_error.abs())
        q_mean = 0.5 * (q1_pred.mean() + q2_pred.mean())

        return {
            "policy_objective": float(policy_objective.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "alpha_loss": alpha_loss_value,
            "q_mean": float(q_mean.detach().cpu().item()),
            "td_error": float(td_error.mean().detach().cpu().item()),
        }

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps: int = 2000,
        warm_up: int = 1000,
        update_interval: int = 1,
        n_updates: int = 1,
        updates_per_interval: int | None = None,
        batch_size: int = 32,
        lr: float = 3e-4,
        alpha_lr: float | None = None,
        optim: str = "AdamW",
        init_optimizer: bool = False,
        store_results: bool = True,
        max_grad_norm: float | None = None,
    ):
        if updates_per_interval is not None:
            n_updates = int(updates_per_interval)

        if self.optim is None or init_optimizer:
            self.init_optimizer(lr, alpha_lr=alpha_lr, optim=optim)

        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            if hasattr(self.buffer, "end_episode"):
                self.buffer.end_episode()

            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])
            self.reset()

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
                        action.detach() if isinstance(action, torch.Tensor) else torch.tensor(action),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        next_state.to_tensor().detach(),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    train_ready = len(self.buffer) >= batch_size
                    terminal = done or (step >= max_steps)

                    if train_ready and ((step % update_interval == 1) or terminal):
                        batch_losses = []
                        for _ in range(n_updates):
                            batch_losses.append(self.update(batch_size=batch_size, max_grad_norm=max_grad_norm))

                        avg_loss_dict = {
                            key: sum(item[key] for item in batch_losses) / len(batch_losses)
                            for key in batch_losses[0]
                        }

                        if store_results:
                            episode_loss.append(avg_loss_dict)

                        msg = (
                            f"episode {episode} [{100 * step / max_steps:.1f}%] - "
                            f"reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
                        )
                        for key, value in avg_loss_dict.items():
                            msg += f" - {key}: {value:.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break

                else:
                    if step % update_interval == 0:
                        print(
                            f"episode {episode} [{100 * step / max_steps:.1f}%] - warm up in progress...",
                            end="\r",
                        )

                state = next_state
                step += 1

            if hasattr(self.buffer, "end_episode"):
                self.buffer.end_episode()

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = (
                f"episode {episode} [{100 * step / max_steps:.1f}%] - "
                f"avg. reward: {sum(episode_reward) / len(episode_reward) if store_results else 0.0:.5f} - "
                f"portfolio: {info['V']:.2f}"
            )
            if store_results and len(episode_loss) > 0:
                for key in episode_loss[0]:
                    msg += f" - {key}: {sum(loss_dict[key] for loss_dict in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def save(self, path: str, include_buffer: bool = False):
        checkpoint = {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optim.state_dict() if self.optim is not None else None,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "autotune_alpha": self.autotune_alpha,
            "target_entropy": self.target_entropy,
            "alpha": float(self.alpha.detach().cpu().item()),
            "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
            "alpha_optimizer": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
        }

        if include_buffer:
            checkpoint["replay_buffer"] = {
                "memory": self.buffer.memory,
                "idx": self.buffer.idx,
                "full": self.buffer.full,
                "memory_size": self.buffer.memory_size,
            }

        torch.save(checkpoint, path)

    def load(self, path: str, map_location=None, load_buffer: bool = False):
        checkpoint = torch.load(path, map_location=map_location or self.device)

        self.net.load_state_dict(checkpoint["net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if self.optim is not None and checkpoint["optimizer"] is not None:
            self.optim.load_state_dict(checkpoint["optimizer"])

        if self.autotune_alpha and checkpoint.get("log_alpha") is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(device=self.device, dtype=self.dtype))
            if self.alpha_opt is not None and checkpoint.get("alpha_optimizer") is not None:
                self.alpha_opt.load_state_dict(checkpoint["alpha_optimizer"])
        elif not self.autotune_alpha:
            self._alpha_fixed = float(checkpoint.get("alpha", self._alpha_fixed))

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.tau = checkpoint.get("tau", self.tau)
        self.batch_size = int(checkpoint.get("batch_size", self.batch_size))
        self.target_entropy = float(checkpoint.get("target_entropy", self.target_entropy))

        if load_buffer and "replay_buffer" in checkpoint:
            replay_buffer = checkpoint["replay_buffer"]
            self.buffer.memory = replay_buffer["memory"]
            self.buffer.idx = replay_buffer["idx"]
            self.buffer.full = replay_buffer["full"]
            self.buffer.memory_size = replay_buffer.get("memory_size", self.buffer.memory_size)

        self.net.to(self.device, dtype=self.dtype)
        self.target_net.to(self.device, dtype=self.dtype)
        self.reset()


class SACAgent(_BaseDiscreteSACAgent):
    """
    Soft Actor-Critic for discrete action spaces using:
      - categorical policy over actions
      - twin Q heads
      - exact expectation over the discrete policy
      - soft target updates
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.999,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float | None = None,
        batch_size: int = 64,
        replay_capacity: int = 200_000,
        hidden_dims: tuple[int, ...] = (256, 256),
        hidden_dims_actor: tuple[int, ...] = (),
        hidden_dims_critic: tuple[int, ...] = (),
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        network = DiscreteSACNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_critic=hidden_dims_critic,
            activation=activation,
        )
        super().__init__(
            network=network,
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            autotune_alpha=autotune_alpha,
            target_entropy=target_entropy,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            dtype=dtype,
            device=device,
        )


class RecurrentSACAgent(_BaseDiscreteSACAgent):
    """
    Recurrent discrete SAC with one shared recurrent trunk and separate
    categorical policy / twin-Q heads.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.999,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float | None = None,
        batch_size: int = 64,
        replay_capacity: int = 200_000,
        sequence_length: int = 16,
        burn_in: int = 128,
        hidden_dims: tuple[int, ...] = (256, 256),
        hidden_dims_actor: tuple[int, ...] = (),
        hidden_dims_critic: tuple[int, ...] = (),
        activation: callable = torch.relu,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        network = RecurrentDiscreteSACNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_critic=hidden_dims_critic,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
        )
        super().__init__(
            network=network,
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            autotune_alpha=autotune_alpha,
            target_entropy=target_entropy,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            dtype=dtype,
            device=device,
        )
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")

        self.buffer = SequenceReplayBuffer(replay_capacity)

    def _unroll_piq_sequence(
        self,
        network: RecurrentDiscreteSACNetwork,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_seq = []
        q1_seq = []
        q2_seq = []

        for obs_t in observations.unbind(dim=1):
            logits_t, q1_t, q2_t = network.piq(obs_t)
            logits_seq.append(logits_t)
            q1_seq.append(q1_t)
            q2_seq.append(q2_t)

        return (
            torch.stack(logits_seq, dim=1),
            torch.stack(q1_seq, dim=1),
            torch.stack(q2_seq, dim=1),
        )

    def update(
        self,
        batch_size: int | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, float]:
        if self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        batch_size = self.batch_size if batch_size is None else int(batch_size)
        if len(self.buffer) < batch_size:
            return {
                "policy_objective": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "alpha": float(self.alpha.detach().cpu().item()),
                "alpha_loss": 0.0,
                "q_mean": 0.0,
                "td_error": 0.0,
            }

        rollout_state = self._snapshot_rollout_state()

        observations, actions, rewards, dones, loss_mask = self.buffer.sample_sequences(
            batch_size=batch_size,
            burn_in=self.burn_in,
            train_length=self.sequence_length,
            device=self.device,
            dtype=self.dtype,
        )

        alpha = self.alpha.detach()
        valid_steps = loss_mask.sum().clamp_min(1.0)

        self._reset_learning_state(batch_size)
        logits_seq, q1_seq, q2_seq = self._unroll_piq_sequence(self.net, observations)

        with torch.no_grad():
            self._reset_target_state(batch_size)
            _, target_q1_seq, target_q2_seq = self._unroll_piq_sequence(self.target_net, observations)

        logits = logits_seq[:, :-1]
        next_logits = logits_seq[:, 1:].detach()
        q1_all = q1_seq[:, :-1]
        q2_all = q2_seq[:, :-1]
        next_q1 = target_q1_seq[:, 1:]
        next_q2 = target_q2_seq[:, 1:]

        probs, log_probs, entropy = _categorical_stats(logits)
        next_probs, next_log_probs, _ = _categorical_stats(next_logits)

        q1_pred = q1_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q2_pred = q2_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        next_q = torch.minimum(next_q1, next_q2)
        next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1)
        targets = rewards + self.gamma * (1.0 - dones) * next_v

        q1_error = targets - q1_pred
        q2_error = targets - q2_pred
        critic_loss = 0.5 * (
            ((q1_error.pow(2) + q2_error.pow(2)) * loss_mask).sum() / valid_steps
        )

        min_q = torch.minimum(q1_all, q2_all).detach()
        policy_terms = (probs * (min_q - alpha * log_probs)).sum(dim=-1)
        policy_objective = (policy_terms * loss_mask).sum() / valid_steps
        entropy_mean = (entropy * loss_mask).sum() / valid_steps

        loss = -policy_objective + critic_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

        self.optim.step()

        alpha_loss_value = 0.0
        if self.autotune_alpha and self.alpha_opt is not None:
            alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy) * loss_mask).sum() / valid_steps
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_loss_value = float(alpha_loss.detach().cpu().item())

        self.soft_update()
        self._restore_rollout_state(rollout_state)

        td_error = 0.5 * (q1_error.abs() + q2_error.abs())
        q_mean = 0.5 * (
            (q1_pred * loss_mask).sum() / valid_steps +
            (q2_pred * loss_mask).sum() / valid_steps
        )

        return {
            "policy_objective": float(policy_objective.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy_mean.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "alpha_loss": alpha_loss_value,
            "q_mean": float(q_mean.detach().cpu().item()),
            "td_error": float(((td_error * loss_mask).sum() / valid_steps).detach().cpu().item()),
        }

    def save(self, path: str, include_buffer: bool = False):
        checkpoint = {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optim.state_dict() if self.optim is not None else None,
            "gamma": self.gamma,
            "tau": self.tau,
            "batch_size": self.batch_size,
            "autotune_alpha": self.autotune_alpha,
            "target_entropy": self.target_entropy,
            "alpha": float(self.alpha.detach().cpu().item()),
            "log_alpha": self.log_alpha.detach().cpu() if self.log_alpha is not None else None,
            "alpha_optimizer": self.alpha_opt.state_dict() if self.alpha_opt is not None else None,
            "sequence_length": self.sequence_length,
            "burn_in": self.burn_in,
        }

        if include_buffer:
            checkpoint["replay_buffer"] = self.buffer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path: str, map_location=None, load_buffer: bool = False):
        checkpoint = torch.load(path, map_location=map_location or self.device)

        self.net.load_state_dict(checkpoint["net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if self.optim is not None and checkpoint["optimizer"] is not None:
            self.optim.load_state_dict(checkpoint["optimizer"])

        if self.autotune_alpha and checkpoint.get("log_alpha") is not None:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].to(device=self.device, dtype=self.dtype))
            if self.alpha_opt is not None and checkpoint.get("alpha_optimizer") is not None:
                self.alpha_opt.load_state_dict(checkpoint["alpha_optimizer"])
        elif not self.autotune_alpha:
            self._alpha_fixed = float(checkpoint.get("alpha", self._alpha_fixed))

        self.gamma = checkpoint.get("gamma", self.gamma)
        self.tau = checkpoint.get("tau", self.tau)
        self.batch_size = int(checkpoint.get("batch_size", self.batch_size))
        self.target_entropy = float(checkpoint.get("target_entropy", self.target_entropy))
        self.sequence_length = int(checkpoint.get("sequence_length", self.sequence_length))
        self.burn_in = int(checkpoint.get("burn_in", self.burn_in))

        if load_buffer and "replay_buffer" in checkpoint:
            self.buffer.load_state_dict(checkpoint["replay_buffer"])

        self.net.to(self.device, dtype=self.dtype)
        self.target_net.to(self.device, dtype=self.dtype)
        self.reset()


def train_on_historical(agent, env, data, *args, **kwargs):
    return agent.train_on_historical(env, data, *args, **kwargs)
