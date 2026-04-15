import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Noisy Linear
# =========================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device=None):
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features, self.weight_mu.device)
        eps_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_eps.copy_(eps_out.outer(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# =========================================================
# Prioritized Replay Buffer with n-step support
# =========================================================

@dataclass
class NStepTransition:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dtype=torch.float32,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-6,
        eps: float = 1e-6,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.n_step = n_step
        self.gamma = gamma
        self.state_dtype = state_dtype

        self.storage = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

        self.nstep_buffer = deque(maxlen=n_step)

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        self.priorities[:] = 0.0
        self.pos = 0
        self.nstep_buffer.clear()

    def _make_nstep_transition(self):
        """
        Build one n-step transition from the left edge of the n-step deque.
        """
        state, action = self.nstep_buffer[0].state, self.nstep_buffer[0].action

        reward = torch.zeros_like(self.nstep_buffer[0].reward)
        next_state = self.nstep_buffer[-1].next_state
        done = self.nstep_buffer[-1].done

        for i, tr in enumerate(self.nstep_buffer):
            reward = reward + (self.gamma ** i) * tr.reward
            next_state = tr.next_state
            done = tr.done
            if bool(tr.done.item()):
                break

        return state, action, reward, next_state, done

    def store(self, state, action, reward, next_state, done):
        """
        Stores 1-step transitions internally and commits n-step transitions to replay.
        """
        tr = NStepTransition(
            state=state.detach().cpu(),
            action=action.detach().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action),
            reward=reward.detach().cpu() if isinstance(reward, torch.Tensor) else torch.tensor(reward),
            next_state=next_state.detach().cpu(),
            done=done.detach().cpu() if isinstance(done, torch.Tensor) else torch.tensor(done),
        )
        self.nstep_buffer.append(tr)

        if len(self.nstep_buffer) < self.n_step and not bool(tr.done.item()):
            return

        state, action, reward, next_state, done = self._make_nstep_transition()
        data = (state, action.long().view(()), reward.view(()), next_state, done.view(()))

        max_prio = self.priorities.max() if len(self.storage) > 0 else 1.0

        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.pos] = data

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

        if bool(tr.done.item()):
            self.nstep_buffer.clear()

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        if len(self.storage) == 0:
            raise RuntimeError("Cannot sample from empty replay buffer.")

        prios = self.priorities[:len(self.storage)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.storage), batch_size, p=probs)
        samples = [self.storage[i] for i in indices]

        total = len(self.storage)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states).to(device=device, dtype=dtype)
        actions = torch.stack(actions).to(device=device, dtype=torch.long)
        rewards = torch.stack(rewards).to(device=device, dtype=dtype)
        next_states = torch.stack(next_states).to(device=device, dtype=dtype)
        dones = torch.stack(dones).to(device=device, dtype=dtype)
        weights = torch.tensor(weights, device=device, dtype=dtype)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        priorities = np.asarray(priorities, dtype=np.float32)
        for i, p in zip(indices, priorities):
            self.priorities[i] = float(abs(p) + self.eps)


# =========================================================
# Recurrent Rainbow Network
# shared recurrent base + dueling distributional heads
# =========================================================

class RecurrentRainbowQNetwork(nn.Module):
    """
    Assumes state -> shared recurrent base -> dueling categorical Q heads.
    Output:
      dist: [B, action_dim, num_atoms]
      q:    [B, action_dim]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        hidden_dims_value: list[int] = [256],
        hidden_dims_advantage: list[int] = [256],
        activation: callable = torch.LeakyReLU(negative_slope=0.02),
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h
        self.base_layers = nn.ModuleList(layers)

        self.rnn = nn.GRUCell(in_dim, in_dim)
        self._h = None
        trunk_dim = in_dim

        # value stream
        v_layers = []
        in_dim_v = trunk_dim
        for h in hidden_dims_value:
            v_layers.append(NoisyLinear(in_dim_v, h))
            in_dim_v = h
        self.value_layers = nn.ModuleList(v_layers)
        self.value_out = NoisyLinear(in_dim_v, num_atoms)

        # advantage stream
        a_layers = []
        in_dim_a = trunk_dim
        for h in hidden_dims_advantage:
            a_layers.append(NoisyLinear(in_dim_a, h))
            in_dim_a = h
        self.adv_layers = nn.ModuleList(a_layers)
        self.adv_out = NoisyLinear(in_dim_a, action_dim * num_atoms)

        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

    def reset(self, batch_size: int = 1):
        hidden_dim = self.rnn.hidden_size
        self._h = torch.zeros(
            batch_size, hidden_dim,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype
        )

    def get_states(self, clone: bool = True, detach: bool = True) -> dict:
        h = self._h
        if h is None:
            return {"h": None}
        if clone:
            h = h.clone()
        if detach:
            h = h.detach()
        return {"h": h}

    def set_states(self, h0: dict | None, strict: bool = False, clone: bool = False, detach: bool = False):
        if h0 is None:
            return
        if "h" not in h0:
            if strict:
                raise KeyError("Missing key 'h'")
            return
        h = h0["h"]
        if h is None:
            self._h = None
            return
        if clone:
            h = h.clone()
        if detach:
            h = h.detach()
        self._h = h

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = state
        for layer in self.base_layers:
            x = self.activation(layer(x))

        if self._h is None or self._h.shape[0] != x.shape[0]:
            self.reset(x.shape[0])

        self._h = self.rnn(x, self._h)
        return self._h

    def dist_from_feat(self, feat: torch.Tensor) -> torch.Tensor:
        v = feat
        for layer in self.value_layers:
            v = self.activation(layer(v))
        v = self.value_out(v).view(-1, 1, self.num_atoms)

        a = feat
        for layer in self.adv_layers:
            a = self.activation(layer(a))
        a = self.adv_out(a).view(-1, self.action_dim, self.num_atoms)

        logits = v + a - a.mean(dim=1, keepdim=True)
        probs = torch.softmax(logits, dim=-1)
        probs = probs.clamp(min=1e-5)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def q_from_dist(self, probs: torch.Tensor) -> torch.Tensor:
        return (probs * self.support.view(1, 1, -1)).sum(dim=-1)

    def q(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.encode(state)
        probs = self.dist_from_feat(feat)
        q = self.q_from_dist(probs)
        return q.squeeze(0) if q.shape[0] == 1 else q

    def dist(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.encode(state)
        probs = self.dist_from_feat(feat)
        return probs.squeeze(0) if probs.shape[0] == 1 else probs

    def qd(self, state: torch.Tensor):
        """
        Forward both Q values and categorical distributions
        in one pass through the recurrent trunk.
        """
        feat = self.encode(state)
        probs = self.dist_from_feat(feat)
        q = self.q_from_dist(probs)
        if q.shape[0] == 1:
            return q.squeeze(0), probs.squeeze(0)
        return q, probs


# =========================================================
# Rainbow DQN Agent
# =========================================================

class RecurrentRainbowDQNAgent:

    def __init__(
        self,
        self_state_dim: int,
        action_dim: int,
        # rainbow properties
        gamma: float = 0.999,
        n_step: int = 3,
        replay_capacity: int = 200_000,
        batch_size: int = 64,
        target_update_interval: int = 250,
        double_dqn: bool = True,
        # PER
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_increment: float = 1e-6,
        per_eps: float = 1e-6,
        # C51
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # network properties
        hidden_dims: tuple[int, ...] = (256, 256),
        hidden_dims_value: list[int] = [256],
        hidden_dims_advantage: list[int] = [256],
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.n_step = n_step
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.double_dqn = double_dqn
        self.dtype = dtype
        self.device = torch.device(device)

        self.state_dim = self_state_dim
        self.action_dim = action_dim

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.net = RecurrentRainbowQNetwork(
            state_dim=self_state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_value=hidden_dims_value,
            hidden_dims_advantage=hidden_dims_advantage,
            activation=activation,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
        ).to(device=self.device, dtype=self.dtype)

        self.target_net = RecurrentRainbowQNetwork(
            state_dim=self_state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_value=hidden_dims_value,
            hidden_dims_advantage=hidden_dims_advantage,
            activation=activation,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
        ).to(device=self.device, dtype=self.dtype)

        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.buffer = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            state_dtype=dtype,
            alpha=per_alpha,
            beta=per_beta,
            beta_increment=per_beta_increment,
            eps=per_eps,
            n_step=n_step,
            gamma=gamma,
        )

        self.net.reset(1)
        self.target_net.reset(1)
        self.update_step_count = 0

    def infer_q(self, state: torch.Tensor) -> torch.Tensor:
        return self.net.q(state)

    def infer_dist(self, state: torch.Tensor) -> torch.Tensor:
        return self.net.dist(state)

    def infer_q_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> torch.Tensor:
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        qs = []
        for state in state_seq:
            q = self.net.q(state)
            qs.append(q.squeeze())
        return torch.stack(qs, dim=0)

    def infer_qd_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None):
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        qs, dists = [], []
        for state in state_seq:
            q, dist = self.net.qd(state)
            qs.append(q.squeeze())
            dists.append(dist.squeeze())
        return torch.stack(qs, dim=0), torch.stack(dists, dim=0)

    @torch.no_grad()
    def act(self, state: torch.Tensor, explore: bool = True, grad_enabled: bool = False):
        """
        Returns discrete action index.
        With NoisyNet, exploration is already inside the network, so `explore`
        mainly controls train/eval mode semantics.
        """
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)

            if explore:
                self.net.train()
                self.net.reset_noise()
            else:
                self.net.eval()

            q = self.infer_q(state)
            action = torch.argmax(q, dim=-1)

        if grad_enabled:
            return action
        return action.cpu()

    def store(self, state, action, reward, next_state, done):
        self.buffer.store(state, action, reward, next_state, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def _project_distribution(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        C51 projection onto fixed support.
        rewards, dones: [B]
        next_dist: [B, num_atoms]
        """
        batch_size = rewards.shape[0]
        support = self.net.support.to(device=self.device, dtype=self.dtype)
        projected = torch.zeros(batch_size, self.num_atoms, device=self.device, dtype=self.dtype)

        tz = rewards[:, None] + (1.0 - dones[:, None]) * (self.gamma ** self.n_step) * support[None, :]
        tz = tz.clamp(self.v_min, self.v_max)

        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = (
            torch.arange(batch_size, device=self.device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms) * self.num_atoms
        )

        projected.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (next_dist * (u.float() - b)).view(-1),
        )
        projected.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (next_dist * (b - l.float())).view(-1),
        )

        eq_mask = (u == l)
        if eq_mask.any():
            projected.view(-1).index_add_(
                0,
                (l[eq_mask] + offset[eq_mask]).view(-1),
                next_dist[eq_mask].view(-1),
            )

        projected = projected.clamp(min=1e-5)
        projected = projected / projected.sum(dim=-1, keepdim=True)
        return projected

    def update(self, h0: dict | None = None, max_grad_norm: float | None = None) -> dict[str, float]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0, "td_error": 0.0}

        # save current recurrent state for rollout continuity
        h_t = self.net.get_states(clone=True, detach=True)

        self.net.train()
        self.target_net.eval()
        self.net.reset_noise()
        self.target_net.reset_noise()

        states, actions, rewards, next_states, dones, weights, indices = self.buffer.sample(
            self.batch_size, self.device, self.dtype
        )

        # step-wise recurrent training: treat samples as independent batch items
        self.net.reset(states.shape[0])
        self.target_net.reset(states.shape[0])

        # current predicted distributions
        q_pred, dist_pred = self.net.qd(states)                     # [B, A], [B, A, Z]
        chosen_dist = dist_pred[torch.arange(states.shape[0], device=self.device), actions]  # [B, Z]

        with torch.no_grad():
            # online net chooses next action (Double DQN)
            self.net.reset(next_states.shape[0])
            next_q_online = self.net.q(next_states)                # [B, A]
            next_actions = next_q_online.argmax(dim=-1)

            # target net evaluates chosen next action
            self.target_net.reset(next_states.shape[0])
            next_dist_target = self.target_net.dist(next_states)   # [B, A, Z]
            next_dist = next_dist_target[
                torch.arange(next_states.shape[0], device=self.device), next_actions
            ]                                                      # [B, Z]

            target_dist = self._project_distribution(rewards, dones, next_dist)

        # cross-entropy / KL-style categorical loss
        log_p = torch.log(chosen_dist)
        per_sample_loss = -(target_dist * log_p).sum(dim=-1)
        loss = (weights * per_sample_loss).mean()

        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

        self.optim.step()

        # update priorities
        with torch.no_grad():
            q_chosen = q_pred[torch.arange(states.shape[0], device=self.device), actions]
            target_q = (target_dist * self.net.support.view(1, -1)).sum(dim=-1)
            td_error = (target_q - q_chosen).abs()
            self.buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        self.update_step_count += 1
        if self.update_step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # restore recurrent state from before learning
        self.net.set_states(h_t, clone=True, detach=True)

        return {
            "loss": float(loss.detach().cpu().item()),
            "q_mean": float(q_pred.mean().detach().cpu().item()),
            "td_error": float(td_error.mean().detach().cpu().item()),
        }

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=1000,
        update_interval=1,
        n_updates=1,
        lr=3e-4,
        optim="AdamW",
        init_optimizer=False,
        store_results=True,
        max_grad_norm=None,
    ):
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        total_reward = []
        total_loss = []
        total_info = []

        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])
            self.net.reset()
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
                        action.detach() if isinstance(action, torch.Tensor) else torch.tensor(action),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        next_state.to_tensor().detach(),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )
                    train_ready = len(self.buffer) >= self.batch_size
                    terminal = done or (step > max_steps)

                    if train_ready and (step % update_interval == 0):
                        batch_losses = []
                        for _ in range(n_updates):
                            loss_dict = self.update(h0=h0, max_grad_norm=max_grad_norm)
                            batch_losses.append(loss_dict)

                        avg_loss_dict = {
                            k: sum(d[k] for d in batch_losses) / len(batch_losses)
                            for k in batch_losses[0].keys()
                        }

                        if store_results:
                            episode_loss.append(avg_loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}€"
                        for key, val in avg_loss_dict.items():
                            msg += f" - {key}: {val:.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break

                else:
                    if step % update_interval == 0:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                state = next_state
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - avg. reward: {sum(episode_reward) / len(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}€"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(ld[key] for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info
    
    def save(self, path: str, include_buffer: bool = False):
        """
        Save agent state.

        Parameters
        ----------
        path : str
            File path.
        include_buffer : bool
            If True, saves replay buffer (can be very large).
        """
        checkpoint = {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optim.state_dict() if hasattr(self, "optim") else None,
            "update_step_count": self.update_step_count,
            "gamma": self.gamma,
            "n_step": self.n_step,
            "batch_size": self.batch_size,
            "num_atoms": self.num_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }

        if include_buffer:
            checkpoint["replay_buffer"] = {
                "storage": self.buffer.storage,
                "priorities": self.buffer.priorities,
                "pos": self.buffer.pos,
                "beta": self.buffer.beta,
            }

        torch.save(checkpoint, path)

    def load(self, path: str, map_location=None, load_buffer: bool = False):
        """
        Load agent state.

        Parameters
        ----------
        path : str
            File path.
        map_location : device
            Torch device mapping.
        load_buffer : bool
            Whether to restore replay buffer.
        """
        checkpoint = torch.load(path, map_location=map_location)

        self.net.load_state_dict(checkpoint["net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if hasattr(self, "optim") and checkpoint["optimizer"] is not None:
            self.optim.load_state_dict(checkpoint["optimizer"])

        self.update_step_count = checkpoint.get("update_step_count", 0)

        if load_buffer and "replay_buffer" in checkpoint:
            rb = checkpoint["replay_buffer"]
            self.buffer.storage = rb["storage"]
            self.buffer.priorities = rb["priorities"]
            self.buffer.pos = rb["pos"]
            self.buffer.beta = rb["beta"]

        self.net.to(self.device, dtype=self.dtype)
        self.target_net.to(self.device, dtype=self.dtype)