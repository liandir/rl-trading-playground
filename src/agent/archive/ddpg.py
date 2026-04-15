from dataclasses import dataclass
from collections import deque, namedtuple
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agent.archive.utils import get_optimizer


class Buffer:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = []
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.memory_size if self.full else self.idx

    def store(self, state, action, next_state, reward, done):
        transition = (state, action, next_state, reward, done)
        if not self.full:
            self.memory.append(transition)
            self.idx += 1
            if self.idx == self.memory_size:
                self.full = True
                self.idx = 0
        else:
            self.memory[self.idx] = transition
            self.idx = (self.idx + 1) % self.memory_size

    def sample(self, batch_size: int, device: str | torch.device, dtype: torch.dtype):
        import random
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        state = torch.stack(state).to(dtype=dtype, device=device)
        action = torch.stack(action).to(dtype=dtype, device=device)
        next_state = torch.stack(next_state).to(dtype=dtype, device=device)
        reward = torch.stack(reward).to(dtype=dtype, device=device)
        done = torch.stack(done).to(dtype=dtype, device=device)
        return state, action, next_state, reward, done


@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 100.0                 # soft target update rate
    noise_std: float = 0.1             # Gaussian exploration noise
    buffer_size: int = 10000
    action_low: float = -1.0
    action_high: float = 1.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class VanillaDDPG:
    """
    Vanilla Deep Deterministic Policy Gradient.

    Expect:
      - actor(s)           : nn.Module, state -> action
      - critic(s,a)        : nn.Module, (state, action) -> Q-value
      - actor_target, critic_target: same architectures, pre-initialized
      - actor_optim, critic_optim: torch.optim.Optimizer
    """
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        config: DDPGConfig = DDPGConfig(),
    ):
        # config and replay buffer
        self.config = config
        self.dtype  = config.dtype
        self.device = torch.device(config.device)
        self.buffer = Buffer(config.buffer_size)

        # networks
        self.a = actor.to(dtype=config.dtype, device=config.device)
        self.q = critic.to(dtype=config.dtype, device=config.device)
        self.a_t = copy.deepcopy(self.a).to(dtype=config.dtype, device=config.device).eval()
        self.q_t = copy.deepcopy(self.q).to(dtype=config.dtype, device=config.device).eval()

        # Hard copy weights into targets
        self.hard_update(self.a_t, self.a)
        self.hard_update(self.q_t, self.q)

    @torch.no_grad()
    def act(self, state: torch.Tensor, explore: bool = False) -> torch.Tensor:
        """
        state: shape (state_dim,) or (1, state_dim)
        returns action in [action_low, action_high]
        """
        state = state.to(dtype=self.dtype, device=self.device)
        a = self.a(state).cpu()

        if explore:
            noise = torch.randn_like(a) * self.config.noise_std
            a = a + noise

        return a.clamp(self.config.action_low, self.config.action_high)

    def store(self, state, action, next_state, reward, done):
        self.buffer.store(state, action, next_state, reward, done)

    def update_optimizers(self, a_lr, q_lr, optim = "AdamW", a_m = 0.0, q_m = 0.0):
        self.a_optim = get_optimizer(self.a, a_lr, optim=optim, m=a_m)
        self.q_optim = get_optimizer(self.q, q_lr, optim=optim, m=q_m)

    def update(self, batch_size: int):
        """
        Single DDPG update: critic step then actor step, followed by soft target updates.
        Returns a dict of scalars for logging.
        """
        (
            states,
            actions,
            next_states,
            rewards,
            dones
        ) = self.buffer.sample(batch_size, self.device, self.dtype)

        # critic update
        with torch.no_grad():
            next_actions = self.a_t(next_states)
            target_q = self.q_t(next_states, next_actions)
            y = rewards + self.config.gamma * (1.0 - dones) * target_q

        current_q = self.q(states, actions)
        critic_loss = F.mse_loss(current_q, y)

        self.q_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optim.step()

        # actor update
        pi = self.a(states)
        actor_loss = -self.q(states, pi).mean()

        self.a_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.a_optim.step()

        # soft target updates
        self.soft_update(self.a_t, self.a, self.config.tau)
        self.soft_update(self.q_t, self.q, self.config.tau)

        return {
            "actor_loss": float(-actor_loss.item()),
            "critic_loss": float(critic_loss.item())
        }

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.lerp_(sp.data, 1 / tau)

    @staticmethod
    def hard_update(target: nn.Module, source: nn.Module):
        target.load_state_dict(source.state_dict())

    def save(self, path: str):
        torch.save(
            {
                "actor": self.a.state_dict(),
                "critic": self.q.state_dict(),
                "actor_target": self.a_t.state_dict(),
                "critic_target": self.q_t.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.a.load_state_dict(ckpt["actor"], strict=strict)
        self.q.load_state_dict(ckpt["critic"], strict=strict)
        self.a_t.load_state_dict(ckpt["actor_target"], strict=strict)
        self.q_t.load_state_dict(ckpt["critic_target"], strict=strict)
