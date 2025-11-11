from dataclasses import dataclass
from collections import deque, namedtuple
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.agent.buffer import Buffer
from src.agent.utils import get_optimizer


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 100.0
    alpha: float = 0.1
    target_entropy: float | None = None
    buffer_size: int = 10000
    action_low: float = -1.0
    action_high: float = 1.0
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class SACAgent:
    """
    Soft Actor Critic - like deep RL agent. Uses entropy regularization.

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
        config: SACConfig = SACConfig(),
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

        # entropy regularization
        log_alpha = torch.log(torch.tensor(config.alpha, dtype=self.dtype))
        self.log_alpha = torch.nn.Parameter(log_alpha, requires_grad=True).to(self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4)
        self.target_entropy = config.target_entropy if config.target_entropy is not None else -self.q.action_size

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
        
        if explore:
            action, _ = self.a.sample(state)
        else:
            action = self.a.action(state)

        return action[0].cpu().clamp(self.config.action_low, self.config.action_high)

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
            next_actions, next_log_probs = self.a_t.sample(next_states)
            next_q_values = self.q_t(next_states, next_actions) - self.log_alpha.exp() * next_log_probs
            q_targets = rewards + self.config.gamma * (1 - dones) * next_q_values

        # Critic Loss (Q-function)
        self.q_optim.zero_grad()
        q_values = self.q(states, actions)
        q_loss = torch.square(q_targets - q_values).mean()
        q_loss.backward()
        self.q_optim.step()

        # Actor Loss (Policy Update)
        self.a_optim.zero_grad()
        actions_, log_probs_ = self.a.sample(states)
        # log_probs_ = torch.nan_to_num(log_probs_, nan=0.0, neginf=-20.0, posinf=0.0)
        q_values_ = self.q(states, actions_)
        a_loss = (self.log_alpha.exp() * log_probs_ - q_values_).mean()
        a_loss.backward()
        self.a_optim.step()

        # Entropy Coefficient (Alpha) Update
        self.alpha_optim.zero_grad()
        alpha_loss = (-self.log_alpha.exp() * (log_probs_ + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()

        # soft target updates
        self.soft_update(self.a_t, self.a, self.config.tau)
        self.soft_update(self.q_t, self.q, self.config.tau)

        return {
            "actor_loss": float(-a_loss.item()),
            "critic_loss": float(q_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
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
                "config": self.config.__dict__
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.a.load_state_dict(ckpt["actor"], strict=strict)
        self.q.load_state_dict(ckpt["critic"], strict=strict)
        self.a_t.load_state_dict(ckpt["actor_target"], strict=strict)
        self.q_t.load_state_dict(ckpt["critic_target"], strict=strict)
