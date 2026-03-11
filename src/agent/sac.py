import copy
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================
# Replay buffer
# =============================

@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: torch.Tensor
    s2: torch.Tensor
    d: torch.Tensor
    h0: dict | None = None
    h1: dict | None = None


class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.capacity = int(capacity)
        self.data: list[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def push(self, tr: Transition):
        if len(self.data) < self.capacity:
            self.data.append(tr)
        else:
            self.data[self.pos] = tr
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.data, k=min(batch_size, len(self.data)))


# =============================
# Gaussian helpers (NO squashing)
# =============================

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def gaussian_logprob(x: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """
    log N(x | mu, std) summed over last dim (action dims).
    Shapes:
      x, mu, log_std: (..., A)
      returns: (...,)
    """
    # stable: -0.5 * [((x-mu)/std)^2 + 2logstd + log(2pi)]
    return (-0.5 * (((x - mu) / (log_std.exp() + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))).sum(dim=-1)


# =============================
# Single recurrent shared network:
#   trunk + actor head + single Q head
# =============================

class RecurrentSACSingleNet(nn.Module):
    """
    One shared recurrent trunk, one actor head (mu, log_std),
    one critic head Q(s,a).

    Use forward_av(...) when you want actor params and critic output
    without advancing the RNN multiple times.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        activation=F.relu,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # trunk
        self.fc = nn.Linear(state_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=False)
        self._h = None

        # actor head
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # critic head Q(s,a)
        self.q_fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    # ---- recurrent state API ----
    def reset(self, batch_size: int = 1):
        self._h = torch.zeros(
            1, batch_size, self.hidden_dim,
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

    def set_states(self, h: dict | None, strict: bool = False, clone: bool = False, detach: bool = False):
        if h is None:
            return
        if "h" not in h:
            if strict:
                raise KeyError("Missing key 'h'.")
            return
        hh = h["h"]
        if hh is None:
            self._h = None
        else:
            if clone:
                hh = hh.clone()
            if detach:
                hh = hh.detach()
            self._h = hh

    # ---- trunk encoder (advances recurrent state once) ----
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        # state: (S,) or (B,S)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.activation(self.fc(state))  # (B,H)
        x = x.unsqueeze(0)                   # (1,B,H)
        if self._h is None or self._h.size(1) != x.size(1):
            self.reset(batch_size=x.size(1))
        y, self._h = self.rnn(x, self._h)    # (1,B,H)
        return y.squeeze(0)                  # (B,H)

    # ---- heads from features (do NOT advance RNN) ----
    def actor_from_feat(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(feat)
        log_std = self.log_std(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def critic_from_feat(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([feat, action], dim=-1)
        x = self.activation(self.q_fc1(x))
        return self.q_out(x).squeeze(-1)  # (B,)

    # ---- PPO-like interface ----
    def a(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encode(state)
        mu, log_std = self.actor_from_feat(feat)
        if mu.size(0) == 1:
            return mu.squeeze(0), log_std.squeeze(0)
        return mu, log_std

    def q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        feat = self.encode(state)
        q = self.critic_from_feat(feat, action)
        return q.squeeze(0) if q.size(0) == 1 else q

    def forward_av(self, state: torch.Tensor, action: torch.Tensor | None = None):
        """
        One trunk pass, then:
          - (mu, log_std)
          - and optionally Q(s,a) if action is provided
        """
        feat = self.encode(state)
        mu, log_std = self.actor_from_feat(feat)
        q = None
        if action is not None:
            q = self.critic_from_feat(feat, action)

        if mu.dim() == 2 and mu.size(0) == 1:
            mu = mu.squeeze(0)
            log_std = log_std.squeeze(0)
            if q is not None:
                q = q.squeeze(0)
        return mu, log_std, q


# =============================
# SAC Agent (single critic, no squashing)
# =============================

class SACAgent:
    """
    SAC with:
      - single shared recurrent network: actor head (mu, log_std) + critic head Q(s,a)
      - single target network (copy) used to compute Q targets
      - optional automatic alpha tuning
      - NO tanh-squash / no logprob correction (env handles action constraints)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        # entropy
        alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: float | None = None,  # default = -action_dim
        # replay
        replay_capacity: int = 500_000,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        # net
        hidden_dim: int = 256,
        activation=F.relu,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.device = torch.device(device)
        self.dtype = dtype

        self.net = RecurrentSACSingleNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            activation=activation,
        ).to(self.device, self.dtype)

        self.net_targ = copy.deepcopy(self.net).eval()
        for p in self.net_targ.parameters():
            p.requires_grad_(False)

        self.optim = torch.optim.AdamW(self.net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_capacity)

        self.autotune_alpha = bool(autotune_alpha)
        if target_entropy is None:
            target_entropy = -float(action_dim)
        self.target_entropy = float(target_entropy)

        if self.autotune_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha), device=self.device, dtype=self.dtype))
            self.alpha_opt = torch.optim.AdamW([self.log_alpha], lr=lr)
        else:
            self.log_alpha = None
            self._alpha_fixed = float(alpha)

        # init recurrent states for interaction
        self.net.reset(1)
        self.net_targ.reset(1)

    @property
    def alpha(self) -> torch.Tensor:
        if self.autotune_alpha:
            return self.log_alpha.exp()
        return torch.tensor(self._alpha_fixed, device=self.device, dtype=self.dtype)

    def _sample_action_and_logp_from_params(
        self,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        if deterministic:
            a = mu
        else:
            a = mu + std * torch.randn_like(mu)
        logp = gaussian_logprob(a, mu, log_std)
        return a, logp

    @torch.no_grad()
    def act(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        s = state.to(self.device, self.dtype)
        mu, log_std = self.net.a(s)
        a, _ = self._sample_action_and_logp_from_params(mu, log_std, deterministic=not explore)
        return a.detach().cpu()

    def store(self, state, action, reward, next_state, done, h0=None, h1=None):
        def to_t(x):
            if isinstance(x, torch.Tensor):
                return x.detach().to(self.device, self.dtype)
            return torch.tensor(x, device=self.device, dtype=self.dtype)

        self.replay.push(
            Transition(
                s=to_t(state),
                a=to_t(action),
                r=to_t(reward).view(()),
                s2=to_t(next_state),
                d=to_t(done).view(()),
                h0=h0,
                h1=h1,
            )
        )

    @torch.no_grad()
    def soft_update(self):
        for p, pt in zip(self.net.parameters(), self.net_targ.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    def update(self, batch_size: int = 256, max_grad_norm: float | None = None) -> dict[str, float]:
        if len(self.replay) < max(10, batch_size):
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "alpha": float(self.alpha.detach().cpu().item()),
            }

        batch = self.replay.sample(batch_size)
        s = torch.stack([b.s for b in batch], 0)
        a = torch.stack([b.a for b in batch], 0)
        r = torch.stack([b.r for b in batch], 0)
        s2 = torch.stack([b.s2 for b in batch], 0)
        d = torch.stack([b.d for b in batch], 0)

        B = s.size(0)

        # Step-wise training: treat samples as independent
        self.net.reset(B)
        self.net_targ.reset(B)

        # ----- Critic target -----
        # y = r + gamma(1-d) * ( Q_targ(s2, a2) - alpha * logpi(a2|s2) ),  a2 ~ pi(s2)
        with torch.no_grad():
            # sample a2 from target policy at s2
            mu2, log_std2, _ = self.net_targ.forward_av(s2, action=None)
            a2, logp2 = self._sample_action_and_logp_from_params(mu2, log_std2, deterministic=False)

            # compute Q_targ(s2, a2) with ONE trunk pass (reset to align hidden state)
            self.net_targ.reset(B)
            _, _, q2 = self.net_targ.forward_av(s2, action=a2)

            y = r + self.gamma * (1.0 - d) * (q2 - self.alpha * logp2)

        # ----- Critic loss -----
        self.net.reset(B)
        _, _, q = self.net.forward_av(s, action=a)
        critic_loss = F.mse_loss(q, y)

        # ----- Actor loss -----
        # Lpi = E[ alpha*logpi(a|s) - Q(s,a) ],  a ~ pi(s)
        self.net.reset(B)
        mu, log_std, _ = self.net.forward_av(s, action=None)
        a_pi, logp = self._sample_action_and_logp_from_params(mu, log_std, deterministic=False)

        self.net.reset(B)
        _, _, q_pi = self.net.forward_av(s, action=a_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        # ----- Optimize shared net -----
        loss = critic_loss + actor_loss
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
        self.optim.step()

        # ----- Alpha tuning -----
        alpha_val = float(self.alpha.detach().cpu().item())
        if self.autotune_alpha:
            # J(alpha) = E[ -alpha * (logpi + target_entropy) ]
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_val = float(self.alpha.detach().cpu().item())

        self.soft_update()

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "alpha": alpha_val,
            "q": float(q.detach().mean().cpu().item()),
            "logp": float(logp.detach().mean().cpu().item()),
        }

    def train_on_historical(
        self,
        env,
        data,
        n_episodes: int,
        max_steps: int = 2000,
        warm_up: int = 0,
        update_interval: int = 1,
        updates_per_interval: int = 1,
        batch_size: int = 256,
        store_results: bool = True,
        max_grad_norm: float | None = None,
    ):
        total_reward, total_loss, total_info = [], [], []

        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])

            self.net.reset(1)

            episode_reward, episode_loss, episode_info = [], [], []
            step = 1

            while True:
                h0 = self.net.get_states(clone=True, detach=True)

                if step > warm_up:
                    action = self.act(state.to_tensor(), explore=True)
                else:
                    action = None

                next_state, reward, done, info = env.step(action, data[start + step])

                if store_results:
                    episode_reward.append(float(reward))
                    episode_info.append(info)

                if step > warm_up:
                    h1 = self.net.get_states(clone=True, detach=True)
                    self.store(
                        state=state.to_tensor(),
                        action=torch.zeros(self.action_dim) if action is None else action,
                        reward=torch.tensor(reward, dtype=self.dtype),
                        next_state=next_state.to_tensor(),
                        done=torch.tensor(done, dtype=self.dtype),
                        h0=h0,
                        h1=h1,
                    )

                    if step % update_interval == 0:
                        for _ in range(updates_per_interval):
                            ld = self.update(batch_size=batch_size, max_grad_norm=max_grad_norm)
                            if store_results:
                                episode_loss.append(ld)

                        last = episode_loss[-1] if episode_loss else {}
                        print(
                            f"episode {episode} [{100*step/max_steps:.1f}%] - "
                            f"reward: {float(reward):.5f} - portfolio: {info['V']:.2f}€ - "
                            f"actor_loss: {last.get('actor_loss', 0.0):.5f} - "
                            f"critic_loss: {last.get('critic_loss', 0.0):.5f} - "
                            f"alpha: {last.get('alpha', float(self.alpha.detach().cpu().item())):.5f}",
                            end="\r"
                        )

                terminal = done or (step >= max_steps)
                if terminal:
                    break

                state = next_state
                step += 1

            if store_results:
                total_reward.append(episode_reward)
                total_loss.append(episode_loss)
                total_info.append(episode_info)

            avg_r = sum(episode_reward) / max(1, len(episode_reward)) if store_results else 0.0
            lastV = episode_info[-1]["V"] if store_results and len(episode_info) else float("nan")

            if store_results and len(episode_loss):
                al = sum(d["actor_loss"] for d in episode_loss) / len(episode_loss)
                cl = sum(d["critic_loss"] for d in episode_loss) / len(episode_loss)
                av = sum(d["alpha"] for d in episode_loss) / len(episode_loss)
                print(f"episode {episode} - avg. reward: {avg_r:.5f} - portfolio: {lastV:.2f}€ - actor_loss: {al:.5f} - critic_loss: {cl:.5f} - alpha: {av:.5f}")
            else:
                print(f"episode {episode} - avg. reward: {avg_r:.5f} - portfolio: {lastV:.2f}€")

        return total_loss, total_reward, total_info