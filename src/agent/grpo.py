import copy
import torch
import torch.nn.functional as F
from torch import distributions


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class GRPORolloutBuffer:
    """
    Stores (state, action, reward, log_prob_old) tuples from on-policy rollouts.

    Unlike VAQAgent's buffer, log_prob_old is recorded at act-time so that
    importance ratios rho = pi_theta / pi_theta_old can be computed at update
    time without needing a separate old-policy forward pass.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.states       = []
        self.actions      = []
        self.rewards      = []
        self.log_probs_old = []

    def store(self, state, action, reward, log_prob_old):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs_old.append(log_prob_old)

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states        = torch.stack(self.states).to(device=device, dtype=dtype)
        actions       = torch.stack(self.actions).to(device=device, dtype=dtype)
        rewards       = torch.stack(self.rewards).to(device=device, dtype=dtype)
        log_probs_old = torch.stack(self.log_probs_old).to(device=device, dtype=dtype)
        return states, actions, rewards, log_probs_old


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class GRPOAgent:
    """
    On-policy discrete actor with Group Relative Policy Optimization (GRPO).

    No value / Q network is used. Advantages are estimated entirely from
    reward comparisons within local groups of size `group_size`:

        A_i = (r_i - mu_g) / (sigma_g + eps)

    where mu_g and sigma_g are the mean and std of the G consecutive rewards
    that contain step i. This replaces the critic while still reducing variance.

    The policy is updated with a PPO-style clipped surrogate objective plus
    an exact KL penalty against a frozen reference policy pi_ref:

        J(theta) = E[ min(rho*A, clip(rho, 1-eps, 1+eps)*A) ]
                 - beta * KL[pi_theta || pi_ref]

        rho_t = pi_theta(a_t | s_t) / pi_theta_old(a_t | s_t)

    The reference policy pi_ref is a snapshot of the network at construction
    time (or after an explicit call to `update_reference()`). It is kept frozen
    and never trained. Its role is to stop the policy drifting too far from the
    supervised pre-training distribution — analogous to its use in LLM RLHF.

    `update()` clears the buffer internally and returns per-epoch metric lists,
    matching the interface of VAQAgent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # GRPO hyperparameters
        group_size: int = 8,        # G — window for group-relative normalisation
        clip_eps: float = 0.2,      # epsilon for PPO importance-ratio clipping
        kl_coef: float = 0.04,      # beta — weight of KL penalty
        # optional entropy bonus (disabled by default; GRPO relies on KL instead)
        ent_coef: float = 0.0,
        # network
        hidden_dims: list[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.group_size = group_size
        self.clip_eps   = clip_eps
        self.kl_coef    = kl_coef
        self.ent_coef   = ent_coef
        self.dtype  = dtype
        self.device = torch.device(device)

        # Policy network — actor head only (no Q / value head needed)
        self.net = _PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        # Frozen reference policy pi_ref — never updated by the optimiser
        self.ref_net = copy.deepcopy(self.net)
        for p in self.ref_net.parameters():
            p.requires_grad_(False)
        self.ref_net.eval()

        self.buffer = GRPORolloutBuffer()
        self.optim  = None

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def update_reference(self):
        """
        Sync pi_ref <- pi_theta.

        Call this periodically (e.g. every N episodes) to allow the reference
        to track the policy as it improves, without letting it drift unboundedly.
        In LLM GRPO this is typically done once per training phase.
        """
        self.ref_net.load_state_dict(self.net.state_dict())
        self.ref_net.eval()

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(
        self,
        state: torch.Tensor,
        explore: bool = True,
        grad_enabled: bool = False,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (explore=True) or act greedily.

        Returns
        -------
        action      : scalar int tensor on CPU
        log_prob_old: log pi_theta(a | s) — must be stored alongside the
                      transition so update() can compute importance ratios.
        """
        with torch.set_grad_enabled(grad_enabled):
            state  = state.to(dtype=self.dtype, device=self.device)
            logits = self.net(state)
            logits = _masked_logits(logits, mask)
            dist   = distributions.Categorical(logits=logits)

            if explore:
                action = dist.sample().squeeze()
            else:
                action = logits.argmax(dim=-1).squeeze()

            log_prob = dist.log_prob(action)

        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.detach().cpu()

    def store(self, state, action, reward, log_prob_old):
        self.buffer.store(state, action, reward, log_prob_old)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def init_optimizer(self, lr: float, optim: str = "AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Group-relative advantage estimation
    # ------------------------------------------------------------------

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Partition the reward vector into non-overlapping windows of size G and
        standardise within each window.

            A_i = (r_i - mu_g) / (sigma_g + 1e-7)    for i in group g

        This is the GRPO equivalent of a critic baseline: the group mean acts
        as a learned-free value estimate, and the std normalises scale.

        A trailing window smaller than G is standardised on its own.

        Shape: (N,) -> (N,)
        """
        G = self.group_size
        N = rewards.shape[0]
        advantages = torch.empty_like(rewards)

        for start in range(0, N, G):
            end = min(start + G, N)
            grp   = rewards[start:end]
            mu    = grp.mean()
            sigma = grp.std(unbiased=False) + 1e-7
            advantages[start:end] = (grp - mu) / sigma

        return advantages

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    @staticmethod
    def _kl(logits: torch.Tensor, logits_ref: torch.Tensor) -> torch.Tensor:
        """
        Exact KL[pi_theta || pi_ref] for discrete distributions, averaged
        over the batch.

            KL = sum_a  pi_theta(a) * [log pi_theta(a) - log pi_ref(a)]

        This is equivalent to the unbiased token-level estimator used in
        the original GRPO paper:  r - log(r) - 1,  r = pi_ref / pi_theta,
        but computed exactly since the full action distribution is available.
        """
        log_pi     = F.log_softmax(logits,     dim=-1)
        log_pi_ref = F.log_softmax(logits_ref, dim=-1)
        kl = (log_pi.exp() * (log_pi - log_pi_ref)).sum(dim=-1)
        return kl.mean()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        k_epochs: int = 1,
        max_grad_norm: float | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, list[float]]:
        """
        Run k_epochs of GRPO on the current buffer, then clear it.

        The objective for each epoch:

            L = -E[min(rho*A, clip(rho, 1-eps, 1+eps)*A)]
              +  beta * KL[pi_theta || pi_ref]
              -  ent_coef * H[pi_theta]

        where rho_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t).

        Advantages are computed once (before the epoch loop) from group-
        relative reward normalisation and are kept fixed across epochs, the
        same as in PPO.
        """
        if self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")
        if len(self.buffer) == 0:
            return {}

        states, actions, rewards, log_probs_old = self.buffer.to_tensors(
            self.device, self.dtype
        )
        actions_long = actions.long()

        # Advantages are fixed for all epochs (computed before the loop)
        with torch.no_grad():
            advantages = self.compute_group_advantages(rewards)

        metrics = {
            "policy_objective": [],
            "kl":               [],
            "entropy":          [],
            "loss":             [],
            "adv_mean":         [],
            "adv_std":          [],
        }

        for _ in range(k_epochs):
            # Current policy
            logits = self.net(states)
            logits = _masked_logits(logits, mask)
            dist   = distributions.Categorical(logits=logits)

            log_probs = dist.log_prob(actions_long)
            entropy   = dist.entropy().mean()

            # Importance ratio rho = pi_theta / pi_theta_old
            rho = (log_probs - log_probs_old).exp()

            # Clipped surrogate objective (PPO-style)
            adv    = advantages.detach()
            surr1  = rho * adv
            surr2  = rho.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_objective = torch.min(surr1, surr2).mean()

            # KL penalty against frozen reference
            with torch.no_grad():
                logits_ref = self.ref_net(states)
                logits_ref = _masked_logits(logits_ref, mask)
            kl = self._kl(logits, logits_ref)

            loss = (
                -policy_objective
                + self.kl_coef * kl
                - self.ent_coef * entropy
            )

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
            self.optim.step()

            metrics["policy_objective"].append(policy_objective.item())
            metrics["kl"].append(kl.item())
            metrics["entropy"].append(entropy.item())
            metrics["loss"].append(loss.item())
            metrics["adv_mean"].append(adv.mean().item())
            metrics["adv_std"].append(adv.std().item())

        self.buffer.clear()
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "network":   self.net.state_dict(),
            "reference": self.ref_net.state_dict(),
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)
        if "reference" in ckpt:
            self.ref_net.load_state_dict(ckpt["reference"], strict=strict)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_on_historical(
        self,
        env,
        data,
        n_episodes: int,
        max_steps: int = 2000,
        warm_up: int = 0,
        update_interval: int = 100,         # steps between updates
        n_updates: int = 1,                 # k_epochs per update call
        lr: float = 3e-4,
        optim: str = "AdamW",
        init_optimizer: bool = False,
        store_results: bool = True,
        max_grad_norm: float | None = None,
        update_reference_every: int | None = None,  # episodes; None = never
    ):
        """
        Train on contiguous historical rollouts and collect episode-level
        metric histories. Interface is identical to VAQAgent.train_on_historical,
        with one extra argument:

        update_reference_every : int | None
            If set, calls update_reference() every N episodes so that pi_ref
            slowly tracks the improving policy. A common choice is every 5-10
            episodes.
        """
        if not hasattr(self, "optim") or self.optim is None or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        total_reward, total_loss, total_info = [], [], []

        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])

            episode_reward, episode_info, episode_loss = [], [], []
            self.buffer.clear()

            step = 1
            while True:
                if step > warm_up:
                    mask   = env.valid_action_mask() if hasattr(env, "valid_action_mask") else None
                    action, log_prob_old = self.act(state.to_tensor(), explore=True, mask=mask)
                else:
                    action = log_prob_old = mask = None

                next_state, reward, done, info = env.step(action, data[start + step])

                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                if step > warm_up:
                    ftype = getattr(env, "dtype", torch.float32)
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        torch.tensor(reward, dtype=ftype),
                        log_prob_old.detach(),
                    )

                    terminal = done or (step >= max_steps)
                    if (len(self.buffer) >= update_interval or terminal) and len(self.buffer) > 1:
                        loss_dict = self.update(k_epochs=n_updates, max_grad_norm=max_grad_norm)
                        # buffer is cleared inside update()

                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = (
                            f"episode {episode} [{100*step/max_steps:.1f}%]"
                            f" - reward: {float(reward):.5f}"
                            f" - portfolio: {info['V']:.2f}"
                        )
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val)/len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break

                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up...", end="\r")

                state = next_state
                step += 1

            # Optionally advance the reference policy
            if update_reference_every and episode % update_reference_every == 0:
                self.update_reference()

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = (
                f"episode {episode} [{100*step/max_steps:.1f}%]"
                f" - total reward: {sum(episode_reward) if store_results else 0.0:.5f}"
                f" - portfolio: {info['V']:.2f}"
            )
            if store_results and episode_loss:
                for key in episode_loss[0]:
                    msg += (
                        f" - {key}: "
                        f"{sum(sum(ld[key])/len(ld[key]) for ld in episode_loss)/len(episode_loss):.5f}"
                    )
            print(msg)

        return total_loss, total_reward, total_info


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _PolicyNetwork(torch.nn.Module):
    """
    Feedforward policy network: shared trunk -> actor head.

    Structurally analogous to ActionQNetwork but without the Q head.
    Drop in your own architecture here — GRPO only requires logits over actions.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        hidden_dims_actor: list[int],
        activation: callable,
    ):
        super().__init__()
        self.activation = activation

        # Shared trunk
        trunk_layers = []
        in_dim = state_dim
        for h in hidden_dims:
            trunk_layers += [torch.nn.Linear(in_dim, h)]
            in_dim = h
        self.trunk = torch.nn.ModuleList(trunk_layers)

        # Actor head
        actor_layers = []
        for h in hidden_dims_actor:
            actor_layers += [torch.nn.Linear(in_dim, h)]
            in_dim = h
        actor_layers += [torch.nn.Linear(in_dim, action_dim)]
        self.actor = torch.nn.Sequential(*actor_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.trunk:
            x = self.activation(layer(x))
        return self.actor(x)


# ---------------------------------------------------------------------------
# Helpers (shared with VAQAgent codebase)
# ---------------------------------------------------------------------------

def _masked_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return logits
    return logits.masked_fill(~mask.bool(), float("-inf"))
