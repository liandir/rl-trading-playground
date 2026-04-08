import torch
from torch import distributions

from src.network.vanilla import ActionValueNetwork
from src.network.recurrent import RecurrentActionValueNetwork


UPDATE_METRIC_NAMES = (
    "actor_loss",
    "critic_loss",
    "entropy",
    "total_loss",
    "ratio_mean",
    "ratio_max",
    "clip_frac",
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
    ratios: torch.Tensor,
    eps_clip: float,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values_pred: torch.Tensor,
    actor_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    entropy: torch.Tensor,
    loss: torch.Tensor,
):
    ratios_d = ratios.detach()
    adv_d = advantages.detach()
    ret_d = returns.detach()
    val_d = values_pred.detach()
    residuals = ret_d - val_d

    metrics = {
        "actor_loss": actor_loss.detach(),
        "critic_loss": critic_loss.detach(),
        "entropy": entropy.detach(),
        "total_loss": loss.detach(),
        "ratio_mean": ratios_d.mean(),
        "ratio_max": ratios_d.max(),
        "clip_frac": ((ratios_d - 1.0).abs() > eps_clip).to(dtype=ret_d.dtype).mean(),
        "explained_variance": _explained_variance(ret_d, residuals),
        "adv_abs_mean": adv_d.abs().mean(),
        "adv_pos_frac": (adv_d > 0).to(dtype=ret_d.dtype).mean(),
    }
    for name, value in metrics.items():
        metric_store[name].append(float(value.cpu().item()))


class RolloutBuffer:

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def store(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def to_tensors(self, device, dtype):
        states = torch.stack(self.states).to(device=device, dtype=dtype)
        actions = torch.stack(self.actions).to(device=device, dtype=dtype)
        log_probs = torch.stack(self.log_probs).to(device=device, dtype=dtype)
        rewards = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones = torch.stack(self.dones).to(device=device, dtype=dtype)
        return states, actions, log_probs, rewards, dones


class PPOAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # basic properties
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        normalize_returns: bool = False,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        hidden_dims_value: list[int] = [256],
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantages = normalize_advantages
        self.normalize_returns = normalize_returns
        self.dtype = dtype
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        self.log_std = torch.nn.Parameter(
            torch.zeros(1, self.action_dim, device=self.device, dtype=self.dtype),
            requires_grad=True,
        )

        self.buffer = RolloutBuffer()

    def infer_dist_from_seq(self, state: torch.Tensor) -> tuple[distributions.Normal, torch.Tensor]:
        mean, val = self.net.av(state)
        std = self.log_std.exp().expand_as(mean)
        return distributions.Normal(mean, std), val

    def act(
        self,
        state: torch.Tensor,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            dist, _ = self.infer_dist_from_seq(state)
            action = dist.rsample().squeeze() if explore else dist.mean.squeeze()
            log_prob = dist.log_prob(action).sum(dim=-1)

        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def store(self, state, action, log_prob, reward, done):
        self.buffer.store(state, action, log_prob, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls([
            {"params": self.net.parameters(), "lr": lr},
            {"params": [self.log_std], "lr": lr},
        ], lr=lr)

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 4,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()

        advantages, returns, states, actions, old_log_probs = self.compute_gae(
            last_next_state=last_next_state,
        )

        for _ in range(k_epochs):
            dist, values_pred = self.infer_dist_from_seq(states)
            values_pred = values_pred.squeeze(-1) if values_pred.dim() > 1 else values_pred

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(new_log_probs - old_log_probs)

            adv = advantages.detach()
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            ret = returns.detach()
            critic_loss = torch.square(ret - values_pred).mean()

            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + [self.log_std],
                    max_grad_norm,
                )

            self.optim.step()

            _append_update_metrics(
                metrics,
                ratios=ratios,
                eps_clip=self.eps_clip,
                advantages=adv,
                returns=ret,
                values_pred=values_pred,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                loss=loss,
            )

        return metrics

    def compute_gae(
        self,
        last_next_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, old_log_probs, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)

        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > 1 else dones
        old_log_probs = old_log_probs.squeeze(-1) if old_log_probs.dim() > 1 else old_log_probs

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            _, values = self.infer_dist_from_seq(states)
            values = values.squeeze(-1) if values.dim() > 1 else values

            last_ns = last_next_state.unsqueeze(0) if last_next_state.dim() == 1 else last_next_state
            _, last_value = self.infer_dist_from_seq(last_ns)
            last_value = last_value.squeeze()

            # append bootstrap value as V(s_{T+1})
            values_ext = torch.cat([values, last_value.unsqueeze(0)], dim=0)

            advantages = torch.zeros_like(rewards)
            gae = torch.zeros((), device=self.device, dtype=self.dtype)

            for t in range(rewards.shape[0] - 1, -1, -1):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * mask * values_ext[t + 1] - values_ext[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae

            returns = advantages + values

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            if self.normalize_returns:
                returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        return advantages, returns, states, actions, old_log_probs

    def save(self, path: str):
        torch.save({
            "network": self.net.state_dict(),
            "log_std": self.log_std.detach().cpu(),
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)
        with torch.no_grad():
            self.log_std.copy_(ckpt["log_std"].to(device=self.device, dtype=self.dtype))

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=4,
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
            self.buffer.clear()

            if store_results:
                episode_reward = []
                episode_loss = []
                episode_info = []

            step = 1
            while True:
                if step > warm_up:
                    action, log_prob = self.act(state.to_tensor(), explore=True, grad_enabled=False)
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
                        log_prob.detach(),
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
                        )
                        self.buffer.clear()

                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}€"
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

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - total reward: {sum(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}€"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info


class RecurrentPPOAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # ppo properties
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        normalize_advantages: bool = True,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        hidden_dims_value: list[int] = [256],
        activation: callable = torch.relu,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.normalize_advantages = normalize_advantages
        self.dtype = dtype
        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = RecurrentActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
        ).to(device=self.device, dtype=self.dtype)

        self.log_std = torch.nn.Parameter(
            torch.zeros(1, self.action_dim, device=self.device, dtype=self.dtype),
            requires_grad=True,
        )

        self.buffer = RolloutBuffer()
        self.net.reset(1)

    def infer_dist(self, state: torch.Tensor) -> distributions.Normal:
        act = self.net.a(state)
        std = self.log_std.exp().expand_as(act)
        return distributions.Normal(act, std)

    def infer_vals(self, state: torch.Tensor) -> torch.Tensor:
        return self.net.v(state)

    def infer_vals_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> torch.Tensor:
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        vals = []
        for state in state_seq:
            val = self.net.v(state)
            vals.append(val.squeeze())
        return torch.stack(vals, dim=0)

    def infer_dist_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> tuple[distributions.Normal, torch.Tensor]:
        if h0 is not None:
            self.net.set_states(h0, strict=False)
        acts, vals = [], []
        for state in state_seq:
            act, val = self.net.av(state)
            acts.append(act.squeeze())
            vals.append(val.squeeze())
        acts = torch.stack(acts, dim=0)
        vals = torch.stack(vals, dim=0)
        stds = self.log_std.exp().expand_as(acts)
        return distributions.Normal(acts, stds), vals

    def act(
        self,
        state: torch.Tensor,
        explore: bool = False,
        grad_enabled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            dist = self.infer_dist(state)
            action = dist.rsample().squeeze() if explore else dist.mean.squeeze()
            log_prob = dist.log_prob(action).sum(dim=-1)

        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def store(self, state, action, log_prob, reward, done):
        self.buffer.store(state, action, log_prob, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls([
            {"params": self.net.parameters(), "lr": lr},
            {"params": [self.log_std], "lr": lr},
        ], lr=lr)

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 4,
        h0: dict | None = None,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return _empty_update_metrics()

        metrics = _init_update_metrics()

        # save recurrent state to restore after updates
        h_t = self.net.get_states(clone=True, detach=True)

        advantages, returns, states, actions, old_log_probs = self.compute_gae(
            last_next_state=last_next_state, h0=h0
        )

        for _ in range(k_epochs):
            dist, values_pred = self.infer_dist_from_seq(states, h0=h0)

            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(new_log_probs - old_log_probs)

            adv = advantages.detach()
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()

            ret = returns.detach()
            critic_loss = torch.square(ret - values_pred).mean()

            loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + [self.log_std],
                    max_grad_norm,
                )

            self.optim.step()

            _append_update_metrics(
                metrics,
                ratios=ratios,
                eps_clip=self.eps_clip,
                advantages=adv,
                returns=ret,
                values_pred=values_pred,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                entropy=entropy,
                loss=loss,
            )

        # restore recurrent state for continued rollout
        self.net.set_states(h_t, clone=True, detach=True)

        return metrics

    def compute_gae(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, old_log_probs, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            values = self.infer_vals_from_seq(
                torch.cat([states, last_next_state[None]], dim=0), h0=h0
            )

            advantages = torch.zeros_like(rewards)
            gae = torch.zeros(values.shape[1:], device=self.device, dtype=self.dtype)

            for t in range(rewards.shape[0] - 1, -1, -1):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * mask * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae

            returns = advantages + values[:-1]

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, returns, states, actions, old_log_probs

    def save(self, path: str):
        torch.save({
            "network": self.net.state_dict(),
            "log_std": self.log_std.detach().cpu(),
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)
        with torch.no_grad():
            self.log_std.copy_(ckpt["log_std"].to(device=self.device, dtype=self.dtype))

    def train_on_historical(
        self,
        env,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=4,
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
            self.buffer.clear()
            h0 = self.net.get_states(clone=True, detach=True)

            if store_results:
                episode_reward = []
                episode_loss = []
                episode_info = []

            step = 1
            while True:
                if step > warm_up:
                    action, log_prob = self.act(state.to_tensor(), explore=True, grad_enabled=False)
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
                        log_prob.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = done or (step >= max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_state.to_tensor(),
                            k_epochs=n_updates,
                            h0=h0,
                            max_grad_norm=max_grad_norm,
                        )
                        self.buffer.clear()

                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}€"
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

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - avg. reward: {sum(episode_reward) / len(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f}€"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def train_on_historical_parallel(
        self,
        envs,
        data,
        n_episodes,
        max_steps=2000,
        warm_up=0,
        update_interval=100,
        n_updates=4,
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
            start = torch.randint(len(data) - max_steps, size=[envs.n_envs])
            states, _ = envs.reset([data[j] for j in start])
            self.buffer.clear()
            self.net.reset(envs.n_envs)
            h0 = self.net.get_states(clone=True, detach=True)

            if store_results:
                episode_reward = []
                episode_loss = []
                episode_info = []

            step = 1
            while True:
                if step > warm_up:
                    actions, log_probs = self.act(states, explore=True, grad_enabled=False)
                else:
                    actions = None

                next_states, rewards, dones, infos = envs.step(actions, [data[j + step] for j in start])

                if store_results:
                    episode_reward.append(rewards)
                    episode_info.append(infos)

                if step > warm_up:
                    self.store(
                        states.detach(),
                        actions.detach(),
                        log_probs.detach(),
                        rewards.detach(),
                        dones.detach(),
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = dones.any() or (step >= max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_states,
                            k_epochs=n_updates,
                            h0=h0,
                            max_grad_norm=max_grad_norm,
                        )
                        self.buffer.clear()

                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(rewards.mean()):.5f}"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break

                else:
                    if step % update_interval == 1:
                        print(f"episode {episode} [{100*step/max_steps:.1f}%] - warm up in progress...", end="\r")

                states = next_states
                step += 1

            if store_results:
                total_loss.append(episode_loss)
                total_reward.append(episode_reward)
                total_info.append(episode_info)

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - average reward: {sum([float(r.mean()) for r in episode_reward]) if store_results else 0.0:.5f}"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(sum(ld[key]) / len(ld[key]) for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info
