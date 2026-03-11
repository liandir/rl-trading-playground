import torch
from torch import distributions

from src.network.vanilla import ActionValueNetwork
from src.network.recurrent import RecurrentActionValueNetwork
from .buffer import RolloutBuffer


class PPOAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # basic properties
        gamma: float = 0.999,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float | None = None,
        # advantages and returns
        normalize_returns: bool = False,
        normalize_advantages: bool = True,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        hidden_dims_value: list[int] = [256],
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_returns = normalize_returns
        self.normalize_advantages = normalize_advantages
        self.dtype = dtype
        self.device = torch.device(device)

        # environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # networks
        self.net = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        # learnable log std (one per action dim), like your prototype
        self.log_std = torch.nn.Parameter(
            torch.zeros(1, self.action_dim, device=self.device, dtype=self.dtype),
            requires_grad=True
        )

        # on-policy buffer
        self.buffer = RolloutBuffer()

    def infer_dist_from_seq(self, state: torch.Tensor) -> distributions.Normal:
        mean, val = self.net.av(state)
        std = self.log_std.exp().expand_as(mean)
        return distributions.Normal(mean, std), val

    def act(self, state: torch.Tensor, explore: bool = False, grad_enabled: bool = False) -> tuple[torch.Tensor, float]:
        """
        state: shape (state_dim,) or (1, state_dim)
        returns:
          - action tensor shape (action_dim,)
          - log_prob float (sum over action dims)
        """
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)

            dist, _ = self.infer_dist_from_seq(state)
            if explore:
                action = dist.rsample().squeeze()
            else:
                action = dist.mean.squeeze()

            log_prob = dist.log_prob(action).sum(dim=-1)
        
        if grad_enabled:
            return action, log_prob
        else:
            return action.cpu(), log_prob.cpu()

    def store(self, state, action, log_prob, reward, done):
        self.buffer.store(state, action, log_prob, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls([
            {"params": self.net.parameters(), "lr": lr},
            {"params": [self.log_std], "lr": lr},
        ], lr=lr)

    def update(self, last_next_state: torch.Tensor, last_done: bool | torch.Tensor) -> dict[str, float]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        advantages, returns, states, actions, old_log_probs = self.compute_gae(
            last_next_state=last_next_state,
            last_done=last_done,
        )

        # PPO epochs (multiple passes over the same on-policy data)
        for _ in range(self.k_epochs):
            dist, values_pred = self.infer_dist_from_seq(states)

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

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.net.parameters()) + [self.log_std],
                    self.max_grad_norm
                )

            self.optim.step()

        self.buffer.clear()

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
        }

    def compute_gae(
        self,
        last_next_state: torch.Tensor,
        last_done: bool | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # unpack rollout
        states, actions, old_log_probs, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)

        # ensure shapes are (T,)
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > 1 else dones
        old_log_probs = old_log_probs.squeeze(-1) if old_log_probs.dim() > 1 else old_log_probs

        # last_done to tensor on device/dtype
        if isinstance(last_done, bool):
            last_done_t = torch.tensor(float(last_done), device=self.device, dtype=self.dtype)
        else:
            last_done_t = last_done.to(device=self.device, dtype=self.dtype)
            if last_done_t.numel() != 1:
                last_done_t = last_done_t.reshape(-1)[0]

        with torch.no_grad():
            # V(s_t)
            _, values = self.infer_dist_from_seq(states)
            values = values.squeeze(-1) if values.dim() > 1 else values  # (T,)

            # bootstrap last value V(s_T) unless terminal
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            if last_next_state.dim() == 1:
                last_next_state_b = last_next_state.unsqueeze(0)
            else:
                last_next_state_b = last_next_state

            _, last_value = self.infer_dist_from_seq(last_next_state_b)
            last_value = last_value.squeeze()

            # next_values[t] = V(s_{t+1}), constructed by shift + bootstrap
            next_values = torch.empty_like(values)
            if values.shape[0] > 1:
                next_values[:-1] = values[1:]
            next_values[-1] = (1.0 - last_done_t) * last_value

            # GAE recursion
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros((), device=self.device, dtype=self.dtype)
            lam = getattr(self, "gae_lambda", 0.95)

            for t in range(rewards.shape[0] - 1, -1, -1):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * mask * next_values[t] - values[t]
                gae = delta + self.gamma * lam * mask * gae
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
            self, env, data,
            n_episodes,
            update_interval=100,    # rollout length before an update
            n_updates=8,            # mapped to PPO k_epochs (like "reuse buffer for training")
            max_steps=2000,
            warm_up=0,              # env warm up steps before acting
            lr=3e-4,
            optim="AdamW",
            init_optimizer=True,
            store_results=True,
        ):
        # initialize optimizer
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        # Map n_updates -> PPO epochs if the self supports it (mirrors SAC's "n_updates" intent)
        if hasattr(self, "k_epochs"):
            self.k_epochs = int(n_updates)

        # result storage
        total_reward = []
        total_loss = []
        total_info = []

        # main training loop
        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])

            if store_results:
                episode_reward = []
                episode_info = []
                episode_loss = []

            # Make sure the PPO rollout buffer is empty at episode start
            if hasattr(self, "buffer") and hasattr(self.buffer, "clear"):
                self.buffer.clear()
            elif hasattr(self, "buffer"):
                self.buffer = []

            step = 0
            while True:
                # action selection
                if step > warm_up:
                    action, log_prob = self.act(state.to_tensor(), explore=True, grad_enabled=False)
                else:
                    action = None

                # environment step
                next_state, reward, done, info = env.step(action, data[start + step])
                
                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                # store rollout transition for PPO
                if step > warm_up:
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        log_prob.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    # update when rollout chunk is ready, or at episode end
                    rollout_ready = ((step + 1) % update_interval == 0)
                    terminal = done or (step > max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        # PPO update consumes current rollout buffer
                        loss_dict = self.update(last_next_state=next_state.to_tensor(), last_done=done)

                        if store_results:
                            episode_loss.append(loss_dict)

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}€"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {val:.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break
                
                else:
                    if (step + 1) % update_interval == 0:
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
                    msg += f" - {key}: {sum(ld[key] for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info
    

class RecurrentPPOAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # ppo properties
        gamma: float = 0.999,
        eps_clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        hidden_dims_actor: list[int] = [256],
        hidden_dims_value: list[int] = [256],
        activation: callable = torch.relu,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.dtype = dtype
        self.device = torch.device(device)

        # environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # network
        self.net = RecurrentActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        # learnable log std (one per action dim), like your prototype
        self.log_std = torch.nn.Parameter(
            torch.zeros(1, self.action_dim, device=self.device, dtype=self.dtype),
            requires_grad=True
        )

        # on-policy buffer
        self.buffer = RolloutBuffer()
        self.net.reset(1)

    def infer_dist(self, state: torch.Tensor) -> distributions.Normal:
        act = self.net.a(state)
        std = self.log_std.exp().expand_as(act)
        return distributions.Normal(act, std)
    
    def infer_vals(self, state: torch.Tensor) -> distributions.Normal:
        return self.net.v(state)

    def infer_vals_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> torch.Tensor:
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        vals = []
        for state in state_seq:
            val = self.net.v(state)
            vals.append(val.squeeze())
        vals = torch.stack(vals, dim=0)

        return vals

    def infer_dist_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> distributions.Normal:
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

    def act(self, state: torch.Tensor, explore: bool = False, grad_enabled: bool = False) -> tuple[torch.Tensor, float]:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)

            dist = self.infer_dist(state)
            if explore:
                action = dist.rsample().squeeze()
            else:
                action = dist.mean.squeeze()

            log_prob = dist.log_prob(action).sum(dim=-1)
        
        if grad_enabled:
            return action, log_prob
        else:
            return action.cpu(), log_prob.cpu()

    def store(self, state, action, log_prob, reward, done):
        self.buffer.store(state, action, log_prob, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls([
            {"params": self.net.parameters(), "lr": lr},
            {"params": [self.log_std], "lr": lr},
        ], lr=lr)

    def update(self, last_next_state: torch.Tensor, k_epochs: int = 4, h0: dict | None = None, max_grad_norm: float | None = None) -> dict[str, float]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # save momentary network states to restore network memory after updates
        h_t = self.net.get_states(clone=True, detach=True)
        
        # store diagnostics
        actor_losses = []
        critic_losses = []
        entropy_losses = []

        # fetch generalized advantages
        advantages, returns, states, actions, old_log_probs = self.compute_gae(
            last_next_state=last_next_state, h0=h0
        )

        # PPO epochs (multiple passes over the same on-policy data)
        for _ in range(k_epochs):
            dists, values_pred = self.infer_dist_from_seq(states, h0=h0)
            
            new_log_probs = dists.log_prob(actions).sum(dim=-1)
            entropy = dists.entropy().sum(dim=-1).mean()

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
                    max_grad_norm
                )

            self.optim.step()

            actor_losses.append(float(actor_loss.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            entropy_losses.append(float(entropy.detach().cpu().item()))
        
        # restore network states to before update for next rollout chunk (important for recurrent nets)
        self.net.set_states(h0 := h_t, clone=True, detach=True)

        return {
            "actor_loss": actor_losses,
            "critic_loss": critic_losses,
            "entropy": entropy_losses,
        }

    def compute_gae(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # unpack rollout
        states, actions, old_log_probs, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)

        # last_done to tensor on device/dtype
        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            values = self.infer_vals_from_seq(
                torch.concat([states, last_next_state[None]], dim=0), h0=h0
            ) # V(s_t), t = 0, 1, 2, ..., T

            # GAE recursion
            advantages = torch.zeros_like(rewards)
            gae = torch.zeros(values.shape[1:], device=self.device, dtype=self.dtype)

            for t in range(rewards.shape[0] - 1, -1, -1):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * mask * values[t+1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages[t] = gae

            returns = advantages + values[:-1]

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, returns, states, actions, old_log_probs

    def train_on_historical(
            self, env, data,
            n_episodes,
            max_steps=2000,
            warm_up=0,              # env warm up steps before acting
            update_interval=100,    # rollout length before an update
            n_updates=8,            # mapped to PPO k_epochs (like "reuse buffer for training")
            lr=3e-4,
            optim="AdamW",
            init_optimizer=False,
            store_results=True,
            max_grad_norm=None
        ):
        # initialize optimizer
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        # result storage
        total_reward = []
        total_loss   = []
        total_info   = []

        # main training loop
        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[1]).item()
            state = env.reset(data[start])
            self.net.reset(), self.buffer.clear()
            h0 = self.net.get_states(clone=True, detach=True)

            if store_results:
                episode_reward = []
                episode_loss   = []
                episode_info   = []

            step = 1
            while True:
                # action selection
                if step > warm_up:
                    action, log_prob = self.act(state.to_tensor(), explore=True, grad_enabled=False)
                else:
                    action = None

                # environment step
                next_state, reward, done, info = env.step(action, data[start + step])
                
                if store_results:
                    episode_reward.append(reward)
                    episode_info.append(info)

                # store rollout transition for PPO
                if step > warm_up:
                    self.store(
                        state.to_tensor().detach(),
                        action.detach(),
                        log_prob.detach(),
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    # update when rollout chunk is ready, or at episode end
                    rollout_ready = (step % update_interval == 0)
                    terminal = done or (step > max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        # PPO update consumes current rollout buffer
                        loss_dict = self.update(
                            last_next_state=next_state.to_tensor(),
                            k_epochs=n_updates,
                            h0=h0,
                            max_grad_norm=max_grad_norm
                        )
                        self.buffer.clear()
                        
                        if store_results:
                            episode_loss.append(loss_dict)

                        # logging
                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}€"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
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
                    msg += f" - {key}: {sum(
                        sum(ld[key]) / len(ld[key]) for ld in episode_loss
                    ) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info
    
    def train_on_historical_parallel(
            self, envs, data,
            n_episodes,
            max_steps=2000,
            warm_up=0,              # env warm up steps before acting
            update_interval=100,    # rollout length before an update
            n_updates=8,            # mapped to PPO k_epochs (like "reuse buffer for training")
            lr=3e-4,
            optim="AdamW",
            init_optimizer=False,
            store_results=True,
            max_grad_norm=None
        ):
        # initialize optimizer
        if not hasattr(self, "optim") or init_optimizer:
            self.init_optimizer(lr, optim=optim)

        # result storage
        total_reward = []
        total_loss   = []
        total_info   = []

        # main training loop
        for episode in range(1, n_episodes + 1):
            start = torch.randint(len(data) - max_steps, size=[envs.n_envs])
            states, _ = envs.reset([data[j] for j in start])
            self.buffer.clear()
            self.net.reset(envs.n_envs)
            h0 = self.net.get_states(clone=True, detach=True)

            if store_results:
                episode_reward = []
                episode_loss   = []
                episode_info   = []

            step = 1
            while True:
                # action selection
                if step > warm_up:
                    actions, log_probs = self.act(states, explore=True, grad_enabled=False)
                else:
                    actions = None

                # environment step
                next_states, rewards, dones, infos = envs.step(actions, [data[j+step] for j in start])
                
                if store_results:
                    episode_reward.append(rewards)
                    episode_info.append(infos)

                # store rollout transition for PPO
                if step > warm_up:
                    self.store(
                        states.detach(),
                        actions.detach(),
                        log_probs.detach(),
                        rewards.detach(),
                        dones.detach(),
                    )

                    # update when rollout chunk is ready, or at episode end
                    rollout_ready = (step % update_interval == 0)
                    terminal = dones.any() or (step > max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        # PPO update consumes current rollout buffer
                        loss_dict = self.update(
                            last_next_state=next_states,
                            k_epochs=n_updates,
                            h0=h0,
                            max_grad_norm=max_grad_norm
                        )
                        self.buffer.clear()
                        
                        if store_results:
                            episode_loss.append(loss_dict)


                        # logging
                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(rewards.mean()):.5f}"
                        for key, val in loss_dict.items():
                            msg += f" - {key}: {sum(val) / len(val):.5f}"
                        print(msg, end="\r")

                    if terminal:
                        break
                
                else:
                    if step % update_interval == 0:
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
                    msg += f" - {key}: {sum(
                        sum(ld[key]) / len(ld[key]) for ld in episode_loss
                    ) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

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
