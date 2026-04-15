import torch
from torch import distributions

from src.network.vanilla import ActionValueNetwork
from src.network.recurrent import RecurrentActionValueNetwork


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
        states     = torch.stack(self.states).to(device=device, dtype=dtype)
        actions    = torch.stack(self.actions).to(device=device, dtype=dtype)
        rewards    = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones      = torch.stack(self.dones).to(device=device, dtype=dtype)
        return states, actions, rewards, dones


class VACAgent:
    """
    Vanilla Actor-Critic for discrete actions (on-policy):
      - Policy: Categorical(logits)
      - Critic: TD(0) bootstrap target y_t = r_t + gamma * (1-done_t) * V(s_{t+1})
      - Actor:  -(logpi(a_t|s_t) * A_t) - ent_coef * H[pi(.|s_t)]
      - Critic: 0.5 * (y_t - V(s_t))^2
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # basic properties
        gamma: float = 0.999,
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

        # network
        self.net = ActionValueNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            hidden_dims_actor=hidden_dims_actor,
            hidden_dims_value=hidden_dims_value,
            activation=activation,
        ).to(device=self.device, dtype=self.dtype)

        # on-policy buffer
        self.buffer = RolloutBuffer()

    def infer_dist_from_seq(self, state: torch.Tensor) -> tuple[distributions.Categorical, torch.Tensor]:
        logits, val = self.net.av(state)
        val = val.squeeze(-1) if val.dim() > 1 else val
        return distributions.Categorical(logits=logits), val

    def act(self, state: torch.Tensor, explore: bool = False, grad_enabled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: shape (state_dim,) or (1, state_dim)
        returns:
          - action index tensor
          - log_prob tensor
        """
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)

            dist, _ = self.infer_dist_from_seq(state)
            if explore:
                action = dist.sample().squeeze()
            else:
                action = torch.argmax(dist.logits, dim=-1).squeeze()

            log_prob = dist.log_prob(action)

        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def store(self, state, action, reward, done):
        self.buffer.store(state, action, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def update(
        self,
        last_next_state: torch.Tensor,
        last_done: bool | torch.Tensor | None = None,
        k_epochs: int = 1,
        max_grad_norm: float | None = None,
    ) -> dict[str, list[float]]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return {"policy_objective": [0.0], "critic_loss": [0.0], "entropy": [0.0]}

        policy_objectives = []
        critic_losses = []
        entropy_losses = []

        advantages, targets, states, actions = self.compute_td0_advantages(
            last_next_state=last_next_state,
            last_done=last_done,
        )

        for _ in range(k_epochs):
            dist, values_pred = self.infer_dist_from_seq(states)
            actions_long = actions.long()

            log_probs = dist.log_prob(actions_long)
            entropy = dist.entropy().mean()

            adv = advantages.detach()
            policy_objective = (log_probs * adv).mean()

            td = targets.detach() - values_pred
            critic_loss = 0.5 * td.pow(2).mean()

            loss = -policy_objective + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

            self.optim.step()

            policy_objectives.append(float(policy_objective.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            entropy_losses.append(float(entropy.detach().cpu().item()))

        return {
            "policy_objective": policy_objectives,
            "critic_loss": critic_losses,
            "entropy": entropy_losses,
        }

    def compute_td0_advantages(
        self,
        last_next_state: torch.Tensor,
        last_done: bool | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TD(0) targets:
          y_t = r_t + gamma * (1-done_t) * V(s_{t+1})
        Advantage:
          A_t = y_t - V(s_t)
        """
        # unpack rollout
        states, actions, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)

        # ensure shapes are (T,)
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > 1 else dones
        with torch.no_grad():
            last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)
            values = self.net.v(torch.concat([states, last_next_state[None]], dim=0))
            values = values.squeeze(-1) if values.dim() > 1 else values  # (T+1,)

            v_t = values[:-1]
            v_tp1 = values[1:]

            targets = rewards + self.gamma * (1.0 - dones) * v_tp1
            advantages = targets - v_t

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            if self.normalize_returns:
                targets = (targets - targets.mean()) / (targets.std() + 1e-7)

        return advantages, targets, states, actions

    def save(self, path: str):
        torch.save({
            "network": self.net.state_dict(),
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)

    def train_on_historical(
            self, env, data,
            n_episodes,
            update_interval=100,
            n_updates=1,
            max_steps=2000,
            warm_up=0,
            lr=3e-4,
            optim="AdamW",
            init_optimizer=True,
            store_results=True,
        ):
        # initialize optimizer
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
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = done or (step >= max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        loss_dict = self.update(
                            last_next_state=next_state.to_tensor(),
                            last_done=done,
                            k_epochs=n_updates,
                            max_grad_norm=self.max_grad_norm,
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
                    if step % update_interval == 0:
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


class RecurrentVACAgent:
    """
    Recurrent Vanilla Actor-Critic for discrete actions:
      - Policy: Categorical(logits) from recurrent actor head
      - Critic: recurrent TD(0) bootstrap with sequence-consistent V(s_t), V(s_{t+1})
      - Actor:  -(logpi(a_t|s_t) * A_t) - ent_coef * H[pi(.|s_t)]
      - Critic: 0.5 * (y_t - V(s_t))^2
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # vac properties
        gamma: float = 0.999,
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
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
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
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
        ).to(device=self.device, dtype=self.dtype)

        # on-policy buffer
        self.buffer = RolloutBuffer()
        self.net.reset(1)

    def infer_dist(self, state: torch.Tensor) -> distributions.Categorical:
        logits = self.net.a(state)
        
        return distributions.Categorical(logits=logits)

    def infer_vals(self, state: torch.Tensor) -> torch.Tensor:
        val = self.net.v(state)

        return val.squeeze(-1) if val.dim() > 1 else val

    def infer_vals_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> torch.Tensor:
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        vals = []
        for state in state_seq:
            val = self.net.v(state)
            vals.append(val.squeeze())
        vals = torch.stack(vals, dim=0)

        return vals

    def infer_dist_from_seq(self, state_seq: torch.Tensor, h0: dict | None = None) -> tuple[distributions.Categorical, torch.Tensor]:
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        logits_seq, vals = [], []
        for state in state_seq:
            logits, val = self.net.av(state)
            logits_seq.append(logits.squeeze())
            vals.append(val.squeeze())
        logits_seq = torch.stack(logits_seq, dim=0)
        vals = torch.stack(vals, dim=0)

        return distributions.Categorical(logits=logits_seq), vals

    def act(self, state: torch.Tensor, explore: bool = False, grad_enabled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)

            dist = self.infer_dist(state)
            if explore:
                action = dist.sample().squeeze()
            else:
                action = torch.argmax(dist.logits, dim=-1).squeeze()

            log_prob = dist.log_prob(action)

        if grad_enabled:
            return action, log_prob
        return action.cpu(), log_prob.cpu()

    def store(self, state, action, reward, done):
        self.buffer.store(state, action, reward, done)

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def update(
        self,
        last_next_state: torch.Tensor,
        k_epochs: int = 1,
        h0: dict | None = None,
        max_grad_norm: float | None = None
    ) -> dict[str, float]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        actor_losses = []
        critic_losses = []
        entropy_losses = []

        advantages, targets, states, actions = self.compute_td0_advantages(
            last_next_state=last_next_state, h0=h0
        )

        for _ in range(k_epochs):
            dists, values_pred = self.infer_dist_from_seq(states, h0=h0)
            actions_long = actions.long()

            log_probs = dists.log_prob(actions_long)
            entropy = dists.entropy().mean()

            adv = advantages.detach()
            policy_objective = (log_probs * adv).mean()

            td = targets.detach() - values_pred
            critic_loss = 0.5 * td.pow(2).mean()

            loss = -policy_objective + self.vf_coef * critic_loss - self.ent_coef * entropy

            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

            self.optim.step()

            actor_losses.append(float(policy_objective.detach().cpu().item()))
            critic_losses.append(float(critic_loss.detach().cpu().item()))
            entropy_losses.append(float(entropy.detach().cpu().item()))

        return {
            "policy_objective": actor_losses,
            "critic_loss": critic_losses,
            "entropy": entropy_losses,
        }

    def compute_td0_advantages(
        self,
        last_next_state: torch.Tensor,
        h0: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        TD(0) with recurrence:
          y_t = r_t + gamma * (1-done_t) * V(s_{t+1})
          A_t = y_t - V(s_t)
        """
        states, actions, rewards, dones = self.buffer.to_tensors(self.device, self.dtype)
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions
        rewards = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
        dones = dones.squeeze(-1) if dones.dim() > 1 else dones

        last_next_state = last_next_state.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # values for s_0..s_T (append last_next_state for bootstrap)
            values = self.infer_vals_from_seq(
                torch.concat([states, last_next_state[None]], dim=0), h0=h0
            )  # shape (T+1,)

            v_t = values[:-1]
            v_tp1 = values[1:]

            targets = rewards + self.gamma * (1.0 - dones) * v_tp1
            advantages = targets - v_t

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        return advantages, targets, states, actions

    def train_on_historical(
            self, env, data,
            n_episodes,
            max_steps=2000,
            warm_up=0,
            update_interval=100,
            n_updates=1,
            lr=3e-4,
            optim="AdamW",
            init_optimizer=False,
            store_results=True,
            max_grad_norm=None
        ):
        # initialize optimizer
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
                        torch.tensor(reward, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                        torch.tensor(done, dtype=env.dtype if hasattr(env, "dtype") else torch.float32),
                    )

                    rollout_ready = len(self.buffer) >= update_interval
                    terminal = done or (step >= max_steps)
                    if (rollout_ready or terminal) and len(self.buffer) > 2:
                        h_t = self.net.get_states(clone=True, detach=True)
                        loss_dict = self.update(
                            last_next_state=next_state.to_tensor(),
                            k_epochs=n_updates,
                            h0=h0,
                            max_grad_norm=max_grad_norm
                        )
                        self.net.set_states(h0 := h_t, clone=True, detach=True)
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
                    if step % update_interval == 0:
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
                    msg += f" - {key}: {sum(
                        sum(ld[key]) / len(ld[key]) for ld in episode_loss
                    ) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def save(self, path: str):
        torch.save({
            "network": self.net.state_dict(),
        }, path)

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["network"], strict=strict)
