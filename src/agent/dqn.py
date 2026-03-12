import copy

import torch
import torch.nn.functional as F

from src.agent.buffer import Buffer
from src.network.recurrent import RecurrentNetwork, RecurrentState


class RecurrentDQNAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # dqn properties
        gamma: float = 0.999,
        batch_size: int = 64,
        replay_capacity: int = 200_000,
        target_update_interval: int = 250,
        double_dqn: bool = True,
        # epsilon-greedy
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        # network properties
        hidden_dims: tuple[int] = [256, 256],
        activation: callable = torch.tanh,
        recurrent_type: str = "simple",
        recurrent_kwargs: dict | None = None,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu"
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.double_dqn = double_dqn

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = max(1, int(epsilon_decay_steps))
        self.epsilon = self.epsilon_start
        self.action_step_count = 0

        self.dtype = dtype
        self.device = torch.device(device)

        # environment dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # networks
        self.net = RecurrentNetwork(
            n_in=state_dim,
            n_out=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            recurrent_type=recurrent_type,
            recurrent_kwargs=recurrent_kwargs,
        ).to(device=self.device, dtype=self.dtype)

        self.target_net = copy.deepcopy(self.net).to(device=self.device, dtype=self.dtype)
        self.target_net.eval()

        # replay buffer
        self.buffer = Buffer(replay_capacity)

        self.net.reset(1)
        self.target_net.reset(1)
        self.update_step_count = 0

    def _update_epsilon(self):
        t = min(1.0, self.action_step_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def infer_q(self, state: torch.Tensor) -> torch.Tensor:
        q = self.net(state)
        return q.squeeze(0) if q.shape[0] == 1 else q

    def infer_q_from_seq(self, state_seq: torch.Tensor, h0: list[RecurrentState] | None = None) -> torch.Tensor:
        if h0 is not None:
            self.net.set_states(h0, strict=False)

        qs = []
        for state in state_seq:
            q = self.infer_q(state)
            qs.append(q.squeeze())
        return torch.stack(qs, dim=0)

    def act(self, state: torch.Tensor, explore: bool = True, grad_enabled: bool = False):
        with torch.set_grad_enabled(grad_enabled):
            state = state.to(dtype=self.dtype, device=self.device)
            q = self.infer_q(state)
            greedy_action = torch.argmax(q, dim=-1).long()

            if explore:
                self.action_step_count += 1
                self._update_epsilon()

                random_pick = torch.rand((), device=self.device).item() < self.epsilon
                if random_pick:
                    if greedy_action.dim() == 0:
                        action = torch.randint(self.action_dim, (), device=self.device)
                    else:
                        action = torch.randint(self.action_dim, greedy_action.shape, device=self.device)
                else:
                    action = greedy_action
            else:
                action = greedy_action

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

    def init_optimizer(self, lr, optim="AdamW"):
        opt_cls = torch.optim.AdamW if optim.lower() == "adamw" else torch.optim.Adam
        self.optim = opt_cls(self.net.parameters(), lr=lr)

    def update(self, h0: list[RecurrentState] | None = None, max_grad_norm: float | None = None) -> dict[str, float]:
        if not hasattr(self, "optim") or self.optim is None:
            raise RuntimeError("Call init_optimizer(...) before update().")

        if len(self.buffer) < self.batch_size:
            return {"loss": 0.0, "q_mean": 0.0, "td_error": 0.0, "epsilon": float(self.epsilon)}

        # save current recurrent state for rollout continuity
        h_t = self.net.get_states(clone=True, detach=True)

        self.net.train()
        self.target_net.eval()

        states, actions, next_states, rewards, dones = self.buffer.sample(
            self.batch_size, self.device, self.dtype
        )

        rewards = rewards.view(-1)
        dones = dones.view(-1)
        actions = actions.long().view(-1, 1)

        # step-wise recurrent training: treat samples as independent batch items
        self.net.reset(states.shape[0])
        q_pred_all = self.net(states)
        q_pred = q_pred_all.gather(1, actions).squeeze(-1)

        with torch.no_grad():
            if self.double_dqn:
                # online net chooses next action
                self.net.reset(next_states.shape[0])
                next_q_online = self.net(next_states)
                next_actions = next_q_online.argmax(dim=-1, keepdim=True)

                # target net evaluates chosen next action
                self.target_net.reset(next_states.shape[0])
                next_q_target = self.target_net(next_states)
                next_q = next_q_target.gather(1, next_actions).squeeze(-1)
            else:
                self.target_net.reset(next_states.shape[0])
                next_q_target = self.target_net(next_states)
                next_q = next_q_target.max(dim=-1).values

            target_q = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.smooth_l1_loss(q_pred, target_q)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)

        self.optim.step()

        td_error = (target_q - q_pred).abs()

        self.update_step_count += 1
        if self.update_step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # restore recurrent state from before learning
        self.net.set_states(h_t, clone=True, detach=True, strict=False)

        return {
            "loss": float(loss.detach().cpu().item()),
            "q_mean": float(q_pred.mean().detach().cpu().item()),
            "td_error": float(td_error.mean().detach().cpu().item()),
            "epsilon": float(self.epsilon),
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

                        msg = f"episode {episode} [{100*step/max_steps:.1f}%] - reward: {float(reward):.5f} - portfolio: {info['V']:.2f}"
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

            msg = f"episode {episode} [{100*step/max_steps:.1f}%] - avg. reward: {sum(episode_reward) / len(episode_reward) if store_results else 0.0:.5f} - portfolio: {info['V']:.2f} - epsilon: {self.epsilon:.5f}"
            if store_results and len(episode_loss) > 0:
                keys = episode_loss[0].keys()
                for key in keys:
                    msg += f" - {key}: {sum(ld[key] for ld in episode_loss) / len(episode_loss):.5f}"
            print(msg)

        return total_loss, total_reward, total_info

    def save(self, path: str, include_buffer: bool = False):
        checkpoint = {
            "net": self.net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optim.state_dict() if hasattr(self, "optim") else None,
            "update_step_count": self.update_step_count,
            "action_step_count": self.action_step_count,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "target_update_interval": self.target_update_interval,
            "double_dqn": self.double_dqn,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
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
        checkpoint = torch.load(path, map_location=map_location)

        self.net.load_state_dict(checkpoint["net"])
        self.target_net.load_state_dict(checkpoint["target_net"])

        if hasattr(self, "optim") and checkpoint["optimizer"] is not None:
            self.optim.load_state_dict(checkpoint["optimizer"])

        self.update_step_count = checkpoint.get("update_step_count", 0)
        self.action_step_count = checkpoint.get("action_step_count", 0)
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon_start))

        if load_buffer and "replay_buffer" in checkpoint:
            rb = checkpoint["replay_buffer"]
            self.buffer.memory = rb["memory"]
            self.buffer.idx = rb["idx"]
            self.buffer.full = rb["full"]
            self.buffer.memory_size = rb.get("memory_size", self.buffer.memory_size)

        self.net.to(self.device, dtype=self.dtype)
        self.target_net.to(self.device, dtype=self.dtype)
