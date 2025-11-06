import copy
import torch


class Buffer:

    def __init__(self, memory_size, state_size, action_size):
        self.memory_size = memory_size
        self.full = 0
        self.idx = 0

        self.states = torch.zeros([memory_size, state_size])
        self.actions = torch.zeros([memory_size, action_size])
        self.next_states = torch.zeros([memory_size, state_size])
        self.rewards = torch.zeros([memory_size])
        self.dones = torch.zeros([memory_size])

    def __len__(self):
        return self.memory_size if self.full else self.idx

    def store(self, state, action, next_state, reward, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        new_idx = (self.idx+1) % self.memory_size
        if new_idx == 0:
            self.full = 1
        self.idx = new_idx

    def sample(self, batch_size, device):
        n_memories = len(self)
        batch_idx = torch.randint(0, n_memories, size=[batch_size])
        return (
            self.states[batch_idx].to(device),
            self.actions[batch_idx].to(device),
            self.next_states[batch_idx].to(device),
            self.rewards[batch_idx].to(device),
            self.dones[batch_idx].to(device)
        )
        

class EntropyRegAgent:
    def __init__(self,
                 state_size,
                 action_size,
                 actor,
                 critic,
                 actor_lr=1e-3,
                 critic_lr=1e-3,
                 actor_m=0.0,
                 critic_m=0.0,
                 gamma=0.99,
                 tau=100.0,
                 alpha=0.01,  # Initial entropy coefficient
                 target_entropy=None,  # SAC-style target entropy
                 buffer_size=10000,
                 device="cpu"):
        """
        Soft Actor-Critic (SAC)-style agent with learnable entropy coefficient.
        """
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Actor & Critic Networks
        self.a   = actor.to(device)
        self.a_t = copy.deepcopy(actor).to(device)
        self.q   = critic.to(device)
        self.q_t = copy.deepcopy(critic).to(device)

        self.update_target_parameters()
        self.update_optimizers(actor_lr, critic_lr, actor_m, critic_m)

        # Replay Buffer
        self.buffer = Buffer(buffer_size, self.state_size, self.action_size)

        # Learnable Entropy Coefficient
        self.log_alpha = torch.tensor(torch.log(torch.tensor(max(alpha, 1e-6))), requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=1e-4)

        # Target entropy (if None, use default from SAC)
        self.target_entropy = target_entropy if target_entropy is not None else -self.action_size

    def act(self, state, deterministic=True):
        state = state.to(self.device)
        if deterministic:
            action = self.a.action(state)
        else:
            action, _ = self.a.sample(state)

        return action[0].cpu()

    def update_optimizers(self, a_lr, q_lr, a_m, q_m):
        self.a_optim = torch.optim.SGD(self.a.parameters(), lr=a_lr, momentum=a_m)
        self.q_optim = torch.optim.SGD(self.q.parameters(), lr=q_lr, momentum=q_m)

    def update_target_parameters(self):
        """Polyak averaging for target networks."""
        with torch.no_grad():
            for target_param, param in zip(self.a_t.parameters(), self.a.parameters()):
                target_param.data.lerp_(param.data, 1 / self.tau)
            for target_param, param in zip(self.q_t.parameters(), self.q.parameters()):
                target_param.data.lerp_(param.data, 1 / self.tau)

    def train_step(self, batch_size):
        """Perform one SAC training step."""
        states, actions, next_states, rewards, dones = self.buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.a_t.sample(next_states)
            next_q_values = self.q_t(next_states, next_actions) - self.log_alpha.exp() * next_log_probs
            q_targets = rewards + self.gamma * (1 - dones) * next_q_values.detach()

        # Critic Loss (Q-function)
        self.q_optim.zero_grad()
        q_values = self.q(states, actions)
        q_loss = torch.square(q_targets - q_values).mean()
        q_loss.backward()
        self.q_optim.step()

        # Actor Loss (Policy Update)
        self.a_optim.zero_grad()
        actions_, log_probs_ = self.a.sample(states)
        log_probs_ = torch.nan_to_num(log_probs_, nan=0.0, neginf=-20.0, posinf=0.0)
        q_values_ = self.q(states, actions_)
        a_loss = (self.log_alpha.exp() * log_probs_ - q_values_).mean()
        a_loss.backward()
        self.a_optim.step()

        # Entropy Coefficient (Alpha) Update
        self.alpha_optim.zero_grad()
        alpha_loss = (-self.log_alpha.exp() * (log_probs_ + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Update Target Networks
        self.update_target_parameters()

        return -a_loss.item(), q_loss.item(), alpha_loss.item()

    def train(self, env,
              n_episodes,
              batch_size=32,
              update_interval=100,
              n_updates=8,
              max_steps=1000,
              store=1,
              actor_lr=1e-3,
              critic_lr=1e-3,
              actor_m=0.0,
              critic_m=0.0):
        """Train the agent in the given environment."""
        self.update_optimizers(actor_lr, critic_lr, actor_m, critic_m)

        if store:
            total_loss_a = []
            total_loss_c = []
            total_loss_alpha = []
            total_reward = []

        for episode in range(1, n_episodes+1):
            state  = env.reset().to_tensor()

            if store:
                episode_loss_a = []
                episode_loss_c = []
                episode_loss_alpha = []
                episode_reward = []

            counter = 1
            while True:
                action = self.act(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                self.buffer.store(state.detach(),
                                  action.detach(),
                                  next_state.to_tensor().detach(),
                                  reward,
                                  done)
                episode_reward.append(reward)
            
                if counter % update_interval == 0:
                    for _ in range(n_updates):
                        loss_a, loss_c, loss_alpha = self.train_step(batch_size)
                        episode_loss_a.append(loss_a)
                        episode_loss_c.append(loss_c)
                        episode_loss_alpha.append(loss_alpha)
                    print(f"episode {episode} - reward: {sum(episode_reward):.2f}", end="\r")

                if done or counter > max_steps:
                    break

                state = next_state
                counter += 1

            if store:
                total_loss_a.append(torch.tensor(episode_loss_a))
                total_loss_c.append(torch.tensor(episode_loss_c))
                total_loss_alpha.append(torch.tensor(episode_loss_alpha))
                total_reward.append(torch.tensor(episode_reward))

            print(f"episode {episode} - loss_a: {sum(episode_loss_a)/counter:.5f} - loss_c: {sum(episode_loss_c)/counter:.5f} - loss_alpha: {sum(episode_loss_alpha)/counter:.5f} - reward: {sum(episode_reward):.5f}")
        
        if store:
            return (
                torch.stack(total_loss_a),
                torch.stack(total_loss_c),
                torch.stack(total_loss_alpha),
                torch.stack(total_reward),
            )