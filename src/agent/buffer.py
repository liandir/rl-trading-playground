import random
import torch


class Buffer:
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory = []
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.memory_size if self.full else self.idx

    def store(self, state, action, next_state, reward, done):
        """Store a transition (as tensors or arrays)."""
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
        """Sample random transitions as tensors on given device."""
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = zip(*batch)

        state = torch.stack(state).to(dtype=dtype, device=device)
        action = torch.stack(action).to(dtype=dtype, device=device)
        next_state = torch.stack(next_state).to(dtype=dtype, device=device)
        reward = torch.stack(reward).to(dtype=dtype, device=device)
        done = torch.stack(done).to(dtype=dtype, device=device)

        return state, action, next_state, reward, done


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
        states     = torch.stack(self.states).to(device=device, dtype=dtype)
        actions    = torch.stack(self.actions).to(device=device, dtype=dtype)
        log_probs  = torch.stack(self.log_probs).to(device=device, dtype=dtype)
        rewards    = torch.stack(self.rewards).to(device=device, dtype=dtype)
        dones      = torch.stack(self.dones).to(device=device, dtype=dtype)
        return states, actions, log_probs, rewards, dones