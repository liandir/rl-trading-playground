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
            # Overwrite oldest
            self.memory[self.idx] = transition
            self.idx = (self.idx + 1) % self.memory_size

    def sample(self, batch_size: int, device: str | torch.device):
        """Sample random transitions as tensors on given device."""
        batch = random.sample(self.memory, batch_size)

        state, action, next_state, reward, done = zip(*batch)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        done = torch.stack(done)

        return state, action, next_state, reward, done
