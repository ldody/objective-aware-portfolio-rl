from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size FIFO replay buffer for off-policy RL."""

    def __init__(self, obs_dim: int, action_dim: int, size: int = 100_000):
        self.size = size
        self.ptr = 0
        self.full = False

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size: int, device: str):
        max_idx = self.size if self.full else self.ptr
        idxs = np.random.randint(0, max_idx, size=batch_size)
        obs = torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=device)
        action = torch.tensor(self.action_buf[idxs], dtype=torch.float32, device=device)
        reward = torch.tensor(self.reward_buf[idxs], dtype=torch.float32, device=device)
        next_obs = torch.tensor(self.next_obs_buf[idxs], dtype=torch.float32, device=device)
        done = torch.tensor(self.done_buf[idxs], dtype=torch.float32, device=device)
        return obs, action, reward, next_obs, done
