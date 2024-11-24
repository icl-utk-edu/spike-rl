import numpy as np
import torch
from spikerl.utils import combined_shape

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer with running mean and variance for normalization.
    """
    def __init__(self, obs_dim, act_dim, size, clip_limit, norm_update_every=1000):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # Running z-score normalization parameters
        self.clip_limit = clip_limit
        self.norm_update_every = norm_update_every
        self.norm_update_batch = np.zeros(combined_shape(norm_update_every, obs_dim), dtype=np.float32)
        self.norm_update_count = 0
        self.norm_total_count = np.finfo(np.float32).eps.item()
        self.mean = np.zeros(obs_dim, dtype=np.float32)
        self.var = np.ones(obs_dim, dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        def extract_array(data):
            if isinstance(data, tuple):
                for item in data:
                    if isinstance(item, np.ndarray):
                        return item
                raise ValueError("No array found in the tuple.")
            return data
        obs = extract_array(obs)
        next_obs = extract_array(next_obs)
        if obs.shape != self.obs_buf[self.ptr].shape:
            raise ValueError(f"Obs shape mismatch. Expected: {self.obs_buf[self.ptr].shape}, Got: {obs.shape}")
        if next_obs.shape != self.obs2_buf[self.ptr].shape:
            raise ValueError(f"Next_obs shape mismatch. Expected: {self.obs2_buf[self.ptr].shape}, Got: {next_obs.shape}")
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # Update Mean and Variance
        self.norm_update_batch[self.norm_update_count] = obs
        self.norm_update_count += 1
        if self.norm_update_count == self.norm_update_every:
            self.norm_update_count = 0
            batch_mean = self.norm_update_batch.mean(axis=0)
            batch_var = self.norm_update_batch.var(axis=0)
            tmp_total_count = self.norm_total_count + self.norm_update_every
            delta_mean = batch_mean - self.mean
            self.mean += delta_mean * (self.norm_update_every / tmp_total_count)
            m_a = self.var * self.norm_total_count
            m_b = batch_var * self.norm_update_every
            m_2 = m_a + m_b + (delta_mean ** 2) * self.norm_total_count * self.norm_update_every / tmp_total_count
            self.var = m_2 / tmp_total_count
            self.norm_total_count = tmp_total_count

    def sample_batch(self, device, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.normalize_obs(self.obs_buf[idxs]),
                     obs2=self.normalize_obs(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def normalize_obs(self, obs):
        def extract_array(data):
            if isinstance(data, tuple):
                for item in data:
                    if isinstance(item, np.ndarray):
                        return item
                raise ValueError("No array found in the tuple.")
            return data
        obs = extract_array(obs)
        eps = np.finfo(np.float32).eps.item()
        norm_obs = np.clip((obs - self.mean) / np.sqrt(self.var + eps), -self.clip_limit, self.clip_limit)
        return norm_obs
