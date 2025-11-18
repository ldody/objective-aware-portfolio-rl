from typing import List, Tuple, Any

import numpy as np

from drl_portfolio.envs import PortfolioEnv


class VectorPortfolioEnv:
	"""
	Simple vectorized wrapper around multiple PortfolioEnv instances.

	- Holds a list of independent PortfolioEnv
	- reset()  -> [num_envs, obs_dim]
	- step(A)  -> next_obs [num_envs, obs_dim], rewards [num_envs], dones [num_envs]
	- Auto-resets envs that are done and returns the reset obs for those slots.
	"""

	def __init__(self, envs: List[PortfolioEnv]):
		if len(envs) == 0:
			raise ValueError("VectorPortfolioEnv requires at least one environment.")
		self.envs = envs
		self.num_envs = len(envs)
		self.obs_dim = envs[0].observation_space.shape[0]
		self.action_dim = envs[0].action_space.shape[0]

	def reset(self) -> np.ndarray:
		obs_batch = []
		for env in self.envs:
			obs, _ = env.reset()
			obs_batch.append(obs)
		return np.stack(obs_batch, axis=0)  # [num_envs, obs_dim]

	def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
		"""
		actions: [num_envs, action_dim]

		Returns:
			obs_batch: [num_envs, obs_dim]
			rewards:   [num_envs]
			dones:     [num_envs] (True if this env terminated or truncated this step)
			infos:     list of info dicts for each env
		"""
		obs_batch, rew_batch, done_batch, info_batch = [], [], [], []

		for i, env in enumerate(self.envs):
			action = actions[i]
			obs, rew, terminated, truncated, info = env.step(action)
			done = bool(terminated or truncated)

			# Auto-reset done envs and use the reset obs as next_obs
			if done:
				obs, _ = env.reset()

			obs_batch.append(obs)
			rew_batch.append(rew)
			done_batch.append(done)
			info_batch.append(info)

		return (
			np.stack(obs_batch, axis=0),
			np.array(rew_batch, dtype=np.float32),
			np.array(done_batch, dtype=bool),
			info_batch,
		)
