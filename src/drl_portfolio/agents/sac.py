from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from drl_portfolio.models import CNNGRUTransformerActor, QNetwork
from drl_portfolio.utils.replay_buffer import ReplayBuffer


class SACAgent:
	"""Soft Actor-Critic agent with CNN+GRU+Transformer actor."""

	def __init__(
		self,
		obs_dim: int,
		action_dim: int,
		n_assets: int,
		n_features: int,
		window: int,
		device: str = "cpu",
		actor_kwargs: Optional[Dict] = None,
		gamma: float = 0.99,
		tau: float = 0.005,
		alpha: float = 0.2,
		lr: float = 3e-4,
		hidden_dim: int = 256,
	):
		self.device = device
		self.gamma = gamma
		self.tau = tau
		self.alpha = alpha

		actor_kwargs = actor_kwargs or {}
		self.actor = CNNGRUTransformerActor(
			n_assets=n_assets,
			n_features=n_features,
			window=window,
			action_dim=action_dim,
			**actor_kwargs,
		).to(device)

		self.q1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
		self.q2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
		self.q1_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
		self.q2_target = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
		self.q1_target.load_state_dict(self.q1.state_dict())
		self.q2_target.load_state_dict(self.q2.state_dict())

		self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
		self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
		self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)

	def sample_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
		obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		with torch.no_grad():
			mean, log_std = self.actor(obs_t)
			if deterministic:
				action = torch.tanh(mean)
			else:
				std = log_std.exp()
				normal = Normal(mean, std)
				z = normal.rsample()
				action = torch.tanh(z)
		return action.cpu().numpy()[0]

	def _evaluate_actions(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		mean, log_std = self.actor(obs)
		std = log_std.exp()
		normal = Normal(mean, std)
		z = normal.rsample()
		action = torch.tanh(z)
		log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
		log_prob = log_prob.sum(dim=-1, keepdim=True)
		return action, log_prob

	def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
		obs, action, reward, next_obs, done = replay_buffer.sample(batch_size, self.device)
		reward = reward.unsqueeze(-1)
		done = done.unsqueeze(-1)

		with torch.no_grad():
			next_action, next_log_prob = self._evaluate_actions(next_obs)
			q1_next = self.q1_target(next_obs, next_action)
			q2_next = self.q2_target(next_obs, next_action)
			q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
			target_q = reward + (1.0 - done) * self.gamma * q_next

		q1_val = self.q1(obs, action)
		q2_val = self.q2(obs, action)
		q1_loss = F.mse_loss(q1_val, target_q)
		q2_loss = F.mse_loss(q2_val, target_q)

		self.q1_optim.zero_grad()
		q1_loss.backward()
		self.q1_optim.step()

		self.q2_optim.zero_grad()
		q2_loss.backward()
		self.q2_optim.step()

		new_action, log_prob = self._evaluate_actions(obs)
		q1_new = self.q1(obs, new_action)
		q2_new = self.q2(obs, new_action)
		q_new = torch.min(q1_new, q2_new)
		actor_loss = (self.alpha * log_prob - q_new).mean()

		self.actor_optim.zero_grad()
		actor_loss.backward()
		self.actor_optim.step()

		self._soft_update(self.q1, self.q1_target)
		self._soft_update(self.q2, self.q2_target)

		return {
			"q1_loss": q1_loss.item(),
			"q2_loss": q2_loss.item(),
			"actor_loss": actor_loss.item(),
		}

	def _soft_update(self, net: torch.nn.Module, target_net: torch.nn.Module) -> None:
		for param, target_param in zip(net.parameters(), target_net.parameters()):
			target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
