from dataclasses import dataclass
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


@dataclass
class ConstraintConfig:
	min_return: float
	max_turnover: float
	max_volatility: float
	max_dd: float
	max_delta_return: float
	min_sharpe: float
	weights: Dict[str, float]


class PortfolioEnv(gym.Env):
	"""Portfolio environment for intraday DRL with constraints."""

	metadata = {"render_modes": []}

	def __init__(
		self,
		features: pd.DataFrame,
		returns: pd.DataFrame,
		assets_df: pd.DataFrame,
		params_df: pd.DataFrame,
		window: int = 20,
		trading_cost: float = 0.0005,
		mode: str = "train",        # <-- NEW: "train" or "test"
		penalty_scale: float = 0.1, # <-- NEW: échelle globale des pénalités
	):
		super().__init__()
		self.features = features
		self.returns = returns
		self.assets = assets_df
		self.window = window
		self.trading_cost = trading_cost
		self.mode = mode
		self.penalty_scale = penalty_scale


		# Real assets from CSV
		real_asset_codes: List[str] = list(self.assets["Local Code"])

		# Add synthetic CASH asset (no price history, return = 0, features = 0)
		self.cash_code = "CASH"
		self.asset_codes: List[str] = real_asset_codes + [self.cash_code]

		self.n_assets: int = len(self.asset_codes)

		self.feature_cols: List[str] = list(self.features.columns)
		self.n_features: int = len(self.feature_cols)

		weights = {p: params_df.loc[p, "Weight"] for p in params_df.index}
		self.constraints = ConstraintConfig(
			min_return=params_df.loc["Return", "Value"],
			max_turnover=params_df.loc["Turnover", "Value"],
			max_volatility=params_df.loc["Volatility", "Value"],
			max_dd=params_df.loc["Drowdown", "Value"],
			max_delta_return=params_df.loc["Delta_return", "Value"],
			min_sharpe=params_df.loc["Sharp", "Value"],
			weights=weights,
		)

		obs_dim = self.n_assets * self.window * self.n_features + self.n_assets
		self.observation_space = spaces.Box(
			low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
		)
		self.action_space = spaces.Box(
			low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
		)

		self.timestamps = sorted(features.index.get_level_values("Timestamp").unique())
		self.dates = np.array([ts.date() for ts in self.timestamps])  # one date per timestep
		self._build_arrays()
		self._build_day_indices()
		self.reset(seed=None)

	def _build_day_indices(self) -> None:
		"""Pré-calcul des indices de début/fin pour chaque journée."""
		day_start_indices = []
		day_end_indices = []

		current_date = self.dates[0]
		start_idx = 0

		for i in range(1, len(self.timestamps)):
			if self.dates[i] != current_date:
				# fin du jour précédent
				day_start_indices.append(start_idx)
				day_end_indices.append(i - 1)
				# nouveau jour
				current_date = self.dates[i]
				start_idx = i

		# dernier jour
		day_start_indices.append(start_idx)
		day_end_indices.append(len(self.timestamps) - 1)

		self.day_start_indices = np.array(day_start_indices, dtype=np.int32)
		self.day_end_indices = np.array(day_end_indices, dtype=np.int32)
		self.n_days = len(self.day_start_indices)
		
		if self.days_per_episode > self.n_days:
			raise ValueError(
				f"days_per_episode={self.days_per_episode} > n_days={self.n_days}"
			)

	def _build_arrays(self) -> None:
		T = len(self.timestamps)
		feat_arr = np.zeros((T, self.n_assets, self.n_features), dtype=np.float32)
		ret_exec_arr = np.zeros((T, self.n_assets), dtype=np.float32)
		ret_mid_arr = np.zeros((T, self.n_assets), dtype=np.float32)

		for t_idx, ts in enumerate(self.timestamps):
			for a_idx, code in enumerate(self.asset_codes):
				idx = (ts, code)
				if idx in self.features.index:
					feat_arr[t_idx, a_idx, :] = self.features.loc[idx].values
				if idx in self.returns.index:
					row = self.returns.loc[idx]
					# assumes columns ["log_return_exec","log_return_mid"]
					ret_exec_arr[t_idx, a_idx] = float(row["log_return_exec"])
					ret_mid_arr[t_idx, a_idx] = float(row["log_return_mid"])

		self.feat_arr = feat_arr
		self.ret_exec_arr = ret_exec_arr
		self.ret_mid_arr = ret_mid_arr


	def _get_obs(self) -> np.ndarray:
		start = max(0, self.t_idx - self.window)
		end = self.t_idx
		window_feat = self.feat_arr[start:end, :, :]  # [window, n_assets, n_features] or shorter at very beginning

		# If start==0 and t_idx<window, you get a shorter history; in practice with our reset
		# conditions we usually have full windows, but this is safe.

		# If you want STRICTLY fixed-size windows, you can enforce t_idx >= window everywhere.
		if end - start < self.window:
			# Pad at the beginning with the earliest available rows
			pad_len = self.window - (end - start)
			first_slice = np.repeat(self.feat_arr[start:start+1, :, :], pad_len, axis=0)
			window_feat = np.concatenate([first_slice, window_feat], axis=0)

		window_feat = np.transpose(window_feat, (1, 0, 2))  # [n_assets, window, n_features]
		obs = window_feat.reshape(-1)
		obs = np.concatenate([obs, self.prev_weights], axis=0)
		return obs.astype(np.float32)


	def reset(self, seed: Optional[int] = None, options=None):
		super().reset(seed=seed)

		if self.mode == "train":
			# --- 1) Choose a random *start* day whose first bar has at least `window` bars before it ---
			# This ensures that when t_idx = day_start_idx, we can take a window
			# [t_idx - window, t_idx) that may include the previous day.
			day_start_indices = self.day_start_indices
			# valid days: require that there are >= window timesteps BEFORE day_start_idx
			valid_days = np.where(day_start_indices >= self.window)[0]

			if len(valid_days) == 0:
				# No day has enough global history to provide a full window including prev day
				raise ValueError(
					f"No training day has at least {self.window} prior timesteps "
					f"(min day_start_idx={int(day_start_indices.min())}). "
					f"Try reducing `window`."
				)

			# We also need `days_per_episode` consecutive days
			max_start_day = self.n_days - self.days_per_episode
			valid_start_days = valid_days[valid_days <= max_start_day]
			if len(valid_start_days) == 0:
				raise ValueError(
					f"No training day can be used as episode start with "
					f"days_per_episode={self.days_per_episode} and window={self.window}."
				)

			start_day_idx = int(self.np_random.choice(valid_start_days))
			end_day_idx = start_day_idx + self.days_per_episode - 1

			self.episode_start_day_idx = start_day_idx
			self.episode_end_day_idx = end_day_idx
			self.current_day_idx = start_day_idx

			self.day_start_idx = int(self.day_start_indices[start_day_idx])
			self.day_end_idx = int(self.day_end_indices[end_day_idx])

			# Start the episode at the FIRST bar of the chosen block of days
			self.t_idx = self.day_start_idx
			else:
				# mode "test": single long episode over full test period
				self.day_start_idx = 0
				self.day_end_idx = len(self.timestamps) - 1
				self.current_day_idx = 0
				self.t_idx = self.window

		else:
			# mode "test": single long episode over full test period
			self.day_start_idx = 0
			self.day_end_idx = len(self.timestamps) - 1
			self.current_day_idx = 0
			self.t_idx = self.window  # first step has full global history

		# --- 2) Init portfolio state ---
		self.prev_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
		self.equity = 1.0
		self.equity_history: List[float] = [self.equity]
		self.return_history: List[float] = []
		self.turnover_history: List[float] = []
		self.done_flag = False

		# --- 3) Daily tracking (start of episode) ---
		self.current_day = self.dates[self.t_idx]
		self.day_start_equity = self.equity
		self.daily_return_history: List[float] = []
		self.daily_equity_history: List[float] = [1.0]  # relative to day start
		self.daily_turnover_sum: float = 0.0

		return self._get_obs(), {}


	def step(self, action: np.ndarray):
		if self.done_flag:
			return self._get_obs(), 0.0, True, False, {}

		# ---- 1) Softmax -> weights ----
		action = np.clip(action, -1.0, 1.0)
		logits = action.astype(np.float64)
		exp_logits = np.exp(logits - logits.max())
		weights = exp_logits / exp_logits.sum()
		weights = weights.astype(np.float32)

		# ---- 2) Turnover & cost ----
		turnover = float(np.sum(np.abs(weights - self.prev_weights)))
		trading_cost = self.trading_cost * turnover

		# ---- 3) Step return from log returns ----
		# a) mid-price portfolio return (no spread, no fee)
		step_ret_mid_vec = self.ret_mid_arr[self.t_idx, :]
		portfolio_log_ret_mid = float((weights * step_ret_mid_vec).sum())
		step_return_mid = np.exp(portfolio_log_ret_mid) - 1.0  # gross, no cost

		# b) execution portfolio return (BID/ASK-based, includes spread)
		step_ret_exec_vec = self.ret_exec_arr[self.t_idx, :]
		portfolio_log_ret_exec = float((weights * step_ret_exec_vec).sum())
		step_return_exec = np.exp(portfolio_log_ret_exec) - 1.0  # gross, but with spread/slippage

		# c) explicit spread/slippage cost: mid - exec
		spread_cost = step_return_mid - step_return_exec  # ≥ 0 on average

		# d) trading fee cost from turnover
		fee_cost = trading_cost  # you already computed trading_cost = self.trading_cost * turnover

		# e) net return actually hitting the account
		step_return_after_cost = step_return_exec - fee_cost

		# ---- 4) Update global equity & histories ----
		self.equity *= (1.0 + step_return_after_cost)
		self.equity_history.append(self.equity)
		self.return_history.append(step_return_after_cost)
		self.turnover_history.append(turnover)

		# ---- 5) Daily tracking ----
		current_date = self.dates[self.t_idx]

		# Update daily returns/equity relative to day start
		self.daily_return_history.append(step_return_after_cost)
		daily_equity = self.daily_equity_history[-1] * (1.0 + step_return_after_cost)
		self.daily_equity_history.append(daily_equity)
		self.daily_turnover_sum += turnover

		# ---- 6) Fin de journée (pour les pénalités) ----
		is_last_global_step = (self.t_idx == len(self.timestamps) - 1)
		if not is_last_global_step:
			next_date = self.dates[self.t_idx + 1]
			end_of_day = (next_date != current_date)
		else:
			end_of_day = True

		# Pénalité journalière calculée à la fin de chaque jour
		if end_of_day:
			raw_penalty = self._compute_daily_penalty()
		else:
			raw_penalty = 0.0

		penalty = self.penalty_scale * raw_penalty
		reward = step_return_after_cost - penalty

		# ---- 7) Move on ----
		self.prev_weights = weights
		self.t_idx += 1

		# Gestion de la fin d'épisode
		terminated = False
		if self.mode == "train":
			# En train: 1 épisode = `days_per_episode` journées consécutives
			# On termine l'épisode quand on a fini la dernière journée de ce bloc.
			# Note: self.t_idx has already been incremented by 1 just above.
			if end_of_day and self.t_idx > self.day_end_idx:
				terminated = True
				self.done_flag = True
		else:
			# En test: 1 épisode = toute la période
			if self.t_idx >= len(self.timestamps):
				terminated = True
				self.done_flag = True

		# Si on a fini la journée mais pas l'épisode (mode test),
		# on reset juste les stats journalières pour le jour suivant.
		if end_of_day and not self.done_flag:
			self.current_day = self.dates[self.t_idx]  # première date du jour suivant
			self.day_start_equity = self.equity
			self.daily_return_history = []
			self.daily_equity_history = [1.0]
			self.daily_turnover_sum = 0.0

		obs = self._get_obs()
		truncated = False
		info = {
			"step_return": step_return_after_cost,
			"step_return_mid": step_return_mid,
			"step_return_exec": step_return_exec,
			"spread_cost": spread_cost,
			"fee_cost": fee_cost,
			"turnover": turnover,
			"weights": weights,
			"equity": self.equity,
			"penalty": penalty,
			"end_of_day": end_of_day,
			"daily_equity": daily_equity,
		}

		return obs, reward, terminated, truncated, info

	def _compute_daily_penalty(self) -> float:
		"""
		Compute penalties based on *daily* metrics:
		  - cumulative daily return
		  - daily turnover
		  - daily intraday volatility
		  - daily intraday drawdown
		  - daily intraday Sharpe
		  - daily stability (delta_return)
		Called once at end-of-day.
		"""
		c = self.constraints
		w = c.weights
		penalty = 0.0

		# If no daily data (should not happen), no penalty
		if len(self.daily_return_history) == 0:
			return 0.0

		ret_arr = np.asarray(self.daily_return_history)
		eq_arr = np.asarray(self.daily_equity_history)  # starts at 1.0

		# 1) Daily cumulative return relative to day start
		daily_return_cum = eq_arr[-1] - 1.0  # (equity_day - 1)

		if daily_return_cum < c.min_return:
			penalty += w.get("Return", 0.0) * (c.min_return - daily_return_cum)

		# 2) Daily total turnover
		daily_turnover = self.daily_turnover_sum
		if daily_turnover > c.max_turnover:
			penalty += w.get("Turnover", 0.0) * (daily_turnover - c.max_turnover)

		# 3) Intraday volatility (std of 5-min returns within the day)
		vol = float(ret_arr.std())
		if vol > c.max_volatility:
			penalty += w.get("Volatility", 0.0) * (vol - c.max_volatility)

		# 4) Stability of returns (delta_return)
		dr = float(np.diff(ret_arr).std()) if len(ret_arr) > 1 else 0.0
		if dr > c.max_delta_return:
			penalty += w.get("Delta_return", 0.0) * (dr - c.max_delta_return)

		# 5) Intraday maximum drawdown (relative to day start)
		running_max = np.maximum.accumulate(eq_arr)
		drawdowns = (running_max - eq_arr) / (running_max + 1e-8)
		max_dd = float(drawdowns.max())
		if max_dd > c.max_dd:
			penalty += w.get("Drowdown", 0.0) * (max_dd - c.max_dd)

		# 6) Intraday Sharpe for the day
		mean_ret = float(ret_arr.mean())
		std_ret = float(ret_arr.std() + 1e-8)
		sharpe = mean_ret / std_ret
		if sharpe < c.min_sharpe:
			penalty += w.get("Sharp", 0.0) * (c.min_sharpe - sharpe)

		return float(penalty)

	def render(self):
		pass


class PortfolioEnvNoConstraints(PortfolioEnv):
	"""Same environment but without constraint penalties (baseline DRL)."""

	def _compute_daily_penalty(self) -> float:
		return 0.0
