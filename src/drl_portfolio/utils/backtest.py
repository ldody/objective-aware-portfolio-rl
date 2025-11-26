from typing import Dict, List

import numpy as np
import pandas as pd

from drl_portfolio.agents import SACAgent
from drl_portfolio.envs import PortfolioEnv
from drl_portfolio.utils.metrics import compute_weekly_metrics


def _build_allocation_and_perf(
	env: PortfolioEnv,
	weights_hist: List[np.ndarray],
	returns_net_hist: List[float],
	returns_mid_hist: List[float],
	returns_exec_hist: List[float],
	spread_hist: List[float],
	fee_hist: List[float],
	equity_hist: List[float],
	turnover_hist: List[float],
	ts_hist: List[pd.Timestamp],
) -> Dict:
	"""
	Helper to build a consistent output dictionary and DataFrames for all backtests.
	"""
	if len(weights_hist) == 0:
		raise ValueError(
			"No history collected in backtest – did the environment terminate immediately?"
		)

	# Convert to arrays
	weights_arr = np.asarray(weights_hist, dtype=np.float32)
	returns_net_arr = np.asarray(returns_net_hist, dtype=np.float32)
	returns_mid_arr = np.asarray(returns_mid_hist, dtype=np.float32)
	returns_exec_arr = np.asarray(returns_exec_hist, dtype=np.float32)
	spread_arr = np.asarray(spread_hist, dtype=np.float32)
	fee_arr = np.asarray(fee_hist, dtype=np.float32)
	equity_arr = np.asarray(equity_hist, dtype=np.float32)
	turnover_arr = np.asarray(turnover_hist, dtype=np.float32)
	ts_arr = np.asarray(ts_hist)

	ts_index = pd.to_datetime(ts_hist)

	allocation_df = pd.DataFrame(
		data=weights_arr,
		index=ts_index,
		columns=getattr(env, "asset_codes", None),
	)

	performance_df = pd.DataFrame(
		{
			"step_return": returns_net_arr,
			"step_return_mid": returns_mid_arr,
			"step_return_exec": returns_exec_arr,
			"spread_cost": spread_arr,
			"fee_cost": fee_arr,
			"equity": equity_arr,
			"turnover": turnover_arr,
		},
		index=ts_index,
	)
	performance_df.index.name = "Timestamp"

	return {
		"weights": weights_arr,
		"returns": returns_net_arr,
		"returns_mid": returns_mid_arr,
		"returns_exec": returns_exec_arr,
		"spread_cost": spread_arr,
		"fee_cost": fee_arr,
		"equity": equity_arr,
		"turnover": turnover_arr,
		"timestamps": ts_arr,
		"allocation": allocation_df,
		"performance": performance_df,
	}


def backtest_policy(env: PortfolioEnv, agent: SACAgent, deterministic: bool = True) -> Dict:
	"""
	Backtest the trained SAC policy in the given environment.

	We use:
	  - info["step_return"]       : net return after spread and fees
	  - info["step_return_mid"]   : gross mid-price return
	  - info["step_return_exec"]  : gross exec-price return
	  - info["spread_cost"]       : mid - exec
	  - info["fee_cost"]          : fee from turnover
	  - info["turnover"], "weights", "equity"
	"""
	obs, _ = env.reset()
	done = False

	weights_hist: List[np.ndarray] = []
	returns_net_hist: List[float] = []
	returns_mid_hist: List[float] = []
	returns_exec_hist: List[float] = []
	spread_hist: List[float] = []
	fee_hist: List[float] = []
	equity_hist: List[float] = []
	turnover_hist: List[float] = []
	ts_hist: List[pd.Timestamp] = []

	while not done:
		action = agent.sample_action(obs, deterministic=deterministic)
		obs, _, terminated, truncated, info = env.step(action)
		done = terminated or truncated

		weights_hist.append(info["weights"])
		returns_net_hist.append(info["step_return"])
		returns_mid_hist.append(info.get("step_return_mid", info["step_return"]))
		returns_exec_hist.append(info.get("step_return_exec", info["step_return"]))
		spread_hist.append(info.get("spread_cost", 0.0))
		fee_hist.append(info.get("fee_cost", 0.0))
		equity_hist.append(info["equity"])
		turnover_hist.append(info["turnover"])

		current_ts = env.timestamps[env.t_idx - 1]
		ts_hist.append(current_ts)

	return _build_allocation_and_perf(
		env,
		weights_hist,
		returns_net_hist,
		returns_mid_hist,
		returns_exec_hist,
		spread_hist,
		fee_hist,
		equity_hist,
		turnover_hist,
		ts_hist,
	)


def backtest_equal_weight(env: PortfolioEnv) -> Dict:
	"""
	Backtest a static equal-weight portfolio in the given environment, using
	the same pricing and cost model as the RL agent.

	We construct weights w = 1/N and keep them constant; actions are obtained
	by inverting the env's softmax mapping approximately via logits = w.
	"""
	obs, _ = env.reset()
	done = False

	n_assets = env.n_assets
	w = np.ones(n_assets, dtype=np.float32) / n_assets
	# Ensure the environment sees these as previous weights (for turnover calc)
	env.prev_weights = w.copy()

	weights_hist: List[np.ndarray] = []
	returns_net_hist: List[float] = []
	returns_mid_hist: List[float] = []
	returns_exec_hist: List[float] = []
	spread_hist: List[float] = []
	fee_hist: List[float] = []
	equity_hist: List[float] = []
	turnover_hist: List[float] = []
	ts_hist: List[pd.Timestamp] = []

	while not done:
		# Env interprets actions as logits and applies softmax -> weights,
		# so we use logits=w (clipped) to get approximately equal weights.
		action = np.clip(w, -1.0, 1.0)
		obs, _, terminated, truncated, info = env.step(action)
		done = terminated or truncated

		weights_hist.append(info["weights"])
		returns_net_hist.append(info["step_return"])
		returns_mid_hist.append(info.get("step_return_mid", info["step_return"]))
		returns_exec_hist.append(info.get("step_return_exec", info["step_return"]))
		spread_hist.append(info.get("spread_cost", 0.0))
		fee_hist.append(info.get("fee_cost", 0.0))
		equity_hist.append(info["equity"])
		turnover_hist.append(info["turnover"])

		current_ts = env.timestamps[env.t_idx - 1]
		ts_hist.append(current_ts)

	return _build_allocation_and_perf(
		env,
		weights_hist,
		returns_net_hist,
		returns_mid_hist,
		returns_exec_hist,
		spread_hist,
		fee_hist,
		equity_hist,
		turnover_hist,
		ts_hist,
	)


def mean_variance_weights(ret_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""
	Simple long-only mean-variance weights with identity regularization.
	Input:
		ret_matrix: [T, N] matrix of asset returns (log or arithmetic).
	"""
	if ret_matrix.ndim != 2 or ret_matrix.shape[0] < 2:
		raise ValueError("ret_matrix must be 2D with at least 2 time steps")

	mu = ret_matrix.mean(axis=0)  # [N]
	Sigma = np.cov(ret_matrix, rowvar=False) + eps * np.eye(ret_matrix.shape[1])
	inv_Sigma = np.linalg.pinv(Sigma)
	w = inv_Sigma @ mu
	w = np.maximum(w, 0.0)
	s = w.sum()
	if s == 0.0:
		w = np.ones_like(w) / len(w)
	else:
		w = w / s
	return w.astype(np.float32)


def backtest_mean_variance(env: PortfolioEnv, window_steps: int = 96) -> Dict:
	"""
	Mean-variance benchmark that uses the environment's price data directly
	(without calling env.step) but reproduces the same mid/exec/spread/fee
	decomposition.

	We approximate:
	  - ret_mid_arr: mid-price log returns
	  - ret_exec_arr: exec-price log returns (includes spread)
	  - trading_cost: env.trading_cost * turnover, with turnover from MV weights.
	"""
	# Arrays: [T, N]
	ret_mid = env.ret_mid_arr
	ret_exec = env.ret_exec_arr
	timestamps = env.timestamps
	trading_cost_per_unit = getattr(env, "trading_cost", 0.0)

	start_idx = env.window
	T, n_assets = ret_mid.shape

	weights_hist: List[np.ndarray] = []
	returns_net_hist: List[float] = []
	returns_mid_hist: List[float] = []
	returns_exec_hist: List[float] = []
	spread_hist: List[float] = []
	fee_hist: List[float] = []
	equity_hist: List[float] = []
	turnover_hist: List[float] = []
	ts_hist: List[pd.Timestamp] = []

	equity = 1.0
	prev_w = np.ones(n_assets, dtype=np.float32) / n_assets

	for t in range(start_idx, T):
		# 1) Recompute MV weights every 'window_steps' steps
		if (t - start_idx) % window_steps == 0:
			start_win = max(env.window, t - window_steps)
			R_win = ret_mid[start_win:t, :]
			if R_win.shape[0] < 2:
				w = prev_w
			else:
				w = mean_variance_weights(R_win)
		else:
			w = prev_w

		# 2) Mid / Exec returns at time t
		step_ret_mid_vec = ret_mid[t, :]
		step_ret_exec_vec = ret_exec[t, :]

		portfolio_log_ret_mid = float((w * step_ret_mid_vec).sum())
		portfolio_log_ret_exec = float((w * step_ret_exec_vec).sum())

		step_return_mid = np.exp(portfolio_log_ret_mid) - 1.0
		step_return_exec = np.exp(portfolio_log_ret_exec) - 1.0

		spread_cost = step_return_mid - step_return_exec

		# 3) Turnover and fee
		turnover = float(np.sum(np.abs(w - prev_w)))
		fee_cost = trading_cost_per_unit * turnover

		# 4) Net return and equity
		step_return_after_cost = step_return_exec - fee_cost
		equity *= (1.0 + step_return_after_cost)

		weights_hist.append(w)
		returns_net_hist.append(step_return_after_cost)
		returns_mid_hist.append(step_return_mid)
		returns_exec_hist.append(step_return_exec)
		spread_hist.append(spread_cost)
		fee_hist.append(fee_cost)
		equity_hist.append(equity)
		turnover_hist.append(turnover)

		prev_w = w
		ts_hist.append(timestamps[t])

	return _build_allocation_and_perf(
		env,
		weights_hist,
		returns_net_hist,
		returns_mid_hist,
		returns_exec_hist,
		spread_hist,
		fee_hist,
		equity_hist,
		turnover_hist,
		ts_hist,
	)


def summarize_strategy(name: str, bt_result: Dict) -> Dict[str, float]:
	"""
	Compute weekly performance metrics + average turnover from a backtest result.
	"""
	metrics = compute_weekly_metrics(bt_result["returns"], bt_result["timestamps"])
	avg_turnover = float(bt_result["turnover"].mean()) if len(bt_result["turnover"]) > 0 else 0.0
	metrics["avg_turnover"] = avg_turnover

	print(f"\n=== {name} ===")
	for k, v in metrics.items():
		print(f"{k}: {v:.4f}")
	return metrics
