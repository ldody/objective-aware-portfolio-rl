from typing import Dict, List, Optional

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


def project_simplex_with_bounds(v: np.ndarray, ub: np.ndarray, tol: float = 1e-12, max_iter: int = 200) -> np.ndarray:
	"""
	Projection: min ||w - v||^2 s.c. 0<=w<=ub and sum(w)=1
	"""
	v = np.asarray(v, dtype=np.float64)
	ub = np.asarray(ub, dtype=np.float64)

	if ub.sum() < 1.0 - 1e-12:
		# infeasible -> fallback no leverage
		w = ub / max(ub.sum(), 1e-12)
		return w

	lo, hi = -1e6, 1e6
	for _ in range(max_iter):
		lam = 0.5 * (lo + hi)
		w = np.clip(v - lam, 0.0, ub)
		s = float(w.sum())
		if abs(s - 1.0) < tol:
			return w
		if s > 1.0:
			lo = lam
		else:
			hi = lam

	lam = 0.5 * (lo + hi)
	return np.clip(v - lam, 0.0, ub)


def mv_direction_risky_only(ret_matrix_risky: np.ndarray, ub_risky: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""
	Mean-variance direction on risky sleeve only, then project to:
	- long-only
	- sum(w_risky)=1
	- w_risky <= ub_risky
	"""
	R = np.asarray(ret_matrix_risky, dtype=np.float64)
	if R.ndim != 2 or R.shape[0] < 2:
		# fallback equal weights
		w0 = np.ones(R.shape[1], dtype=np.float64) / max(R.shape[1], 1)
		return project_simplex_with_bounds(w0, ub_risky)

	# robustify NaNs
	R = np.where(np.isfinite(R), R, np.nan)
	col_ok = np.nanmean(np.isfinite(R), axis=0) > 0.9
	if not np.all(col_ok):
		R[:, ~col_ok] = 0.0

	mu = np.nanmean(R, axis=0)  # [N_risky]
	X = R - mu
	X = np.where(np.isfinite(X), X, 0.0)

	T = R.shape[0]
	Sigma = (X.T @ X) / (T - 1) + eps * np.eye(R.shape[1])
	inv_Sigma = np.linalg.pinv(Sigma)

	w_raw = inv_Sigma @ mu
	# project to long-only simplex with bounds (sum=1 on risky sleeve)
	w_dir = project_simplex_with_bounds(w_raw, ub_risky)
	return w_dir


def ex_ante_mu_sigma_bar(ret_matrix_risky: np.ndarray, w_risky: np.ndarray, eps: float = 1e-6) -> tuple[float, float]:
	"""
	Estimate ex-ante mean and vol per BAR for risky portfolio w_risky.
	"""
	R = np.asarray(ret_matrix_risky, dtype=np.float64)
	w = np.asarray(w_risky, dtype=np.float64)

	mu = np.nanmean(R, axis=0)
	X = R - mu
	X = np.where(np.isfinite(X), X, 0.0)

	T = R.shape[0]
	if T < 2:
		return float(mu @ w), 0.0

	Sigma = (X.T @ X) / (T - 1) + eps * np.eye(R.shape[1])
	mu_p = float(mu @ w)
	sig_p = float(np.sqrt(max(0.0, w @ Sigma @ w)))
	return mu_p, sig_p


def apply_turnover_cap_risky_only(
	w_target: np.ndarray,
	w_prev: np.ndarray,
	cap_step: Optional[float],
	cap_day_remaining: Optional[float],
) -> tuple[np.ndarray, float]:
	"""
	Apply hard turnover cap on risky sleeve only: sum(|w_risky_new - w_risky_prev|) <= cap_eff
	where cap_eff = min(cap_step, cap_day_remaining) if provided.

	Returns (w_new, turnover_risky_effective)
	"""
	w_target = np.asarray(w_target, dtype=np.float64)
	w_prev = np.asarray(w_prev, dtype=np.float64)

	n = w_target.size
	if n < 2:
		return w_target.astype(np.float32), 0.0

	# risky sleeves
	rt = np.clip(w_target[:-1], 0.0, np.inf)
	rp = np.clip(w_prev[:-1], 0.0, np.inf)

	# effective cap
	caps = []
	if cap_step is not None and np.isfinite(cap_step) and cap_step >= 0:
		caps.append(float(cap_step))
	if cap_day_remaining is not None and np.isfinite(cap_day_remaining) and cap_day_remaining >= 0:
		caps.append(float(cap_day_remaining))
	cap_eff = min(caps) if len(caps) > 0 else None

	if cap_eff is None:
		risky_new = rt
		to_eff = float(np.sum(np.abs(risky_new - rp)))
	else:
		to = float(np.sum(np.abs(rt - rp)))
		if to <= cap_eff or to <= 1e-12:
			risky_new = rt
			to_eff = to
		else:
			alpha = cap_eff / to
			risky_new = rp + alpha * (rt - rp)
			to_eff = float(np.sum(np.abs(risky_new - rp)))

	risky_new = np.clip(risky_new, 0.0, np.inf)

	# cash adjusts to keep sum=1
	sr = float(risky_new.sum())
	cash = 1.0 - sr
	if cash < 0.0:
		# scale down risky to fit cash>=0
		if sr > 0:
			risky_new = risky_new / sr
			sr = 1.0
		cash = 0.0

	w_new = np.empty(n, dtype=np.float64)
	w_new[:-1] = risky_new
	w_new[-1] = cash

	# final normalize (tiny numerical)
	total = float(w_new.sum())
	if total > 0 and abs(total - 1.0) > 1e-10:
		w_new /= total

	return w_new.astype(np.float32), float(to_eff)


def estimate_bars_per_day_median(timestamps: np.ndarray) -> float:
	idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
	s = pd.Series(1, index=idx)
	bpd = float(s.groupby(s.index.date).sum().median())
	return max(1.0, bpd)


def weights_to_action_logits(w: np.ndarray, clip: float = 20.0) -> np.ndarray:
	"""
	Convert weights -> env logits so that softmax(logits) = weights.
	Required because env.step() expects logits, not weights.
	"""
	w = np.asarray(w, dtype=np.float64)
	w = np.clip(w, 1e-12, 1.0)
	logits = np.log(w)
	logits = logits - np.max(logits)
	return np.clip(logits, -clip, clip).astype(np.float32)


def backtest_mean_variance(
	env,
	window_steps: int = 96,
	turnover_cap_step: Optional[float] = None,
	turnover_cap_daily: Optional[float] = None,
	target_daily_return_pct: Optional[float] = None,
	target_daily_vol_pct: Optional[float] = None,
	w_max_risky: Optional[float] = 0.25,
	w_max_cash: float = 1,
	eps: float = 1e-6,
) -> Dict:
	"""
	Mean-variance benchmark aligned with EW execution model:

	- Uses env.step() (same spread / fees / turnover model as EW)
	- Long-only, sum=1
	- CASH = last asset
	- Turnover caps (risky sleeve)
	- Optional vol targeting (k <= 1, no leverage)
	"""

	obs, _ = env.reset()
	done = False

	# Dimensions
	n_assets = env.n_assets
	if n_assets < 2:
		raise ValueError("Need at least 2 assets (including CASH).")
	n_risky = n_assets - 1

	# Bounds
	if w_max_risky is None:
		ub_risky = np.full(n_risky, np.inf, dtype=np.float64)
		ub_all = np.array([np.inf] * n_risky + [w_max_cash], dtype=np.float64)
	else:
		ub_risky = np.full(n_risky, float(w_max_risky), dtype=np.float64)
		ub_all = np.array([float(w_max_risky)] * n_risky + [float(w_max_cash)], dtype=np.float64)

	# Targets
	target_vol_day = None if target_daily_vol_pct is None else float(target_daily_vol_pct) / 100.0

	timestamps = pd.to_datetime(env.timestamps)
	bars_per_day = estimate_bars_per_day_median(timestamps)

	# Histories
	weights_hist = []
	returns_net_hist = []
	returns_mid_hist = []
	returns_exec_hist = []
	spread_hist = []
	fee_hist = []
	equity_hist = []
	turnover_hist = []
	ts_hist = []

	# Init weights
	prev_w = project_simplex_with_bounds(
		np.ones(n_assets) / n_assets, ub_all
	).astype(np.float32)

	env.prev_weights = prev_w.copy()
	prev_action = weights_to_action_logits(prev_w)

	# Daily turnover tracking
	last_day = None
	day_used = 0.0

	ret_mid = env.ret_mid_arr

	while not done:

		t = env.t_idx
		if t >= len(timestamps):
			break

		is_rebal = (t >= env.window) and ((t - env.window) % window_steps == 0)

		# Reset daily budget
		day = timestamps[t].date()
		if last_day is None or day != last_day:
			last_day = day
			day_used = 0.0

		action = prev_action

		if is_rebal:

			start_idx = env.window
			start_win = max(start_idx, t - window_steps)
			R_win_all = ret_mid[start_win:t, :]

			if R_win_all.shape[0] >= 2:

				R_win_risky = R_win_all[:, :n_risky]

				# ===== 1) MV direction =====
				w_dir_risky = mv_direction_risky_only(
					R_win_risky, ub_risky, eps=eps
				)

				# ===== 2) Ex-ante stats =====
				mu_bar, sig_bar = ex_ante_mu_sigma_bar(
					R_win_risky, w_dir_risky, eps=eps
				)

				mu_day_dir = mu_bar * bars_per_day
				sig_day_dir = sig_bar * np.sqrt(bars_per_day)

				# ===== 3) Scaling k =====
				if mu_day_dir <= 0:
					k = 0.0
				else:
					if target_vol_day is not None:
						k_vol = min(
							1.0,
							target_vol_day / (sig_day_dir + 1e-12),
						)
					else:
						k_vol = 1.0

					k = float(np.clip(k_vol, 0.0, 1.0))

				# ===== 4) Build target =====
				w_target = np.zeros(n_assets, dtype=np.float64)
				w_target[:n_risky] = k * w_dir_risky
				w_target[-1] = 1.0 - w_target[:n_risky].sum()

				w_target = project_simplex_with_bounds(
					w_target, ub_all
				).astype(np.float32)

				# ===== 5) Turnover caps =====
				day_remaining = None
				if (
					turnover_cap_daily is not None
					and np.isfinite(turnover_cap_daily)
					and turnover_cap_daily >= 0
				):
					day_remaining = (
						float(turnover_cap_daily) - float(day_used)
					)

				w_new, to_eff = apply_turnover_cap_risky_only(
					w_target=w_target,
					w_prev=prev_w,
					cap_step=turnover_cap_step,
					cap_day_remaining=day_remaining,
				)

				w_new = project_simplex_with_bounds(
					w_new.astype(np.float64), ub_all
				).astype(np.float32)

				day_used += float(to_eff)

				action = weights_to_action_logits(w_new)

				prev_w = w_new
				prev_action = action

		# ===== STEP ENV =====
		obs, _, terminated, truncated, info = env.step(action)
		done = terminated or truncated

		weights_hist.append(info["weights"])
		returns_net_hist.append(info["step_return"])
		returns_mid_hist.append(
			info.get("step_return_mid", info["step_return"])
		)
		returns_exec_hist.append(
			info.get("step_return_exec", info["step_return"])
		)
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
