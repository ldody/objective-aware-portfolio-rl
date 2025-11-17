from typing import Dict

import numpy as np
import pandas as pd

from drl_portfolio.agents import SACAgent
from drl_portfolio.envs import PortfolioEnv
from drl_portfolio.utils.metrics import compute_weekly_metrics


def backtest_policy(env: PortfolioEnv, agent: SACAgent, deterministic: bool = True) -> Dict:
    obs, _ = env.reset()
    done = False
    weights_hist, returns_hist, equity_hist, turnover_hist = [], [], [], []

    while not done:
        action = agent.sample_action(obs, deterministic=deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        weights_hist.append(info["weights"])
        returns_hist.append(info["step_return"])
        equity_hist.append(info["equity"])
        turnover_hist.append(info["turnover"])

    return {
        "weights": np.asarray(weights_hist),
        "returns": np.asarray(returns_hist),
        "equity": np.asarray(equity_hist),
        "turnover": np.asarray(turnover_hist),
        "timestamps": env.timestamps[env.window :],
    }


def backtest_equal_weight(env: PortfolioEnv) -> Dict:
    obs, _ = env.reset()
    done = False
    weights_hist, returns_hist, equity_hist, turnover_hist = [], [], [], []

    w = np.ones(env.n_assets, dtype=np.float32) / env.n_assets
    env.prev_weights = w.copy()

    while not done:
        action = np.clip(w, -1.0, 1.0)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        weights_hist.append(info["weights"])
        returns_hist.append(info["step_return"])
        equity_hist.append(info["equity"])
        turnover_hist.append(info["turnover"])

    return {
        "weights": np.asarray(weights_hist),
        "returns": np.asarray(returns_hist),
        "equity": np.asarray(equity_hist),
        "turnover": np.asarray(turnover_hist),
        "timestamps": env.timestamps[env.window :],
    }


def mean_variance_weights(ret_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = ret_matrix.mean(axis=0)
    Sigma = np.cov(ret_matrix, rowvar=False) + eps * np.eye(ret_matrix.shape[1])
    inv_Sigma = np.linalg.pinv(Sigma)
    w = inv_Sigma @ mu
    w = np.maximum(w, 0.0)
    if w.sum() == 0.0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return w.astype(np.float32)


def backtest_mean_variance(env: PortfolioEnv, window_steps: int = 96) -> Dict:
    returns = env.ret_arr
    timestamps = env.timestamps
    start_idx = env.window
    T = len(timestamps)

    weights_hist, returns_hist, equity_hist, turnover_hist = [], [], [], []
    equity = 1.0
    prev_w = np.ones(env.n_assets, dtype=np.float32) / env.n_assets

    for t in range(start_idx, T):
        if (t - start_idx) % window_steps == 0:
            start_win = max(env.window, t - window_steps)
            R = returns[start_win:t, :]
            if R.shape[0] < 2:
                w = prev_w
            else:
                w = mean_variance_weights(R)
        else:
            w = prev_w

        step_ret_vec = returns[t, :]
        portfolio_log_ret = float((w * step_ret_vec).sum())
        step_return = np.exp(portfolio_log_ret) - 1.0
        turnover = float(np.sum(np.abs(w - prev_w)))

        equity *= (1.0 + step_return)
        equity_hist.append(equity)
        weights_hist.append(w)
        returns_hist.append(step_return)
        turnover_hist.append(turnover)

        prev_w = w

    return {
        "weights": np.asarray(weights_hist),
        "returns": np.asarray(returns_hist),
        "equity": np.asarray(equity_hist),
        "turnover": np.asarray(turnover_hist),
        "timestamps": timestamps[start_idx:],
    }


def summarize_strategy(name: str, bt_result: Dict) -> Dict[str, float]:
    metrics = compute_weekly_metrics(bt_result["returns"], bt_result["timestamps"])
    avg_turnover = float(bt_result["turnover"].mean()) if len(bt_result["turnover"]) > 0 else 0.0
    metrics["avg_turnover"] = avg_turnover

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics
