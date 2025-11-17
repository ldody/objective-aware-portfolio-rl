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
    ):
        super().__init__()
        self.features = features
        self.returns = returns
        self.assets = assets_df
        self.window = window
        self.trading_cost = trading_cost

        self.asset_codes: List[str] = list(self.assets["Local Code"])
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
        self._build_arrays()
        self.reset(seed=None)

    def _build_arrays(self) -> None:
        T = len(self.timestamps)
        feat_arr = np.zeros((T, self.n_assets, self.n_features), dtype=np.float32)
        ret_arr = np.zeros((T, self.n_assets), dtype=np.float32)

        for t_idx, ts in enumerate(self.timestamps):
            for a_idx, code in enumerate(self.asset_codes):
                idx = (ts, code)
                if idx in self.features.index:
                    feat_arr[t_idx, a_idx, :] = self.features.loc[idx].values
                    ret_arr[t_idx, a_idx] = self.returns.loc[idx].values[0]

        self.feat_arr = feat_arr
        self.ret_arr = ret_arr

    def _get_obs(self) -> np.ndarray:
        start = self.t_idx - self.window
        end = self.t_idx
        window_feat = self.feat_arr[start:end, :, :]  # [window, n_assets, n_features]
        window_feat = np.transpose(window_feat, (1, 0, 2))  # [n_assets, window, n_features]
        obs = window_feat.reshape(-1)
        obs = np.concatenate([obs, self.prev_weights], axis=0)
        return obs.astype(np.float32)

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.t_idx = self.window
        self.prev_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.equity = 1.0
        self.equity_history: List[float] = [self.equity]
        self.return_history: List[float] = []
        self.turnover_history: List[float] = []
        self.done_flag = False
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        if self.done_flag:
            return self._get_obs(), 0.0, True, False, {}

        action = np.clip(action, -1.0, 1.0)
        logits = action.astype(np.float64)
        exp_logits = np.exp(logits - logits.max())
        weights = exp_logits / exp_logits.sum()
        weights = weights.astype(np.float32)

        turnover = float(np.sum(np.abs(weights - self.prev_weights)))
        trading_cost = self.trading_cost * turnover

        step_ret_vec = self.ret_arr[self.t_idx, :]
        portfolio_log_ret = float((weights * step_ret_vec).sum())
        step_return = np.exp(portfolio_log_ret) - 1.0

        step_return_after_cost = step_return - trading_cost

        self.equity *= (1.0 + step_return_after_cost)
        self.equity_history.append(self.equity)
        self.return_history.append(step_return_after_cost)
        self.turnover_history.append(turnover)

        penalty = self._compute_penalty(step_return_after_cost, turnover)
        reward = step_return_after_cost - penalty

        self.prev_weights = weights
        self.t_idx += 1
        if self.t_idx >= len(self.timestamps):
            self.done_flag = True

        obs = self._get_obs()
        terminated = self.done_flag
        truncated = False
        info = {
            "step_return": step_return_after_cost,
            "turnover": turnover,
            "weights": weights,
            "equity": self.equity,
            "penalty": penalty,
        }
        return obs, reward, terminated, truncated, info

    def _compute_penalty(self, step_return: float, turnover: float) -> float:
        c = self.constraints
        w = c.weights
        penalty = 0.0

        if step_return < c.min_return:
            penalty += w.get("Return", 0.0) * (c.min_return - step_return)

        if turnover > c.max_turnover:
            penalty += w.get("Turnover", 0.0) * (turnover - c.max_turnover)

        if len(self.return_history) > 5:
            ret_arr = np.asarray(self.return_history[-50:])
            vol = float(ret_arr.std())
            if vol > c.max_volatility:
                penalty += w.get("Volatility", 0.0) * (vol - c.max_volatility)

            dr = float(np.diff(ret_arr).std()) if len(ret_arr) > 2 else 0.0
            if dr > c.max_delta_return:
                penalty += w.get("Delta_return", 0.0) * (dr - c.max_delta_return)

            eq_arr = np.asarray(self.equity_history)
            running_max = np.maximum.accumulate(eq_arr)
            drawdowns = (running_max - eq_arr) / (running_max + 1e-8)
            max_dd = float(drawdowns.max())
            if max_dd > c.max_dd:
                penalty += w.get("Drowdown", 0.0) * (max_dd - c.max_dd)

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

    def _compute_penalty(self, step_return: float, turnover: float) -> float:
        return 0.0
