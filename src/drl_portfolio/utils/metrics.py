from typing import Dict, List

import numpy as np
import pandas as pd


def compute_weekly_metrics(
    returns: np.ndarray,
    timestamps: List[pd.Timestamp],
    weekly_target: float = 0.03,
) -> Dict[str, float]:
    """Compute weekly performance metrics from step returns and timestamps."""
    ts_index = pd.to_datetime(timestamps)
    df = pd.DataFrame({"ret": returns}, index=ts_index)
    weekly = (1.0 + df["ret"]).resample("W-FRI").prod() - 1.0
    if len(weekly) == 0:
        return {}

    avg_weekly = float(weekly.mean())
    prob_target = float((weekly >= weekly_target).mean())

    equity = (1.0 + weekly).cumprod()
    roll_max = np.maximum.accumulate(equity)
    dd = (roll_max - equity) / (roll_max + 1e-8)
    max_dd = float(dd.max())

    mean_w = float(weekly.mean())
    std_w = float(weekly.std() + 1e-8)
    sharpe = mean_w / std_w

    downside = weekly.copy()
    downside[downside > 0] = 0.0
    downside_std = float(downside.std() + 1e-8)
    sortino = mean_w / downside_std

    return {
        "avg_weekly_return": avg_weekly,
        "prob_reach_3pct_weekly": prob_target,
        "max_drawdown": max_dd,
        "sharpe_weekly": sharpe,
        "sortino_weekly": sortino,
    }
