from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features_and_returns(
    panel_df: pd.DataFrame,
    return_bid_col: str = "BID",
    return_ask_col: str = "ASK",
    window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use all numeric columns as features and build bid/ask-based returns.

    Execution-style returns per asset:
        - buy at ASK at t-1
        - sell at BID at t
        -> log_return_exec_t = log(BID_t / ASK_{t-1})
    """
    df = panel_df.copy()

    for col in [return_bid_col, return_ask_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' must be present in price data for bid/ask returns.")

    df["ask_prev"] = df.groupby(level="Local Code")[return_ask_col].shift(1)
    df["log_return_exec"] = np.log(
        (df[return_bid_col] / df["ask_prev"]).clip(lower=1e-12)
    )
    df["log_return_exec"] = df["log_return_exec"].fillna(0.0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "ask_prev" in numeric_cols:
        numeric_cols.remove("ask_prev")

    feature_df = df[numeric_cols].copy().fillna(0.0)
    return_df = df[["log_return_exec"]].copy()
    return feature_df, return_df


def train_test_split_time_based(
    feature_df: pd.DataFrame,
    return_df: pd.DataFrame,
    train_ratio: float = 0.7,
):
    """Time-based train/test split on the Timestamp level."""
    timestamps = feature_df.index.get_level_values("Timestamp").unique()
    split_idx = int(len(timestamps) * train_ratio)
    train_ts = timestamps[:split_idx]
    test_ts = timestamps[split_idx:]

    def _mask(ts_set):
        return feature_df.index.get_level_values("Timestamp").isin(ts_set)

    train_feat = feature_df.loc[_mask(train_ts)]
    test_feat = feature_df.loc[_mask(test_ts)]
    train_ret = return_df.loc[_mask(train_ts)]
    test_ret = return_df.loc[_mask(test_ts)]
    return train_feat, test_feat, train_ret, test_ret


def normalize_features(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Global z-score normalization using training-set statistics."""
    scaler = StandardScaler()
    train_vals = scaler.fit_transform(train_feat.values)
    test_vals = scaler.transform(test_feat.values)

    train_norm = pd.DataFrame(train_vals, index=train_feat.index, columns=train_feat.columns)
    test_norm = pd.DataFrame(test_vals, index=test_feat.index, columns=test_feat.columns)
    return train_norm, test_norm, scaler
