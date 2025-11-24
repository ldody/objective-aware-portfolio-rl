from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# TA-lib in pure Python
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def engineer_features_and_returns(
	panel_df: pd.DataFrame,
	return_bid_col: str = "BID",
	return_ask_col: str = "ASK",
	window: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Build:
	  - execution-style log returns from BID/ASK
	  - technical indicators (RSI, MACD, Bollinger, ATR, ADX, StochRSI, SMAs, EMAs)
	per asset, using TA library.

	Index expected: MultiIndex (Timestamp, Local Code).
	Columns expected (from your data): 
	  'ACVOL_UNS','BID_HIGH_1','BID_LOW_1','OPEN_BID','BID',
	  'ASK_HIGH_1','ASK_LOW_1','OPEN_ASK','ASK',
	  'MID_HIGH','MID_LOW','MID_OPEN','MID_PRICE'
	"""
	df = panel_df.copy()

	# ---------- 1) execution log-return using BID_t / ASK_{t-1} ----------
	for col in [return_bid_col, return_ask_col]:
		if col not in df.columns:
			raise ValueError(f"Column '{col}' must be present in price data.")

	# Shift ask by 1 within each asset
	df["ask_prev"] = df.groupby(level="Local Code")[return_ask_col].shift(1)
	df["log_return_exec"] = np.log(
		(df[return_bid_col] / df["ask_prev"]).clip(lower=1e-12)
	)
	df["log_return_exec"] = df["log_return_exec"].fillna(0.0)

	# ---------- 2) define OHLCV for TA ------------

	# We'll use MID_* as main prices (more stable than best bid/ask extremes)
	if "MID_PRICE" in df.columns:
		close_col = "MID_PRICE"
	else:
		# fallback to mid of BID/ASK
		df["MID_PRICE"] = (df[return_bid_col] + df[return_ask_col]) / 2.0
		close_col = "MID_PRICE"

	high_col = "MID_HIGH"
	low_col = "MID_LOW"
	open_col = "MID_OPEN"
	vol_col = "ACVOL_UNS"

	# make sure they exist (you said they do in your spec)
	for col in [high_col, low_col, open_col, vol_col]:
		if col not in df.columns:
			raise ValueError(f"Required column '{col}' is missing from data.")

	# ---------- 3) allocate columns for indicators ----------
	df["ta_rsi_14"] = 0.0
	df["ta_macd"] = 0.0
	df["ta_macd_signal"] = 0.0
	df["ta_macd_diff"] = 0.0
	df["ta_bb_high"] = 0.0
	df["ta_bb_low"] = 0.0
	df["ta_bb_pct"] = 0.0
	df["ta_atr_14"] = 0.0
	df["ta_adx_14"] = 0.0
	df["ta_stochrsi"] = 0.0
	df["ta_sma_20"] = 0.0
	df["ta_sma_50"] = 0.0
	df["ta_ema_20"] = 0.0
	df["ta_ema_50"] = 0.0

	# ---------- 4) compute TA indicators per asset ----------
	# groupby asset code
	for code, g in df.groupby(level="Local Code", sort=False):
		close = g[close_col]
		high = g[high_col]
		low = g[low_col]
		volume = g[vol_col]

		# RSI
		rsi_14 = RSIIndicator(close=close, window=14).rsi()

		# MACD (12, 26, 9 classic)
		macd_ind = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
		macd = macd_ind.macd()
		macd_signal = macd_ind.macd_signal()
		macd_diff = macd_ind.macd_diff()

		# Bollinger Bands 20, 2
		bb_ind = BollingerBands(close=close, window=20, window_dev=2)
		bb_high = bb_ind.bollinger_hband()
		bb_low = bb_ind.bollinger_lband()
		# %b: where price is within the band
		bb_pct = bb_ind.bollinger_pband()

		# ATR, ADX use high/low/close
		atr_14 = AverageTrueRange(
			high=high, low=low, close=close, window=14
		).average_true_range()
		adx_14 = ADXIndicator(
			high=high, low=low, close=close, window=14
		).adx()

		# StochRSI
		stochrsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3).stochrsi()

		# Moving averages
		sma_20 = SMAIndicator(close=close, window=20).sma_indicator()
		sma_50 = SMAIndicator(close=close, window=50).sma_indicator()
		ema_20 = EMAIndicator(close=close, window=20).ema_indicator()
		ema_50 = EMAIndicator(close=close, window=50).ema_indicator()

		# assign back into df (align by MultiIndex)
		idx = g.index
		df.loc[idx, "ta_rsi_14"] = rsi_14.values
		df.loc[idx, "ta_macd"] = macd.values
		df.loc[idx, "ta_macd_signal"] = macd_signal.values
		df.loc[idx, "ta_macd_diff"] = macd_diff.values
		df.loc[idx, "ta_bb_high"] = bb_high.values
		df.loc[idx, "ta_bb_low"] = bb_low.values
		df.loc[idx, "ta_bb_pct"] = bb_pct.values
		df.loc[idx, "ta_atr_14"] = atr_14.values
		df.loc[idx, "ta_adx_14"] = adx_14.values
		df.loc[idx, "ta_stochrsi"] = stochrsi.values
		df.loc[idx, "ta_sma_20"] = sma_20.values
		df.loc[idx, "ta_sma_50"] = sma_50.values
		df.loc[idx, "ta_ema_20"] = ema_20.values
		df.loc[idx, "ta_ema_50"] = ema_50.values

	# ---------- 5) Build final feature matrix ----------

	# we keep:
	#  - all original numeric cols (prices, volume, etc.)
	#  - plus execution log return
	#  - plus TA features
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

	# "ask_prev" is just a helper; we can drop it from features
	if "ask_prev" in numeric_cols:
		numeric_cols.remove("ask_prev")

	feature_df = df[numeric_cols].copy().fillna(0.0)

	# target: execution log-returns (same as before)
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
