import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import os

from drl_portfolio.agents import SACAgent
from drl_portfolio.config import Paths, TrainingConfig
from drl_portfolio.data import (
	engineer_features_and_returns,
	load_and_merge_prices,
	load_assets,
	load_parameters,
	normalize_features,
	train_test_split_time_based,
)
from drl_portfolio.envs import (
	PortfolioEnv,
	PortfolioEnvNoConstraints,
	VectorPortfolioEnv,
)
from drl_portfolio.utils.backtest import (
	backtest_equal_weight,
	backtest_mean_variance,
	backtest_policy,
	summarize_strategy,
)
from drl_portfolio.utils.replay_buffer import ReplayBuffer


def train_sac_on_env(
	env: PortfolioEnv,
	cfg: TrainingConfig,
	actor_kwargs: Optional[Dict] = None,
) -> SACAgent:
	obs_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]

	agent = SACAgent(
		obs_dim=obs_dim,
		action_dim=action_dim,
		n_assets=env.n_assets,
		n_features=env.n_features,
		window=env.window,
		device=cfg.device,
		actor_kwargs=actor_kwargs or {},
		lr=cfg.learning_rate,
	)

	replay_buffer = ReplayBuffer(obs_dim, action_dim, size=cfg.buffer_size)

	total_steps = 0
	for ep in range(cfg.num_episodes):
		obs, _ = env.reset()
		done = False
		ep_reward = 0.0

		while not done:
			if total_steps < cfg.warmup_steps:
				action = env.action_space.sample()
			else:
				action = agent.sample_action(obs, deterministic=False)

			next_obs, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated

			replay_buffer.store(obs, action, reward, next_obs, float(terminated))
			obs = next_obs
			ep_reward += reward
			total_steps += 1

			if total_steps >= cfg.warmup_steps and replay_buffer.ptr > cfg.batch_size:
				agent.update(replay_buffer, cfg.batch_size)

		print(f"Episode {ep+1}/{cfg.num_episodes} - reward: {ep_reward:.4f}")

	return agent

def train_sac_on_vec_env(
	vec_env: VectorPortfolioEnv,
	cfg: TrainingConfig,
	actor_kwargs: Optional[Dict] = None,
) -> SACAgent:
	"""
	Train SAC using several PortfolioEnv instances in parallel in one process.

	vec_env: VectorPortfolioEnv wrapping N independent PortfolioEnv instances.
	"""
	obs_dim = vec_env.obs_dim
	action_dim = vec_env.action_dim

	base_env = vec_env.envs[0]  # reference for n_assets, n_features, window

	agent = SACAgent(
		obs_dim=obs_dim,
		action_dim=action_dim,
		n_assets=base_env.n_assets,
		n_features=base_env.n_features,
		window=base_env.window,
		device=cfg.device,
		actor_kwargs=actor_kwargs or {},
		lr=cfg.learning_rate,
	)

	replay_buffer = ReplayBuffer(obs_dim, action_dim, size=cfg.buffer_size)
	num_envs = vec_env.num_envs
	total_steps = 0

	for ep in range(cfg.num_episodes):
		obs_batch = vec_env.reset()  # [num_envs, obs_dim]
		done_batch = np.array([False] * num_envs)
		ep_rewards = np.zeros(num_envs, dtype=np.float32)

		# We run until ALL envs have terminated once in this episode.
		while not done_batch.all():
			# 1) sample batched actions: [num_envs, action_dim]
			actions = agent.sample_action(obs_batch, deterministic=False)

			# 2) vector env step
			next_obs_batch, rew_batch, done_step, info_batch = vec_env.step(actions)

			# 3) store transitions for each env
			for i in range(num_envs):
				replay_buffer.store(
					obs_batch[i],
					actions[i],
					rew_batch[i],
					next_obs_batch[i],
					float(done_step[i]),  # 1.0 if done at this step
				)

			obs_batch = next_obs_batch
			ep_rewards += rew_batch
			# we track whether each env has *ever* been done in this episode
			done_batch = np.logical_or(done_batch, done_step)
			total_steps += num_envs

			# 4) SAC updates
			if total_steps >= cfg.warmup_steps and replay_buffer.ptr > cfg.batch_size:
				agent.update(replay_buffer, cfg.batch_size)

		print(
			f"[Parallel] Episode {ep+1}/{cfg.num_episodes} "
			f"- mean reward per env: {ep_rewards.mean():.4f}"
		)

	return agent


def run_full_experiment(
	data_dir: Path,
	results_dir: Path,
	window: int = 20,
	num_episodes: int = 50,
	num_day_per_ep: int = 2,
	**kwargs
) -> None:
	paths = Paths(data_dir=data_dir, results_dir=results_dir)
	cfg = TrainingConfig(window=window, num_episodes=num_episodes)

	print("Loading data...")
	assets_df = load_assets(paths.assets_file)
	params_df = load_parameters(paths.params_file)

	if kwargs:
		assets_df = assets_df[assets_df['user_request'] == kwargs['user_request']]
		params_df = params_df[params_df['user_request'] == kwargs['user_request']]
		
	else:
		assets_df = assets_df[assets_df['user_request'] == 1]
		params_df = params_df[params_df['user_request'] == 1]
	
	panel_df = load_and_merge_prices(assets_df, paths.raw_data_dir, **kwargs)

	print("Engineering features (all columns + bid/ask returns)...")
	feat_df, ret_df = engineer_features_and_returns(panel_df, window=window)

	print("Splitting train/test...")
	train_feat, test_feat, train_ret, test_ret = train_test_split_time_based(
		feat_df, ret_df
	)

	print("Normalizing features...")
	train_norm, test_norm, _ = normalize_features(train_feat, test_feat)
	
	"""debug"""
	# ---------------- FIX ALIGNMENT BEFORE ENV ----------------
	# 1) Normalize asset codes
	assets_df = assets_df.copy()
	assets_df["Local Code"] = (
		assets_df["Local Code"].astype(str).str.strip().str.replace(".0", "", regex=False)
	)

	# 2) Ensure MultiIndex order is (Timestamp, Local Code)
	train_norm = train_norm.reorder_levels(["Timestamp", "Local Code"]).sort_index()
	test_norm  = test_norm.reorder_levels(["Timestamp", "Local Code"]).sort_index()
	train_ret  = train_ret.reorder_levels(["Timestamp", "Local Code"]).sort_index()
	test_ret   = test_ret.reorder_levels(["Timestamp", "Local Code"]).sort_index()

	# 3) Normalize Local Code in the MultiIndex levels (features/returns)
	def _norm_index(mi: pd.MultiIndex) -> pd.MultiIndex:
		ts = mi.get_level_values("Timestamp")
		codes = (
			mi.get_level_values("Local Code")
			.astype(str).str.strip().str.replace(".0", "", regex=False)
		)
		return pd.MultiIndex.from_arrays([ts, codes], names=["Timestamp", "Local Code"])

	train_norm.index = _norm_index(train_norm.index)
	test_norm.index  = _norm_index(test_norm.index)
	train_ret.index  = _norm_index(train_ret.index)
	test_ret.index   = _norm_index(test_ret.index)

	train_norm = train_norm.sort_index()
	test_norm  = test_norm.sort_index()
	train_ret  = train_ret.sort_index()
	test_ret   = test_ret.sort_index()

	# 4) Quick sanity key test (should print True True)
	ts0 = train_norm.index.get_level_values("Timestamp")[0]
	code0 = str(assets_df["Local Code"].iloc[0])
	print("SANITY key:", (ts0, code0),
		  "in train_norm?", (ts0, code0) in train_norm.index,
		  "in train_ret?", (ts0, code0) in train_ret.index)
	# ----------------------------------------------------------

	print("assets codes sample:", assets_df["Local Code"].head(10).tolist())
	print("train_norm index names:", train_norm.index.names)
	print("train_ret  index names:", train_ret.index.names)
	print("train_norm Timestamp sample:", train_norm.index.get_level_values("Timestamp")[:3].tolist())
	print("train_ret  Timestamp sample:", train_ret.index.get_level_values("Timestamp")[:3].tolist())
	print("train_norm Local Code sample:", train_norm.index.get_level_values("Local Code").unique()[:5].tolist())


	print("Building environments...")

	# ---------- ENVS D'ENTRAÎNEMENT ----------
	train_env = PortfolioEnv(
		features=train_norm,
		returns=train_ret,
		assets_df=assets_df,
		params_df=params_df,
		window=window,
		mode="train",
		days_per_episode=num_day_per_ep,
	)
	train_env_nc = PortfolioEnvNoConstraints(
		features=train_norm,
		returns=train_ret,
		assets_df=assets_df,
		params_df=params_df,
		window=window,
		mode="train",
		days_per_episode=num_day_per_ep,
	)

	num_envs = 4
	train_vec_env = VectorPortfolioEnv(
		[
			PortfolioEnv(
				features=train_norm,
				returns=train_ret,
				assets_df=assets_df,
				params_df=params_df,
				window=window,
				mode="train",
				days_per_episode=num_day_per_ep,
			)
			for _ in range(num_envs)
		]
	)
	train_vec_env_nc = VectorPortfolioEnv(
		[
			PortfolioEnvNoConstraints(
				features=train_norm,
				returns=train_ret,
				assets_df=assets_df,
				params_df=params_df,
				window=window,
				mode="train",
				days_per_episode=num_day_per_ep,
			)
			for _ in range(num_envs)
		]
	)

	# ---------- ENVS DE BACKTEST ----------
	# Ici : un seul gros épisode sur TOUTE la période test
	test_env = PortfolioEnv(
		features=test_norm,
		returns=test_ret,
		assets_df=assets_df,
		params_df=params_df,
		window=window,
		mode="test",           # <--- TRES IMPORTANT
	)
	test_env_nc = PortfolioEnvNoConstraints(
		features=test_norm,
		returns=test_ret,
		assets_df=assets_df,
		params_df=params_df,
		window=window,
		mode="test",           # <--- TRES IMPORTANT
	)

	filename_ext = [str(v) for v in kwargs.values()]

	if (results_dir / f"alloc_sac_constrained_{'_'.join(filename_ext)}.csv").exists():
		print("\nTraining SAC (constraint-aware) already done")
		
	else:
		print("\nTraining SAC (constraint-aware)...")
		sac_agent = train_sac_on_vec_env(train_vec_env, cfg)
		bt_sac = backtest_policy(test_env, sac_agent, deterministic=True)
		# DRL constrained
		bt_sac["allocation"].to_csv(results_dir / f"alloc_sac_constrained_{'_'.join(filename_ext)}.csv")
		bt_sac["performance"].to_csv(results_dir / f"perf_sac_constrained_{'_'.join(filename_ext)}.csv")
		# sac_agent = train_sac_on_env(train_env, cfg)

	if (results_dir / f"alloc_sac_baseline_{'_'.join(filename_ext)}.csv").exists():
		print("\nTraining SAC baseline (no constraints) already done")
		
	else:
		print("\nTraining SAC baseline (no constraints)...")
		sac_agent_base = train_sac_on_vec_env(train_vec_env_nc, cfg)
		bt_sac_base = backtest_policy(test_env_nc, sac_agent_base, deterministic=True)
		# DRL baseline
		bt_sac_base["allocation"].to_csv(results_dir / f"alloc_sac_baseline_{'_'.join(filename_ext)}.csv")
		bt_sac_base["performance"].to_csv(results_dir / f"perf_sac_baseline_{'_'.join(filename_ext)}.csv")
		# sac_agent_base = train_sac_on_env(train_env_nc, cfg)

	print("\nBacktesting (bid/ask-based returns)...")
	# En mode "test", ces envs parcourent toute la période
	bt_eq = backtest_equal_weight(test_env_nc)  # equal-weight sans contraintes
	bt_mv = backtest_mean_variance(test_env_nc)
	
	"""
	res_sac = summarize_strategy("SAC constrained", bt_sac)
	res_sac_base = summarize_strategy("SAC baseline", bt_sac_base)
	res_eq = summarize_strategy("Equal-weight", bt_eq)
	res_mv = summarize_strategy("Mean-variance", bt_mv)

	summary = {
		"SAC_constrained": res_sac,
		"SAC_baseline": res_sac_base,
		"Equal_weight": res_eq,
		"Mean_variance": res_mv,
	}
	"""
	
	results_dir.mkdir(exist_ok=True)

	# Equal-weight
	bt_eq["allocation"].to_csv(results_dir / f"alloc_equal_weight_{'_'.join(filename_ext)}.csv")
	bt_eq["performance"].to_csv(results_dir / f"perf_equal_weight_{'_'.join(filename_ext)}.csv")

	# Mean-variance
	bt_mv["allocation"].to_csv(results_dir / f"alloc_mean_variance_{'_'.join(filename_ext)}.csv")
	bt_mv["performance"].to_csv(results_dir / f"perf_mean_variance_{'_'.join(filename_ext)}.csv")

	print(f"\nSaved metrics to: {results_dir}")
