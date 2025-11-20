from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
	"""Global configuration for training."""

	window: int = 20
	num_episodes: int = 50
	batch_size: int = 64
	buffer_size: int = 100_000
	warmup_steps: int = 2_000
	learning_rate: float = 1e-3

	@property
	def device(self) -> str:
		import torch
		return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Paths:
	"""Centralised paths for data and results."""

	data_dir: Path
	results_dir: Path

	@property
	def assets_file(self) -> Path:
		return self.data_dir / "assets_portfolio.csv"

	@property
	def params_file(self) -> Path:
		return self.data_dir / "parameters_portfolio.csv"

	@property
	def raw_data_dir(self) -> Path:
		return self.data_dir / "raw"
