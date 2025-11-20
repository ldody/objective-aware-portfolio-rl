import sys
from pathlib import Path
import torch
import os

torch.set_num_threads(20)          # PyTorch internal compute threads
torch.set_num_interop_threads(20)  # Cross-op parallelism

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

# Ensure src/ is on the Python path
base_dir = Path(__file__).resolve().parent
src_dir = base_dir / "src"
if str(src_dir) not in sys.path:
	sys.path.insert(0, str(src_dir))

from drl_portfolio.training.train_sac import run_full_experiment


if __name__ == "__main__":
	data_dir = base_dir / "data"
	results_dir = base_dir / "results"
	#results_dir.mkdir(exist_ok=True)

	run_full_experiment(
		data_dir=data_dir,
		results_dir=results_dir,
		window=40,          # 20 previous timestamps
		num_episodes=30,    # increase for real training
	)
