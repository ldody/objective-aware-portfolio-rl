import sys
from pathlib import Path
import torch
import os
import argparse
import itertools
import pandas as pd

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
	results_dir.mkdir(exist_ok=True)
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--task-id", type=int, default=None,
						help="ID de la tâche dans l'array SLURM")
	args = parser.parse_args()
	
	user_request = [1,2,3,4,5]
	timeframe = ['5min','15min','30min','1H','2H']

	combinaisons = list(itertools.product(user_request, timeframe))

	df = pd.DataFrame(combinaisons, columns=["user_request", "timeframe"])
	
	run_full_experiment(
		data_dir=data_dir,
		results_dir=results_dir,
		window=50,          # 20 previous timestamps
		num_episodes=200,    # increase for real training
		**dict(df.iloc[args.task_id]),
	)
