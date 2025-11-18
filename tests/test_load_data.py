import sys
from pathlib import Path
import pandas as pd

# Ensure src/ is on the Python path
base_dir = Path(__file__).resolve().parent.parent
src_dir = base_dir / "src"
if str(src_dir) not in sys.path:
	sys.path.insert(0, str(src_dir))

print(base_dir)
import drl_portfolio.data as data_loader
from drl_portfolio.config import Paths, TrainingConfig


if __name__ == "__main__":
	data_dir = base_dir / "data"
	results_dir = base_dir / "results"
	paths = Paths(data_dir=data_dir, results_dir=results_dir)
	#results_dir.mkdir(exist_ok=True)
	
	assets_df = data_loader.load_assets(paths.assets_file)
	assets_df = assets_df[assets_df['user_request'] == 1]
	print(assets_df)
	params_df = data_loader.load_parameters(paths.params_file)
	params_df = params_df[params_df['user_request'] == 1]
	print(params_df)
	panel_df = data_loader.load_and_merge_prices(assets_df, paths.raw_data_dir)
	print(panel_df)

	print("Engineering features (all columns + bid/ask returns)...")
	feat_df, ret_df = data_loader.engineer_features_and_returns(panel_df, window=20)
	print(feat_df, ret_df)
	
	print("Splitting train/test...")
	train_feat, test_feat, train_ret, test_ret = data_loader.train_test_split_time_based(
		feat_df, ret_df
	)
	print(train_feat, test_feat)

	print("Normalizing features...")
	train_norm, test_norm, _ = data_loader.normalize_features(train_feat, test_feat)
	
	print(train_norm, test_norm)