from pathlib import Path

import pandas as pd


def load_assets(path: Path) -> pd.DataFrame:
	"""Load asset list file with columns: ['Local Code', 'Name']."""
	assets = pd.read_csv(path, index_col=0)
	required = {"Local Code", "Name (English)"}
	if not required.issubset(assets.columns):
		raise ValueError(f"assets_portfolio.csv must contain columns {required}")
	return assets


def load_parameters(path: Path) -> pd.DataFrame:
	"""Load DRL constraint parameters with columns ['Parameters', 'Value', 'Weight']."""
	params = pd.read_csv(path)
	required = {"Parameters", "Value", "Weight"}
	if not required.issubset(params.columns):
		raise ValueError(f"parameters_portfolio.csv must contain columns {required}")
	return params.set_index("Parameters")


def load_and_merge_prices(
	assets_df: pd.DataFrame,
	raw_data_dir: Path,
	**kwargs,
) -> pd.DataFrame:
	"""Load per-asset intraday data and merge into a panel DataFrame.

	Returns
	-------
	DataFrame
		MultiIndex (Timestamp, Local Code) with all original columns.
	"""
	
	assets_df = assets_df.copy()
	assets_df["Local Code"] = assets_df["Local Code"].apply(lambda x: str(int(x)))
	
	dfs = []
	start_date = pd.Timestamp("2025-06-01")
	for _, row in assets_df.iterrows():
		code = row["Local Code"]
		file_path = raw_data_dir / f"{code}.T.csv"
		if not file_path.exists():
			print(f"[WARN] Missing price file for {code}: {file_path}")
			continue

		df = pd.read_csv(file_path, header=[0,1], index_col=0)
		df.columns = df.columns.get_level_values(1)
		df.index = pd.to_datetime(df.index)
		df = df.loc[df.index >= start_date]
		df.ffill(inplace=True)

		if kwargs:
			df = df.between_time("00:00", "06:30")
			
			df = df.resample(kwargs['timeframe']).agg({'ACVOL_UNS':'sum',
													   'BID_HIGH_1':'max',
													   'BID_LOW_1':'min',
													   'OPEN_BID':'first',
													   'BID':'last',
													   'ASK_HIGH_1':'max',
													   'ASK_LOW_1':'min',
													   'OPEN_ASK':'first',
													   'ASK':'last',
													   'MID_HIGH':'max',
													   'MID_LOW':'min',
													   'MID_OPEN':'first',
													   'MID_PRICE':'last'})
			
			# fill gaps created by resampling
			df = df.ffill()

			print(
				"NaN rows after resample:",
				df[["BID", "ASK", "MID_PRICE"]].isna().all(axis=1).mean()
			)

			# drop only truly empty bars
			df = df.dropna(how="all", subset=["BID", "ASK", "MID_PRICE"])

			# (optional but often good) if a bar has any of these missing, you cannot compute returns safely:
			df = df.dropna(subset=["BID", "ASK", "MID_PRICE"])
			
		df.reset_index(drop=False, inplace=True)
		df["Local Code"] = code
		dfs.append(df)
		print(df)

	if not dfs:
		raise ValueError("No price files loaded. Check data/raw/<LocalCode>.T.csv files.")
	
	data = pd.concat(dfs, ignore_index=True)
	data.set_index(["Timestamp", "Local Code"], inplace=True)
	data.sort_index(inplace=True)
	print("Codes in final panel:", sorted(data.index.get_level_values("Local Code").unique()))
	return data
