from pathlib import Path

import pandas as pd


def load_assets(path: Path) -> pd.DataFrame:
    """Load asset list file with columns: ['Local Code', 'Name']."""
    assets = pd.read_csv(path)
    required = {"Local Code", "Name"}
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
) -> pd.DataFrame:
    """Load per-asset intraday data and merge into a panel DataFrame.

    Returns
    -------
    DataFrame
        MultiIndex (Timestamp, Local Code) with all original columns.
    """
    dfs = []
    for _, row in assets_df.iterrows():
        code = row["Local Code"]
        file_path = raw_data_dir / f"{code}.T.csv"
        if not file_path.exists():
            print(f"[WARN] Missing price file for {code}: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df["Local Code"] = code
        dfs.append(df)

    if not dfs:
        raise ValueError("No price files loaded. Check data/raw/<LocalCode>.T.csv files.")

    data = pd.concat(dfs, ignore_index=True)
    data.set_index(["Timestamp", "Local Code"], inplace=True)
    data.sort_index(inplace=True)
    return data
