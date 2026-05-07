import os
import pandas as pd
import numpy as np

DATASET_DIR = "dataset"
INSIGHTS_DIR = "insights"

def check_nan_across_files():
    """
    Step 1.2 (pre-check):
    Scan all dataset/*.csv files and report NaN statistics.

    Outputs:
    - Prints summary per file
    - Returns a dataframe summary
    - Optionally saves dataset/nan_report.csv
    """

    summary_rows = []

    print("=== NaN CHECK ACROSS ALL FILES ===\n")

    for file in os.listdir(DATASET_DIR):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(DATASET_DIR, file)
        df = pd.read_csv(file_path)

        total_cells = df.shape[0] * df.shape[1]
        total_nans = df.isna().sum().sum()
        nan_pct = (total_nans / total_cells) * 100 if total_cells > 0 else 0

        # Column-wise NaNs
        col_nan_counts = df.isna().sum()
        col_nan_pct = (col_nan_counts / len(df)) * 100

        print(f"File: {file}")
        print(f"Shape: {df.shape}")
        print(f"Total NaNs: {total_nans} ({nan_pct:.4f}%)")

        # Only print columns that actually have NaNs
        problematic_cols = col_nan_counts[col_nan_counts > 0]

        if len(problematic_cols) > 0:
            print("Columns with NaNs:")
            for col in problematic_cols.index:
                print(f"  - {col}: {col_nan_counts[col]} ({col_nan_pct[col]:.2f}%)")
        else:
            print("No NaNs found.")

        print("-" * 50)

        summary_rows.append({
            "file": file,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "total_nans": total_nans,
            "nan_pct": nan_pct
        })

    summary_df = pd.DataFrame(summary_rows)

    # Sort worst files on top
    summary_df = summary_df.sort_values(by="nan_pct", ascending=False)

    print("\n=== SUMMARY (Worst First) ===")
    print(summary_df)

    return summary_df

# --- FX mapping for non-USD assets ---
FX_MAP = {
    "^GSPTSE": "CAD=X",
    "^HSI": "HKD=X",
    "^BVSP": "BRL=X",
    "^AXJO": "AUD=X"
}


def _get_price_column(df: pd.DataFrame) -> str:
    """Pick best available price column with auto_adjust compatibility."""

    for name in ("adj_close", "close"):
        matches = [c for c in df.columns if name in c]
        if matches:
            return matches[0]

    raise ValueError("No price column found (expected adj_close or close)")


def _load_fx_series(fx_ticker):
    """
    Load FX series and return (date, rate)
    Uses adj_close or close as FX rate
    """
    path = os.path.join(DATASET_DIR, f"{fx_ticker}.csv")
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    price_col = _get_price_column(df)

    return df[["date", price_col]].rename(columns={price_col: "fx_rate"})


def _convert_to_usd(df, ticker):
    """
    Convert local currency → USD for non-USD assets
    """
    fx_ticker = FX_MAP.get(ticker)

    if fx_ticker is None:
        return df  # already USD or INR

    fx_df = _load_fx_series(fx_ticker)

    df = df.merge(fx_df, on="date", how="left")

    price_cols = [c for c in df.columns if any(x in c for x in ["open", "high", "low", "close", "adj_close"])]

    for col in price_cols:
        df[col] = df[col] * df["fx_rate"]  # local → USD

    df.drop(columns=["fx_rate"], inplace=True)

    return df


def convert_all_to_inr():
    """
    Convert all dataset files to INR (in place)

    Steps:
    - Local → USD (if needed)
    - USD → INR
    """

    print("=== CONVERTING ALL ASSETS TO INR ===\n")

    # Load USDINR once
    usdinr = _load_fx_series("USDINR=X")

    for file in os.listdir(DATASET_DIR):
        if not file.endswith(".csv"):
            continue

        ticker = file.replace(".csv", "")
        path = os.path.join(DATASET_DIR, file)

        # Skip FX files themselves
        if ticker in ["USDINR=X", "CAD=X", "HKD=X", "BRL=X", "AUD=X"]:
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        print(f"Processing {ticker}...")

        # --- Step 1: Convert to USD if needed ---
        df = _convert_to_usd(df, ticker)

        # --- Step 2: Convert USD → INR ---
        df = df.merge(usdinr, on="date", how="left")

        price_cols = [c for c in df.columns if any(x in c for x in ["open", "high", "low", "close"])]

        for col in price_cols:
            df[col] = df[col] * df["fx_rate"]

        df.drop(columns=["fx_rate"], inplace=True)

        # Save back (in-place)
        df.to_csv(path, index=False)

        print(f"✔ Converted {ticker} to INR")

    print("\n=== ALL FILES CONVERTED TO INR ===")


def check_calendar_alignment():
    """
    Check calendar alignment across all asset CSVs.

    Outputs:
    - Common date range
    - Missing dates per asset
    - Overlap statistics
    - Returns summary dataframe
    """

    print("=== CALENDAR ALIGNMENT CHECK ===\n")

    date_sets = {}
    all_dates = set()

    # --- Load all dates ---
    for file in os.listdir(DATASET_DIR):
        if not file.endswith(".csv"):
            continue

        ticker = file.replace(".csv", "")
        path = os.path.join(DATASET_DIR, file)

        df = pd.read_csv(path, usecols=["date"])
        df["date"] = pd.to_datetime(df["date"])

        dates = set(df["date"])
        date_sets[ticker] = dates
        all_dates.update(dates)

    # --- Global date index ---
    all_dates = sorted(all_dates)
    all_dates_set = set(all_dates)

    print(f"Global date range: {min(all_dates).date()} → {max(all_dates).date()}")
    print(f"Total unique dates (union): {len(all_dates)}\n")

    # --- Intersection (perfect overlap) ---
    common_dates = set.intersection(*date_sets.values())
    print(f"Common dates across ALL assets: {len(common_dates)}")
    print(f"Overlap ratio: {len(common_dates) / len(all_dates):.4f}\n")

    # --- Per asset gaps ---
    summary = []

    for ticker, dates in date_sets.items():
        missing = all_dates_set - dates
        missing_pct = (len(missing) / len(all_dates)) * 100

        print(f"{ticker}:")
        print(f"  available: {len(dates)}")
        print(f"  missing: {len(missing)} ({missing_pct:.2f}%)")

        # Show first few missing dates for debugging
        if len(missing) > 0:
            sample = sorted(list(missing))[:5]
            print(f"  sample missing: {[d.date() for d in sample]}")

        print("-" * 40)

        summary.append({
            "ticker": ticker,
            "available_dates": len(dates),
            "missing_dates": len(missing),
            "missing_pct": missing_pct
        })

    summary_df = pd.DataFrame(summary).sort_values(by="missing_pct", ascending=False)

    print("\n=== SUMMARY (Worst Alignment First) ===")
    print(summary_df)

    return summary_df

def keep_common_dates():
    """
    Restrict all asset files to only common overlapping dates.

    This ensures:
    - Perfect calendar alignment
    - No need for forward-fill later
    - Clean input for VAR / DCC

    WARNING:
    - Drops non-overlapping dates (weekends, holidays, etc.)
    """

    print("=== KEEPING ONLY COMMON DATES ACROSS ALL ASSETS ===\n")

    date_sets = {}
    dfs = {}

    # --- Load all datasets ---
    for file in os.listdir(DATASET_DIR):
        if not file.endswith(".csv"):
            continue

        ticker = file.replace(".csv", "")
        path = os.path.join(DATASET_DIR, file)

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        dfs[ticker] = df
        date_sets[ticker] = set(df["date"])

    # --- Compute intersection ---
    common_dates = sorted(set.intersection(*date_sets.values()))
    common_dates = pd.to_datetime(common_dates)

    print(f"Common dates retained: {len(common_dates)}")

    # --- Filter each dataset ---
    for ticker, df in dfs.items():
        before = len(df)

        df_filtered = df[df["date"].isin(common_dates)].copy()
        after = len(df_filtered)

        path = os.path.join(DATASET_DIR, f"{ticker}.csv")
        df_filtered.to_csv(path, index=False)

        print(f"{ticker}: {before} → {after} rows")

    print("\n=== ALL FILES NOW PERFECTLY ALIGNED ===")

def build_unified_matrix(window: int = 21, save: bool = True):
    """
    Build unified feature matrix for all assets.

    Includes:
    - Log returns
    - Realized volatility

    Assets:
    - Crypto
    - Global equities
    - NSEI
    - Gold
    - USDINR

    Saves:
    - insights/unified.csv
    """

    os.makedirs(INSIGHTS_DIR, exist_ok=True)

    TARGET = [
        "BTC-USD",
        "ETH-USD",
        "^GSPC",
        "^GSPTSE",
        "^HSI",
        "^BVSP",
        "^AXJO",
        "^NSEI",
        "GC=F",
        "USDINR=X"
    ]

    unified_df = None

    for ticker in TARGET:

        path = os.path.join(DATASET_DIR, f"{ticker}.csv")

        if not os.path.exists(path):
            print(f"Missing: {ticker}")
            continue

        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        price_col = _get_price_column(df)

        # --- Log Returns ---
        df[f"ret_{ticker}"] = np.log(
            df[price_col] / df[price_col].shift(1)
        )

        # --- Realized Volatility ---
        df[f"vol_{ticker}"] = (
            df[f"ret_{ticker}"]
            .rolling(window)
            .std()
            * np.sqrt(252)
        )

        df = df[
            [
                "date",
                f"ret_{ticker}",
                f"vol_{ticker}"
            ]
        ]

        # --- Merge ---
        if unified_df is None:
            unified_df = df
        else:
            unified_df = unified_df.merge(
                df,
                on="date",
                how="inner"
            )

    # --- Final cleanup ---
    unified_df = unified_df.dropna().reset_index(drop=True)

    # --- Save ---
    if save:

        save_path = os.path.join(
            INSIGHTS_DIR,
            "unified.csv"
        )

        unified_df.to_csv(save_path, index=False)

        print(f"Saved unified matrix → {save_path}")

    print("\n=== UNIFIED MATRIX INFO ===")
    print(unified_df.shape)
    print(unified_df.head())

    return unified_df