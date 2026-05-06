import os
import pandas as pd

DATASET_DIR = "dataset"


def check_nan_across_files(save_report: bool = True):
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

    if save_report:
        output_path = os.path.join(DATASET_DIR, "nan_report.csv")
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved report → {output_path}")

    return summary_df

# --- FX mapping for non-USD assets ---
FX_MAP = {
    "^GSPTSE": "CAD=X",
    "^HSI": "HKD=X",
    "^BVSP": "BRL=X",
    "^AXJO": "AUD=X"
}


def _load_fx_series(fx_ticker):
    """
    Load FX series and return (date, rate)
    Uses adj_close as FX rate
    """
    path = os.path.join(DATASET_DIR, f"{fx_ticker}.csv")
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    # Find adj_close column dynamically
    adj_col = [c for c in df.columns if "adj_close" in c][0]

    return df[["date", adj_col]].rename(columns={adj_col: "fx_rate"})


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

        price_cols = [c for c in df.columns if any(x in c for x in ["open", "high", "low", "close", "adj_close"])]

        for col in price_cols:
            df[col] = df[col] * df["fx_rate"]

        df.drop(columns=["fx_rate"], inplace=True)

        # Save back (in-place)
        df.to_csv(path, index=False)

        print(f"✔ Converted {ticker} to INR")

    print("\n=== ALL FILES CONVERTED TO INR ===")