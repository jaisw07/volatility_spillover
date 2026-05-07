import os
import yfinance as yf
import pandas as pd
from datetime import datetime

ASSETS = {
    # --- Crypto ---
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",

    # --- Equity Indices ---
    "^GSPC": "equity",     # USA (USD)
    "^GSPTSE": "equity",   # Canada (CAD)
    "^HSI": "equity",      # Hong Kong (HKD)
    "^BVSP": "equity",     # Brazil (BRL)
    "^AXJO": "equity",     # Australia (AUD)
    "^NSEI": "equity",     # India (INR)

    # --- Commodities ---
    "GC=F": "commodity",   # Gold (USD)

    # --- FX (CRITICAL for conversion) ---
    "USDINR=X": "fx",  # USD → INR (base conversion anchor)

    # Cross rates → USD
    "CAD=X": "fx",     # CAD → USD
    "HKD=X": "fx",     # HKD → USD
    "BRL=X": "fx",     # BRL → USD
    "AUD=X": "fx"      # AUD → USD
}

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

DATASET_DIR = "dataset"


def fetch_and_store_ticker(ticker: str, start: str = START_DATE, end: str = END_DATE):
    """
    Fetch daily OHLCV data for a single ticker and store as CSV.

    Output path: dataset/<ticker>.csv
    """

    try:
        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=True
        )

        if df.empty:
            print(f"Warning: No data for {ticker}")
            return

        # Reset index to have Date as column
        df.reset_index(inplace=True)

        # Ensure consistent column naming (handle MultiIndex/tuple columns from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(part) for part in col if part)
                .lower()
                .replace(" ", "_")
                for col in df.columns
            ]
        else:
            normalized_cols = []
            for col in df.columns:
                if isinstance(col, tuple):
                    col = "_".join(str(part) for part in col if part)
                normalized_cols.append(str(col).lower().replace(" ", "_"))
            df.columns = normalized_cols

        # Standardize OHLCV column names to include ticker suffix
        # Example: close_^axjo, high_^axjo, ...
        suffix = ticker.lower()
        rename_map = {}
        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col in df.columns:
                rename_map[col] = f"{col}_{suffix}"

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # Add ticker column for traceability
        df["ticker"] = ticker

        # Create directory if not exists
        os.makedirs(DATASET_DIR, exist_ok=True)

        # Save file
        file_path = os.path.join(DATASET_DIR, f"{ticker}.csv")
        df.to_csv(file_path, index=False)

        print(f"Saved → {file_path}")

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")


def download_assets():
    """
    Execute Step 1.1:
    - Fetch daily OHLCV data for all assets
    - Store each as separate CSV in dataset/
    """

    print("=== Asset Price Data Collection ===")

    for ticker in ASSETS.keys():
        fetch_and_store_ticker(ticker)

    print("=== Data Collection Complete ===")