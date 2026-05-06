import os
import yfinance as yf
import pandas as pd
from datetime import datetime

# Step 1.1 asset universe EXACTLY as per methodology
ASSETS = {
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",
    "^GSPC": "equity",
    "^GSPTSE": "equity",
    "^HSI": "equity",
    "^BVSP": "equity",
    "^AXJO": "equity",
    "^NSEI": "equity",
    "GC=F": "commodity",
    "USDINR=X": "fx"
}

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

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
            auto_adjust=False,
            progress=False
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


def run_step_1_1():
    """
    Execute Step 1.1:
    - Fetch daily OHLCV data for all assets
    - Store each as separate CSV in dataset/
    """

    print("=== STEP 1.1: Asset Price Data Collection START ===")

    for ticker in ASSETS.keys():
        fetch_and_store_ticker(ticker)

    print("=== STEP 1.1 COMPLETE ===")