import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"


def plot_global_log_returns(batch_size: int = 4):
    """
    Plot log returns ONLY for:
    - BTC, ETH
    - 5 global equity indices (no India)
    - Gold

    Excludes:
    - FX
    - India (^NSEI)

    Produces plots in batches of size batch_size
    """

    TARGET_TICKERS = [
        "BTC-USD",
        "ETH-USD",
        "^GSPC",
        "^GSPTSE",
        "^HSI",
        "^BVSP",
        "^AXJO",
        "GC=F"
    ]

    asset_returns = {}

    # --- Load & compute returns ---
    for ticker in TARGET_TICKERS:
        path = os.path.join(DATASET_DIR, f"{ticker}.csv")

        if not os.path.exists(path):
            print(f"Missing file: {ticker}.csv")
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Find adj_close column dynamically
        adj_col = [c for c in df.columns if "adj_close" in c][0]

        df["log_ret"] = np.log(df[adj_col] / df[adj_col].shift(1))

        asset_returns[ticker] = df.set_index("date")["log_ret"]

    tickers = list(asset_returns.keys())

    # --- Plot in batches ---
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        plt.figure(figsize=(14, 7))

        for ticker in batch:
            plt.plot(asset_returns[ticker], linewidth=0.6, label=ticker)

        plt.title(
            "Log Returns (Adj Close) — Global Assets",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Log Return", fontsize=11)

        plt.legend(loc="upper right", fontsize=10, frameon=False)
        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()