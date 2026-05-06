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

def plot_realized_volatility(window: int = 5, batch_size: int = 4):
    """
    Plot realized volatility (rolling std of log returns)

    - Uses adj_close
    - Annualized volatility (sqrt(252))
    - Same asset universe as global EDA
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

    asset_vol = {}

    for ticker in TARGET_TICKERS:
        path = os.path.join(DATASET_DIR, f"{ticker}.csv")

        if not os.path.exists(path):
            print(f"Missing file: {ticker}.csv")
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        adj_col = [c for c in df.columns if "adj_close" in c][0]

        # --- Log returns ---
        df["log_ret"] = np.log(df[adj_col] / df[adj_col].shift(1))

        # --- Realized volatility ---
        df["rv"] = df["log_ret"].rolling(window).std() * np.sqrt(252)

        asset_vol[ticker] = df.set_index("date")["rv"]

    tickers = list(asset_vol.keys())

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    # --- Plot ---
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        plt.figure(figsize=(14, 7))

        for ticker in batch:
            plt.plot(asset_vol[ticker], linewidth=1.2, label=ticker)

        plt.title(
            f"Realized Volatility ({window}-Day Rolling, Annualized)",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Volatility", fontsize=11)

        plt.legend(loc="upper right", fontsize=10, frameon=False)
        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

def detect_crypto_crashes(window: int = 5, min_duration: int = 5, plot: bool = True):
    """
    Detect crash periods for BTC and ETH using realized volatility.

    Returns:
    - summary_df (threshold + total crash days)
    - crash_periods_dict (start/end per asset)
    """

    TARGET = ["BTC-USD", "ETH-USD"]

    summary = []
    crash_periods = {}

    for ticker in TARGET:
        path = os.path.join(DATASET_DIR, f"{ticker}.csv")

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        adj_col = [c for c in df.columns if "adj_close" in c][0]

        # --- returns ---
        df["log_ret"] = np.log(df[adj_col] / df[adj_col].shift(1))

        # --- realized volatility ---
        df["rv"] = df["log_ret"].rolling(window).std() * np.sqrt(252)

        # drop initial NaNs (important)
        df = df.dropna(subset=["rv"]).copy()

        # --- threshold ---
        threshold = df["rv"].quantile(0.90)

        df["high_vol"] = (df["rv"] > threshold).astype(int)

        # --- consecutive grouping ---
        df["group"] = (df["high_vol"] != df["high_vol"].shift()).cumsum()

        df["crash_flag"] = 0
        periods = []

        for _, g in df.groupby("group"):
            if g["high_vol"].iloc[0] == 1 and len(g) >= min_duration:
                df.loc[g.index, "crash_flag"] = 1

                periods.append({
                    "start": g["date"].iloc[0],
                    "end": g["date"].iloc[-1],
                    "length": len(g)
                })

        crash_periods[ticker] = pd.DataFrame(periods)

        summary.append({
            "asset": ticker,
            "threshold": threshold,
            "crash_days": df["crash_flag"].sum(),
            "num_periods": len(periods)
        })

        print(f"\n=== {ticker} ===")
        print(f"Threshold: {threshold:.4f}")
        print(f"Crash days: {df['crash_flag'].sum()}")
        print(f"Crash periods: {len(periods)}")

        if len(periods) > 0:
            print("Sample periods:")
            print(crash_periods[ticker].head())

        # --- Plot ---
        if plot:
            plt.figure(figsize=(14, 6))

            plt.plot(df["date"], df["rv"], linewidth=1.2, label="Volatility")
            plt.axhline(threshold, linestyle="--", linewidth=1, label="90th pct")

            crash_df = df[df["crash_flag"] == 1]
            plt.scatter(
                crash_df["date"],
                crash_df["rv"],
                s=10,
                label="Crash",
                alpha=0.7,
                color="red"
            )

            plt.title(f"{ticker} Realized Volatility & Crash Periods",
                      fontsize=13, fontweight="bold")

            plt.legend(frameon=False)
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.show()

    summary_df = pd.DataFrame(summary)

    print("\n=== SUMMARY ===")
    print(summary_df)

    return summary_df, crash_periods