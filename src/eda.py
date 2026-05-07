import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
INSIGHTS_DIR = "insights"

CRASH_WINDOWS = {
    "COVID Crash": ("2020-03-01", "2020-04-15"),
    "China Ban + Tesla": ("2021-05-01", "2021-06-30"),
    "LUNA Collapse": ("2022-05-01", "2022-06-30"),
    "FTX Collapse": ("2022-11-01", "2022-12-15"),
    "Yen Carry Trade": ("2024-08-01", "2024-09-01")
}

def _highlight_crashes(ax):
    """
    Highlight predefined crash periods on plots.
    """

    for label, (start, end) in CRASH_WINDOWS.items():

        ax.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color="red",
            alpha=0.15
        )

def _get_price_column(df: pd.DataFrame) -> str:
    """Pick best available price column with auto_adjust compatibility."""

    for name in ("adj_close", "close"):
        matches = [c for c in df.columns if name in c]
        if matches:
            return matches[0]

    raise ValueError("No price column found (expected adj_close or close)")

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
        "^NSEI",
        "GC=F",
        "USDINR=X"
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

        price_col = _get_price_column(df)

        df["log_ret"] = np.log(df[price_col] / df[price_col].shift(1))

        asset_returns[ticker] = df.set_index("date")["log_ret"]

    tickers = list(asset_returns.keys())

    # --- Plot in batches ---
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        plt.figure(figsize=(14, 7))

        for ticker in batch:
            plt.plot(asset_returns[ticker], linewidth=0.5, label=ticker)

        plt.title(
            "Log Returns (Adj Close) — Global Assets",
            fontsize=14,
            fontweight="bold"
        )
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Log Return", fontsize=11)

        plt.legend(loc="upper right", fontsize=10, frameon=False)
        ax = plt.gca()
        _highlight_crashes(ax)
        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()

def plot_realized_volatility(window: int = 21, batch_size: int = 4):
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

        price_col = _get_price_column(df)

        # --- Log returns ---
        df["log_ret"] = np.log(df[price_col] / df[price_col].shift(1))

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

def plot_normalized_prices(batch_size: int = 4):

    TARGET_TICKERS = [
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

    normalized = {}

    for ticker in TARGET_TICKERS:

        path = os.path.join(DATASET_DIR, f"{ticker}.csv")

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        price_col = _get_price_column(df)

        series = df.set_index("date")[price_col]

        normalized[ticker] = (
            series / series.iloc[0]
        ) * 100

    tickers = list(normalized.keys())

    for i in range(0, len(tickers), batch_size):

        batch = tickers[i:i + batch_size]

        plt.figure(figsize=(15, 7))

        for ticker in batch:

            plt.plot(
                normalized[ticker],
                linewidth=1.5,
                label=ticker
            )

        ax = plt.gca()

        _highlight_crashes(ax)

        plt.title(
            "Normalized Asset Prices (Base = 100)",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlabel("Date")
        plt.ylabel("Normalized Price")

        plt.legend(frameon=False)

        plt.grid(alpha=0.25)

        plt.tight_layout()
        plt.show()