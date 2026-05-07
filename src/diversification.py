# src/diversification.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INSIGHTS_DIR = "insights"

CRASH_WINDOWS = {
    "COVID Crash": ("2020-03-01", "2020-04-15"),
    "LUNA Collapse": ("2022-05-01", "2022-06-30"),
    "FTX Collapse": ("2022-11-01", "2022-12-15")
}

PAIRS = [
    ("BTC-USD", "GC=F"),
    ("BTC-USD", "^GSPC"),
    ("ETH-USD", "^NSEI")
]


# =========================================================
# LOADERS
# =========================================================

def load_unified():

    path = os.path.join(
        INSIGHTS_DIR,
        "unified.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_dcc():

    path = os.path.join(
        INSIGHTS_DIR,
        "dcc_correlations.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    return df


def load_garch_vol():

    path = os.path.join(
        INSIGHTS_DIR,
        "garch_volatility.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    return df


# =========================================================
# HELPERS
# =========================================================

def resolve_dcc_pair(columns, a1, a2):
    patterns = [
        f"dcc_{a1}__{a2}",
        f"dcc_{a2}__{a1}",
        f"{a1}__{a2}",
        f"{a2}__{a1}",
    ]

    for pattern in patterns:
        if pattern in columns:
            return pattern

    return None


def compute_covariance(
    corr,
    vol1,
    vol2
):

    return corr * vol1 * vol2


def create_crash_mask(df):

    mask = pd.Series(False, index=df.index)

    for _, (start, end) in CRASH_WINDOWS.items():

        regime_mask = (
            (df["date"] >= pd.to_datetime(start))
            &
            (df["date"] <= pd.to_datetime(end))
        )

        mask |= regime_mask

    return mask


# =========================================================
# 1. DYNAMIC PORTFOLIO WEIGHTS
# =========================================================

def compute_dynamic_weights(save=True):

    dcc_df = load_dcc()
    vol_df = load_garch_vol()

    merged = dcc_df.merge(
        vol_df,
        on="date",
        how="inner"
    )

    output = pd.DataFrame({
        "date": merged["date"]
    })

    for a1, a2 in PAIRS:

        pair_col = resolve_dcc_pair(
            merged.columns,
            a1,
            a2
        )

        if pair_col is None:
            continue

        vol1 = merged[f"garch_vol_{a1}"]
        vol2 = merged[f"garch_vol_{a2}"]

        corr = merged[pair_col]

        cov12 = compute_covariance(
            corr,
            vol1,
            vol2
        )

        h11 = vol1 ** 2
        h22 = vol2 ** 2

        # Minimum variance portfolio weight
        weight_1 = (
            h22 - cov12
        ) / (
            h11 + h22 - 2 * cov12
        )

        weight_1 = weight_1.clip(0, 1)

        output[f"weight_{a1}"] = weight_1
        output[f"weight_{a2}"] = 1 - weight_1

    if save:

        output.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "dynamic_portfolio_weights.csv"
            ),
            index=False
        )

    return output


# =========================================================
# 2. HEDGE RATIOS
# =========================================================

def compute_hedge_ratios(save=True):

    dcc_df = load_dcc()
    vol_df = load_garch_vol()

    merged = dcc_df.merge(
        vol_df,
        on="date",
        how="inner"
    )

    output = pd.DataFrame({
        "date": merged["date"]
    })

    for a1, a2 in PAIRS:

        pair_col = resolve_dcc_pair(
            merged.columns,
            a1,
            a2
        )

        if pair_col is None:
            continue

        vol1 = merged[f"garch_vol_{a1}"]
        vol2 = merged[f"garch_vol_{a2}"]

        corr = merged[pair_col]

        cov12 = compute_covariance(
            corr,
            vol1,
            vol2
        )

        h22 = vol2 ** 2

        hedge_ratio = cov12 / h22

        output[f"hedge_{a1}_with_{a2}"] = hedge_ratio

    if save:

        output.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "hedge_ratios.csv"
            ),
            index=False
        )

    return output


# =========================================================
# 3. HEDGING EFFECTIVENESS
# =========================================================

def compute_hedging_effectiveness(save=True):

    unified = load_unified()
    hedge_df = compute_hedge_ratios(save=False)

    df = unified.merge(
        hedge_df,
        on="date",
        how="inner"
    )

    crash_mask = create_crash_mask(df)

    summary_rows = []

    for a1, a2 in PAIRS:

        ret1 = df[f"ret_{a1}"]
        ret2 = df[f"ret_{a2}"]

        hedge_col = f"hedge_{a1}_with_{a2}"

        hedge_ratio = df[hedge_col]

        hedged_returns = (
            ret1 - hedge_ratio * ret2
        )

        unhedged_var_normal = (
            ret1[~crash_mask].var()
        )

        hedged_var_normal = (
            hedged_returns[~crash_mask].var()
        )

        he_normal = (
            1 -
            (
                hedged_var_normal
                /
                unhedged_var_normal
            )
        ) * 100

        unhedged_var_crash = (
            ret1[crash_mask].var()
        )

        hedged_var_crash = (
            hedged_returns[crash_mask].var()
        )

        he_crash = (
            1 -
            (
                hedged_var_crash
                /
                unhedged_var_crash
            )
        ) * 100

        summary_rows.append({
            "pair": f"{a1}__{a2}",
            "normal_HE_pct": he_normal,
            "crash_HE_pct": he_crash
        })

    summary_df = pd.DataFrame(summary_rows)

    if save:

        summary_df.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "hedging_effectiveness.csv"
            ),
            index=False
        )

    print("\n=== HEDGING EFFECTIVENESS ===")
    print(summary_df)

    return summary_df


# =========================================================
# 4. PLOTS
# =========================================================

def plot_dynamic_weights():

    path = os.path.join(
        INSIGHTS_DIR,
        "dynamic_portfolio_weights.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    cols = [
        c for c in df.columns
        if c.startswith("weight_")
    ]

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=(14, 3 * len(cols)),
        sharex=True
    )

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):

        ax.plot(
            df["date"],
            df[col],
            linewidth=2
        )

        for _, (start, end) in CRASH_WINDOWS.items():

            ax.axvspan(
                pd.to_datetime(start),
                pd.to_datetime(end),
                color="red",
                alpha=0.20
            )

        ax.set_title(col)
        ax.set_ylabel("Weight")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_hedge_ratios():

    path = os.path.join(
        INSIGHTS_DIR,
        "hedge_ratios.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    cols = [
        c for c in df.columns
        if c.startswith("hedge_")
    ]

    fig, axes = plt.subplots(
        len(cols),
        1,
        figsize=(14, 3 * len(cols)),
        sharex=True
    )

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):

        ax.plot(
            df["date"],
            df[col],
            linewidth=2
        )

        for _, (start, end) in CRASH_WINDOWS.items():

            ax.axvspan(
                pd.to_datetime(start),
                pd.to_datetime(end),
                color="red",
                alpha=0.20
            )

        ax.set_title(col)
        ax.set_ylabel("Hedge Ratio")
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_hedging_effectiveness():

    path = os.path.join(
        INSIGHTS_DIR,
        "hedging_effectiveness.csv"
    )

    df = pd.read_csv(path)

    df = df.set_index("pair")

    df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("HE (%)")

    plt.title(
        "Hedging Effectiveness: Normal vs Crash"
    )

    plt.grid(True)

    plt.tight_layout()
    plt.show()