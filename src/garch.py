import os
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
import seaborn as sns

INSIGHTS_DIR = "insights"


def run_univariate_garch(save: bool = True):
    """
    Step 2.1: Univariate GARCH(1,1) per asset (GLOBAL ONLY)

    Updates:
    - Removes NSEI and USDINR
    - Uses expanding mean (lagged) as expected return
    - Handles scaling cleanly

    Outputs:
    - standardized residuals
    - conditional volatility
    - summary diagnostics
    """

    path = os.path.join(INSIGHTS_DIR, "unified.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # --- Select ONLY global assets ---
    ret_cols = [
        c for c in df.columns
        if c.startswith("ret_")
        and "NSEI" not in c
        and "USDINR" not in c
    ]

    residuals_df = pd.DataFrame({"date": df["date"]})
    vol_df = pd.DataFrame({"date": df["date"]})

    summary_rows = []

    print("=== RUNNING GARCH(1,1) FOR GLOBAL ASSETS ===\n")

    for col in ret_cols:
        asset = col.replace("ret_", "")

        print(f"Processing {asset}...")

        series = df[col].copy()

        # --- Expected return (expanding mean, lagged) ---
        exp_mean = series.expanding().mean().shift(1)

        # demeaned returns
        series_dm = (series - exp_mean).dropna()

        # --- ARCH test ---
        arch_test = het_arch(series_dm, nlags=5)
        arch_pval = arch_test[1]

        # --- GARCH fit (scaled) ---
        model = arch_model(
            series_dm * 100,   # scale for stability
            mean="Zero",       # already demeaned
            vol="GARCH",
            p=1,
            q=1,
            dist="normal"
        )

        res = model.fit(disp="off")

        # --- Extract ---
        cond_vol = res.conditional_volatility / 100
        std_resid = res.resid / res.conditional_volatility

        # Align index properly
        cond_vol.index = series_dm.index
        std_resid.index = series_dm.index

        # --- Store ---
        residuals_df[f"resid_{asset}"] = std_resid
        vol_df[f"garch_vol_{asset}"] = cond_vol

        summary_rows.append({
            "asset": asset,
            "arch_pval": arch_pval,
            "omega": res.params.get("omega", np.nan),
            "alpha": res.params.get("alpha[1]", np.nan),
            "beta": res.params.get("beta[1]", np.nan),
            "alpha_beta": res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0)
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n=== GARCH SUMMARY ===")
    print(summary_df)

    # --- Save ---
    if save:
        residuals_path = os.path.join(INSIGHTS_DIR, "garch_residuals.csv")
        vol_path = os.path.join(INSIGHTS_DIR, "garch_volatility.csv")
        summary_path = os.path.join(INSIGHTS_DIR, "garch_summary.csv")

        residuals_df.to_csv(residuals_path, index=False)
        vol_df.to_csv(vol_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSaved → {residuals_path}")
        print(f"Saved → {vol_path}")
        print(f"Saved → {summary_path}")

    return residuals_df, vol_df, summary_df

def run_dcc_garch(save: bool = True):
    """
    Step 2.2: Dynamic Conditional Correlation (DCC-GARCH)

    Uses standardized residuals from Step 2.1.

    Outputs:
    - Dynamic correlation matrices
    - Average market correlation over time
    - Crash vs non-crash correlation comparison

    Saves:
    - insights/dcc_correlations.csv
    - insights/dcc_summary.csv
    """

    # --- Load standardized residuals ---
    resid_path = os.path.join(INSIGHTS_DIR, "garch_residuals.csv")
    unified_path = os.path.join(INSIGHTS_DIR, "unified.csv")

    resid_df = pd.read_csv(resid_path)
    unified_df = pd.read_csv(unified_path)

    resid_df["date"] = pd.to_datetime(resid_df["date"])
    unified_df["date"] = pd.to_datetime(unified_df["date"])

    # --- Residual matrix ---
    resid_cols = [c for c in resid_df.columns if c.startswith("resid_")]

    Z = resid_df[resid_cols].dropna().values

    T, N = Z.shape

    print("=== RUNNING DCC-GARCH ===")
    print(f"Observations: {T}")
    print(f"Assets: {N}")

    # --- Unconditional covariance ---
    S = np.cov(Z.T)

    # --- DCC parameters ---
    # Typical stable values used in empirical finance
    a = 0.02
    b = 0.97

    # Stability check
    if a + b >= 1:
        raise ValueError("DCC parameters unstable: a + b must be < 1")

    # --- Initialize ---
    Qt = S.copy()

    avg_corr_series = []
    dates = resid_df.loc[resid_df[resid_cols].dropna().index, "date"].reset_index(drop=True)

    pairwise_corrs = []

    asset_names = [c.replace("resid_", "") for c in resid_cols]

    # --- Recursive DCC ---
    for t in range(T):

        z_prev = Z[t - 1].reshape(-1, 1) if t > 0 else np.zeros((N, 1))

        # DCC recursion
        Qt = (
            (1 - a - b) * S
            + a * (z_prev @ z_prev.T)
            + b * Qt
        )

        # Convert to correlation matrix
        diag_q = np.sqrt(np.diag(Qt))

        Rt = Qt / np.outer(diag_q, diag_q)

        # Numerical safety
        Rt = np.clip(Rt, -1, 1)

        # --- Average off-diagonal correlation ---
        avg_corr = (
            Rt[np.triu_indices(N, k=1)].mean()
        )

        avg_corr_series.append(avg_corr)

        # --- Store pairwise correlations ---
        row = {"date": dates.iloc[t]}

        for i in range(N):
            for j in range(i + 1, N):

                pair_name = f"{asset_names[i]}__{asset_names[j]}"
                row[pair_name] = Rt[i, j]

        pairwise_corrs.append(row)

    # --- Correlation dataframe ---
    corr_df = pd.DataFrame(pairwise_corrs)

    # --- Add average system correlation ---
    corr_df["avg_system_corr"] = avg_corr_series

    # --- Merge crash regime ---
    crash_cols = ["date", "crash_combined"]

    crash_df = unified_df[crash_cols].copy()

    corr_df = corr_df.merge(crash_df, on="date", how="inner")

    # --- Crash vs non-crash summary ---
    summary_rows = []

    pair_cols = [
        c for c in corr_df.columns
        if "__" in c
    ]

    for pair in pair_cols:

        crash_mean = corr_df.loc[
            corr_df["crash_combined"] == 1,
            pair
        ].mean()

        normal_mean = corr_df.loc[
            corr_df["crash_combined"] == 0,
            pair
        ].mean()

        delta = crash_mean - normal_mean

        summary_rows.append({
            "pair": pair,
            "normal_corr": normal_mean,
            "crash_corr": crash_mean,
            "delta_corr": delta
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n=== DCC SUMMARY ===")
    print(summary_df.sort_values("delta_corr", ascending=False).head())

    # --- Save ---
    if save:

        corr_path = os.path.join(INSIGHTS_DIR, "dcc_correlations.csv")
        summary_path = os.path.join(INSIGHTS_DIR, "dcc_summary.csv")

        corr_df.to_csv(corr_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        print(f"\nSaved → {corr_path}")
        print(f"Saved → {summary_path}")

    return corr_df, summary_df

# =========================================================
# VISUALIZATION HELPERS
# =========================================================

GLOBAL_STOCKS = [
    "^GSPC",
    "^GSPTSE",
    "^HSI",
    "^BVSP",
    "^AXJO"
]


def _load_dcc_data():
    path = os.path.join(INSIGHTS_DIR, "dcc_correlations.csv")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    return df


def _resolve_pair_column(columns, asset1, asset2):
    """
    Resolve pair column irrespective of ordering.
    """

    pair1 = f"{asset1}__{asset2}"
    pair2 = f"{asset2}__{asset1}"

    if pair1 in columns:
        return pair1

    if pair2 in columns:
        return pair2

    return None


def _highlight_crashes(ax, df):
    """
    Highlight crash periods using shaded regions.
    """

    crash_mask = df["crash_combined"] == 1

    in_crash = False
    start_date = None

    for i in range(len(df)):

        if crash_mask.iloc[i] and not in_crash:
            in_crash = True
            start_date = df["date"].iloc[i]

        elif not crash_mask.iloc[i] and in_crash:
            end_date = df["date"].iloc[i]

            ax.axvspan(
                start_date,
                end_date,
                alpha=0.2,
                color="#DB2020"
            )

            in_crash = False

    # handle ending crash regime
    if in_crash:
        ax.axvspan(
            start_date,
            df["date"].iloc[-1],
            alpha=0.2,
            color="#DB2020"
        )


# =========================================================
# 1. AVERAGE SYSTEM CORRELATION
# =========================================================

def plot_avg_system_correlation():
    """
    Plot average system-wide correlation over time.
    """

    df = _load_dcc_data()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["date"],
        df["avg_system_corr"],
        linewidth=2
    )

    _highlight_crashes(ax, df)

    ax.set_title("Average System Correlation Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Correlation")

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =========================================================
# 2. DELTA CORRELATION HEATMAPS
# =========================================================

def plot_dcc_delta_heatmaps():
    """
    Create 3 side-by-side heatmaps:
    1. BTC vs all
    2. ETH vs all
    3. GOLD vs all

    Uses shared/global color scale.
    """

    path = os.path.join(INSIGHTS_DIR, "dcc_summary.csv")
    df = pd.read_csv(path)

    def get_delta(asset1, asset2):

        pair1 = f"{asset1}__{asset2}"
        pair2 = f"{asset2}__{asset1}"

        row = df[df["pair"] == pair1]

        if row.empty:
            row = df[df["pair"] == pair2]

        if row.empty:
            return np.nan

        return row["delta_corr"].values[0]

    btc_assets = GLOBAL_STOCKS + ["GC=F", "ETH-USD"]
    eth_assets = GLOBAL_STOCKS + ["GC=F", "BTC-USD"]
    gold_assets = GLOBAL_STOCKS + ["BTC-USD", "ETH-USD"]

    btc_vals = [get_delta("BTC-USD", a) for a in btc_assets]
    eth_vals = [get_delta("ETH-USD", a) for a in eth_assets]
    gold_vals = [get_delta("GC=F", a) for a in gold_assets]

    vmin = min(
        np.nanmin(btc_vals),
        np.nanmin(eth_vals),
        np.nanmin(gold_vals)
    )

    vmax = max(
        np.nanmax(btc_vals),
        np.nanmax(eth_vals),
        np.nanmax(gold_vals)
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharey=False
    )

    heatmaps = [
        (
            axes[0],
            pd.DataFrame(
                {"Delta": btc_vals},
                index=btc_assets
            ),
            "BTC vs Assets"
        ),
        (
            axes[1],
            pd.DataFrame(
                {"Delta": eth_vals},
                index=eth_assets
            ),
            "ETH vs Assets"
        ),
        (
            axes[2],
            pd.DataFrame(
                {"Delta": gold_vals},
                index=gold_assets
            ),
            "Gold vs Assets"
        )
    ]

    for ax, data, title in heatmaps:

        sns.heatmap(
            data,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".3f",
            vmin=vmin,
            vmax=vmax,
            cbar=ax == axes[-1],
            ax=ax
        )

        ax.set_title(title)

    plt.suptitle(
        "Crash vs Normal Correlation Change (Delta)",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# =========================================================
# 3. BTC vs ETH DYNAMIC CORRELATION
# =========================================================

def plot_btc_eth_dcc():
    """
    Plot BTC-ETH dynamic correlation.
    """

    df = _load_dcc_data()

    pair = _resolve_pair_column(
        df.columns,
        "BTC-USD",
        "ETH-USD"
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["date"],
        df[pair],
        linewidth=2
    )

    _highlight_crashes(ax, df)

    ax.set_title("BTC vs ETH Dynamic Correlation")
    ax.set_xlabel("Date")
    ax.set_ylabel("DCC Correlation")

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =========================================================
# 4. CRYPTO VS EQUITY (6 SEPARATE PLOTS)
# =========================================================

def plot_crypto_equity_dcc():
    """
    Create 6 plots:
    BTC vs Asset
    ETH vs Asset

    One subplot per asset:
    - 5 global equities
    - Gold
    """

    df = _load_dcc_data()

    assets = GLOBAL_STOCKS + ["GC=F"]

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
        sharex=True
    )

    axes = axes.flatten()

    for idx, asset in enumerate(assets):

        ax = axes[idx]

        btc_pair = _resolve_pair_column(
            df.columns,
            "BTC-USD",
            asset
        )

        eth_pair = _resolve_pair_column(
            df.columns,
            "ETH-USD",
            asset
        )

        if btc_pair:
            ax.plot(
                df["date"],
                df[btc_pair],
                label="BTC",
                linewidth=2
            )

        if eth_pair:
            ax.plot(
                df["date"],
                df[eth_pair],
                label="ETH",
                linewidth=2
            )

        _highlight_crashes(ax, df)

        ax.set_title(asset)
        ax.set_ylabel("Correlation")

        ax.grid(True)
        ax.legend()

    plt.suptitle(
        "Crypto vs Equity Dynamic Correlations",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# =========================================================
# 5. GOLD SAFE HAVEN PLOT
# =========================================================

def plot_gold_safe_haven():
    """
    Plot Gold correlations against all equities.
    """

    df = _load_dcc_data()

    fig, ax = plt.subplots(figsize=(14, 6))

    for asset in GLOBAL_STOCKS:

        pair = _resolve_pair_column(
            df.columns,
            "GC=F",
            asset
        )

        if pair is None:
            continue

        ax.plot(
            df["date"],
            df[pair],
            label=asset
        )

    _highlight_crashes(ax, df)

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title("Gold vs Equity Dynamic Correlations")
    ax.set_xlabel("Date")
    ax.set_ylabel("DCC Correlation")

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()