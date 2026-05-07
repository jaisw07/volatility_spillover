import os
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import matplotlib.pyplot as plt
import seaborn as sns

INSIGHTS_DIR = "insights"

CRASH_WINDOWS = {
    "COVID Crash": ("2020-03-01", "2020-04-15"),
    "LUNA Collapse": ("2022-05-01", "2022-06-30"),
    "FTX Collapse": ("2022-11-01", "2022-12-15")
}

def run_univariate_garch(save: bool = True):

    """
    Univariate GARCH(1,1) for ALL assets.

    Includes:
    - Crypto
    - Global equities
    - NSEI
    - Gold
    - USDINR

    Outputs:
    - standardized residuals
    - conditional volatility
    - diagnostics summary
    """

    path = os.path.join(INSIGHTS_DIR, "unified.csv")

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    ret_cols = [
        c for c in df.columns
        if c.startswith("ret_")
    ]

    residuals_df = pd.DataFrame({
        "date": df["date"]
    })

    vol_df = pd.DataFrame({
        "date": df["date"]
    })

    summary_rows = []

    print("=== RUNNING UNIVARIATE GARCH ===\n")

    for col in ret_cols:

        asset = col.replace("ret_", "")

        print(f"Processing {asset}...")

        series = df[col].copy()

        # expanding mean
        exp_mean = (
            series
            .expanding()
            .mean()
            .shift(1)
        )

        series_dm = (
            series - exp_mean
        ).dropna()

        # ARCH effect test
        arch_test = het_arch(
            series_dm,
            nlags=5
        )

        arch_pval = arch_test[1]

        # GARCH fit
        model = arch_model(
            series_dm * 100,
            mean="Zero",
            vol="GARCH",
            p=1,
            q=1,
            dist="t"
        )

        res = model.fit(
            disp="off"
        )

        cond_vol = (
            res.conditional_volatility / 100
        )

        std_resid = (
            res.resid / res.conditional_volatility
        )

        cond_vol.index = series_dm.index
        std_resid.index = series_dm.index

        residuals_df[
            f"resid_{asset}"
        ] = std_resid

        vol_df[
            f"garch_vol_{asset}"
        ] = cond_vol

        alpha = res.params.get(
            "alpha[1]",
            np.nan
        )

        beta = res.params.get(
            "beta[1]",
            np.nan
        )

        summary_rows.append({
            "asset": asset,
            "arch_pval": arch_pval,
            "omega": res.params.get(
                "omega",
                np.nan
            ),
            "nu": res.params.get(
                "nu",
                np.nan
            ),
            "alpha": alpha,
            "beta": beta,
            "alpha_beta": alpha + beta
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    print("\n=== GARCH SUMMARY ===")
    print(summary_df)

    if save:

        residuals_df.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "garch_residuals.csv"
            ),
            index=False
        )

        vol_df.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "garch_volatility.csv"
            ),
            index=False
        )

        summary_df.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "garch_summary.csv"
            ),
            index=False
        )

    return (
        residuals_df,
        vol_df,
        summary_df
    )

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

    summary_rows = []

    pair_cols = [
        c for c in corr_df.columns
        if "__" in c
    ]

    for pair in pair_cols:

        full_mean = corr_df[pair].mean()

        row = {
            "pair": pair,
            "full_sample_corr": full_mean
        }

        for regime, (start, end) in CRASH_WINDOWS.items():

            mask = (
                (corr_df["date"] >= pd.to_datetime(start))
                &
                (corr_df["date"] <= pd.to_datetime(end))
            )

            regime_mean = corr_df.loc[
                mask,
                pair
            ].mean()

            row[
                regime.replace(" ", "_").lower()
            ] = regime_mean

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== DCC SUMMARY ===")
    print(summary_df.head())
    
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
    "^AXJO",
    "^NSEI",
    "USDINR=X"
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


def _highlight_crashes(ax):

    for label, (start, end) in CRASH_WINDOWS.items():

        ax.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color="#DB2020",
            alpha=0.20
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

    _highlight_crashes(ax)

    ax.set_title("Average System Correlation Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Correlation")

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =========================================================
# 2. DELTA CORRELATION HEATMAPS
# =========================================================

def plot_regime_correlation_heatmaps():

    """
    Heatmaps of average DCC correlations
    during each stress regime.
    """

    path = os.path.join(
        INSIGHTS_DIR,
        "dcc_summary.csv"
    )

    df = pd.read_csv(path)

    regimes = [
        "covid_crash",
        "luna_collapse",
        "ftx_collapse"
    ]

    assets = [
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

    for regime in regimes:

        matrix = pd.DataFrame(
            np.nan,
            index=assets,
            columns=assets
        )

        for _, row in df.iterrows():

            pair = row["pair"]

            a1, a2 = pair.split("__")

            val = row[regime]

            matrix.loc[a1, a2] = val
            matrix.loc[a2, a1] = val

        np.fill_diagonal(matrix.values, 1)

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            matrix,
            cmap="coolwarm",
            center=0,
            annot=False
        )

        plt.title(
            f"DCC Correlations — {regime}"
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

    _highlight_crashes(ax)

    ax.set_title("BTC vs ETH Dynamic Correlation")
    ax.set_xlabel("Date")
    ax.set_ylabel("DCC Correlation")

    ax.grid(True)

    plt.tight_layout()
    plt.show()


# =========================================================
# 4. CRYPTO VS EQUITY (DYNAMIC SUBPLOTS)
# =========================================================

def plot_crypto_equity_dcc():
    """
    Create one subplot per asset showing BTC and ETH correlations.
    """

    df = _load_dcc_data()

    assets = GLOBAL_STOCKS + ["GC=F"]
    total_assets = len(assets)
    cols = 2
    rows = int(np.ceil(total_assets / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(16, 4 * rows + 2),
        sharex=True
    )

    axes = np.array(axes).flatten()

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

        _highlight_crashes(ax)

        ax.set_title(asset)
        ax.set_ylabel("Correlation")

        ax.grid(True)
        ax.legend()

    for ax in axes[total_assets:]:
        ax.set_visible(False)

    plt.suptitle(
        "Crypto vs Equity Dynamic Correlations",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# =========================================================
# 5. GOLD SAFE HAVEN PLOT
# =========================================================

def plot_gold_safe_haven(batch_size: int = 5):
    """
    Plot Gold correlations against all equities in batches.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    df = _load_dcc_data()

    for i in range(0, len(GLOBAL_STOCKS), batch_size):
        batch = GLOBAL_STOCKS[i:i + batch_size]

        fig, ax = plt.subplots(figsize=(14, 6))

        for asset in batch:
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

        _highlight_crashes(ax)

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