import os
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

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