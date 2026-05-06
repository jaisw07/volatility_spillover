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