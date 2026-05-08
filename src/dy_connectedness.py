import os
import numpy as np
import pandas as pd

from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import seaborn as sns

INSIGHTS_DIR = "insights"

CRASH_WINDOWS = {
    "COVID Crash": ("2020-03-01", "2020-04-15"),
    "LUNA Collapse": ("2022-05-01", "2022-06-30"),
    "FTX Collapse": ("2022-11-01", "2022-12-15")
}

def load_residuals():

    path = os.path.join(
        INSIGHTS_DIR,
        "garch_residuals.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    resid_cols = [
        c for c in df.columns
        if c.startswith("resid_")
    ]

    return df, resid_cols


def run_full_sample_dy(
    lags=1,
    horizon=10,
    save=True
):

    df, resid_cols = load_residuals()

    resid_data = (
        df[resid_cols]
        .dropna()
    )

    fevd_df, conn_table, tci = (
        compute_dy_connectedness(
            resid_data,
            lags,
            horizon
        )
    )

    print("\n=== TOTAL CONNECTEDNESS INDEX ===")
    print(f"TCI: {tci:.2f}%")

    print("\n=== CONNECTEDNESS TABLE ===")
    print(conn_table)

    if save:

        fevd_df.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "dy_fevd.csv"
            )
        )

        conn_table.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "dy_connectedness.csv"
            )
        )

    return (
        fevd_df,
        conn_table,
        tci
    )


def compute_dy_connectedness(
    data,
    lags=1,
    horizon=10
):
    """
    Compute Diebold-Yilmaz connectedness.

    Parameters
    ----------
    data : DataFrame
        standardized residuals

    lags : int
        VAR lag order

    horizon : int
        FEVD horizon

    Returns
    -------
    fevd_df
    connectedness_table
    tci
    """

    model = VAR(data)

    results = model.fit(lags)
    
    if not results.is_stable():
        raise ValueError(
            "VAR system unstable."
        )

    fevd = results.fevd(horizon)

    fevd_matrix = fevd.decomp[:, horizon - 1, :]

    assets = data.columns.tolist()

    fevd_df = pd.DataFrame(
        fevd_matrix,
        index=assets,
        columns=assets
    )

    # normalize rows
    fevd_df = fevd_df.div(
        fevd_df.sum(axis=1),
        axis=0
    )

    # total connectedness index
    off_diag_sum = (
        fevd_df.values.sum()
        - np.trace(fevd_df.values)
    )

    tci = (
        off_diag_sum
        / fevd_df.values.sum()
    ) * 100

    # directional spillovers
    to_others = (
        fevd_df.sum(axis=0)
        - np.diag(fevd_df)
    ) * 100

    from_others = (
        fevd_df.sum(axis=1)
        - np.diag(fevd_df)
    ) * 100

    net_spillover = (
        to_others - from_others
    )

    connectedness_table = pd.DataFrame({
        "TO": to_others,
        "FROM": from_others,
        "NET": net_spillover
    })

    return (
        fevd_df,
        connectedness_table,
        tci
    )

def run_regime_dy(
    lags=1,
    horizon=10
):

    df, resid_cols = load_residuals()

    summary_rows = []

    for regime, (start, end) in CRASH_WINDOWS.items():

        mask = (
            (df["date"] >= pd.to_datetime(start))
            &
            (df["date"] <= pd.to_datetime(end))
        )

        subset = (
            df.loc[mask, resid_cols]
            .dropna()
        )

        _, conn_table, tci = (
            compute_dy_connectedness(
                subset,
                lags,
                horizon
            )
        )

        row = {
            "regime": regime,
            "tci": tci
        }

        summary_rows.append(row)

        print(f"\n=== {regime} ===")
        print(f"TCI: {tci:.2f}%")

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        os.path.join(
            INSIGHTS_DIR,
            "dy_regime_summary.csv"
        ),
        index=False
    )

    return summary_df

def rolling_tci(
    window=60,
    lags=1,
    horizon=10,
    save=True
):

    df, resid_cols = load_residuals()

    rows = []

    for i in range(window, len(df)):

        subset = (
            df.iloc[i-window:i]
            [resid_cols]
            .dropna()
        )

        try:

            _, _, tci = (
                compute_dy_connectedness(
                    subset,
                    lags,
                    horizon
                )
            )

            rows.append({
                "date": df.iloc[i - window // 2]["date"],
                "tci": tci
            })

        except:
            continue

    tci_df = pd.DataFrame(rows)

    if save:

        tci_df.to_csv(
            os.path.join(
                INSIGHTS_DIR,
                "rolling_tci.csv"
            ),
            index=False
        )

    return tci_df

def plot_rolling_tci():

    path = os.path.join(
        INSIGHTS_DIR,
        "rolling_tci.csv"
    )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        df["date"],
        df["tci"],
        linewidth=2
    )

    for label, (start, end) in CRASH_WINDOWS.items():

        ax.axvspan(
            pd.to_datetime(start),
            pd.to_datetime(end),
            color="red",
            alpha=0.20
        )

    ax.set_title(
        "Rolling Total Connectedness Index"
    )

    ax.set_ylabel("TCI (%)")

    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_connectedness_heatmap():

    path = os.path.join(
        INSIGHTS_DIR,
        "dy_fevd.csv"
    )

    df = pd.read_csv(
        path,
        index_col=0
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        df,
        cmap="Reds",
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Diebold-Yilmaz Spillover Matrix"
    )

    plt.tight_layout()
    plt.show()