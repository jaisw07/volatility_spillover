# src/lstm.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

INSIGHTS_DIR = "insights"

# =========================================================
# CRASH WINDOWS
# =========================================================

CRASH_WINDOWS = {
    "COVID Crash": ("2020-03-01", "2020-04-15"),
    "LUNA Collapse": ("2022-05-01", "2022-06-30"),
    "FTX Collapse": ("2022-11-01", "2022-12-15")
}


# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    unified = pd.read_csv(
        os.path.join(
            INSIGHTS_DIR,
            "unified.csv"
        )
    )

    dcc = pd.read_csv(
        os.path.join(
            INSIGHTS_DIR,
            "dcc_correlations.csv"
        )
    )

    tci = pd.read_csv(
        os.path.join(
            INSIGHTS_DIR,
            "rolling_tci.csv"
        )
    )

    unified["date"] = pd.to_datetime(unified["date"])
    dcc["date"] = pd.to_datetime(dcc["date"])
    tci["date"] = pd.to_datetime(tci["date"])

    dcc = dcc.rename(
        columns={
            c: f"dcc_{c}"
            for c in dcc.columns
            if c != "date"
        }
    )

    dcc_cols = [
        "date",
        "dcc_BTC-USD__ETH-USD",
        "dcc_BTC-USD__^GSPC",
        "dcc_BTC-USD__GC=F",
        "dcc_ETH-USD__^NSEI"
    ]

    df = unified.merge(
        dcc[dcc_cols],
        on="date",
        how="inner"
    )

    df = df.merge(
        tci,
        on="date",
        how="inner"
    )

    return df


# =========================================================
# LABELS
# =========================================================

def create_crash_labels(df):

    df["crash_label"] = 0

    for _, (start, end) in CRASH_WINDOWS.items():

        mask = (
            (df["date"] >= pd.to_datetime(start))
            &
            (df["date"] <= pd.to_datetime(end))
        )

        df.loc[mask, "crash_label"] = 1

    return df


# =========================================================
# BUILD DATASET
# =========================================================

def build_lstm_dataset():

    df = load_dataset()

    df = create_crash_labels(df)

    feature_cols = [

        # Returns
        "ret_BTC-USD",
        "ret_ETH-USD",
        "ret_^GSPC",
        "ret_^NSEI",

        # Volatility
        "vol_BTC-USD",
        "vol_ETH-USD",

        # DCC
        "dcc_BTC-USD__ETH-USD",
        "dcc_BTC-USD__^GSPC",
        "dcc_BTC-USD__GC=F",
        "dcc_ETH-USD__^NSEI",

        # Connectedness
        "tci"
    ]

    df = df[
        ["date"]
        + feature_cols
        + ["crash_label"]
    ]

    df = df.dropna().reset_index(drop=True)

    print("\n=== DATASET ===")
    print(df.shape)

    print("\n=== LABEL DISTRIBUTION ===")
    print(df["crash_label"].value_counts())

    return df


# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(
    X,
    y,
    lookback=10
):

    X_seq = []
    y_seq = []

    for i in range(lookback, len(X)):

        X_seq.append(
            X[i-lookback:i]
        )

        y_seq.append(
            y[i]
        )

    return (
        np.array(X_seq),
        np.array(y_seq)
    )


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_lstm_data(
    lookback=10
):

    df = build_lstm_dataset()

    feature_cols = [
        c for c in df.columns
        if c not in ["date", "crash_label"]
    ]

    y = df["crash_label"].values
    dates = df["date"].values

    # =====================================================
    # NEW SPLITS
    # =====================================================

    # TRAIN:
    # Includes COVID + LUNA

    train_rows = (
        (dates >= np.datetime64("2022-02-28"))
        &
        (dates <= np.datetime64("2022-06-30"))
    )

    # VALIDATION:
    # Includes FTX

    val_rows = (
        (dates >= np.datetime64("2020-01-15"))
        &
        (dates <= np.datetime64("2020-04-15"))
    )

    # TEST:
    # Fully unseen later regime

    ttest_rows = (
        (dates >= np.datetime64("2022-09-15"))
        &
        (dates <= np.datetime64("2022-12-15"))
    )

    # =====================================================
    # SCALER
    # =====================================================

    scaler = StandardScaler()

    scaler.fit(
        df.loc[train_rows, feature_cols]
    )

    X_scaled = scaler.transform(
        df[feature_cols]
    )

    # =====================================================
    # SEQUENCES
    # =====================================================

    X_seq, y_seq = create_sequences(
        X_scaled,
        y,
        lookback=lookback
    )

    seq_dates = dates[lookback:]

    train_mask = (
        (seq_dates >= np.datetime64("2022-02-28"))
        &
        (seq_dates <= np.datetime64("2022-06-30"))
    )

    val_mask = (
        (seq_dates >= np.datetime64("2020-01-15"))
        &
        (seq_dates <= np.datetime64("2020-04-15"))
    )

    test_mask = (
        (seq_dates >= np.datetime64("2022-09-15"))
        &
        (seq_dates <= np.datetime64("2022-12-15"))
    )

    X_train = X_seq[train_mask]
    y_train = y_seq[train_mask]

    X_val = X_seq[val_mask]
    y_val = y_seq[val_mask]

    X_test = X_seq[test_mask]
    y_test = y_seq[test_mask]

    print("\n=== DATA DISTRIBUTION ===")

    print(
        f"Train crashes: {y_train.sum()} / {len(y_train)}"
    )

    print(
        f"Validation crashes: {y_val.sum()} / {len(y_val)}"
    )

    print(
        f"Test crashes: {y_test.sum()} / {len(y_test)}"
    )

    print("\n=== UNIQUE LABELS ===")

    print("Train:", np.unique(y_train))
    print("Validation:", np.unique(y_val))
    print("Test:", np.unique(y_test))

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

# =========================================================
# MODEL
# =========================================================

def build_lstm_model(
    input_shape
):

    model = Sequential()

    model.add(
        LSTM(
            32,
            input_shape=input_shape
        )
    )

    model.add(
        Dropout(0.2)
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =========================================================
# TRAIN
# =========================================================

def train_lstm(
    lookback=10,
    epochs=30,
    batch_size=16,
    best_model_path=None
):

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = prepare_lstm_data(
        lookback=lookback
    )

    model = build_lstm_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    if best_model_path is None:
        best_model_path = os.path.join(
            INSIGHTS_DIR,
            "lstm_best_model.keras"
        )

    os.makedirs(
        os.path.dirname(best_model_path),
        exist_ok=True
    )

    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    history = model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=epochs,

        batch_size=batch_size,

        callbacks=[early_stop, checkpoint],

        verbose=1
    )

    if os.path.exists(best_model_path):
        model = load_model(best_model_path)

    return (
        model,
        history,
        X_test,
        y_test
    )

# =========================================================
# EVALUATE
# =========================================================

def evaluate_lstm(
    model,
    X_test,
    y_test,
    threshold=0.5
):

    y_prob = model.predict(
        X_test
    ).flatten()

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    metrics_df = pd.DataFrame([{

        "auc_roc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1

    }])

    print("\n=== LSTM EVALUATION ===")
    print(metrics_df)

    print("\n=== CLASSIFICATION REPORT ===")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Normal", "Crash"],
            zero_division=0
        )
    )

    return (
        metrics_df,
        cm,
        y_prob,
        y_pred
    )


# =========================================================
# PLOTS
# =========================================================

def plot_training_history(history):

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Train Loss"
    )

    if "val_loss" in history.history:
        plt.plot(
            history.history["val_loss"],
            label="Val Loss"
        )

    plt.title(
        "LSTM Training and Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(cm):

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Crash"],
        yticklabels=["Normal", "Crash"]
    )

    plt.title(
        "LSTM Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()