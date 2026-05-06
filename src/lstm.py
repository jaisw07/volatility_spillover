import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import ta

DATASET_DIR = "dataset"
INSIGHTS_DIR = "insights"

def feature_engineering():
    print("--- 1. Feature Engineering ---")
    
    # Load unified dataset
    unified_path = os.path.join(INSIGHTS_DIR, "unified.csv")
    if not os.path.exists(unified_path):
        raise FileNotFoundError(f"Run data pipeline first. Missing: {unified_path}")
        
    df_uni = pd.read_csv(unified_path)
    df_uni["date"] = pd.to_datetime(df_uni["date"])
    df_uni = df_uni.sort_values("date").reset_index(drop=True)
    
    # We need BTC and ETH price data for RSI and MACD
    btc_path = os.path.join(DATASET_DIR, "BTC-USD.csv")
    eth_path = os.path.join(DATASET_DIR, "ETH-USD.csv")
    
    df_btc = pd.read_csv(btc_path)
    df_eth = pd.read_csv(eth_path)
    df_btc["date"] = pd.to_datetime(df_btc["date"])
    df_eth["date"] = pd.to_datetime(df_eth["date"])
    
    adj_col_btc = [c for c in df_btc.columns if "adj_close" in c][0]
    adj_col_eth = [c for c in df_eth.columns if "adj_close" in c][0]
    
    # Calculate Tech Indicators for BTC
    df_btc['btc_rsi'] = ta.momentum.RSIIndicator(df_btc[adj_col_btc], window=14).rsi()
    macd_btc = ta.trend.MACD(df_btc[adj_col_btc])
    df_btc['btc_macd_signal'] = macd_btc.macd_signal()
    
    # Realized Volatility variations (5d, 10d, 20d)
    df_btc["log_ret"] = np.log(df_btc[adj_col_btc] / df_btc[adj_col_btc].shift(1))
    df_btc["btc_rv_5d"] = df_btc["log_ret"].rolling(5).std() * np.sqrt(252)
    df_btc["btc_rv_10d"] = df_btc["log_ret"].rolling(10).std() * np.sqrt(252)
    df_btc["btc_rv_20d"] = df_btc["log_ret"].rolling(20).std() * np.sqrt(252)
    
    # Merge BTC tech indicators to unified
    btc_feats = df_btc[["date", "btc_rsi", "btc_macd_signal", "btc_rv_5d", "btc_rv_10d", "btc_rv_20d"]]
    df = df_uni.merge(btc_feats, on="date", how="left")
    
    # Calculate Tech Indicators for ETH
    df_eth['eth_rsi'] = ta.momentum.RSIIndicator(df_eth[adj_col_eth], window=14).rsi()
    macd_eth = ta.trend.MACD(df_eth[adj_col_eth])
    df_eth['eth_macd_signal'] = macd_eth.macd_signal()
    
    df_eth["log_ret"] = np.log(df_eth[adj_col_eth] / df_eth[adj_col_eth].shift(1))
    df_eth["eth_rv_5d"] = df_eth["log_ret"].rolling(5).std() * np.sqrt(252)
    df_eth["eth_rv_10d"] = df_eth["log_ret"].rolling(10).std() * np.sqrt(252)
    df_eth["eth_rv_20d"] = df_eth["log_ret"].rolling(20).std() * np.sqrt(252)
    
    eth_feats = df_eth[["date", "eth_rsi", "eth_macd_signal", "eth_rv_5d", "eth_rv_10d", "eth_rv_20d"]]
    df = df.merge(eth_feats, on="date", how="left")
    
    # Load DCC correlations
    dcc_path = os.path.join(INSIGHTS_DIR, "dcc_correlations.csv")
    if os.path.exists(dcc_path):
        df_dcc = pd.read_csv(dcc_path)
        df_dcc["date"] = pd.to_datetime(df_dcc["date"])
        # Take BTC-GSPC (S&P 500) and average system correlation (TCI proxy)
        dcc_cols = ["date", "avg_system_corr"]
        pair_col = [c for c in df_dcc.columns if ("BTC-USD" in c and "^GSPC" in c)][0]
        dcc_cols.append(pair_col)
        df = df.merge(df_dcc[dcc_cols], on="date", how="left")
        df.rename(columns={pair_col: "dcc_btc_gspc"}, inplace=True)
    else:
        print("Warning: DCC correlations missing.")
        
    # Forward fill missing values
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    
    # --- Define Target Variable ---
    # Binary crash indicator: Y = 1 if BTC realized volatility in next 21 trading days exceeds 90th percentile
    # First calculate the rolling 21-day forward max volatility
    df["future_max_rv_21d"] = df["btc_rv_5d"].shift(-21).rolling(21).max()
    threshold = df["btc_rv_5d"].quantile(0.90)
    
    df["target"] = (df["future_max_rv_21d"] > threshold).astype(int)
    
    # Drop rows at the end where future is unknown
    df.dropna(subset=["target"], inplace=True)
    
    print(f"Dataset shape after feature engineering: {df.shape}")
    print(f"Target class distribution:\n{df['target'].value_counts(normalize=True)}")
    
    return df

def create_sequences(features, targets, seq_length=60):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i : i + seq_length])
        y.append(targets[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def train_and_evaluate():
    df = feature_engineering()
    
    # Features to use
    drop_cols = ["date", "future_max_rv_21d", "target", "crash_btc", "crash_eth", "crash_combined"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Ensure chronological split
    dates = df["date"].values
    X_raw = df[feature_cols].values
    y_raw = df["target"].values
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Sequences (60 trading days)
    seq_length = 60
    X_seq, y_seq = create_sequences(X_scaled, y_raw, seq_length)
    
    # Split 70% Train, 15% Val, 15% Test
    n = len(X_seq)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    # Class weights for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    print(f"Class Weights: {class_weights}")
    
    # Train LSTM
    print("\n--- 2. Training LSTM Model ---")
    model = build_lstm_model((seq_length, len(feature_cols)))
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate LSTM
    print("\n--- 3. Evaluating LSTM Model ---")
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    lstm_auc = roc_auc_score(y_test, y_pred_prob)
    lstm_prec = precision_score(y_test, y_pred)
    lstm_rec = recall_score(y_test, y_pred)
    lstm_f1 = f1_score(y_test, y_pred)
    
    print("\nLSTM Evaluation:")
    print(f"AUC-ROC:   {lstm_auc:.4f}")
    print(f"Precision: {lstm_prec:.4f}")
    print(f"Recall:    {lstm_rec:.4f}")
    print(f"F1-Score:  {lstm_f1:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Baseline Logistic Regression (on non-sequential scaled features)
    print("\n--- 4. Baseline: Logistic Regression ---")
    # Need to match the exact test indices to compare properly
    X_train_lr = X_scaled[seq_length:train_end+seq_length]
    y_train_lr = y_raw[seq_length:train_end+seq_length]
    X_test_lr = X_scaled[val_end+seq_length:]
    y_test_lr = y_raw[val_end+seq_length:]
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr.fit(X_train_lr, y_train_lr)
    lr_pred_prob = lr.predict_proba(X_test_lr)[:, 1]
    lr_pred = lr.predict(X_test_lr)
    
    print("LR Baseline Evaluation:")
    print(f"AUC-ROC:   {roc_auc_score(y_test_lr, lr_pred_prob):.4f}")
    print(f"Precision: {precision_score(y_test_lr, lr_pred):.4f}")
    print(f"Recall:    {recall_score(y_test_lr, lr_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test_lr, lr_pred):.4f}")
    
    # Save Confusion Matrix plot
    os.makedirs(INSIGHTS_DIR, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('LSTM Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(INSIGHTS_DIR, "lstm_confusion_matrix.png"))
    plt.close()
    
    print(f"\nSaved confusion matrix to {INSIGHTS_DIR}/lstm_confusion_matrix.png")
    
    # Save Model
    model.save(os.path.join(INSIGHTS_DIR, "crash_prediction_lstm.h5"))

if __name__ == "__main__":
    train_and_evaluate()
