"""
train_lstm.py -- CryptoPulse Phase 3
Trains a 2-layer stacked LSTM to predict next-day BTC RETURN (pct change).
TARGET = Return (not Close) to fix the lazy predictor / 50% direction problem.
"""

import os, sys, pickle
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_data, chronological_split

FEATURE_COLS  = ["Return","Volume","RSI","EMA20","EMA50","BB_upper","BB_lower","MACD"]
TARGET_COL    = "Return"
LOOKBACK      = 60
TRAIN_RATIO   = 0.80
EPOCHS        = 100
BATCH_SIZE    = 32
PATIENCE      = 15
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.20
LEARNING_RATE = 0.001
MODEL_DIR     = "models"
MODEL_PATH    = os.path.join(MODEL_DIR, "lstm_model.h5")
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
HISTORY_PATH  = os.path.join(MODEL_DIR, "train_history.pkl")
METRICS_PATH  = os.path.join(MODEL_DIR, "metrics.pkl")


def fit_and_save_scaler(train_df):
    """
    Fit MinMaxScaler on training data only.
    WHY: Fitting on full data leaks future price statistics into training
    (data leakage), inflating all metrics artificially.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[FEATURE_COLS])
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[train_lstm] Scaler fitted on {len(train_df)} rows -> '{SCALER_PATH}'")
    return scaler


def apply_scaler(df, scaler):
    """Apply pre-fitted scaler without re-fitting (prevents leakage)."""
    out = df.copy()
    out[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return out


def build_sequences(df):
    """
    Convert flat DataFrame into (X, y) LSTM input pairs.
    X shape : (n_samples, 60, 8)  -- 60-day window, 8 features
    y shape : (n_samples,)        -- next-day Return value
    """
    data   = df[FEATURE_COLS].values
    target = df[TARGET_COL].values
    X, y   = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[i - LOOKBACK : i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_model(n_features):
    """
    2-layer stacked LSTM architecture.
    WHY return_sequences=True on layer 1: passes full sequence to layer 2.
    WHY Dropout(0.2): prevents overfitting by zeroing 20% of neurons each batch.
    WHY Dense(1) no activation: regression output, Return can be +/-.
    """
    model = Sequential([
        Input(shape=(LOOKBACK, n_features)),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mean_squared_error",
        metrics=["mae"],
    )
    return model


def train_model(model, X_tr, y_tr, X_val, y_val):
    """
    Train with EarlyStopping (patience=15) and ModelCheckpoint.
    WHY separate val set from test: prevents EarlyStopping from
    indirectly tuning to test data, which biases final metrics.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    print(f"\n[train_lstm] Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Shape: {X_tr.shape}\n")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)
    best = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\n[train_lstm] Best epoch: {best}/{len(history.history['loss'])}")
    print(f"[train_lstm] Model saved -> '{MODEL_PATH}'")
    return history


def inverse_returns(pred_sc, actual_sc, scaler):
    """
    Undo MinMaxScaler on Return column (index 0).
    Uses dummy array pattern because scaler was fitted on all 8 features.
    """
    n = scaler.n_features_in_
    def _inv(arr):
        d = np.zeros((len(arr), n), dtype=np.float32)
        d[:, 0] = arr.flatten()
        return scaler.inverse_transform(d)[:, 0]
    return _inv(pred_sc), _inv(actual_sc)


def returns_to_prices(returns, start_price):
    """
    Reconstruct price series from returns.
    price_t = price_{t-1} * (1 + return_t)
    Used to compute dollar-level MAE/RMSE for the report.
    """
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return np.array(prices[1:], dtype=np.float32)


def compute_metrics(actuals, predictions):
    """
    MAE, RMSE, MAPE, Directional Accuracy.
    Directional Accuracy: % of days where sign(pred) == sign(actual).
    50% = coin flip. Above 55% = model has predictive value.
    """
    mae  = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    nz   = actuals != 0
    mape = np.mean(np.abs((actuals[nz] - predictions[nz]) / actuals[nz])) * 100
    dir_acc = np.mean(np.sign(actuals) == np.sign(predictions)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Directional Accuracy": dir_acc}


def print_metrics(m, label):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    if m["MAE"] > 100:
        print(f"  MAE              : ${m['MAE']:>12,.2f}")
        print(f"  RMSE             : ${m['RMSE']:>12,.2f}")
    else:
        print(f"  MAE  (return)    :  {m['MAE']:>12.6f}")
        print(f"  RMSE (return)    :  {m['RMSE']:>12.6f}")
    print(f"  MAPE             :  {m['MAPE']:>11.4f}%")
    print(f"  Directional Acc. :  {m['Directional Accuracy']:>10.2f}%")
    print(f"{'─'*50}")


def main(csv_path="data/BTC-USD.csv"):

    print("=" * 60)
    print("CryptoPulse -- train_lstm.py  (TARGET = Return)")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = load_data(csv_path)

    # Step 2: Split
    print("\n[2/6] Chronological 80/20 split...")
    train_df, test_df = chronological_split(df, train_ratio=TRAIN_RATIO)

    # Step 3: Scale (fit on train only)
    print("\n[3/6] Fitting scaler on training data only...")
    scaler   = fit_and_save_scaler(train_df)
    train_sc = apply_scaler(train_df, scaler)
    test_sc  = apply_scaler(test_df,  scaler)

    # Step 4: Build sequences
    print("\n[4/6] Building sequences (lookback=60)...")
    X_train, y_train = build_sequences(train_sc)
    X_test,  y_test  = build_sequences(test_sc)

    # Carve 10% of train for validation (NOT from test set)
    val_n    = int(len(X_train) * 0.10)
    X_val    = X_train[-val_n:]
    y_val    = y_train[-val_n:]
    X_train_ = X_train[:-val_n]
    y_train_ = y_train[:-val_n]
    print(f"  Train: {X_train_.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # Step 5: Build and train
    print("\n[5/6] Building and training model...")
    model = build_model(n_features=len(FEATURE_COLS))
    model.summary()
    train_model(model, X_train_, y_train_, X_val, y_val)

    # Step 6: Evaluate
    print("\n[6/6] Evaluating on held-out test set...")
    y_pred_sc          = model.predict(X_test, verbose=0).flatten()
    y_pred_ret, y_test_ret = inverse_returns(y_pred_sc, y_test, scaler)

    # Return-level metrics
    ret_metrics = compute_metrics(y_test_ret, y_pred_ret)
    print_metrics(ret_metrics, "Return-level metrics")

    # Reconstruct prices for dollar-level metrics
    start_price  = float(test_df["Close"].iloc[LOOKBACK])
    y_pred_price = returns_to_prices(y_pred_ret, start_price)
    y_test_price = returns_to_prices(y_test_ret, start_price)
    price_metrics = compute_metrics(y_test_price, y_pred_price)
    print_metrics(price_metrics, "Price-level metrics (USD)")

    # Naive baseline: predict return = 0 every day
    naive_metrics = compute_metrics(y_test_ret, np.zeros_like(y_test_ret))
    print_metrics(naive_metrics, "Naive baseline (predict return = 0)")

    # Summary
    lift = ret_metrics["Directional Accuracy"] - naive_metrics["Directional Accuracy"]
    print(f"\n  Directional Acc. lift vs naive: {lift:>+.2f}%")
    print("  Positive = LSTM beats naive baseline")

    # Save everything for dashboard
    with open(METRICS_PATH, "wb") as f:
        pickle.dump({
            "return_metrics": ret_metrics,
            "price_metrics":  price_metrics,
            "y_test_ret":     y_test_ret,
            "y_pred_ret":     y_pred_ret,
            "y_test_price":   y_test_price,
            "y_pred_price":   y_pred_price,
            "test_dates":     test_df["Date"].iloc[LOOKBACK:].values,
        }, f)
    print(f"\n[train_lstm] Metrics saved -> '{METRICS_PATH}'")

    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Model  -> {MODEL_PATH}")
    print(f"  Scaler -> {SCALER_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/BTC-USD.csv"
    main(csv)
