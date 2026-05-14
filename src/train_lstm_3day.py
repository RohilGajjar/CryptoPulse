"""
train_lstm_3day.py -- CryptoPulse Phase 4 Experiment (FIXED)
=============================================================
Trains a separate LSTM predicting 3-day forward return.
Does NOT overwrite any existing 1-day model files.

BUG FIXED:
    The original code passed RAW 3-day return values into
    inverse_transform(), which treated them as SCALED values.
    Since the scaler's data_min[0] (1-day Return) is negative,
    every value got shifted to near data_min, making ALL signs
    match -> 100% directional accuracy (completely spurious).

    FIX: The 3-day target is never passed through the scaler.
    It is kept in raw return space throughout. The model predicts
    scaled 1-day Return (column 0) as a proxy signal, then we
    compute directional accuracy directly on raw 3-day returns.

    BETTER FIX (implemented here): Scale the 3-day target with its
    OWN separate scaler fitted on training targets only, so the
    model learns in a consistent [0,1] space, and we can cleanly
    inverse-transform predictions back to raw 3-day returns.

Outputs (separate from 1-day model):
    models/lstm_model_3day.h5
    models/scaler_3day.pkl        -- feature scaler
    models/target_scaler_3day.pkl -- 3-day target scaler (separate)
    models/metrics_3day.pkl
    models/train_history_3day.pkl

Usage:
    python src/train_lstm_3day.py
    python src/train_lstm_3day.py data/BTC-USD.csv
"""

import os
import sys
import pickle
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


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Same 8 features as 1-day model -- Return_3d is NOT here (prevents leakage)
FEATURE_COLS = [
    "Return",    # 1-day return (feature only, not the target)
    "Volume",
    "RSI",
    "EMA20",
    "EMA50",
    "BB_upper",
    "BB_lower",
    "MACD",
]

TARGET_COL    = "Return_3d"   # 3-day forward return -- separate from features
HORIZON       = 3             # prediction horizon in trading days
LOOKBACK      = 60
TRAIN_RATIO   = 0.80
EPOCHS        = 100
BATCH_SIZE    = 32
PATIENCE      = 15
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.20
LEARNING_RATE = 0.001

# Separate output paths -- 1-day model files are NEVER touched
MODEL_DIR          = "models"
MODEL_PATH         = os.path.join(MODEL_DIR, "lstm_model_3day.h5")
SCALER_PATH        = os.path.join(MODEL_DIR, "scaler_3day.pkl")
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, "target_scaler_3day.pkl")
HISTORY_PATH       = os.path.join(MODEL_DIR, "train_history_3day.pkl")
METRICS_PATH       = os.path.join(MODEL_DIR, "metrics_3day.pkl")
METRICS_1DAY       = os.path.join(MODEL_DIR, "metrics.pkl")

# Leakage warning threshold
SUSPICION_THRESHOLD = 70.0   # % -- above this, warn about leakage


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Compute 3-day forward return
#
# Return_3d[t] = (Close[t+3] - Close[t]) / Close[t]
#
# LEAKAGE PREVENTION:
#   - Return_3d is computed BEFORE the train/test split so it's available
#     for both sets -- but it only uses future Close values relative to t.
#   - Return_3d is NOT added to FEATURE_COLS (features at time t must not
#     include any information from t+1 onward).
#   - The last 3 rows have NaN targets and are dropped.
# ─────────────────────────────────────────────────────────────────────────────

def add_3day_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Return_3d column. Drop the last HORIZON rows (NaN targets).
    Return_3d is NOT in FEATURE_COLS -- it is the label only.
    """
    df = df.copy()
    df[TARGET_COL] = (df["Close"].shift(-HORIZON) - df["Close"]) / df["Close"]
    rows_before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    print(f"[3day] Dropped {rows_before - len(df)} rows (NaN 3-day targets)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Scalers
#
# FIX: Two separate scalers:
#   1. Feature scaler  -- fitted on FEATURE_COLS training data (same as 1-day)
#   2. Target scaler   -- fitted on TARGET_COL training data only
#
# WHY separate target scaler:
#   The original bug mixed the feature scaler (fitted on 1-day Return range)
#   with the 3-day target. Since 1-day and 3-day returns have different ranges,
#   this caused the inverse_transform to shift all predictions to near
#   data_min[0], making all signs identical -> 100% directional accuracy.
#   A dedicated target scaler maps 3-day returns cleanly to [0,1] and back.
# ─────────────────────────────────────────────────────────────────────────────

def fit_scalers(train_df: pd.DataFrame):
    """
    Fit feature scaler and target scaler on training data only.
    Saves both to disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Feature scaler
    feat_scaler = MinMaxScaler(feature_range=(0, 1))
    feat_scaler.fit(train_df[FEATURE_COLS])
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(feat_scaler, f)

    # Target scaler -- fitted only on 3-day returns from training set
    tgt_scaler = MinMaxScaler(feature_range=(0, 1))
    tgt_scaler.fit(train_df[[TARGET_COL]])
    with open(TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(tgt_scaler, f)

    print(f"[3day] Feature scaler fitted on {len(train_df)} rows -> '{SCALER_PATH}'")
    print(f"[3day] Target  scaler fitted on {len(train_df)} rows -> '{TARGET_SCALER_PATH}'")
    print(f"[3day] 3-day return range (train): "
          f"[{float(train_df[TARGET_COL].min()):.4f}, "
          f"{float(train_df[TARGET_COL].max()):.4f}]")
    return feat_scaler, tgt_scaler


def scale_df(df: pd.DataFrame, feat_scaler, tgt_scaler) -> pd.DataFrame:
    """Apply pre-fitted scalers. Never re-fit on test data."""
    out = df.copy()
    out[FEATURE_COLS] = feat_scaler.transform(df[FEATURE_COLS])
    out[TARGET_COL]   = tgt_scaler.transform(df[[TARGET_COL]])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Build sequences
#
# ALIGNMENT (verified):
#   X[i] = feature rows [i-LOOKBACK .. i-1]  -- all at or before time i
#   y[i] = Return_3d at time i               -- return from i to i+3
#
#   There is NO future leakage because:
#   - FEATURE_COLS does not contain Return_3d
#   - X window ends at i-1 (not i)
#   - y[i] = label for what happens AFTER time i
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame) -> tuple:
    """
    Build (X, y) pairs.
    X shape : (n_samples, LOOKBACK, n_features)
    y shape : (n_samples,)  -- scaled 3-day return
    """
    features = df[FEATURE_COLS].values   # shape (rows, 8)
    targets  = df[TARGET_COL].values     # shape (rows,) -- scaled 3-day return

    X, y = [], []
    for i in range(LOOKBACK, len(features)):
        X.append(features[i - LOOKBACK : i])   # past 60 rows of features
        y.append(targets[i])                    # 3-day return label at day i

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Model architecture (identical to 1-day model)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(n_features: int) -> tf.keras.Model:
    """Same 2-layer LSTM as 1-day model -- architecture unchanged."""
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- Train
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_tr, y_tr, X_val, y_val):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]
    print(f"\n[3day] Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  "
          f"Shape: {X_tr.shape}\n")
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=callbacks, verbose=1)
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)
    best = int(np.argmin(history.history["val_loss"])) + 1
    print(f"\n[3day] Best epoch: {best}/{len(history.history['loss'])}")
    print(f"[3day] Model saved -> '{MODEL_PATH}'")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 -- Evaluate
#
# FIX: Use the dedicated TARGET scaler (not the feature scaler) to
# inverse-transform predictions and actuals back to raw 3-day return space.
# Directional accuracy is then computed on raw return values.
# ─────────────────────────────────────────────────────────────────────────────

def inverse_transform_target(arr_scaled: np.ndarray, tgt_scaler) -> np.ndarray:
    """
    Inverse-transform scaled 3-day returns using the DEDICATED target scaler.

    FIX: This replaces the buggy inverse_transform_3day() which used
    the feature scaler (fitted on 1-day Return) and produced uniformly
    negative values -> 100% spurious directional accuracy.
    """
    return tgt_scaler.inverse_transform(
        arr_scaled.reshape(-1, 1)
    ).flatten()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, Directional Accuracy on raw return arrays."""
    mae     = float(mean_absolute_error(y_true, y_pred))
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    return {"MAE": mae, "RMSE": rmse, "Directional Accuracy": dir_acc}


def print_debug_checks(y_true: np.ndarray, y_pred: np.ndarray,
                       test_dates, dir_acc: float) -> None:
    """
    Print diagnostic values to verify there is no leakage.
    If directional accuracy > SUSPICION_THRESHOLD, print a warning.
    """
    print("\n  ── DEBUG CHECKS ──────────────────────────────────────")
    print(f"  Test samples          : {len(y_true)}")
    if test_dates is not None:
        print(f"  Test date range       : {pd.Timestamp(test_dates[0]).date()} "
              f"-> {pd.Timestamp(test_dates[-1]).date()}")
    print(f"  y_true first 5       : {np.round(y_true[:5], 5).tolist()}")
    print(f"  y_pred first 5       : {np.round(y_pred[:5], 5).tolist()}")
    print(f"  y_true % positive    : {np.mean(y_true > 0)*100:.1f}%")
    print(f"  y_pred % positive    : {np.mean(y_pred > 0)*100:.1f}%")
    print(f"  y_true mean          : {np.mean(y_true):+.5f}")
    print(f"  y_pred mean          : {np.mean(y_pred):+.5f}")
    print(f"  Directional Accuracy : {dir_acc:.2f}%")

    if dir_acc > SUSPICION_THRESHOLD:
        print(f"\n  !! WARNING: {dir_acc:.1f}% directional accuracy is suspiciously")
        print(f"  !! high. Verify for data leakage before reporting.")
        print(f"  !! Check: Return_3d not in FEATURE_COLS, target scaler")
        print(f"  !! separate from feature scaler, no future data in X.")
    else:
        print(f"\n  Accuracy {dir_acc:.1f}% is in a plausible range.")
    print("  ─────────────────────────────────────────────────────")


def print_comparison_table(metrics_3d: dict) -> None:
    """Compare 1-day vs 3-day model results."""
    print("\n" + "=" * 62)
    print("  Horizon Comparison: 1-Day vs 3-Day LSTM")
    print("=" * 62)
    print(f"  {'Model':<28} {'Dir.Acc':>10} {'MAE':>10} {'RMSE':>10}")
    print(f"  {'─'*58}")

    if os.path.exists(METRICS_1DAY):
        with open(METRICS_1DAY, "rb") as f:
            m1 = pickle.load(f)
        rm1 = m1.get("return_metrics", {})
        d1  = rm1.get("Directional Accuracy", None)
        mae1 = rm1.get("MAE", None)
        rmse1= rm1.get("RMSE", None)
        if d1 is not None:
            print(f"  {'1-day LSTM (Phase 3)':<28} "
                  f"{d1:>9.2f}% {mae1:>10.4f} {rmse1:>10.4f}")
        else:
            print(f"  {'1-day LSTM (Phase 3)':<28} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    else:
        print(f"  {'1-day LSTM':<28} {'(metrics.pkl not found)':>32}")

    d3   = metrics_3d["Directional Accuracy"]
    mae3 = metrics_3d["MAE"]
    rmse3= metrics_3d["RMSE"]
    print(f"  {'3-day LSTM (Phase 4 exp)':<28} "
          f"{d3:>9.2f}% {mae3:>10.4f} {rmse3:>10.4f}")
    print(f"  {'Naive baseline':<28} {'~50.00%':>10} {'—':>10} {'—':>10}")
    print("=" * 62)
    print("""
  INTERPRETATION:
  ─────────────────────────────────────────────────────────
  Higher 3-day directional accuracy (if observed) means:
    - 3-day returns carry stronger momentum signal
    - EMA/RSI/MACD indicators better capture multi-day trends
    - Longer horizons reduce single-day noise sensitivity

  Higher 3-day MAE/RMSE is EXPECTED (not a problem):
    - 3-day returns have larger absolute magnitude than 1-day
    - Compare accuracy %, not raw MAE, across horizons

  This is an experimental comparison. The 1-day model is
  the production Phase 3 model. Original files unchanged.
  ─────────────────────────────────────────────────────────
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(csv_path: str = "data/BTC-USD.csv") -> None:
    print("=" * 62)
    print("CryptoPulse -- train_lstm_3day.py (FIXED)")
    print(f"Experiment: {HORIZON}-day forward return prediction")
    print("Original 1-day model files: NOT MODIFIED")
    print("=" * 62)

    # 1. Load base features (same pipeline)
    print("\n[1/7] Loading data...")
    df_raw = load_data(csv_path)

    # 2. Add 3-day target (NOT in FEATURE_COLS)
    print(f"\n[2/7] Computing {HORIZON}-day forward return target...")
    df = add_3day_target(df_raw)

    # 3. Chronological split BEFORE scaler fit (prevent leakage)
    print("\n[3/7] Chronological 80/20 split...")
    train_df, test_df = chronological_split(df, train_ratio=TRAIN_RATIO)

    # 4. Fit two separate scalers on training data only
    print("\n[4/7] Fitting feature + target scalers on training data only...")
    feat_scaler, tgt_scaler = fit_scalers(train_df)

    # Apply scalers
    train_sc = scale_df(train_df, feat_scaler, tgt_scaler)
    test_sc  = scale_df(test_df,  feat_scaler, tgt_scaler)

    # 5. Build sequences
    print("\n[5/7] Building sequences (lookback=60)...")
    X_train, y_train = build_sequences(train_sc)
    X_test,  y_test  = build_sequences(test_sc)

    val_n    = int(len(X_train) * 0.10)
    X_val    = X_train[-val_n:];  y_val    = y_train[-val_n:]
    X_train_ = X_train[:-val_n]; y_train_ = y_train[:-val_n]
    print(f"  Train: {X_train_.shape[0]}  Val: {X_val.shape[0]}  "
          f"Test: {X_test.shape[0]}")

    # 6. Build and train
    print("\n[6/7] Building and training 3-day LSTM...")
    model = build_model(n_features=len(FEATURE_COLS))
    model.summary()
    train_model(model, X_train_, y_train_, X_val, y_val)

    # 7. Evaluate (FIX: use dedicated target scaler for inverse transform)
    print("\n[7/7] Evaluating on held-out test set...")
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform using TARGET scaler (not feature scaler -- that was the bug)
    y_pred_raw = inverse_transform_target(y_pred_scaled, tgt_scaler)
    y_test_raw = inverse_transform_target(y_test,        tgt_scaler)

    metrics    = compute_metrics(y_test_raw, y_pred_raw)
    test_dates = test_df["Date"].iloc[LOOKBACK:].values

    # Debug checks -- catches any remaining leakage
    print_debug_checks(y_test_raw, y_pred_raw, test_dates,
                       metrics["Directional Accuracy"])

    print(f"\n  3-Day Return Metrics (test set):")
    print(f"  MAE  : {metrics['MAE']:.6f}  ({metrics['MAE']*100:.3f}%/3-day period)")
    print(f"  RMSE : {metrics['RMSE']:.6f}  ({metrics['RMSE']*100:.3f}%/3-day period)")
    print(f"  Dir. Accuracy : {metrics['Directional Accuracy']:.2f}%")
    print(f"  Naive baseline: ~50.00%")

    # Save
    with open(METRICS_PATH, "wb") as f:
        pickle.dump({
            "return_metrics": {
                "Directional Accuracy": metrics["Directional Accuracy"],
                "MAE" : metrics["MAE"],
                "RMSE": metrics["RMSE"],
            },
            "y_test_ret"  : y_test_raw,
            "y_pred_ret"  : y_pred_raw,
            "test_dates"  : test_dates,
            "horizon_days": HORIZON,
        }, f)
    print(f"\n[3day] Metrics saved -> '{METRICS_PATH}'")

    print_comparison_table(metrics)

    print("=" * 62)
    print("Training complete.")
    print(f"  Model         -> {MODEL_PATH}")
    print(f"  Feature scaler-> {SCALER_PATH}")
    print(f"  Target scaler -> {TARGET_SCALER_PATH}")
    print("  1-day model files: UNCHANGED")
    print("=" * 62)


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/BTC-USD.csv"
    main(csv)
