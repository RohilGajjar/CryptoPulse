"""
model_inference.py -- CryptoPulse Phase 3
==========================================
Loads trained LSTM + scaler, runs next-day prediction, returns results as dict.
Plug directly into Streamlit via: from src.model_inference import predict_next_day
"""

import os
import sys
import pickle
import numpy as np

# Suppress ALL TensorFlow logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["PYTHONUNBUFFERED"]       = "1"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model

# Make sure src/ is on the path so data_pipeline imports correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import load_data, FEATURE_COLS, LOOKBACK

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/lstm_model.h5"
SCALER_PATH = "models/scaler.pkl"
CSV_PATH    = "data/BTC-USD.csv"

# ── Module-level cache (load once, reuse every call) ──────────────────────────
_model  = None
_scaler = None


def load_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load model and scaler from disk. Cached after first call."""
    global _model, _scaler

    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: '{model_path}' -- "
                f"run: python src/train_lstm.py data/BTC-USD.csv"
            )
        _model = load_model(model_path, compile=False)
        print(f"[inference] Model loaded  <- '{model_path}'")

    if _scaler is None:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found: '{scaler_path}' -- "
                f"run: python src/train_lstm.py data/BTC-USD.csv"
            )
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        print(f"[inference] Scaler loaded <- '{scaler_path}'")

    return _model, _scaler


def build_latest_sequence(df, scaler, lookback=LOOKBACK):
    """
    Take the last `lookback` rows, scale them, reshape to (1, 60, 8).
    WHY same scaler: model expects inputs in the same [0,1] range it trained on.
    """
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(df)}")

    window        = df.tail(lookback).copy()
    window_scaled = window.copy()
    window_scaled[FEATURE_COLS] = scaler.transform(window[FEATURE_COLS])

    sequence = window_scaled[FEATURE_COLS].values                     # (60, 8)
    sequence = sequence.reshape(1, lookback, len(FEATURE_COLS))       # (1, 60, 8)
    return sequence.astype(np.float32), df.iloc[-1]


def inverse_transform_return(scaled_value, scaler):
    """
    Undo scaler on the Return column (index 0).
    Uses dummy array because scaler was fitted on all 8 features jointly.
    """
    dummy      = np.zeros((1, scaler.n_features_in_), dtype=np.float32)
    dummy[0,0] = scaled_value
    return float(scaler.inverse_transform(dummy)[0, 0])


def predict_next_day(csv_path=CSV_PATH, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    Full inference pipeline. Returns a plain dict for easy Streamlit use.

    Usage in app.py:
        from src.model_inference import predict_next_day
        result = predict_next_day()
        st.metric("Predicted Price", f"${result['predicted_price']:,.2f}")

    Returns:
        {
            predicted_return : float   e.g.  0.0213
            predicted_price  : float   e.g.  76482.50
            last_close       : float   e.g.  74676.42
            last_date        : str     e.g. "2026-04-16"
            direction        : str     "UP" or "DOWN"
            change_pct       : str     "+2.13%" or "-1.05%"
        }
    """
    model,  scaler   = load_artifacts(model_path, scaler_path)
    df               = load_data(csv_path)
    sequence, last   = build_latest_sequence(df, scaler)
    scaled_pred      = model.predict(sequence, verbose=0)[0][0]
    predicted_return = inverse_transform_return(scaled_pred, scaler)
    last_close       = float(last["Close"])
    predicted_price  = last_close * (1 + predicted_return)

    return {
        "predicted_return" : predicted_return,
        "predicted_price"  : round(predicted_price, 2),
        "last_close"       : last_close,
        "last_date"        : str(last["Date"])[:10],
        "direction"        : "UP" if predicted_return >= 0 else "DOWN",
        "change_pct"       : f"{predicted_return * 100:+.2f}%",
    }


def predict_n_days(n_days=7, csv_path=CSV_PATH, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    Autoregressive multi-day forecast.
    Each predicted return feeds into the next window.
    Accuracy degrades with each step (compounding uncertainty).

    Returns list of dicts:
        [{"day":1, "predicted_price":76482.50, "change_pct":"+2.13%"}, ...]
    """
    model,  scaler = load_artifacts(model_path, scaler_path)
    df             = load_data(csv_path)

    window        = df.tail(LOOKBACK).copy()
    window_scaled = window.copy()
    window_scaled[FEATURE_COLS] = scaler.transform(window[FEATURE_COLS])
    seq           = window_scaled[FEATURE_COLS].values.astype(np.float32)  # (60, 8)

    last_close = float(df["Close"].iloc[-1])
    results    = []

    for day in range(1, n_days + 1):
        inp          = seq.reshape(1, LOOKBACK, len(FEATURE_COLS))
        scaled_pred  = model.predict(inp, verbose=0)[0][0]
        ret          = inverse_transform_return(scaled_pred, scaler)
        price        = last_close * (1 + ret)

        results.append({
            "day"             : day,
            "predicted_price" : round(price, 2),
            "predicted_return": ret,
            "change_pct"      : f"{ret * 100:+.2f}%",
        })

        # Roll window: drop oldest row, append new row with predicted return
        new_row    = seq[-1].copy()
        new_row[0] = scaled_pred          # update Return column (index 0)
        seq        = np.vstack([seq[1:], new_row])
        last_close = price

    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Change working directory to project root so relative paths work
    # regardless of where the script is called from
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print("=" * 55)
    print("CryptoPulse -- model_inference.py self-test")
    print("=" * 55)

    # --- Next-day prediction ---
    print("\n--- Next-day prediction ---")
    result = predict_next_day()
    print(f"  Last date        : {result['last_date']}")
    print(f"  Last close       : ${result['last_close']:>12,.2f}")
    print(f"  Predicted return : {result['change_pct']}")
    print(f"  Predicted price  : ${result['predicted_price']:>12,.2f}")
    print(f"  Direction        : {result['direction']}")

    # --- 7-day forecast ---
    print("\n--- 7-day forecast ---")
    forecast = predict_n_days(n_days=7)
    print(f"  {'Day':<6} {'Price':>14}  {'Change':>8}")
    print(f"  {'─'*6} {'─'*14}  {'─'*8}")
    for f in forecast:
        arrow = "▲" if f["predicted_return"] >= 0 else "▼"
        print(f"  Day {f['day']:<3}  ${f['predicted_price']:>12,.2f}  {arrow} {f['change_pct']:>7}")

    print("\nSelf-test complete.")
    print("=" * 55)
