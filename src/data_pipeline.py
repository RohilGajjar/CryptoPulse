"""
data_pipeline.py -- CryptoPulse Phase 3
========================================
Loads BTC-USD CSV, computes technical indicators, adds daily Return column.
Return column is the KEY FIX -- target for LSTM instead of raw Close price.

Outputs clean DataFrame with columns:
Date, Open, High, Low, Close, Volume,
EMA20, EMA50, RSI, BB_upper, BB_mid, BB_lower,
MACD, MACD_signal, MACD_hist, Return
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Shared constants (imported by train_lstm.py and app.py) ──────────────────
FEATURE_COLS = [
    "Return",    # Daily pct change -- PRIMARY feature (fixes direction problem)
    "Volume",    # Trading volume
    "RSI",       # Momentum oscillator 0-100
    "EMA20",     # Short-term trend
    "EMA50",     # Medium-term trend
    "BB_upper",  # Bollinger upper band
    "BB_lower",  # Bollinger lower band
    "MACD",      # Trend momentum
]
TARGET_COL = "Return"
LOOKBACK   = 60


# ── Indicator functions (pandas/numpy only, no external TA library) ──────────

def compute_ema(series, span):
    """Exponential Moving Average. alpha = 2/(span+1)."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series, period=14):
    """RSI using Wilder smoothing (matches TradingView)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_bollinger(series, window=20, num_std=2.0):
    """Bollinger Bands: SMA +/- 2 standard deviations."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=0)
    return sma + num_std * std, sma, sma - num_std * std


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD = EMA(fast) - EMA(slow). Signal = EMA(MACD, signal)."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_data(csv_path="data/BTC-USD.csv"):
    """
    Load CSV -> clean -> sort -> indicators -> Return -> validate.

    KEY ADDITION: Return = pct_change(Close)
    This fixes the lazy predictor problem where LSTM predicts raw price
    and achieves low RMSE by just copying yesterday's price without
    learning direction. Predicting Return forces directional learning.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File not found: '{csv_path}'\n"
            f"Run: python download_data.py"
        )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Handle yfinance v0.2 multi-level header
    if df.columns[0].lower() in ("price", "ticker", ""):
        df = pd.read_csv(csv_path, header=[0, 1])
        df.columns = [f"{a}_{b}".strip("_") for a, b in df.columns]
        df.columns = [c.split("_")[0] for c in df.columns]

    # Validate required columns
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Parse dates and sort
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    df = df.reset_index(drop=True)

    # Cast OHLCV to float
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = df.reset_index(drop=True)

    # Compute indicators
    df["EMA20"]    = compute_ema(df["Close"], span=20)
    df["EMA50"]    = compute_ema(df["Close"], span=50)
    df["RSI"]      = compute_rsi(df["Close"], period=14)
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = compute_bollinger(df["Close"])
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["Close"])

    # KEY FIX: Add Return column
    # Return_t = (Close_t - Close_{t-1}) / Close_{t-1}
    # Dashboard converts predictions back via: price = last_close * (1 + return)
    df["Return"] = df["Close"].pct_change()

    # Drop NaN rows from indicator warm-up + first Return row
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(
        f"[data_pipeline] Dropped {rows_before - len(df)} NaN warm-up rows "
        f"({rows_before} -> {len(df)} rows)"
    )

    # Validate all columns present and NaN-free
    expected = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "EMA20", "EMA50", "RSI", "BB_upper", "BB_mid", "BB_lower",
        "MACD", "MACD_signal", "MACD_hist", "Return"
    ]
    for col in expected:
        assert col in df.columns,         f"Missing column: {col}"
        assert df[col].isna().sum() == 0, f"NaNs found in: {col}"

    print(
        f"[data_pipeline] Ready: {len(df)} rows | "
        f"{df['Date'].iloc[0].date()} -> {df['Date'].iloc[-1].date()}"
    )
    return df


# ── Train/test split ──────────────────────────────────────────────────────────

def chronological_split(df, train_ratio=0.80):
    """
    Strict chronological split -- NO random shuffle.
    Shuffling leaks future data into training (lookahead bias).
    """
    split_idx = int(len(df) * train_ratio)
    train_df  = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df   = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(
        f"[data_pipeline] Train: {len(train_df)} rows "
        f"({train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()})\n"
        f"[data_pipeline] Test : {len(test_df)} rows "
        f"({test_df['Date'].iloc[0].date()} -> {test_df['Date'].iloc[-1].date()})"
    )
    return train_df, test_df


def build_sequences(df, feature_cols=None, target_col=TARGET_COL, lookback=LOOKBACK):
    """
    Convert flat DataFrame into (X, y) LSTM pairs.
    X shape : (n_samples, 60, 8)
    y shape : (n_samples,)  -- next-day Return
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    data   = df[feature_cols].values
    target = df[target_col].values
    X, y   = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(target[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/BTC-USD.csv"

    print("=" * 60)
    print("CryptoPulse -- data_pipeline.py self-test")
    print("=" * 60)

    df = load_data(csv_path)
    print("\nColumn summary (last 5 rows):")
    print(df[["Date", "Close", "Return", "RSI", "EMA20", "MACD"]].tail(5).to_string(index=False))

    train_df, test_df = chronological_split(df)
    X_train, y_train  = build_sequences(train_df)
    X_test,  y_test   = build_sequences(test_df)

    print(f"\nX_train : {X_train.shape}  (samples, timesteps, features)")
    print(f"y_train : {y_train.shape}  target = daily Return")
    print(f"X_test  : {X_test.shape}")
    print(f"y_test  : {y_test.shape}")

    assert X_train.ndim == 3
    assert X_train.shape[1] == LOOKBACK
    assert X_train.shape[2] == len(FEATURE_COLS)
    assert not np.isnan(X_train).any()
    assert not np.isnan(y_train).any()

    print("\nAll checks passed.")
    print("=" * 60)
