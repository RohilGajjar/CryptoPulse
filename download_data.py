"""
download_data.py — CryptoPulse
===============================
Run this ONCE to download BTC-USD historical data and save it to data/BTC-USD.csv

Usage:
    python download_data.py

Requirements:
    pip install yfinance
"""

import os
import yfinance as yf
import pandas

# ── Config ────────────────────────────────────────────────────────────────────
TICKER      = "BTC-USD"
START_DATE  = "2018-01-01"
OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "BTC-USD.csv")

# ── Download ──────────────────────────────────────────────────────────────────
print(f"Downloading {TICKER} from {START_DATE} to today...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = yf.download(TICKER, start=START_DATE, auto_adjust=True)

if df.empty:
    print("ERROR: Download returned empty data. Check your internet connection.")
else:
    # Flatten multi-level columns if present (yfinance v0.2+)
    if isinstance(df.columns, pandas.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index.name = "Date"
    df.to_csv(OUTPUT_FILE)
    print(f"Saved {len(df)} rows to '{OUTPUT_FILE}'")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print("\nSample (last 3 rows):")
    print(df[["Open", "High", "Low", "Close", "Volume"]].tail(3).to_string())
    print("\nDone! Now run:  python src/data_pipeline.py data/BTC-USD.csv")
