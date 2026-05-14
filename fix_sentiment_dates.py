"""
fix_sentiment_dates.py -- CryptoPulse Phase 4
===============================================
Fixes the date alignment between FinBERT-scored headlines and the
LSTM test set (Oct 2024 - Apr 2026).

Problem:
    NewsAPI free tier returns articles grouped to today's date only.
    Your test set covers Oct 2024 - Apr 2026, so there is zero date
    overlap with today's headlines.

Solution:
    1. Read real FinBERT scores from data/bitcoin_headlines.csv
       to extract the real distribution (mean, std).
    2. Generate FinBERT-calibrated synthetic scores for all 542 test
       days using an AR(1) process that mimics realistic news sentiment
       persistence (sentiment carries over day to day).
    3. Inject any real FinBERT scores on dates where real headlines exist.
    4. Save corrected data/daily_sentiment.csv aligned to test dates.

Outputs:
    data/daily_sentiment.csv  -- rows aligned to test set dates

Usage:
    python fix_sentiment_dates.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_PATH   = "models/metrics.pkl"
HEADLINES_PATH = "data/bitcoin_headlines.csv"
SENTIMENT_PATH = "data/daily_sentiment.csv"

# ── AR(1) config ──────────────────────────────────────────────────────────────
AR_PHI     = 0.3     # sentiment persistence (30% carries forward each day)
SPIKE_PROB = 0.05    # 5% of days have a sudden sentiment event
SPIKE_SIZE = 0.4     # magnitude of sentiment spike
SEED       = 42


def load_test_dates():
    """Load test set dates from models/metrics.pkl."""
    if not os.path.exists(METRICS_PATH):
        print(f"ERROR: '{METRICS_PATH}' not found.")
        print("Run: python src/train_lstm.py data/BTC-USD.csv")
        sys.exit(1)

    with open(METRICS_PATH, "rb") as f:
        data = pickle.load(f)

    if "test_dates" not in data:
        print("ERROR: 'test_dates' missing from metrics.pkl.")
        print("Re-run: python src/train_lstm.py data/BTC-USD.csv")
        sys.exit(1)

    dates = pd.to_datetime(data["test_dates"]).normalize()
    print(f"[fix] Test period : {dates[0].date()} -> {dates[-1].date()}")
    print(f"[fix] Test samples: {len(dates)} days")
    return dates


def load_finbert_distribution():
    """
    Read real FinBERT scores from bitcoin_headlines.csv.
    Returns (mean, std, date_to_score_map).
    Falls back to sensible defaults if file is missing.
    """
    if not os.path.exists(HEADLINES_PATH):
        print(f"[fix] '{HEADLINES_PATH}' not found -- using defaults.")
        return 0.10, 0.20, {}

    df = pd.read_csv(HEADLINES_PATH)

    if "sentiment_score" not in df.columns or df["sentiment_score"].dropna().empty:
        print("[fix] No sentiment scores found -- using defaults.")
        return 0.10, 0.20, {}

    scores    = df["sentiment_score"].dropna().values
    real_mean = float(np.mean(scores))
    real_std  = float(np.std(scores))
    print(f"[fix] Real FinBERT: mean={real_mean:+.4f}  std={real_std:.4f}  "
          f"n={len(scores)} headlines")

    # Build date -> mean score map for real headline dates
    real_map = {}
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        real_map   = df.groupby("date")["sentiment_score"].mean().to_dict()
        print(f"[fix] Real headline dates: {len(real_map)}")

    return real_mean, real_std, real_map


def generate_ar1_scores(n, mean, std):
    """
    AR(1) autoregressive sentiment:
        score[t] = phi * score[t-1] + noise[t] + mean * (1 - phi)

    Produces realistic day-to-day sentiment persistence.
    Clips result to [-1, +1] (FinBERT output range).
    """
    np.random.seed(SEED)
    scores    = np.zeros(n)
    scores[0] = mean
    noise_std = std * np.sqrt(1 - AR_PHI ** 2)
    noise     = np.random.normal(0, noise_std, n)

    for i in range(1, n):
        scores[i] = AR_PHI * scores[i-1] + noise[i] + mean * (1 - AR_PHI)

    # Add occasional spikes (major news events)
    spike_idx = np.random.choice(n, size=max(1, int(n * SPIKE_PROB)), replace=False)
    scores[spike_idx] += np.random.choice([-SPIKE_SIZE, SPIKE_SIZE],
                                          size=len(spike_idx))
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


def build_daily_sentiment(test_dates, synth_scores, real_map):
    """
    Build daily_sentiment DataFrame.
    Real FinBERT scores override synthetic on matched dates.
    headline_count = 0 marks synthetic rows.
    """
    rows      = []
    real_days = 0

    for i, d in enumerate(test_dates):
        if d in real_map:
            score = float(real_map[d])
            count = 1
            real_days += 1
        else:
            score = float(synth_scores[i])
            count = 0
        rows.append({
            "date"           : d.date(),
            "sentiment_score": round(score, 6),
            "headline_count" : count,
        })

    print(f"[fix] Real FinBERT days  : {real_days}")
    print(f"[fix] Synthetic days     : {len(rows) - real_days}")
    return pd.DataFrame(rows)


def print_summary(df):
    s       = df["sentiment_score"].values
    bullish = float(np.mean(s >  0.1) * 100)
    bearish = float(np.mean(s < -0.1) * 100)
    neutral = 100 - bullish - bearish

    print("\n" + "=" * 52)
    print("  Corrected Sentiment Distribution")
    print("=" * 52)
    print(f"  Total days       : {len(df)}")
    print(f"  Real FinBERT days: {int((df['headline_count'] > 0).sum())}")
    print(f"  Synthetic days   : {int((df['headline_count'] == 0).sum())}")
    print(f"  Mean score       : {np.mean(s):+.4f}")
    print(f"  Std dev          : {np.std(s):.4f}")
    print(f"  Min / Max        : {np.min(s):+.4f} / {np.max(s):+.4f}")
    print(f"  Bullish (> +0.1) : {bullish:.1f}%")
    print(f"  Bearish (< -0.1) : {bearish:.1f}%")
    print(f"  Neutral (+-0.1)  : {neutral:.1f}%")
    print("=" * 52)
    print("\nNext step: python phase4_evaluation.py")


def main():
    print("[fix_sentiment_dates] Starting...")

    # 1. Load test dates
    test_dates = load_test_dates()
    n          = len(test_dates)

    # 2. Get real FinBERT distribution
    real_mean, real_std, real_map = load_finbert_distribution()

    # 3. Generate calibrated synthetic scores
    print(f"\n[fix] Generating AR(1) scores  "
          f"phi={AR_PHI}  mean={real_mean:+.4f}  std={real_std:.4f} ...")
    synth = generate_ar1_scores(n, real_mean, real_std)

    # 4. Build aligned DataFrame
    df = build_daily_sentiment(test_dates, synth, real_map)

    # 5. Save
    os.makedirs("data", exist_ok=True)
    df.to_csv(SENTIMENT_PATH, index=False)
    print(f"[fix] Saved -> '{SENTIMENT_PATH}'")

    # 6. Summary
    print_summary(df)


if __name__ == "__main__":
    main()
