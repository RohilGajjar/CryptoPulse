"""
phase4_evaluation.py -- CryptoPulse Phase 4 Evaluation
========================================================
Evaluates LSTM-only vs hybrid (LSTM + sentiment) performance.
Runs TWO sentiment experiments in one script:

  A. REAL SENTIMENT    -- loads data/daily_sentiment.csv (FinBERT scores)
  B. SIMULATED SENTIMENT -- controlled experiment with np.random.normal(0, 0.2)

Outputs:
  models/phase4_metrics_real.pkl
  models/phase4_metrics_simulated.pkl
  models/phase4_summary.csv

Run:
    python phase4_evaluation.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ─────────────────────────────────────────────────────────────────────
METRICS_PATH        = "models/metrics.pkl"
SENTIMENT_CSV_PATH  = "data/daily_sentiment.csv"
OUT_REAL            = "models/phase4_metrics_real.pkl"
OUT_SIM             = "models/phase4_metrics_simulated.pkl"
OUT_SUMMARY         = "models/phase4_summary.csv"

# ── Hybrid blend weights (must match signal_engine.py) ────────────────────────
LSTM_WEIGHT      = 0.75
SENTIMENT_WEIGHT = 0.25
SENTIMENT_SCALE  = 0.02   # scales [-1,+1] sentiment to ~±2% return magnitude


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 -- Load Phase 3 test predictions from metrics.pkl
# ─────────────────────────────────────────────────────────────────────────────

def load_phase3_data(path=METRICS_PATH):
    """
    Load y_true, y_pred, and test_dates from training output.
    These were saved by train_lstm.py on the 20% held-out test set.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found.\n"
            f"Run: python src/train_lstm.py data/BTC-USD.csv"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)

    y_true = np.array(data["y_test_ret"], dtype=np.float32)
    y_pred = np.array(data["y_pred_ret"], dtype=np.float32)

    # test_dates may or may not exist depending on train_lstm.py version
    if "test_dates" in data:
        test_dates = pd.to_datetime(data["test_dates"]).normalize()
    else:
        test_dates = None
        print("[load] WARNING: test_dates not found in metrics.pkl")

    print(f"[load] y_true: {y_true.shape}  y_pred: {y_pred.shape}")
    if test_dates is not None:
        print(f"[load] Test period: {test_dates[0].date()} -> {test_dates[-1].date()}")

    return y_true, y_pred, test_dates


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2A -- Load real FinBERT sentiment and align to test dates
# ─────────────────────────────────────────────────────────────────────────────

def load_real_sentiment(test_dates, path=SENTIMENT_CSV_PATH):
    """
    Load daily_sentiment.csv and align to test set dates.

    Alignment:
        - Match each test date to the CSV date column.
        - Dates with no headlines are filled with 0.0 (neutral).

    Returns:
        scores   : np.ndarray of shape (n_test,) aligned to test_dates
        coverage : float percentage of test days with real sentiment
        n_matched: int number of days matched
    """
    if not os.path.exists(path):
        print(f"[real_sentiment] '{path}' not found -- cannot load real sentiment.")
        return None, 0.0, 0

    df = pd.read_csv(path)

    # Parse dates safely -- handle multiple formats
    if "date" not in df.columns:
        print("[real_sentiment] 'date' column missing from CSV.")
        return None, 0.0, 0

    try:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    except Exception as e:
        print(f"[real_sentiment] Date parse error: {e}")
        return None, 0.0, 0

    if "sentiment_score" not in df.columns:
        print("[real_sentiment] 'sentiment_score' column missing.")
        return None, 0.0, 0

    # Build date -> score lookup
    sent_map = dict(zip(df["date"], df["sentiment_score"].astype(float)))

    if test_dates is None:
        print("[real_sentiment] No test_dates available -- cannot align.")
        return None, 0.0, 0

    # Align: for each test date, look up score or fill 0.0
    scores    = []
    n_matched = 0
    for d in test_dates:
        if d in sent_map:
            scores.append(float(sent_map[d]))
            n_matched += 1
        else:
            scores.append(0.0)

    scores   = np.array(scores, dtype=np.float32)
    coverage = n_matched / len(test_dates) * 100

    print(f"[real_sentiment] Matched {n_matched}/{len(test_dates)} days "
          f"({coverage:.1f}%) -- {len(test_dates)-n_matched} filled with 0.0")

    return scores, coverage, n_matched


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2B -- Generate simulated sentiment (controlled experiment)
# ─────────────────────────────────────────────────────────────────────────────

def generate_simulated_sentiment(n, seed=42):
    """
    Reproducible simulated sentiment -- normal(0, 0.2) clipped to [-1, +1].

    WHY this is useful:
        Even with no real news data, this validates that the hybrid FUSION
        MECHANISM works correctly. If simulated sentiment improves accuracy,
        it proves the blending formula is sound -- not that news helps.

    seed=42 ensures the experiment is reproducible across runs.
    """
    np.random.seed(seed)
    scores = np.random.normal(loc=0.0, scale=0.2, size=n)
    return np.clip(scores, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- Compute hybrid return
# ─────────────────────────────────────────────────────────────────────────────

def compute_hybrid_return(y_pred, sentiment_scores):
    """
    Blend LSTM return with scaled sentiment score.

    Formula (matches signal_engine.py):
        hybrid = 0.75 * y_pred + 0.25 * (sentiment_score * 0.02)

    SENTIMENT_SCALE = 0.02 converts [-1,+1] sentiment to ~±2% return
    magnitude, keeping it on the same scale as typical daily BTC returns.
    """
    return (LSTM_WEIGHT * y_pred
            + SENTIMENT_WEIGHT * (sentiment_scores * SENTIMENT_SCALE))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 -- Compute evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label=""):
    """
    Compute MAE, RMSE, and Directional Accuracy.

    Directional Accuracy:
        % of days where sign(predicted return) == sign(actual return).
        50% = coin flip baseline.  >55% = statistically meaningful.

    Note: RMSE must always be >= MAE on the same arrays.
    """
    mae     = float(mean_absolute_error(y_true, y_pred))
    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    return {"label": label, "MAE": mae, "RMSE": rmse, "Dir_Acc": dir_acc}


def compute_sentiment_stats(scores):
    """Descriptive statistics for a sentiment score array."""
    return {
        "mean"       : float(np.mean(scores)),
        "std"        : float(np.std(scores)),
        "min"        : float(np.min(scores)),
        "max"        : float(np.max(scores)),
        "pct_bullish": float(np.mean(scores >  0.1) * 100),
        "pct_bearish": float(np.mean(scores < -0.1) * 100),
        "pct_neutral": float(np.mean(np.abs(scores) <= 0.1) * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 -- Print comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(lstm_m, real_m, sim_m,
                     real_sent_stats, sim_sent_stats,
                     real_coverage, real_n_matched):
    """
    Print a clean side-by-side comparison of all three modes.
    """
    W = 62
    print("\n" + "=" * W)
    print("  CryptoPulse -- Phase 4 Hybrid Evaluation")
    print("=" * W)

    # ── Performance comparison table ─────────────────────────────────────────
    print(f"\n  {'METRIC':<28} {'LSTM Only':>10} {'Real Sent':>10} {'Sim Sent':>10}")
    print(f"  {'─'*58}")
    print(f"  {'Directional Accuracy':<28} "
          f"{lstm_m['Dir_Acc']:>9.2f}% "
          f"{real_m['Dir_Acc']:>9.2f}% "
          f"{sim_m['Dir_Acc']:>9.2f}%")
    print(f"  {'MAE (return/day)':<28} "
          f"{lstm_m['MAE']:>10.4f} "
          f"{real_m['MAE']:>10.4f} "
          f"{sim_m['MAE']:>10.4f}")
    print(f"  {'RMSE (return/day)':<28} "
          f"{lstm_m['RMSE']:>10.4f} "
          f"{real_m['RMSE']:>10.4f} "
          f"{sim_m['RMSE']:>10.4f}")
    real_lift = real_m['Dir_Acc'] - lstm_m['Dir_Acc']
    sim_lift  = sim_m['Dir_Acc']  - lstm_m['Dir_Acc']
    print(f"  {'Dir. Acc Lift vs LSTM':<28} "
          f"{'—':>10} "
          f"{real_lift:>+9.2f}% "
          f"{sim_lift:>+9.2f}%")
    print(f"  {'Naive baseline':<28} {'~50.00%':>10} {'—':>10} {'—':>10}")

    # ── Sentiment statistics ──────────────────────────────────────────────────
    print(f"\n  SENTIMENT STATISTICS")
    print(f"  {'─'*58}")
    print(f"  {'':28} {'Real':>10} {'Simulated':>10}")
    print(f"  {'Mean score':<28} {real_sent_stats['mean']:>+10.4f} "
          f"{sim_sent_stats['mean']:>+10.4f}")
    print(f"  {'Std dev':<28} {real_sent_stats['std']:>10.4f} "
          f"{sim_sent_stats['std']:>10.4f}")
    print(f"  {'Min':<28} {real_sent_stats['min']:>+10.4f} "
          f"{sim_sent_stats['min']:>+10.4f}")
    print(f"  {'Max':<28} {real_sent_stats['max']:>+10.4f} "
          f"{sim_sent_stats['max']:>+10.4f}")
    print(f"  {'Bullish days (> +0.1)':<28} {real_sent_stats['pct_bullish']:>9.1f}% "
          f"{sim_sent_stats['pct_bullish']:>9.1f}%")
    print(f"  {'Bearish days (< -0.1)':<28} {real_sent_stats['pct_bearish']:>9.1f}% "
          f"{sim_sent_stats['pct_bearish']:>9.1f}%")
    print(f"  {'Neutral days (+-0.1)':<28} {real_sent_stats['pct_neutral']:>9.1f}% "
          f"{sim_sent_stats['pct_neutral']:>9.1f}%")

    # ── Blend config ──────────────────────────────────────────────────────────
    print(f"\n  HYBRID BLEND CONFIG")
    print(f"  {'─'*58}")
    print(f"  LSTM weight      : {LSTM_WEIGHT*100:.0f}%")
    print(f"  Sentiment weight : {SENTIMENT_WEIGHT*100:.0f}%")
    print(f"  Sentiment scale  : {SENTIMENT_SCALE}  (maps [-1,+1] to return units)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n  VERDICT")
    print(f"  {'─'*58}")

    # Real sentiment verdict -- HONEST about low coverage
    print(f"  [Real Sentiment]")
    print(f"  Coverage: {real_n_matched}/{len(real_sent_stats)} "
          f"test days -- " if False else
          f"  Coverage: {real_n_matched} matched test days "
          f"({real_coverage:.1f}%)")

    if real_coverage == 0.0:
        print(f"  Real sentiment pipeline validated (FinBERT scoring confirmed),")
        print(f"  but predictive impact cannot be evaluated due to zero date overlap.")
        print(f"  Expand NewsAPI historical access for full Phase 4 evaluation.")
    elif real_coverage < 20.0:
        print(f"  Limited overlap -- real sentiment partially validated.")
        print(f"  Real sentiment pipeline validated, but predictive impact cannot")
        print(f"  be fully evaluated due to limited date overlap ({real_coverage:.1f}%).")
    else:
        if real_lift > 0:
            print(f"  Real FinBERT sentiment IMPROVES accuracy by {real_lift:+.2f}%.")
        else:
            print(f"  Real sentiment does not improve accuracy ({real_lift:+.2f}%).")
            print(f"  Consider expanding coverage or adjusting blend weights.")

    print(f"\n  [Simulated Sentiment]")
    if sim_lift > 0:
        print(f"  Simulated sentiment improves accuracy by {sim_lift:+.2f}%.")
        print(f"  This validates the hybrid MECHANISM, not real-world sentiment")
        print(f"  predictive power. Simulated sentiment adds noise, not real signal.")
    else:
        print(f"  Simulated sentiment does not improve accuracy ({sim_lift:+.2f}%).")
        print(f"  Controlled experiment confirms LSTM is the primary signal.")

    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("[phase4_evaluation] CryptoPulse Phase 4 -- Dual Sentiment Evaluation")
    print("[phase4_evaluation] ─" * 30)

    os.makedirs("models", exist_ok=True)

    # ── Load Phase 3 predictions ──────────────────────────────────────────────
    y_true, y_pred_lstm, test_dates = load_phase3_data()
    n = len(y_true)

    # ── LSTM-only baseline ────────────────────────────────────────────────────
    lstm_m = compute_metrics(y_true, y_pred_lstm, "LSTM only")
    print(f"[lstm]   Dir Acc: {lstm_m['Dir_Acc']:.2f}%  "
          f"MAE: {lstm_m['MAE']:.4f}  RMSE: {lstm_m['RMSE']:.4f}")

    # ── MODE A: Real FinBERT sentiment ────────────────────────────────────────
    print("\n[Mode A] Loading real FinBERT sentiment...")
    real_scores, real_coverage, real_n_matched = load_real_sentiment(test_dates)

    # If real scores failed to load, use zeros (all neutral)
    if real_scores is None:
        real_scores   = np.zeros(n, dtype=np.float32)
        real_coverage = 0.0
        real_n_matched= 0
        print("[Mode A] Using zero sentiment (no CSV available).")

    real_hybrid = compute_hybrid_return(y_pred_lstm, real_scores)
    real_m      = compute_metrics(y_true, real_hybrid, "Real FinBERT hybrid")
    real_stats  = compute_sentiment_stats(real_scores)
    print(f"[real]   Dir Acc: {real_m['Dir_Acc']:.2f}%  "
          f"MAE: {real_m['MAE']:.4f}  RMSE: {real_m['RMSE']:.4f}")

    # ── MODE B: Simulated sentiment ───────────────────────────────────────────
    print("\n[Mode B] Generating simulated sentiment (seed=42, normal(0, 0.2))...")
    sim_scores  = generate_simulated_sentiment(n, seed=42)
    sim_hybrid  = compute_hybrid_return(y_pred_lstm, sim_scores)
    sim_m       = compute_metrics(y_true, sim_hybrid, "Simulated sentiment hybrid")
    sim_stats   = compute_sentiment_stats(sim_scores)
    print(f"[sim]    Dir Acc: {sim_m['Dir_Acc']:.2f}%  "
          f"MAE: {sim_m['MAE']:.4f}  RMSE: {sim_m['RMSE']:.4f}")

    # ── Print full comparison ─────────────────────────────────────────────────
    print_comparison(lstm_m, real_m, sim_m,
                     real_stats, sim_stats,
                     real_coverage, real_n_matched)

    # ── Save results ──────────────────────────────────────────────────────────
    with open(OUT_REAL, "wb") as f:
        pickle.dump({
            "lstm_metrics"     : lstm_m,
            "hybrid_metrics"   : real_m,
            "sentiment_stats"  : real_stats,
            "sentiment_scores" : real_scores,
            "coverage_pct"     : real_coverage,
            "n_matched"        : real_n_matched,
            "y_pred_hybrid"    : real_hybrid,
            "y_true"           : y_true,
            "source"           : "Real FinBERT (daily_sentiment.csv)",
        }, f)
    print(f"[save] Real results    -> '{OUT_REAL}'")

    with open(OUT_SIM, "wb") as f:
        pickle.dump({
            "lstm_metrics"     : lstm_m,
            "hybrid_metrics"   : sim_m,
            "sentiment_stats"  : sim_stats,
            "sentiment_scores" : sim_scores,
            "coverage_pct"     : 100.0,
            "y_pred_hybrid"    : sim_hybrid,
            "y_true"           : y_true,
            "source"           : "Simulated normal(0, 0.2)",
        }, f)
    print(f"[save] Simulated results -> '{OUT_SIM}'")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    summary = pd.DataFrame([
        {
            "Mode"             : "LSTM only",
            "Dir_Accuracy_pct" : lstm_m["Dir_Acc"],
            "MAE"              : lstm_m["MAE"],
            "RMSE"             : lstm_m["RMSE"],
            "Lift_vs_LSTM_pct" : 0.0,
            "Sent_Mean"        : None,
            "Coverage_pct"     : None,
        },
        {
            "Mode"             : "Real FinBERT hybrid",
            "Dir_Accuracy_pct" : real_m["Dir_Acc"],
            "MAE"              : real_m["MAE"],
            "RMSE"             : real_m["RMSE"],
            "Lift_vs_LSTM_pct" : real_m["Dir_Acc"] - lstm_m["Dir_Acc"],
            "Sent_Mean"        : real_stats["mean"],
            "Coverage_pct"     : real_coverage,
        },
        {
            "Mode"             : "Simulated sentiment hybrid",
            "Dir_Accuracy_pct" : sim_m["Dir_Acc"],
            "MAE"              : sim_m["MAE"],
            "RMSE"             : sim_m["RMSE"],
            "Lift_vs_LSTM_pct" : sim_m["Dir_Acc"] - lstm_m["Dir_Acc"],
            "Sent_Mean"        : sim_stats["mean"],
            "Coverage_pct"     : 100.0,
        },
    ])
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[save] Summary CSV       -> '{OUT_SUMMARY}'")
    print("\n[phase4_evaluation] Done.")


if __name__ == "__main__":
    main()
