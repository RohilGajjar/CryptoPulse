"""
compare_horizon_results.py -- CryptoPulse Phase 4
===================================================
Loads and compares 1-day vs 3-day LSTM evaluation metrics.
Run AFTER both models have been trained.

Usage:
    python compare_horizon_results.py
"""

import os
import pickle
import pandas as pd

METRICS_1DAY = "models/metrics.pkl"
METRICS_3DAY = "models/metrics_3day.pkl"
OUT_CSV      = "models/horizon_comparison.csv"


def load_metrics(path, label):
    """Load a metrics.pkl and extract key values into a flat dict."""
    if not os.path.exists(path):
        print(f"  [{label}] Not found: '{path}'")
        return None

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle both old and new metrics.pkl formats
    rm = data.get("return_metrics", data)
    return {
        "Model"            : label,
        "Horizon"          : f"{data.get('horizon_days', 1)}-day",
        "Dir_Accuracy_pct" : rm.get("Directional Accuracy",
                             rm.get("Directional Acc", None)),
        "MAE"              : rm.get("MAE", None),
        "RMSE"             : rm.get("RMSE", None),
    }


def print_table(rows):
    """Print a clean formatted comparison table."""
    W = 65
    print("\n" + "=" * W)
    print("  CryptoPulse -- Horizon Comparison: 1-Day vs 3-Day LSTM")
    print("=" * W)
    print(f"\n  {'Model':<30} {'Horizon':>8} {'Dir.Acc':>10} "
          f"{'MAE':>10} {'RMSE':>10}")
    print(f"  {'─'*60}")

    for r in rows:
        if r is None:
            continue
        dir_str  = f"{r['Dir_Accuracy_pct']:.2f}%" if r["Dir_Accuracy_pct"] else "N/A"
        mae_str  = f"{r['MAE']:.4f}"               if r["MAE"]              else "N/A"
        rmse_str = f"{r['RMSE']:.4f}"              if r["RMSE"]             else "N/A"
        print(f"  {r['Model']:<30} {r['Horizon']:>8} {dir_str:>10} "
              f"{mae_str:>10} {rmse_str:>10}")

    print(f"  {'Naive baseline':<30} {'any':>8} {'~50.00%':>10} "
          f"{'—':>10} {'—':>10}")
    print("=" * W)

    # Lift analysis
    valid = [r for r in rows if r and r["Dir_Accuracy_pct"] is not None]
    if len(valid) == 2:
        lift = valid[1]["Dir_Accuracy_pct"] - valid[0]["Dir_Accuracy_pct"]
        mae_delta  = valid[0]["MAE"]  - valid[1]["MAE"]
        rmse_delta = valid[0]["RMSE"] - valid[1]["RMSE"]
        print(f"\n  LIFT (3-day vs 1-day):")
        print(f"  {'─'*40}")
        print(f"  Directional Acc. : {lift:>+.2f}%  "
              f"({'↑ 3-day better' if lift > 0 else '↓ 1-day better' if lift < 0 else '= tie'})")
        print(f"  MAE  improvement : {mae_delta:>+.4f}  "
              f"({'↑ 3-day better' if mae_delta > 0 else '↓ 1-day better'})")
        print(f"  RMSE improvement : {rmse_delta:>+.4f}  "
              f"({'↑ 3-day better' if rmse_delta > 0 else '↓ 1-day better'})")

    print("""
  INTERPRETATION:
  ─────────────────────────────────────────────────────────
  If 3-day model shows higher directional accuracy:
    - 3-day returns carry stronger momentum signal
    - Technical indicators (EMA, RSI, MACD) better capture
      multi-day trends than single-day noise
    - This is expected: longer horizons reduce noise

  If 3-day model has higher MAE/RMSE:
    - Expected -- 3-day returns have larger absolute magnitude
      than 1-day returns, so errors are proportionally larger
    - Compare MAE as % of the average return, not raw value

  Note: Both models are valid -- they serve different trading
  horizons. The 1-day model is the production Phase 3 model;
  the 3-day model is a Phase 4 experimental comparison.
  ─────────────────────────────────────────────────────────
    """)


def main():
    print("[compare] Loading metrics from both models...")

    row1 = load_metrics(METRICS_1DAY, "1-day LSTM (Phase 3)")
    row3 = load_metrics(METRICS_3DAY, "3-day LSTM (Phase 4 exp.)")

    if row1 is None and row3 is None:
        print("No metrics files found. Train both models first:")
        print("  python src/train_lstm.py data/BTC-USD.csv")
        print("  python src/train_lstm_3day.py data/BTC-USD.csv")
        return

    print_table([row1, row3])

    # Save CSV
    rows = [r for r in [row1, row3] if r is not None]
    df   = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[compare] Comparison saved -> '{OUT_CSV}'")


if __name__ == "__main__":
    main()
