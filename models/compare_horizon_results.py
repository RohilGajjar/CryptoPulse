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
import numpy as np
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
    rm = data.get("return_metrics", data)
    return {
        "Model"            : label,
        "Horizon"          : f"{data.get('horizon_days', 1)}-day",
        "Dir_Accuracy_pct" : rm.get("Directional Accuracy",
                             rm.get("Directional Acc", None)),
        "MAE"              : rm.get("MAE", None),
        "RMSE"             : rm.get("RMSE", None),
        "y_test"           : data.get("y_test_ret", None),
        "y_pred"           : data.get("y_pred_ret", None),
    }


def debug_check(row, label):
    """Print sanity checks for a loaded metrics row."""
    y_true = row.get("y_test")
    y_pred = row.get("y_pred")
    if y_true is None or y_pred is None:
        return
    print(f"\n  ── DEBUG [{label}] ──────────────────────────")
    print(f"  Samples           : {len(y_true)}")
    print(f"  y_true first 5    : {np.round(y_true[:5], 5).tolist()}")
    print(f"  y_pred first 5    : {np.round(y_pred[:5], 5).tolist()}")
    print(f"  y_true % positive : {np.mean(y_true > 0)*100:.1f}%")
    print(f"  y_pred % positive : {np.mean(y_pred > 0)*100:.1f}%")
    dir_acc = row["Dir_Accuracy_pct"]
    if dir_acc and dir_acc > 70:
        print(f"\n  !! WARNING: {dir_acc:.1f}% directional accuracy is suspicious.")
        print(f"  !! Verify for data leakage before including in report.")
    else:
        print(f"  Accuracy {dir_acc:.2f}% -- plausible range.")


def print_table(rows):
    """Print comparison table and lift analysis."""
    W = 65
    print("\n" + "=" * W)
    print("  CryptoPulse -- Horizon Comparison: 1-Day vs 3-Day LSTM")
    print("=" * W)
    print(f"\n  {'Model':<30} {'Horizon':>8} {'Dir.Acc':>10} "
          f"{'MAE':>10} {'RMSE':>10}")
    print(f"  {'─'*62}")

    clean_rows = []
    for r in rows:
        if r is None:
            continue
        d  = r["Dir_Accuracy_pct"]
        m  = r["MAE"]
        rm = r["RMSE"]
        d_s  = f"{d:.2f}%"  if d  is not None else "N/A"
        m_s  = f"{m:.4f}"   if m  is not None else "N/A"
        rm_s = f"{rm:.4f}"  if rm is not None else "N/A"
        print(f"  {r['Model']:<30} {r['Horizon']:>8} {d_s:>10} "
              f"{m_s:>10} {rm_s:>10}")
        clean_rows.append(r)

    print(f"  {'Naive baseline':<30} {'any':>8} {'~50.00%':>10} "
          f"{'—':>10} {'—':>10}")
    print("=" * W)

    # Lift analysis
    if len(clean_rows) == 2:
        r1, r3 = clean_rows
        if r1["Dir_Accuracy_pct"] and r3["Dir_Accuracy_pct"]:
            lift     = r3["Dir_Accuracy_pct"] - r1["Dir_Accuracy_pct"]
            mae_lift = (r1["MAE"]  - r3["MAE"])  if r1["MAE"]  and r3["MAE"]  else None
            rms_lift = (r1["RMSE"] - r3["RMSE"]) if r1["RMSE"] and r3["RMSE"] else None
            print(f"\n  LIFT (3-day vs 1-day):")
            print(f"  {'─'*40}")
            print(f"  Dir. Acc. lift : {lift:>+.2f}%  "
                  f"({'3-day better' if lift > 0 else '1-day better' if lift < 0 else 'tied'})")
            if mae_lift is not None:
                print(f"  MAE improvement: {mae_lift:>+.4f}  "
                      f"({'3-day better' if mae_lift > 0 else '1-day better'})")
            if rms_lift is not None:
                print(f"  RMSE improvement:{rms_lift:>+.4f}  "
                      f"({'3-day better' if rms_lift > 0 else '1-day better'})")

    print("""
  INTERPRETATION:
  ─────────────────────────────────────────────────────────
  If 3-day accuracy > 1-day:
    - Longer horizons have higher signal-to-noise ratio
    - EMA/RSI/MACD better capture multi-day momentum
  If 3-day MAE/RMSE > 1-day:
    - Expected: 3-day returns have larger magnitude
    - Compare directional accuracy %, not raw error
  ─────────────────────────────────────────────────────────
    """)


def main():
    print("[compare] Loading metrics from both models...")

    row1 = load_metrics(METRICS_1DAY, "1-day LSTM (Phase 3)")
    row3 = load_metrics(METRICS_3DAY, "3-day LSTM (Phase 4 exp)")

    if row1 is None and row3 is None:
        print("No metrics files found. Train both models first:")
        print("  python src/train_lstm.py data/BTC-USD.csv")
        print("  python src/train_lstm_3day.py data/BTC-USD.csv")
        return

    # Debug checks for each model
    if row1: debug_check(row1, "1-day")
    if row3: debug_check(row3, "3-day")

    print_table([row1, row3])

    # Save CSV (drop numpy arrays before saving)
    rows_clean = []
    for r in [row1, row3]:
        if r:
            rows_clean.append({k: v for k, v in r.items()
                               if k not in ("y_test", "y_pred")})
    df = pd.DataFrame(rows_clean)
    df.to_csv(OUT_CSV, index=False)
    print(f"[compare] Saved -> '{OUT_CSV}'")


if __name__ == "__main__":
    main()
