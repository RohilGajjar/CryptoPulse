"""
run_cryptopulse.py -- CryptoPulse Master Runner
=================================================
Runs the full CryptoPulse pipeline in sequence with a single command.

Usage:
    # Full setup (first time ever):
    python run_cryptopulse.py --mode setup --api_key YOUR_NEWSAPI_KEY

    # Daily demo (every time you present):
    python run_cryptopulse.py --mode demo

    # Evaluation only:
    python run_cryptopulse.py --mode eval

    # Everything including training (takes 30+ min):
    python run_cryptopulse.py --mode full --api_key YOUR_NEWSAPI_KEY
"""

import os
import sys
import time
import argparse
import subprocess

# ── Full Python path (fixes msys64 conflict on this machine) ──────────────────
PYTHON = r"C:/Users/Romeo.DESKTOP-IFEH0EL/AppData/Local/Programs/Python/Python313/python.exe"

# ── Color codes for terminal output ──────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def banner():
    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════╗
║           CryptoPulse -- Master Runner               ║
║     AI-Powered Bitcoin Decision Dashboard            ║
║           CECS 551 -- Phase 4                        ║
╚══════════════════════════════════════════════════════╝
{RESET}""")


def log(msg, level="info"):
    ts = time.strftime("%H:%M:%S")
    if level == "ok":
        print(f"  {GREEN}✔{RESET}  [{ts}] {msg}")
    elif level == "run":
        print(f"  {CYAN}▶{RESET}  [{ts}] {msg}")
    elif level == "warn":
        print(f"  {YELLOW}⚠{RESET}  [{ts}] {msg}")
    elif level == "err":
        print(f"  {RED}✘{RESET}  [{ts}] {msg}")
    elif level == "head":
        print(f"\n{BOLD}{YELLOW}── {msg} {RESET}")
    else:
        print(f"     [{ts}] {msg}")


def run(label, cmd, critical=True, capture=False):
    """
    Run a command and stream its output.
    If critical=True, abort the entire pipeline on failure.
    """
    log(label, "run")
    start = time.time()

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=capture,
        text=True,
    )

    elapsed = time.time() - start

    if result.returncode == 0:
        log(f"{label} -- done in {elapsed:.1f}s", "ok")
        return True
    else:
        log(f"{label} FAILED (exit code {result.returncode})", "err")
        if capture and result.stderr:
            print(result.stderr[:500])
        if critical:
            print(f"\n{RED}Pipeline aborted. Fix the error above and retry.{RESET}\n")
            sys.exit(1)
        return False


def check_file(path, label):
    """Check if a required file exists before proceeding."""
    if os.path.exists(path):
        log(f"{label} found: {path}", "ok")
        return True
    else:
        log(f"{label} NOT found: {path}", "warn")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────

def step_download_data():
    log("STEP 1 — Download BTC historical data", "head")
    run("Downloading BTC-USD.csv via yfinance",
        f'"{PYTHON}" download_data.py')


def step_train_1day():
    log("STEP 2 — Train 1-day LSTM (Phase 3 model)", "head")
    log("This takes 10-15 minutes. Do not close the terminal.", "warn")
    run("Training 1-day LSTM",
        f'"{PYTHON}" src/train_lstm.py data/BTC-USD.csv')


def step_train_3day():
    log("STEP 3 — Train 3-day LSTM (Phase 4 experiment)", "head")
    log("This takes 10-15 minutes.", "warn")
    run("Training 3-day LSTM",
        f'"{PYTHON}" src/train_lstm_3day.py data/BTC-USD.csv')


def step_fetch_sentiment(api_key):
    log("STEP 4 — Fetch and score headlines with FinBERT", "head")
    if not api_key:
        log("No API key provided -- skipping fetch, using cached headlines if available.", "warn")
        if not os.path.exists("data/bitcoin_headlines.csv"):
            log("No cached headlines found either. Sentiment will be simulated.", "warn")
            return
        log("Using cached bitcoin_headlines.csv", "ok")
        return
    run("Fetching headlines from NewsAPI and scoring with FinBERT",
        f'"{PYTHON}" fetch_news_sentiment.py --api_key {api_key}')


def step_fix_sentiment():
    log("STEP 5 — Fix sentiment date alignment", "head")
    run("Aligning sentiment scores to test set dates",
        f'"{PYTHON}" fix_sentiment_dates.py')


def step_generate_cache():
    log("STEP 6 — Generate dashboard cache", "head")
    run("Running LSTM inference and saving dashboard_cache.pkl",
        f'"{PYTHON}" generate_cache.py')


def step_launch_dashboard():
    log("STEP 7 — Launch Streamlit dashboard", "head")
    log("Opening at http://localhost:8501", "ok")
    log("Press Ctrl+C to stop the dashboard.", "warn")
    # This one is blocking -- no critical flag, user exits with Ctrl+C
    subprocess.run(
        f'"{PYTHON}" -m streamlit run app.py',
        shell=True,
    )


def step_eval_phase4():
    log("EVAL — Phase 4 dual sentiment evaluation", "head")
    run("Running phase4_evaluation.py",
        f'"{PYTHON}" phase4_evaluation.py')


def step_compare_horizons():
    log("EVAL — Horizon comparison (1-day vs 3-day)", "head")
    run("Running compare_horizon_results.py",
        f'"{PYTHON}" compare_horizon_results.py')


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────

def mode_demo():
    """
    DEMO MODE — run before every presentation.
    Refreshes data and cache, then launches dashboard.
    Assumes models are already trained.
    Steps: download → generate cache → launch
    """
    banner()
    log("MODE: DEMO -- refreshing data and launching dashboard", "head")

    # Verify models exist before proceeding
    missing = []
    for f in ["models/lstm_model.h5", "models/scaler.pkl", "models/metrics.pkl"]:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        log("Required model files not found:", "err")
        for m in missing:
            log(f"  Missing: {m}", "err")
        log("Run:  python run_cryptopulse.py --mode setup  first.", "warn")
        sys.exit(1)

    step_download_data()
    step_generate_cache()
    step_launch_dashboard()


def mode_setup(api_key):
    """
    SETUP MODE — first time only.
    Downloads data, trains both models, fetches sentiment, generates cache.
    Does NOT launch the dashboard (run --mode demo for that).
    Steps: download → train 1day → train 3day → sentiment → fix dates → cache
    """
    banner()
    log("MODE: SETUP -- full first-time pipeline (30-45 min)", "head")
    log("Do not close this terminal until complete.", "warn")

    step_download_data()
    step_train_1day()
    step_train_3day()
    step_fetch_sentiment(api_key)
    step_fix_sentiment()
    step_generate_cache()

    print(f"""
{GREEN}{BOLD}
╔══════════════════════════════════════════════════════╗
║              Setup complete!                         ║
║  Run:  python run_cryptopulse.py --mode demo         ║
║  to launch the dashboard at any time.                ║
╚══════════════════════════════════════════════════════╝
{RESET}""")


def mode_eval():
    """
    EVAL MODE — run Phase 4 evaluation scripts.
    Assumes models are already trained.
    Steps: phase4_evaluation → compare_horizons
    """
    banner()
    log("MODE: EVAL -- running Phase 4 evaluation scripts", "head")

    for f in ["models/metrics.pkl", "models/metrics_3day.pkl"]:
        if not check_file(f, f.split("/")[-1]):
            log(f"Train both models first: python run_cryptopulse.py --mode setup", "warn")

    step_eval_phase4()
    step_compare_horizons()

    log("Evaluation complete. Results saved to models/", "ok")


def mode_full(api_key):
    """
    FULL MODE — everything including training and then launch.
    Steps: all setup steps → launch dashboard
    """
    banner()
    log("MODE: FULL -- complete pipeline + dashboard launch", "head")

    step_download_data()
    step_train_1day()
    step_train_3day()
    step_fetch_sentiment(api_key)
    step_fix_sentiment()
    step_generate_cache()
    step_eval_phase4()
    step_compare_horizons()
    step_launch_dashboard()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CryptoPulse master runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "setup", "eval", "full"],
        default="demo",
        help=(
            "demo   -- refresh data + cache + launch dashboard (daily use)\n"
            "setup  -- first-time: train both models + sentiment + cache\n"
            "eval   -- run Phase 4 evaluation scripts only\n"
            "full   -- everything: setup + eval + launch dashboard"
        ),
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="NewsAPI key for fetching headlines (only needed for setup/full modes)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}CryptoPulse · Mode: {args.mode.upper()}{RESET}")
    print(f"Python: {PYTHON}\n")

    if args.mode == "demo":
        mode_demo()
    elif args.mode == "setup":
        mode_setup(args.api_key)
    elif args.mode == "eval":
        mode_eval()
    elif args.mode == "full":
        mode_full(args.api_key)


if __name__ == "__main__":
    main()
