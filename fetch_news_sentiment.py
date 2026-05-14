"""
fetch_news_sentiment.py -- CryptoPulse Phase 4
================================================
Fetches Bitcoin-related headlines from NewsAPI, scores them with FinBERT,
aggregates into daily sentiment scores, and saves to CSV.

Outputs:
    data/bitcoin_headlines.csv  -- raw headlines with FinBERT scores
    data/daily_sentiment.csv    -- aggregated daily sentiment scores

Usage:
    python fetch_news_sentiment.py --api_key YOUR_NEWSAPI_KEY

Get a free NewsAPI key at: https://newsapi.org/register
Free tier: 100 requests/day, up to 100 articles per request.

Requirements:
    pip install requests transformers torch pandas
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
QUERY        = "bitcoin OR BTC OR cryptocurrency"
LANGUAGE     = "en"
SORT_BY      = "publishedAt"
PAGE_SIZE    = 100            # max per request on free tier
HEADLINES_PATH = "data/bitcoin_headlines.csv"
SENTIMENT_PATH = "data/daily_sentiment.csv"
FINBERT_MODEL  = "ProsusAI/finbert"

# How many days back to fetch (free tier: last 30 days only)
DAYS_BACK = 30


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FETCH HEADLINES FROM NEWSAPI
# ─────────────────────────────────────────────────────────────────────────────

def fetch_headlines(api_key: str, days_back: int = DAYS_BACK) -> pd.DataFrame:
    """
    Fetch Bitcoin headlines from NewsAPI for the last `days_back` days.
    Returns a DataFrame with columns: date, headline, source.
    """
    url = "https://newsapi.org/v2/everything"

    # Build date range — NewsAPI free tier only supports last 30 days
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    all_articles = []

    # Paginate through results (max 100 per page)
    for page in range(1, 4):    # fetch up to 3 pages = 300 articles
        params = {
            "q"        : QUERY,
            "language" : LANGUAGE,
            "sortBy"   : SORT_BY,
            "pageSize" : PAGE_SIZE,
            "page"     : page,
            "from"     : start_date.strftime("%Y-%m-%d"),
            "to"       : end_date.strftime("%Y-%m-%d"),
            "apiKey"   : api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
        except requests.RequestException as e:
            print(f"[fetch] Network error on page {page}: {e}")
            break

        if resp.status_code == 426:
            print("[fetch] NewsAPI free tier limit reached (developer plan).")
            print("        Free accounts only support last 30 days and 100 requests/day.")
            break
        elif resp.status_code == 401:
            print("[fetch] ERROR: Invalid or missing NewsAPI key.")
            print("        Get a free key at: https://newsapi.org/register")
            sys.exit(1)
        elif resp.status_code != 200:
            print(f"[fetch] NewsAPI error {resp.status_code}: {resp.json().get('message','')}")
            break

        data     = resp.json()
        articles = data.get("articles", [])
        if not articles:
            break

        for a in articles:
            title = a.get("title") or ""
            desc  = a.get("description") or ""
            # Use title + description for richer sentiment signal
            headline = (title + " " + desc).strip()
            pub_at   = a.get("publishedAt", "")[:10]   # "YYYY-MM-DD"
            source   = a.get("source", {}).get("name", "Unknown")

            if headline and pub_at:
                all_articles.append({
                    "date"    : pub_at,
                    "headline": headline,
                    "source"  : source,
                })

        print(f"[fetch] Page {page}: {len(articles)} articles fetched")
        time.sleep(0.5)  # be polite to the API

    if not all_articles:
        print("[fetch] No articles fetched.")
        return pd.DataFrame(columns=["date", "headline", "source"])

    df = pd.DataFrame(all_articles)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="headline")
    df = df.sort_values("date").reset_index(drop=True)
    print(f"[fetch] Total unique headlines: {len(df)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SCORE HEADLINES WITH FINBERT
# ─────────────────────────────────────────────────────────────────────────────

def load_finbert():
    """
    Load FinBERT pipeline from HuggingFace.
    First run downloads ~440 MB of weights (cached locally after that).
    """
    print(f"[finbert] Loading model '{FINBERT_MODEL}' ...")
    print("          (First run downloads ~440 MB — takes 1-2 minutes)")

    from transformers import pipeline
    pipe = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        truncation=True,    # truncate headlines longer than 512 tokens
        max_length=512,
    )
    print("[finbert] Model loaded.")
    return pipe


def score_headlines(df: pd.DataFrame, pipe) -> pd.DataFrame:
    """
    Run FinBERT on each headline and add score column.

    FinBERT outputs label in {positive, negative, neutral}.
    We convert to numeric: positive=+1, neutral=0, negative=-1.
    The final score is label_value * confidence, giving a [-1, +1] signal
    that reflects both direction and certainty.
    """
    label_map = {"positive": 1, "neutral": 0, "negative": -1}

    headlines = df["headline"].tolist()
    scores    = []

    # Process in batches of 16 to avoid OOM on CPU
    batch_size = 16
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i : i + batch_size]
        try:
            results = pipe(batch)
        except Exception as e:
            print(f"[finbert] Error on batch {i//batch_size}: {e}")
            results = [{"label": "neutral", "score": 1.0}] * len(batch)

        for r in results:
            label     = r["label"].lower()
            confidence= r["score"]
            numeric   = label_map.get(label, 0)
            # Weighted score: direction × confidence → [-1, +1]
            scores.append(numeric * confidence)

        if (i // batch_size) % 5 == 0:
            print(f"[finbert] Scored {min(i+batch_size, len(headlines))}/{len(headlines)} headlines...")

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = [
        pipe([h])[0]["label"].lower() for h in headlines
    ] if False else df.get("sentiment_label", "")  # skip second pass

    # Simpler: re-derive label from score sign
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda s: "positive" if s > 0.1 else ("negative" if s < -0.1 else "neutral")
    )

    print(f"[finbert] Scoring complete. {len(df)} headlines scored.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — AGGREGATE TO DAILY SCORES
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-headline scores into one score per day.

    daily_sentiment = mean(sentiment_score) across all headlines that day.
    Clipped to [-1, +1] — daily extremes indicate one-sided news.
    """
    daily = (
        df.groupby("date")["sentiment_score"]
        .agg(sentiment_score="mean", headline_count="count")
        .reset_index()
    )
    daily["sentiment_score"] = daily["sentiment_score"].clip(-1, 1)
    daily = daily.sort_values("date").reset_index(drop=True)
    print(f"[aggregate] {len(daily)} days of sentiment computed.")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch and score Bitcoin news sentiment")
    parser.add_argument("--api_key", type=str, default="",
                        help="NewsAPI key (get free at newsapi.org)")
    parser.add_argument("--days",    type=int, default=DAYS_BACK,
                        help="Number of days back to fetch (default 30)")
    parser.add_argument("--skip_fetch", action="store_true",
                        help="Skip fetching and use existing bitcoin_headlines.csv")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    # ── Fetch or load headlines ───────────────────────────────────────────────
    if args.skip_fetch and os.path.exists(HEADLINES_PATH):
        print(f"[main] Loading cached headlines from '{HEADLINES_PATH}'")
        df = pd.read_csv(HEADLINES_PATH, parse_dates=["date"])
    else:
        if not args.api_key:
            if os.path.exists(HEADLINES_PATH):
                print("[main] No API key provided. Using cached headlines.")
                df = pd.read_csv(HEADLINES_PATH, parse_dates=["date"])
            else:
                print("\nERROR: No API key provided and no cached headlines found.")
                print("Please provide your NewsAPI key:")
                print("  python fetch_news_sentiment.py --api_key YOUR_KEY_HERE")
                print("\nGet a free key at: https://newsapi.org/register")
                sys.exit(1)
        else:
            print(f"[main] Fetching headlines for last {args.days} days...")
            df = fetch_headlines(args.api_key, days_back=args.days)

            if df.empty:
                print("[main] No headlines fetched. Check your API key and connection.")
                sys.exit(1)

            # Save raw headlines
            df.to_csv(HEADLINES_PATH, index=False)
            print(f"[main] Headlines saved → '{HEADLINES_PATH}'")

    print(f"[main] {len(df)} headlines loaded.")

    # ── Score with FinBERT ────────────────────────────────────────────────────
    pipe   = load_finbert()
    df_scored = score_headlines(df, pipe)

    # Save scored headlines
    df_scored.to_csv(HEADLINES_PATH, index=False)
    print(f"[main] Scored headlines saved → '{HEADLINES_PATH}'")

    # ── Aggregate to daily ────────────────────────────────────────────────────
    daily = aggregate_daily(df_scored)
    daily.to_csv(SENTIMENT_PATH, index=False)
    print(f"[main] Daily sentiment saved → '{SENTIMENT_PATH}'")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  Sentiment Summary")
    print("="*50)
    print(f"  Headlines scored : {len(df_scored)}")
    print(f"  Days covered     : {len(daily)}")
    print(f"  Mean score       : {daily['sentiment_score'].mean():+.4f}")
    print(f"  Min / Max        : {daily['sentiment_score'].min():+.4f} / {daily['sentiment_score'].max():+.4f}")
    bullish = (daily["sentiment_score"] >  0.1).mean() * 100
    bearish = (daily["sentiment_score"] < -0.1).mean() * 100
    neutral = 100 - bullish - bearish
    print(f"  Bullish days     : {bullish:.1f}%")
    print(f"  Bearish days     : {bearish:.1f}%")
    print(f"  Neutral days     : {neutral:.1f}%")
    print("="*50)
    print(f"\nNext step: python phase4_evaluation.py")


if __name__ == "__main__":
    main()
