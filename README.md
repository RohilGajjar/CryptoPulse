# CryptoPulse

AI-powered Bitcoin decision dashboard for CECS 551.

## Features
- LSTM-based return prediction
- Explainable BUY / HOLD / SELL signals
- Real-time BTC price updates
- Phase 4 evaluation experiments

## Run Demo

```bash
python run_cryptopulse.py --mode demo

## Project Structure
    src/ → core AI modules
    app.py → Streamlit dashboard
    generate_cache.py → inference pipeline
    phase4_evaluation.py → evaluation experiments
    
## Key Insight
    Short-term Bitcoin prediction is highly noisy (~50% accuracy).
    System performance is limited by feature quality, not model architecture.

## Master Runner (Recommended):

        # Every time you demo — refreshes data + cache + launches dashboard (~2 min)
        python run_cryptopulse.py --mode demo

        # First-time setup — trains both models + sentiment + cache (~45 min)
        python run_cryptopulse.py --mode setup --api_key YOUR_NEWSAPI_KEY

        # Run Phase 4 evaluation only
        python run_cryptopulse.py --mode eval

        # Full pipeline + dashboard launch
        python run_cryptopulse.py --mode full --api_key YOUR_NEWSAPI_KEY

## Manual Step-by-Step

        # 1. Download BTC data
        python download_data.py

        # 2. Train 1-day LSTM (10-15 min)
        python src/train_lstm.py data/BTC-USD.csv

        # 3. Train 3-day LSTM experiment (10-15 min)
        python src/train_lstm_3day.py data/BTC-USD.csv

        # 4. Fetch and score headlines with FinBERT
        python fetch_news_sentiment.py --api_key YOUR_NEWSAPI_KEY

        # 5. Fix sentiment date alignment
        python fix_sentiment_dates.py

        # 6. Generate dashboard cache
        python generate_cache.py

        # 7. Launch dashboard
        streamlit run app.py

## Evaluation Scritps

        # Phase 4 dual sentiment evaluation
        python phase4_evaluation.py

        # 1-day vs 3-day horizon comparison
        python compare_horizon_results.py

