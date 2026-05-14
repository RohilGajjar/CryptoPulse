"""
generate_cache.py -- CryptoPulse
Run this to refresh all predictions before launching the dashboard.
Usage: python generate_cache.py
"""
import os, sys, pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
sys.path.insert(0, "src")

from data_pipeline    import load_data
from model_inference  import predict_next_day, predict_n_days
from signal_engine    import generate_signal, estimate_volatility

print("Generating dashboard cache...")
df       = load_data("data/BTC-USD.csv")
infer    = predict_next_day()
forecast = predict_n_days(n_days=7)
vol      = estimate_volatility(df["Close"].values)
signal   = generate_signal(infer["predicted_return"], sentiment_score=0.0, volatility=vol)

with open("models/metrics.pkl", "rb") as f:
    mdata = pickle.load(f)

with open("models/dashboard_cache.pkl", "wb") as f:
    pickle.dump({"df":df,"infer":infer,"forecast":forecast,
                 "signal":signal,"mdata":mdata}, f)

print(f"Done. Predicted price: ${infer['predicted_price']:,.2f} | Signal: {signal['signal']}")
print("Now run: streamlit run app.py")
