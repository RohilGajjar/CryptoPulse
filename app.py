"""
app.py -- CryptoPulse Phase 3 Dashboard (with live price ticker)
=================================================================
Real-time BTC price is fetched from CoinGecko API (free, no key needed)
and displayed separately from the LSTM prediction cache.

Architecture:
    Live price  -> CoinGecko API (refreshes every 60s via st.fragment)
    LSTM output -> dashboard_cache.pkl (pre-computed, stable)

This keeps TensorFlow completely out of Streamlit while still showing
a live price in the header and top metric card.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="CryptoPulse | BTC Intelligence",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#080C14;color:#E8EDF5;}
.stApp{background:#080C14;}
section[data-testid="stSidebar"]{background:#0D1220!important;border-right:1px solid #1C2333;}
section[data-testid="stSidebar"] *{color:#A8B4CC!important;}
#MainMenu,footer{visibility:hidden;}
.header{background:linear-gradient(135deg,#0D1220,#111827);border:1px solid #1C2333;
  border-radius:16px;padding:24px 32px;margin-bottom:20px;position:relative;overflow:hidden;}
.header::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,#F7931A,#FFB347,#F7931A);}
.htitle{font-size:2rem;font-weight:800;color:#FFF;margin:0;}
.htitle span{color:#F7931A;}
.hsub{font-family:'Space Mono',monospace;font-size:.6rem;color:#4A5568;
  letter-spacing:.15em;text-transform:uppercase;margin-top:4px;}
.live-badge{display:inline-block;background:#064E3B;color:#10B981;
  font-family:'Space Mono',monospace;font-size:.55rem;padding:2px 8px;
  border-radius:100px;font-weight:700;letter-spacing:.08em;margin-left:8px;
  animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.5;}}
.card{background:#0D1220;border:1px solid #1C2333;border-radius:12px;
  padding:16px 20px;margin-bottom:8px;}
.card.pos{border-left:3px solid #10B981;}
.card.neg{border-left:3px solid #EF4444;}
.card.ora{border-left:3px solid #F7931A;}
.card.blu{border-left:3px solid #3B82F6;}
.card.gry{border-left:3px solid #4A5568;}
.lbl{font-family:'Space Mono',monospace;font-size:.58rem;color:#4A5568;
  letter-spacing:.1em;text-transform:uppercase;margin-bottom:5px;}
.val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:700;color:#FFF;line-height:1;}
.dlt{font-family:'Space Mono',monospace;font-size:.68rem;margin-top:4px;}
.up{color:#10B981;}.dn{color:#EF4444;}.nt{color:#F7931A;}
.sec{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:.18em;
  text-transform:uppercase;color:#4A5568;margin-bottom:8px;padding-bottom:6px;
  border-bottom:1px solid #1C2333;}
.signal-box{background:linear-gradient(135deg,#0D1220,#111827);
  border:1px solid #1C2333;border-radius:14px;padding:20px;text-align:center;}
.expl{background:#0D1220;border:1px solid #1C2333;border-left:3px solid #F7931A;
  border-radius:8px;padding:12px 16px;font-family:'Space Mono',monospace;
  font-size:.62rem;color:#A8B4CC;line-height:1.6;margin-top:8px;}
.placeholder{background:#0D1220;border:1px dashed #2D3748;border-radius:10px;
  padding:16px;text-align:center;font-family:'Space Mono',monospace;
  font-size:.62rem;color:#4A5568;line-height:1.8;}
.note-box{background:#111827;border:1px solid #2D3748;border-radius:8px;
  padding:10px 14px;font-family:'Space Mono',monospace;font-size:.58rem;
  color:#6B7280;line-height:1.6;margin-top:8px;}
.news-item{background:#0D1220;border:1px solid #1C2333;border-radius:10px;
  padding:10px 14px;margin-bottom:8px;}
.news-item.pos{border-left:3px solid #10B981;}
.news-item.neg{border-left:3px solid #EF4444;}
.news-item.neu{border-left:3px solid #F7931A;}
.news-hl{font-size:.8rem;color:#CBD5E0;line-height:1.4;margin-bottom:4px;}
.news-meta{font-family:'Space Mono',monospace;font-size:.56rem;color:#4A5568;}
.badge{font-family:'Space Mono',monospace;font-size:.5rem;padding:2px 7px;
  border-radius:100px;font-weight:700;}
.bp{background:#064E3B;color:#10B981;}
.bn{background:#450A0A;color:#EF4444;}
.bo{background:#1A1500;color:#F7931A;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PRICE FETCHER
# Uses CoinGecko public API -- free, no key, no rate limit for simple queries.
# Falls back to cache price if the API is unavailable.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_price() -> dict:
    """
    Fetch current BTC price from CoinGecko API.
    Returns dict with price, change_24h, and timestamp.
    Returns None if request fails (fallback to cache price used in UI).
    """
    try:
        url    = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids"               : "bitcoin",
            "vs_currencies"     : "usd",
            "include_24hr_change": "true",
        }
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            data     = resp.json()["bitcoin"]
            return {
                "price"     : data["usd"],
                "change_24h": data["usd_24h_change"],
                "fetched_at": datetime.now().strftime("%H:%M:%S"),
                "live"      : True,
            }
    except Exception:
        pass
    return {"price": None, "change_24h": None, "fetched_at": None, "live": False}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CACHE (LSTM predictions, signal, metrics)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_cache():
    path = "models/dashboard_cache.pkl"
    if not os.path.exists(path):
        st.error("Cache not found. Run: python generate_cache.py")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)


def card(col, cls, label, val, dlt="", dcls="nt"):
    with col:
        st.markdown(f"""
        <div class='card {cls}'>
          <div class='lbl'>{label}</div>
          <div class='val'>{val}</div>
          {"<div class='dlt "+dcls+"'>"+dlt+"</div>" if dlt else ""}
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 18px'>
      <div style='font-size:2rem'>₿</div>
      <div style='font-family:Syne,sans-serif;font-weight:800;
                  font-size:1.05rem;color:#F7931A'>CryptoPulse</div>
      <div style='font-family:Space Mono,monospace;font-size:.52rem;
                  color:#4A5568;letter-spacing:.1em'>
        BITCOIN INTELLIGENCE · CECS 551</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    lookback_opt = st.selectbox(
        "Price History", ["30 Days","90 Days","180 Days","1 Year"], index=2)
    st.markdown("---")
    show_ema   = st.checkbox("EMA 20 / 50",     value=True)
    show_bb    = st.checkbox("Bollinger Bands",  value=True)
    show_pred  = st.checkbox("LSTM Predictions", value=True)
    show_fcast = st.checkbox("Forecast Cone",    value=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:.6rem;
                color:#4A5568;line-height:2.1'>
    Model &#8194;&#8194; LSTM (2-layer)<br>
    Target &#8194;&#8194; Daily Return<br>
    Lookback &#8194; 60 days<br>
    Features &#8194; 8 (Return+TA)<br>
    Split &#8194;&#8194;&#8194; 80/20 chron.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Refresh predictions:")
    st.code("python generate_cache.py", language="bash")
    st.caption("Live price refreshes every 60s automatically.")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

try:
    cache   = load_cache()
    df      = cache["df"]
    infer   = cache["infer"]
    forecast= cache["forecast"]
    signal  = cache["signal"]
    mdata   = cache["mdata"]
except Exception as e:
    st.error(f"Error loading cache: {e}")
    st.stop()

lkb_map    = {"30 Days":30,"90 Days":90,"180 Days":180,"1 Year":365}
dv         = df.tail(lkb_map[lookback_opt]).copy()

# Values from LSTM cache (stable, not live)
pred_price  = infer["predicted_price"]
pred_return = infer["predicted_return"]
change_pct  = infer["change_pct"]
direction   = infer["direction"]
last_date   = infer["last_date"]
rsi_val     = float(df["RSI"].iloc[-1])
vol_val     = float(df["Volume"].iloc[-1])
cache_price = infer["last_close"]   # price from CSV (may be stale)

sig      = signal["signal"]
conf     = signal["confidence"]
expl     = signal["explanation"]
sig_col  = "#10B981" if sig=="BUY" else "#EF4444" if sig=="SELL" else "#F7931A"
sig_bg   = "#064E3B" if sig=="BUY" else "#450A0A" if sig=="SELL" else "#1A1500"
sig_arr  = "▲" if sig=="BUY" else "▼" if sig=="SELL" else "—"
sig_cls  = "pos" if sig=="BUY" else "neg" if sig=="SELL" else "ora"

rm      = mdata.get("return_metrics", {})
mae_v   = f"{rm.get('MAE',0):.4f}"
rmse_v  = f"{rm.get('RMSE',0):.4f}"
dir_v   = f"{rm.get('Directional Accuracy',0):.1f}%"

test_dates  = pd.to_datetime(mdata.get("test_dates", []))
pred_prices = np.array(mdata.get("y_pred_price", []))
recent_rets = df["Return"].dropna().tail(14).values
vol_14d     = float(np.std(recent_rets)) * 100

fc           = forecast
fcast_dates  = [datetime.now() + timedelta(days=f["day"]) for f in fc]
fcast_prices = [f["predicted_price"] for f in fc]
fcast_upper  = [f["predicted_price"] * (1 + 0.025*f["day"]) for f in fc]
fcast_lower  = [f["predicted_price"] * (1 - 0.025*f["day"]) for f in fc]
low_variance = all(abs(f["predicted_return"]) < 0.005 for f in fc)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PRICE SECTION (auto-refreshes every 60 seconds)
# st.fragment with run_every="60s" re-runs only this block,
# leaving the rest of the dashboard (charts, tables) untouched.
# ─────────────────────────────────────────────────────────────────────────────

@st.fragment(run_every="60s")
def live_price_section():
    """
    This block refreshes every 60 seconds automatically.
    It fetches the live BTC price from CoinGecko and renders
    the header + metric cards using the live price.
    Falls back to the cache price if the API is unavailable.
    """
    live = fetch_live_price()

    # Decide which price to show
    if live["live"] and live["price"]:
        current_price = live["price"]
        chg_24h       = live["change_24h"]
        price_label   = f"LIVE · {live['fetched_at']}"
        is_live       = True
    else:
        # Fallback to CSV price
        current_price = cache_price
        chg_24h       = (cache_price - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100
        price_label   = f"CACHED · {last_date}"
        is_live       = False

    cc = "#10B981" if chg_24h >= 0 else "#EF4444"
    cs = "▲" if chg_24h >= 0 else "▼"
    live_badge = "<span class='live-badge'>● LIVE</span>" if is_live else \
                 "<span style='font-family:Space Mono,monospace;font-size:.55rem;" \
                 "color:#4A5568;'>(cached)</span>"

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='header'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start'>
        <div>
          <h1 class='htitle'><span>Crypto</span>Pulse</h1>
          <p class='hsub'>LSTM + Signal Engine · Phase 3 MVP · Real Model Output</p>
        </div>
        <div style='text-align:right'>
          <div style='font-family:Space Mono,monospace;font-size:.58rem;color:#4A5568'>
            BTC/USD {live_badge}</div>
          <div style='font-family:Syne,sans-serif;font-size:1.9rem;
                      font-weight:800;color:#F7931A'>${current_price:,.0f}</div>
          <div style='font-family:Space Mono,monospace;font-size:.7rem;color:{cc}'>
            {cs} {abs(chg_24h):.2f}% (24h)</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Metric cards (top row) ────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)

    # Card 1: Live BTC price
    with c1:
        st.markdown(f"""
        <div class='card {"pos" if chg_24h>=0 else "neg"}'>
          <div class='lbl'>BTC / USD {"🟢 LIVE" if is_live else "📁 CACHED"}</div>
          <div class='val'>${current_price:,.0f}</div>
          <div class='dlt {"up" if chg_24h>=0 else "dn"}'>
            {"▲" if chg_24h>=0 else "▼"} {abs(chg_24h):.2f}% 24h</div>
        </div>""", unsafe_allow_html=True)

    # Card 2: LSTM predicted price (from cache)
    with c2:
        st.markdown(f"""
        <div class='card {"pos" if pred_return>=0 else "neg"}'>
          <div class='lbl'>LSTM PREDICTION</div>
          <div class='val'>${pred_price:,.0f}</div>
          <div class='dlt {"up" if pred_return>=0 else "dn"}'>
            Predicted return: {change_pct}</div>
        </div>""", unsafe_allow_html=True)

    # Card 3: RSI
    with c3:
        rsi_cls = "ora" if 30<=rsi_val<=70 else ("neg" if rsi_val>70 else "pos")
        st.markdown(f"""
        <div class='card {rsi_cls}'>
          <div class='lbl'>RSI (14)</div>
          <div class='val'>{rsi_val:.1f}</div>
          <div class='dlt {"dn" if rsi_val>70 else "up" if rsi_val<30 else "nt"}'>
            {"Overbought" if rsi_val>70 else "Oversold" if rsi_val<30 else "Normal"}</div>
        </div>""", unsafe_allow_html=True)

    # Card 4: Hybrid signal
    with c4:
        st.markdown(f"""
        <div class='card {sig_cls}'>
          <div class='lbl'>HYBRID SIGNAL</div>
          <div class='val'>{sig}</div>
          <div class='dlt {"up" if sig=="BUY" else "dn" if sig=="SELL" else "nt"}'>
            Confidence: {conf}</div>
        </div>""", unsafe_allow_html=True)

    # Card 5: Volatility
    with c5:
        st.markdown(f"""
        <div class='card gry'>
          <div class='lbl'>VOLATILITY (14D)</div>
          <div class='val'>{vol_14d:.2f}%</div>
          <div class='dlt nt'>Daily return std dev</div>
        </div>""", unsafe_allow_html=True)

# Call the live section
live_price_section()
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHART (static -- doesn't need to refresh every 60s)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<p class='sec'>Price Chart · Real OHLCV · LSTM Test Overlay · Forecast</p>",
            unsafe_allow_html=True)

fig = make_subplots(rows=3, cols=1, row_heights=[0.60,0.22,0.18],
                    shared_xaxes=True, vertical_spacing=0.03)

fig.add_trace(go.Candlestick(
    x=dv["Date"], open=dv["Open"], high=dv["High"],
    low=dv["Low"], close=dv["Close"], name="BTC/USD",
    increasing=dict(line=dict(color="#10B981",width=1), fillcolor="#10B981"),
    decreasing=dict(line=dict(color="#EF4444",width=1), fillcolor="#EF4444"),
), row=1, col=1)

if show_ema:
    fig.add_trace(go.Scatter(x=dv["Date"],y=dv["EMA20"],name="EMA 20",
        line=dict(color="#F7931A",width=1.5,dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dv["Date"],y=dv["EMA50"],name="EMA 50",
        line=dict(color="#3B82F6",width=1.5,dash="dot")), row=1, col=1)

if show_bb:
    fig.add_trace(go.Scatter(x=dv["Date"],y=dv["BB_upper"],name="BB Upper",
        line=dict(color="rgba(150,120,255,0.4)",width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dv["Date"],y=dv["BB_lower"],name="BB Lower",
        line=dict(color="rgba(150,120,255,0.4)",width=1),
        fill="tonexty", fillcolor="rgba(150,120,255,0.05)"), row=1, col=1)

if show_pred and len(test_dates) > 0:
    mask = pd.Series(test_dates) >= dv["Date"].iloc[0]
    fd = [d for d,m in zip(test_dates,mask) if m]
    fp = [p for p,m in zip(pred_prices,mask) if m]
    if fd:
        fig.add_trace(go.Scatter(x=fd, y=fp, name="LSTM (test set)",
            line=dict(color="#A78BFA",width=1.5,dash="dash")), row=1, col=1)

if show_fcast:
    fig.add_trace(go.Scatter(
        x=fcast_dates+fcast_dates[::-1], y=fcast_upper+fcast_lower[::-1],
        fill="toself", fillcolor="rgba(247,147,26,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="Forecast Band",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=fcast_dates,y=fcast_prices,name="Forecast",
        line=dict(color="#F7931A",width=2,dash="dot")), row=1, col=1)

vc = ["#10B981" if dv["Close"].iloc[i]>=dv["Open"].iloc[i]
      else "#EF4444" for i in range(len(dv))]
fig.add_trace(go.Bar(x=dv["Date"],y=dv["Volume"],
    name="Volume",marker_color=vc,opacity=0.5), row=2, col=1)

mc = ["#10B981" if v>=0 else "#EF4444" for v in dv["MACD_hist"]]
fig.add_trace(go.Bar(x=dv["Date"],y=dv["MACD_hist"],
    name="MACD Hist",marker_color=mc,opacity=0.7), row=3, col=1)
fig.add_trace(go.Scatter(x=dv["Date"],y=dv["MACD"],name="MACD",
    line=dict(color="#3B82F6",width=1.2)), row=3, col=1)
fig.add_trace(go.Scatter(x=dv["Date"],y=dv["MACD_signal"],name="Signal",
    line=dict(color="#F7931A",width=1.2)), row=3, col=1)

fig.update_layout(
    height=600, plot_bgcolor="#080C14", paper_bgcolor="#080C14",
    font=dict(family="Space Mono",color="#4A5568",size=10),
    legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9,color="#6B7280"),
                orientation="h",y=1.02,x=0),
    margin=dict(l=0,r=0,t=20,b=0),
    xaxis_rangeslider_visible=False,
)
for i in [1,2,3]:
    fig.update_xaxes(showgrid=False,zeroline=False,
                     linecolor="#1C2333",color="#4A5568",row=i,col=1)
    fig.update_yaxes(showgrid=True,gridcolor="#0F1929",zeroline=False,
                     linecolor="#1C2333",color="#4A5568",row=i,col=1)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM ROW
# ─────────────────────────────────────────────────────────────────────────────

left, mid, right = st.columns([1.1, 1.3, 1.6])

with left:
    st.markdown("<p class='sec'>RSI Momentum Gauge</p>", unsafe_allow_html=True)
    rc = "#EF4444" if rsi_val>70 else "#10B981" if rsi_val<30 else "#F7931A"
    rg = go.Figure(go.Indicator(
        mode="gauge+number", value=rsi_val,
        number=dict(font=dict(family="Syne",size=32,color="#FFF")),
        gauge=dict(
            axis=dict(range=[0,100],tickfont=dict(color="#4A5568",size=9)),
            bar=dict(color=rc,thickness=0.25),
            bgcolor="#0D1220", bordercolor="#1C2333",
            steps=[dict(range=[0,30],  color="#0A1A10"),
                   dict(range=[30,70], color="#111827"),
                   dict(range=[70,100],color="#1A0A0A")],
        ),
    ))
    rg.update_layout(height=200,paper_bgcolor="#0D1220",plot_bgcolor="#0D1220",
                     font=dict(color="#4A5568"),margin=dict(l=20,r=20,t=10,b=10))
    st.plotly_chart(rg, use_container_width=True)
    rm_lbl = ("OVERBOUGHT" if rsi_val>70 else "OVERSOLD" if rsi_val<30
              else "NEUTRAL — Trending zone")
    st.markdown(f"<div style='text-align:center;font-family:Space Mono,monospace;"
                f"font-size:.56rem;color:{rc};margin-top:-8px'>{rm_lbl}</div>",
                unsafe_allow_html=True)

with mid:
    st.markdown("<p class='sec'>Decision Signal · signal_engine.py</p>",
                unsafe_allow_html=True)
    ret_color = "#10B981" if pred_return>=0 else "#EF4444"
    st.markdown(f"""
    <div style='background:#0D1220;border:1px solid #1C2333;border-radius:8px;
                padding:10px 16px;font-family:Space Mono,monospace;font-size:.7rem;
                margin-bottom:8px;text-align:center'>
      PREDICTED NEXT-DAY RETURN &nbsp;·&nbsp;
      <span style='color:{ret_color};font-weight:700;font-size:.85rem'>
        {change_pct}</span>
    </div>
    <div class='signal-box'>
      <div style='font-family:Space Mono,monospace;font-size:.55rem;color:#4A5568;
                  letter-spacing:.1em;margin-bottom:6px'>LSTM + RULE-BASED FUSION</div>
      <div style='font-family:Syne,sans-serif;font-size:2.8rem;font-weight:800;
                  color:{sig_col};margin-bottom:4px'>{sig}</div>
      <div style='font-family:Space Mono,monospace;font-size:.58rem;color:#4A5568;
                  margin-bottom:10px'>
        Confidence: <span style='color:{sig_col}'>{conf}</span>
        &nbsp;·&nbsp; Vol: {vol_14d:.2f}% (14d)</div>
      <div style='display:inline-block;background:{sig_bg};color:{sig_col};
                  font-family:Space Mono,monospace;font-size:.56rem;padding:5px 14px;
                  border-radius:6px;letter-spacing:.08em;font-weight:700'>
        {sig_arr} {sig} SIGNAL</div>
    </div>
    <div class='expl'><span style='color:#F7931A;font-weight:700'>Reason: </span>{expl}</div>
    """, unsafe_allow_html=True)

    st.markdown("<br><p class='sec'>7-Day LSTM Forecast</p>", unsafe_allow_html=True)
    if low_variance:
        st.markdown("""<div class='note-box'>⚠ Low return variance in forecast —
        reflects noisy market regime.</div>""", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{
        "Day": f"Day {f['day']}",
        "Price": f"${f['predicted_price']:,.2f}",
        "Return": f['change_pct'],
    } for f in fc]), use_container_width=True, hide_index=True)

with right:
    st.markdown("<p class='sec'>Sentiment · Phase 3 Placeholder</p>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='placeholder'>
      <div style='font-size:1.3rem;margin-bottom:6px'>📰</div>
      <div style='color:#F7931A;font-weight:700;margin-bottom:8px'>
        SENTIMENT MODULE — PHASE 4</div>
      <div style='text-align:left;line-height:2'>
        FinBERT scoring integrated in Phase 4.<br>
        Module: <span style='color:#A8B4CC'>sentiment_pipeline.py</span><br>
        Status: <span style='color:#F7931A'>⏳ Pending full integration</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br><p class='sec'>Sample Headlines (Static)</p>",
                unsafe_allow_html=True)
    HEADLINES = [
        ("Bitcoin ETF inflows surge to record $1.2B",      "pos","Reuters",    "2h ago"),
        ("Fed signals rate cuts, boosting crypto markets",  "pos","Bloomberg",  "4h ago"),
        ("MicroStrategy buys additional $500M in Bitcoin",  "pos","Forbes",     "8h ago"),
        ("Regulatory uncertainty grows in Congress",        "neg","WSJ",        "12h ago"),
        ("Whale transfers 2,000 BTC to exchange",           "neg","CryptoSlate","14h ago"),
        ("Bitcoin mining difficulty hits all-time high",    "neu","The Block",  "10h ago"),
    ]
    BADGE = {"pos":("BULLISH","bp"),"neg":("BEARISH","bn"),"neu":("NEUTRAL","bo")}
    for hl, sent, src, ago in HEADLINES:
        btxt, bcls = BADGE[sent]
        st.markdown(f"""
        <div class='news-item {sent}'>
          <div class='news-hl'>{hl}</div>
          <div class='news-meta'>{src} · {ago} &nbsp;
            <span class='badge {bcls}'>{btxt}</span></div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br><p class='sec'>Return-Based Test Metrics · 20% Held-Out Test Set</p>",
            unsafe_allow_html=True)
p1,p2,p3,p4,p5,p6 = st.columns(6)
for col,cls,lbl,v,sub in [
    (p1,"blu","MAE (Return)",      mae_v,             "Avg daily return error"),
    (p2,"blu","RMSE (Return)",     rmse_v,            "Return RMSE"),
    (p3,"pos","Dir. Accuracy",     dir_v,             "Directional correct"),
    (p4,"gry","Volatility (14D)",  f"{vol_14d:.2f}%", "Daily return std"),
    (p5,"ora","Model",             "LSTM (2-layer)",  "60-day lookback"),
    (p6,"blu","Dataset",           f"{len(df):,} rows","Jan 2018 → present"),
]:
    with col:
        st.markdown(f"""
        <div class='card {cls}' style='text-align:center'>
          <div class='lbl'>{lbl}</div>
          <div style='font-family:Syne,sans-serif;font-size:1.1rem;
                      font-weight:700;color:#FFF'>{v}</div>
          <div style='font-family:Space Mono,monospace;font-size:.54rem;
                      color:#4A5568;margin-top:3px'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown(
    "<br><div style='text-align:center;font-family:Space Mono,monospace;"
    "font-size:.52rem;color:#1C2333'>CRYPTOPULSE · CECS 551 · "
    "FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE</div>",
    unsafe_allow_html=True)
