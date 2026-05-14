"""
signal_engine.py -- CryptoPulse Phase 3
=========================================
Rule-based hybrid signal engine.
Combines LSTM predicted return + optional sentiment + optional volatility
into a BUY / HOLD / SELL decision with confidence and plain-English explanation.

FIX 3 applied: explanation now explicitly connects predicted_return value
to the decision, formatted for easy demo and report explanation.
"""

import numpy as np

# ── Thresholds (tune here without touching logic) ─────────────────────────────
BUY_THRESHOLD    =  0.010   # +1.0% predicted return → lean BUY
SELL_THRESHOLD   = -0.010   # -1.0% predicted return → lean SELL
SENTIMENT_STRONG =  0.30
SENTIMENT_WEAK   = -0.30
SENTIMENT_WEIGHT =  0.25
VOL_HIGH         =  0.035   # > 3.5% daily std = high volatility
VOL_MEDIUM       =  0.020   # > 2.0% daily std = moderate volatility


def estimate_volatility(close_prices, window=20):
    """14 or 20-day rolling std of daily returns. Returns None if insufficient data."""
    prices = np.array(close_prices, dtype=float)
    if len(prices) < 2:
        return None
    prices  = prices[-window:]
    returns = np.diff(prices) / prices[:-1]
    return float(np.std(returns))


def blend_signal(predicted_return, sentiment_score):
    """Blend LSTM return with sentiment at SENTIMENT_WEIGHT (25%)."""
    sentiment_as_return = sentiment_score * 0.02
    return ((1 - SENTIMENT_WEIGHT) * predicted_return
            + SENTIMENT_WEIGHT * sentiment_as_return)


def generate_signal(
    predicted_return,
    sentiment_score=None,
    volatility=None,
    close_prices=None,
):
    """
    Generate BUY / HOLD / SELL from predicted_return + optional inputs.

    FIX 3: explanation now uses the actual predicted_return value and
    volatility level to form a concrete, report-ready sentence.

    Returns dict: {signal, confidence, explanation}
    """

    # Auto-compute volatility if prices provided
    if volatility is None and close_prices is not None:
        volatility = estimate_volatility(close_prices)

    # Blend with sentiment if available
    if sentiment_score is not None:
        effective_return = blend_signal(predicted_return, sentiment_score)
        used_sentiment   = True
    else:
        effective_return = predicted_return
        used_sentiment   = False

    # ── Core signal rule ─────────────────────────────────────────────────────
    if effective_return > BUY_THRESHOLD:
        signal = "BUY"
    elif effective_return < SELL_THRESHOLD:
        signal = "SELL"
    else:
        signal = "HOLD"

    # ── Confidence from margin to threshold ──────────────────────────────────
    if signal == "HOLD":
        margin = 0.0
    elif signal == "BUY":
        margin = effective_return - BUY_THRESHOLD
    else:
        margin = abs(effective_return) - abs(SELL_THRESHOLD)

    if signal == "HOLD":
        confidence = "Low"
    elif margin > 0.015:
        confidence = "High"
    elif margin > 0.005:
        confidence = "Medium"
    else:
        confidence = "Low"

    # ── Sentiment adjustment ──────────────────────────────────────────────────
    sentiment_note = ""
    if used_sentiment:
        if signal == "BUY" and sentiment_score >= SENTIMENT_STRONG:
            confidence     = _upgrade(confidence)
            sentiment_note = f" News sentiment is bullish ({sentiment_score:+.2f}), confirming the signal."
        elif signal == "SELL" and sentiment_score <= SENTIMENT_WEAK:
            confidence     = _upgrade(confidence)
            sentiment_note = f" News sentiment is bearish ({sentiment_score:+.2f}), confirming the signal."
        elif signal == "BUY" and sentiment_score <= SENTIMENT_WEAK:
            confidence     = _downgrade(confidence)
            sentiment_note = f" However, news sentiment is bearish ({sentiment_score:+.2f}), contradicting the signal."
        elif signal == "SELL" and sentiment_score >= SENTIMENT_STRONG:
            confidence     = _downgrade(confidence)
            sentiment_note = f" However, news sentiment is bullish ({sentiment_score:+.2f}), contradicting the signal."

    # ── Volatility adjustment ─────────────────────────────────────────────────
    vol_label = ""
    vol_note  = ""
    if volatility is not None:
        vol_pct = volatility * 100
        if volatility > VOL_HIGH:
            confidence = _downgrade(confidence)
            vol_label  = "high"
            vol_note   = f" Market volatility is elevated ({vol_pct:.1f}% daily std), reducing signal reliability."
        elif volatility > VOL_MEDIUM:
            vol_label  = "moderate"
            vol_note   = f" Market volatility is moderate ({vol_pct:.1f}% daily std)."
        else:
            vol_label  = "low"

    # ── FIX 3: Explicit explanation connecting prediction → decision ──────────
    ret_pct = predicted_return * 100

    if signal == "BUY":
        explanation = (
            f"Model predicts a next-day return of {ret_pct:+.2f}%, "
            f"which exceeds the BUY threshold of +{BUY_THRESHOLD*100:.1f}%. "
            f"A BUY signal is generated with {confidence.lower()} confidence."
        )
    elif signal == "SELL":
        explanation = (
            f"Model predicts a next-day return of {ret_pct:+.2f}%, "
            f"which falls below the SELL threshold of {SELL_THRESHOLD*100:.1f}%. "
            f"A SELL signal is generated with {confidence.lower()} confidence."
        )
    else:
        # HOLD — most common case — make it maximally clear
        explanation = (
            f"Model predicts a next-day return of {ret_pct:+.2f}%, "
            f"which is within the HOLD zone "
            f"({SELL_THRESHOLD*100:.1f}% to +{BUY_THRESHOLD*100:.1f}%). "
            f"No strong directional edge detected — HOLD is recommended."
        )

    # Append sentiment and volatility notes
    explanation += sentiment_note + vol_note

    return {
        "signal"     : signal,
        "confidence" : confidence,
        "explanation": explanation,
    }


def _upgrade(confidence):
    tiers = ["Low", "Medium", "High"]
    return tiers[min(tiers.index(confidence) + 1, 2)]

def _downgrade(confidence):
    tiers = ["Low", "Medium", "High"]
    return tiers[max(tiers.index(confidence) - 1, 0)]


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def show(label, result):
        print(f"\n  [{label}]")
        print(f"  Signal     : {result['signal']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Explanation: {result['explanation']}")

    print("=" * 65)
    print("signal_engine.py self-test (FIX 3 — explicit explanations)")
    print("=" * 65)

    show("Real model output: -0.11%",
         generate_signal(-0.0011, volatility=0.022))

    show("Strong BUY: +3.5%, bullish sentiment",
         generate_signal(0.035, sentiment_score=0.6))

    show("SELL: -2.1%, confirmed by sentiment",
         generate_signal(-0.021, sentiment_score=-0.7))

    show("HOLD: +0.3%, high volatility",
         generate_signal(0.003, volatility=0.042))

    # Verify HOLD explanation contains the actual return value
    r = generate_signal(-0.0011)
    assert "-0.11%" in r["explanation"], "Explanation must contain actual return"
    assert "HOLD" in r["explanation"], "HOLD explanation must mention HOLD"

    r2 = generate_signal(0.035)
    assert "BUY" in r2["explanation"]
    assert "+3.50%" in r2["explanation"]

    print("\nAll assertions passed.")
    print("=" * 65)
