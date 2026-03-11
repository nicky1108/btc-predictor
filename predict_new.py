#!/usr/bin/env python3
"""
BTC Price Prediction - Simple version
"""

import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SKILL_DIR, "models", "btc_model_new.npz")


def load_model():
    if os.path.exists(MODEL_PATH):
        data = np.load(MODEL_PATH)
        return {
            "weights": data["weights"],
            "bias": float(data["bias"]),
            "feat_mean": data["feat_mean"],
            "feat_std": data["feat_std"],
        }
    return None


def fetch_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 100}

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        candles = []
        for kline in data:
            candles.append(
                {
                    "close": float(kline[4]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "open": float(kline[1]),
                    "volume": float(kline[5]),
                }
            )
        return candles
    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_features(candles):
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    # Latest values only
    c = closes[-1]
    c_prev = closes[-2] if len(closes) > 1 else c
    h = highs[-1]
    l = lows[-1]
    v = volumes[-1]

    returns = (c - c_prev) / c_prev
    log_returns = np.log(c / c_prev) if c_prev > 0 else 0
    hl_range = (h - l) / c
    oc_range = (c - closes[0]) / closes[0] if len(closes) > 0 else 0

    ma6 = np.mean(closes[-6:]) / c if len(closes) >= 6 else 1
    ma12 = np.mean(closes[-12:]) / c if len(closes) >= 12 else 1
    ma24 = np.mean(closes[-24:]) / c if len(closes) >= 24 else 1
    ma48 = np.mean(closes[-48:]) / c if len(closes) >= 48 else 1

    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 1
    rsi = (100 - (100 / (1 + rs))) / 100.0

    volatility = (
        np.std(closes[-24:]) / np.mean(closes[-24:]) if len(closes) >= 24 else 0
    )
    volume_ratio = v / np.mean(volumes[-24:]) if len(volumes) >= 24 else 1
    close_pos = (c - l) / (h - l) if h > l else 0.5
    trend_6h = (c - closes[-7]) / closes[-7] if len(closes) >= 7 else 0

    return np.array(
        [
            returns,
            log_returns,
            hl_range,
            oc_range,
            ma6,
            ma12,
            ma24,
            ma48,
            rsi,
            volatility,
            volume_ratio,
            close_pos,
            trend_6h,
        ],
        dtype=np.float32,
    )


def predict():
    print("=" * 50)
    print("BTC Price Prediction")
    print("=" * 50)

    model = load_model()
    if model is None:
        print("Error: Model not found")
        return

    candles = fetch_data()
    if not candles:
        print("Error: Could not fetch data")
        return

    current_price = candles[-1]["close"]
    features = calculate_features(candles)

    X_norm = (features - model["feat_mean"]) / model["feat_std"]
    prediction = np.dot(X_norm, model["weights"]) + model["bias"]
    prediction = np.clip(prediction, -0.5, 0.5)

    future_price = current_price * (1 + prediction)
    ret = prediction * 100

    if ret > 2.0:
        signal, confidence = "STRONG_BUY", 90
    elif ret > 0.5:
        signal, confidence = "BUY", 70
    elif ret > -0.5:
        signal, confidence = "HOLD", 50
    elif ret > -2.0:
        signal, confidence = "SELL", 70
    else:
        signal, confidence = "STRONG_SELL", 90

    print(f"\nCurrent Price: ${current_price:,.2f}")
    print(f"24h Prediction: ${future_price:,.2f}")
    print(f"Expected Return: {ret:+.2f}%")
    print(f"Signal: {signal} ({confidence}%)")

    result = {
        "success": True,
        "current_price": current_price,
        "predicted_price": future_price,
        "predicted_return": ret,
        "signal": signal,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"\nJSON_OUTPUT:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    predict()
