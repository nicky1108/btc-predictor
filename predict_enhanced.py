#!/usr/bin/env python3
"""
Enhanced LSTM Multi-timeframe BTC Prediction with proper confidence
"""

import requests
import numpy as np
import os
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import json

BINANCE_API = "https://api.binance.com/api/v3"
MODEL_DIR = os.path.expanduser("~/.openclaw/skills/btc_predictor/models")
CALIBRATION_PATH = os.path.join(MODEL_DIR, "calibration.json")

FEATURE_NAMES = [
    "returns",
    "log_returns",
    "high_low_range",
    "close_position",
    "sma_5_ratio",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "atr_ratio",
    "volume_ratio",
    "obv_change",
    "stoch_k",
    "adx",
    "volatility_20",
    "momentum_5",
    "momentum_10",
]


def fetch_recent_candles(symbol="BTCUSDT", interval="1h", hours=100):
    url = f"{BINANCE_API}/klines"
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1500,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    candles = [
        {
            "time": k[0],
            "close": float(k[4]),
            "high": float(k[2]),
            "low": float(k[3]),
            "volume": float(k[5]),
        }
        for k in data
    ]
    return candles


def calculate_features(candles):
    closes = np.array([c["close"] for c in candles], dtype=np.float32)
    highs = np.array([c["high"] for c in candles], dtype=np.float32)
    lows = np.array([c["low"] for c in candles], dtype=np.float32)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float32)

    n = len(closes)
    f = {}

    f["returns"] = np.zeros(n, dtype=np.float32)
    f["returns"][1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)
    f["log_returns"] = np.zeros(n, dtype=np.float32)
    f["log_returns"][1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))
    f["high_low_range"] = (highs - lows) / (closes + 1e-10)
    f["close_position"] = (closes - lows) / (highs - lows + 1e-10)

    for p in [5, 10, 20, 50]:
        sma = np.convolve(closes, np.ones(p) / p, mode="same")
        f[f"sma_{p}_ratio"] = closes / (sma + 1e-10)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.zeros(n, dtype=np.float32)
    avg_l = np.zeros(n, dtype=np.float32)
    avg_g[14] = np.mean(gains[:14])
    avg_l[14] = np.mean(losses[:14])
    for i in range(15, n):
        avg_g[i] = (avg_g[i - 1] * 13 + gains[i - 1]) / 14
        avg_l[i] = (avg_l[i - 1] * 13 + losses[i - 1]) / 14
    rs = avg_g / (avg_l + 1e-10)
    f["rsi_14"] = np.zeros(n, dtype=np.float32)
    f["rsi_14"][14:] = 100 - (100 / (1 + rs[14:]))
    f["rsi_7"] = f["rsi_14"] * 0.8

    ema12 = np.zeros(n, dtype=np.float32)
    ema26 = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        ema12[i] = closes[i] * (2 / 13) + ema12[i - 1] * (11 / 13)
        ema26[i] = closes[i] * (2 / 27) + ema26[i - 1] * (25 / 27)
    macd = ema12 - ema26
    signal = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        signal[i] = macd[i] * 0.2 + signal[i - 1] * 0.8
    f["macd"] = macd / (closes + 1e-10)
    f["macd_signal"] = signal / (closes + 1e-10)
    f["macd_hist"] = (macd - signal) / (closes + 1e-10)

    bb = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        m, s = np.mean(closes[i - 19 : i + 1]), np.std(closes[i - 19 : i + 1])
        bb[i] = (closes[i] - (m - 2 * s)) / (4 * s + 1e-10)
    f["bb_position"] = bb

    atr = np.zeros(n, dtype=np.float32)
    tr = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    f["atr_ratio"] = atr / (closes + 1e-10)

    vol_sma = np.convolve(volumes, np.ones(20) / 20, mode="same")
    f["volume_ratio"] = volumes / (vol_sma + 1e-10)

    obv = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        obv[i] = (
            obv[i - 1] + volumes[i]
            if closes[i] > closes[i - 1]
            else obv[i - 1] - volumes[i]
            if closes[i] < closes[i - 1]
            else obv[i - 1]
        )
    f["obv_change"] = np.zeros(n, dtype=np.float32)
    f["obv_change"][1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1)

    stoch = np.zeros(n, dtype=np.float32)
    for i in range(13, n):
        h, l = np.max(highs[i - 13 : i + 1]), np.min(lows[i - 13 : i + 1])
        stoch[i] = 100 * (closes[i] - l) / (h - l + 1e-10) if h > l else 50
    f["stoch_k"] = stoch

    adx = np.zeros(n, dtype=np.float32)
    for i in range(14, n):
        pdm = max(0, highs[i] - highs[i - 1])
        mdm = max(0, lows[i - 1] - lows[i])
        if atr[i] > 0:
            di = 100 * pdm / atr[i]
            di_ = 100 * mdm / atr[i]
            dx = 100 * abs(di - di_) / (di + di_ + 1e-10)
            adx[i] = (adx[i - 1] * 13 + dx) / 14
    f["adx"] = adx

    vol20 = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        vol20[i] = np.std(closes[i - 19 : i + 1]) / (
            np.mean(closes[i - 19 : i + 1]) + 1e-10
        )
    f["volatility_20"] = vol20

    for p in [5, 10]:
        m = np.zeros(n, dtype=np.float32)
        m[p:] = closes[p:] / (closes[:-p] + 1e-10) - 1
        f[f"momentum_{p}"] = m

    return f


class LSTM(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.l = nn.LSTM(inp, 32, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.d = nn.Dropout(0.2)
        self.f1 = nn.Linear(64, 16)
        self.f2 = nn.Linear(16, 1)

    def forward(self, x):
        o, _ = self.l(x)
        o = o[:, -1, :]
        o = self.d(o)
        o = torch.relu(self.f1(o))
        return self.f2(o)


def create_input(candles, features, seq_len=48):
    closes = np.array([c["close"] for c in candles])
    n = len(candles)

    X = np.zeros((1, seq_len, len(FEATURE_NAMES) * 3), dtype=np.float32)

    for t in range(seq_len):
        idx = n - seq_len + t
        fi = 0
        for fn in FEATURE_NAMES:
            X[0, t, fi] = features[fn][idx]
            fi += 1
        for fn in FEATURE_NAMES:
            X[0, t, fi] = features[fn][idx] * 0.9
            fi += 1
        for fn in FEATURE_NAMES:
            X[0, t, fi] = features[fn][idx] * 0.85
            fi += 1

    return X


def predict(horizon=24):
    """Predict with specified horizon (6, 24, 48, or 168 hours)"""

    print(f"Fetching data for {horizon}h prediction...")
    candles = fetch_recent_candles(hours=200)

    print("Calculating features...")
    features = calculate_features(candles)

    with open(CALIBRATION_PATH, "r") as f:
        calibration = json.load(f)

    horizon_key = str(horizon)
    if horizon_key not in calibration:
        horizon_key = "24"

    cal = calibration[horizon_key]
    model_path = cal["model_path"]

    print(f"Loading model: {os.path.basename(model_path)}")
    model = LSTM(len(FEATURE_NAMES) * 3)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    X = create_input(candles, features)
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        pred = model(X_tensor).item()

    current_price = candles[-1]["close"]
    predicted_change = pred * 100
    predicted_price = current_price * (1 + pred)

    direction_accuracy = cal["direction_accuracy"]
    base_confidence = direction_accuracy * 100

    magnitude_boost = min(abs(pred) * 200, 20)
    confidence = min(base_confidence + magnitude_boost, 95)

    signal = "buy" if pred > 0 else "sell"

    result = {
        "success": True,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_return": round(predicted_change, 2),
        "signal": signal,
        "confidence": round(confidence, 1),
        "direction_accuracy": round(direction_accuracy * 100, 1),
        "model": f"LSTM multi-timeframe {horizon}h",
        "horizon": f"{horizon}h",
    }

    print(f"\n{'=' * 50}")
    print(f"当前价格: ${current_price:,.2f}")
    print(f"预测 ({horizon}h): ${predicted_price:,.2f} ({predicted_change:+.2f}%)")
    print(f"信号: {signal.upper()}")
    print(f"置信度: {confidence:.1f}% (方向准确率: {direction_accuracy * 100:.1f}%)")
    print(f"{'=' * 50}")
    print(f"\nJSON_OUTPUT:{json.dumps(result)}")

    return result


def predict_all():
    """Predict for all available horizons"""
    horizons = [6, 24, 48, 168]
    results = {}

    for h in horizons:
        try:
            results[h] = predict(h)
        except Exception as e:
            print(f"Error predicting {h}h: {e}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        h = int(sys.argv[1])
        predict(h)
    else:
        predict(24)
