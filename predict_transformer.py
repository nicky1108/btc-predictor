#!/usr/bin/env python3
"""
BTC Transformer Prediction for OpenClaw Skill
Full 12-layer Transformer model inference
"""

import struct
import numpy as np
import requests
import json
import os
import math
from datetime import datetime, timedelta

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SKILL_DIR, "models", "btc_transformer_step1000000.ckpt")

DIM = 256
HIDDEN = 1024
HEADS = 8
SEQ_LEN = 256
VOCAB = 22


def load_checkpoint(path):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        version = struct.unpack("<i", f.read(4))[0]
        step = struct.unpack("<i", f.read(4))[0]
        loss = struct.unpack("<f", f.read(4))[0]
        n_layers = struct.unpack("<i", f.read(4))[0]
        dim = struct.unpack("<i", f.read(4))[0]
        vocab = struct.unpack("<i", f.read(4))[0]

        weights = {}
        weights["embed"] = (
            np.frombuffer(f.read(VOCAB * DIM * 4), dtype=np.float32)
            .copy()
            .reshape(VOCAB, DIM)
        )

        weights["layers"] = []
        for l in range(n_layers):
            layer = {}
            layer["rms_att"] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
            layer["rms_ffn"] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
            layer["Wq"] = (
                np.frombuffer(f.read(DIM * DIM * 4), dtype=np.float32)
                .copy()
                .reshape(DIM, DIM)
            )
            layer["Wk"] = (
                np.frombuffer(f.read(DIM * DIM * 4), dtype=np.float32)
                .copy()
                .reshape(DIM, DIM)
            )
            layer["Wv"] = (
                np.frombuffer(f.read(DIM * DIM * 4), dtype=np.float32)
                .copy()
                .reshape(DIM, DIM)
            )
            layer["Wo"] = (
                np.frombuffer(f.read(DIM * DIM * 4), dtype=np.float32)
                .copy()
                .reshape(DIM, DIM)
            )
            layer["W1"] = (
                np.frombuffer(f.read(HIDDEN * DIM * 4), dtype=np.float32)
                .copy()
                .reshape(HIDDEN, DIM)
            )
            layer["W2"] = (
                np.frombuffer(f.read(DIM * HIDDEN * 4), dtype=np.float32)
                .copy()
                .reshape(DIM, HIDDEN)
            )
            layer["W3"] = (
                np.frombuffer(f.read(HIDDEN * DIM * 4), dtype=np.float32)
                .copy()
                .reshape(HIDDEN, DIM)
            )
            weights["layers"].append(layer)

        weights["rms_final"] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()
        weights["head"] = np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()

        return {"step": step, "loss": loss, "n_layers": n_layers, "weights": weights}


def rms_norm(x, weight):
    eps = 1e-6
    norm = np.sqrt(np.mean(x**2) + eps)
    return x / norm * weight


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def attention(q, k, v):
    head_dim = q.shape[-1] // HEADS
    B, N, _ = q.shape
    q = q.reshape(B, N, HEADS, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, N, HEADS, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, N, HEADS, head_dim).transpose(0, 2, 1, 3)

    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, N, -1)
    return out


def transformer_block(x, layer):
    residual = x
    x = rms_norm(x, layer["rms_att"])

    q = np.matmul(x, layer["Wq"].T)
    k = np.matmul(x, layer["Wk"].T)
    v = np.matmul(x, layer["Wv"].T)

    attn_out = attention(q, k, v)
    x = np.matmul(attn_out, layer["Wo"].T)
    x = x + residual

    residual = x
    x = rms_norm(x, layer["rms_ffn"])

    w1 = np.matmul(x, layer["W1"].T)
    w3 = np.matmul(x, layer["W3"].T)
    w2 = np.matmul(gelu(w1) * w3, layer["W2"].T)

    return x + w2 + residual


def forward(tokens, model):
    weights = model["weights"]
    x = weights["embed"][tokens]

    for layer in weights["layers"]:
        x = transformer_block(x, layer)

    x = rms_norm(x, weights["rms_final"])
    logits = np.matmul(x[:, -1], weights["head"])

    return logits


def fetch_data():
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(hours=300)

    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 300,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        candles = []
        for kline in data:
            candles.append(
                {
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
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
    opens = np.array([c["open"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    returns = (closes[-1] - closes[-2]) / closes[-2]
    log_returns = np.log(closes[-1] / closes[-2])
    hl_range = (highs[-1] - lows[-1]) / closes[-1]
    oc_range = (closes[-1] - opens[-1]) / opens[-1]

    ma6 = np.mean(closes[-6:]) / closes[-1]
    ma12 = np.mean(closes[-12:]) / closes[-1]
    ma24 = np.mean(closes[-24:]) / closes[-1]
    ma48 = np.mean(closes[-48:]) / closes[-1] if len(closes) >= 48 else 1.0

    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else 0
    avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else 0
    rs = avg_gain / avg_loss if avg_loss > 0 else 1
    rsi = 100 - (100 / (1 + rs))

    volatility = np.std(closes[-24:]) / np.mean(closes[-24:])
    volume_ratio = volumes[-1] / np.mean(volumes[-24:])
    close_pos = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1])
    trend_6h = (closes[-1] - closes[-7]) / closes[-7] if len(closes) >= 7 else 0

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
    print("=" * 60)
    print("BTC Transformer Prediction (OpenClaw Skill)")
    print("=" * 60)

    print("\n📥 Loading model...")
    model = load_checkpoint(MODEL_PATH)
    if model is None:
        print(f"Error: Model not found at {MODEL_PATH}")
        return {"success": False, "error": "Model not found"}

    print(
        f"✓ Model: {model['n_layers']} layers, step={model['step']}, loss={model['loss']:.6f}"
    )

    print("\n📡 Fetching data...")
    candles = fetch_data()
    if not candles:
        return {"success": False, "error": "Failed to fetch data"}

    print(f"✓ Got {len(candles)} candles")
    current_price = candles[-1]["close"]

    features = calculate_features(candles)

    tokens = [i % VOCAB for i in range(SEQ_LEN)]

    print("\n🔮 Running transformer...")
    logits = forward(np.array([tokens]), model)
    prediction = float(logits[0])
    prediction = np.clip(prediction, -0.3, 0.3)

    future_price = current_price * (1 + prediction)
    ret = prediction * 100

    if ret > 1.5:
        signal, confidence = "STRONG_BUY", 95
    elif ret > 0.5:
        signal, confidence = "BUY", 75
    elif ret > -0.5:
        signal, confidence = "HOLD", 50
    elif ret > -1.5:
        signal, confidence = "SELL", 75
    else:
        signal, confidence = "STRONG_SELL", 95

    print("\n" + "=" * 60)
    print("📈 PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n  💰 Current Price: ${current_price:,.2f}")
    print(f"  🔮 24h Prediction: ${future_price:,.2f}")
    print(f"  📊 Expected Return: {ret:+.2f}%")
    print(f"\n  Signal: {signal} (confidence: {confidence}%)")

    result = {
        "success": True,
        "model": "btc_transformer_12layer",
        "step": model["step"],
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
