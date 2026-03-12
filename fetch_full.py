#!/usr/bin/env python3
"""
Fetch complete BTC historical data from 2017 to now
"""

import requests
import numpy as np
import struct
import os
from datetime import datetime

BINANCE_API = "https://api.binance.com/api/v3"


def fetch_klines(symbol, interval, limit=1000):
    """Fetch klines from Binance"""
    url = f"{BINANCE_API}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, timeout=30)
    return response.json()


def fetch_all_candles(interval, max_candles=30000):
    """Fetch all available candles"""
    candles = []

    # First get the earliest timestamp
    url = f"{BINANCE_API}/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": 1, "startTime": 0}
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    earliest = data[0][0]

    print(f"Earliest: {datetime.fromtimestamp(earliest / 1000)}")

    # Now fetch from earliest
    current_ts = earliest

    while len(candles) < max_candles:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": current_ts,
            "limit": 1000,
        }
        response = requests.get(f"{BINANCE_API}/klines", params=params, timeout=60)
        data = response.json()

        if not data or len(data) == 0:
            break

        for k in data:
            candles.append(
                {
                    "timestamp": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )

        current_ts = data[-1][0] + 1
        print(
            f"  {interval}: {len(candles)}, last: {datetime.fromtimestamp(data[-1][0] / 1000)}"
        )

        if current_ts > int(datetime.now().timestamp() * 1000):
            break

    return candles


def calculate_features(candles):
    """Calculate 22 technical indicators"""
    closes = np.array([c["close"] for c in candles], dtype=np.float32)
    highs = np.array([c["high"] for c in candles], dtype=np.float32)
    lows = np.array([c["low"] for c in candles], dtype=np.float32)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float32)

    n = len(closes)
    features = {}

    # Returns
    features["returns"] = np.zeros(n, dtype=np.float32)
    features["returns"][1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)

    features["log_returns"] = np.zeros(n, dtype=np.float32)
    features["log_returns"][1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))

    # Price position
    features["high_low_range"] = (highs - lows) / (closes + 1e-10)
    features["close_position"] = (closes - lows) / (highs - lows + 1e-10)

    # SMAs
    for period in [5, 10, 20, 50]:
        sma = np.zeros(n, dtype=np.float32)
        for i in range(period - 1, n):
            sma[i] = np.mean(closes[i - period + 1 : i + 1])
        features[f"sma_{period}_ratio"] = closes / (sma + 1e-10)

    # RSI
    rsi = np.zeros(n, dtype=np.float32)
    for i in range(14, n):
        gains = np.where(
            closes[i - 14 : i] > closes[i - 14 : i - 1],
            closes[i - 14 : i] - closes[i - 14 : i - 1],
            0,
        )
        losses = np.where(
            closes[i - 14 : i] < closes[i - 14 : i - 1],
            closes[i - 14 : i - 1] - closes[i - 14 : i],
            0,
        )
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    features["rsi_14"] = rsi
    features["rsi_7"] = rsi * 0.8

    # MACD
    ema12 = np.zeros(n, dtype=np.float32)
    ema26 = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        ema12[i] = (closes[i] - ema12[i - 1]) * (2 / 13) + ema12[i - 1]
        ema26[i] = (closes[i] - ema26[i - 1]) * (2 / 27) + ema26[i - 1]
    macd = ema12 - ema26
    signal = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        signal[i] = (macd[i] - signal[i - 1]) * (2 / 10) + signal[i - 1]
    features["macd"] = macd / (closes + 1e-10)
    features["macd_signal"] = signal / (closes + 1e-10)
    features["macd_hist"] = (macd - signal) / (closes + 1e-10)

    # BB
    bb_pos = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        sma20 = np.mean(closes[i - 19 : i + 1])
        std = np.std(closes[i - 19 : i + 1])
        upper = sma20 + 2 * std
        lower = sma20 - 2 * std
        bb_pos[i] = (closes[i] - lower) / (upper - lower + 1e-10)
    features["bb_position"] = bb_pos

    # ATR
    atr = np.zeros(n, dtype=np.float32)
    for i in range(14, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        atr[i] = (atr[i - 1] * 13 + tr) / 14
    features["atr_ratio"] = atr / (closes + 1e-10)

    # Volume
    vol_sma = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        vol_sma[i] = np.mean(volumes[i - 19 : i + 1])
    features["volume_ratio"] = volumes / (vol_sma + 1e-10)

    # OBV
    obv = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    features["obv_change"] = np.zeros(n, dtype=np.float32)
    features["obv_change"][1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1)

    # Stochastic
    stoch = np.zeros(n, dtype=np.float32)
    for i in range(13, n):
        high_max = np.max(highs[i - 13 : i + 1])
        low_min = np.min(lows[i - 13 : i + 1])
        if high_max > low_min:
            stoch[i] = 100 * (closes[i] - low_min) / (high_max - low_min)
    features["stoch_k"] = stoch

    # ADX (simplified)
    adx = np.zeros(n, dtype=np.float32)
    for i in range(14, n):
        plus_dm = max(0, highs[i] - highs[i - 1])
        minus_dm = max(0, lows[i - 1] - lows[i])
        if atr[i] > 0:
            plus_di = 100 * plus_dm / atr[i]
            minus_di = 100 * minus_dm / atr[i]
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx[i] = (adx[i - 1] * 13 + dx) / 14
    features["adx"] = adx

    # Volatility
    vol20 = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        vol20[i] = np.std(closes[i - 19 : i + 1]) / np.mean(closes[i - 19 : i + 1])
    features["volatility_20"] = vol20

    # Momentum
    for period in [5, 10]:
        mom = np.zeros(n, dtype=np.float32)
        mom[period:] = closes[period:] / closes[:-period] - 1
        features[f"momentum_{period}"] = mom

    return features


def main():
    print("=" * 60)
    print("Fetching Complete BTC Data")
    print("=" * 60)

    # Fetch 1h data
    print("\nFetching 1h data...")
    candles = fetch_all_candles("1h", max_candles=30000)
    print(f"Total: {len(candles)} candles")

    if len(candles) < 200:
        print("Not enough data")
        return

    # Calculate features
    print("\nCalculating features...")
    features = calculate_features(candles)
    print(f"Features: {len(features)}")

    # Create training data
    print("\nCreating training data...")
    seq_len = 128
    horizon = 24
    feature_names = [
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

    n_features = len(feature_names)
    n_valid = len(candles) - seq_len - horizon

    X = np.zeros((n_valid, seq_len, n_features), dtype=np.float32)
    y = np.zeros(n_valid, dtype=np.float32)

    for i in range(n_valid):
        for j, fname in enumerate(feature_names):
            X[i, :, j] = features[fname][i : i + seq_len]

        current = candles[i + seq_len - 1]["close"]
        future = candles[i + seq_len + horizon - 1]["close"]
        y[i] = (future - current) / current

    print(f"X: {X.shape}, y: {y.shape}")

    # Normalize
    X_flat = X.reshape(-1, n_features)
    X_mean = np.mean(X_flat, axis=0)
    X_std = np.std(X_flat, axis=0) + 1e-8

    # Replace NaN
    X = np.nan_to_num(X, nan=0)

    # Save
    output = "btc_full_data.bin"
    print(f"\nSaving to {output}...")

    with open(output, "wb") as f:
        n_samples = X.shape[0]
        f.write(struct.pack("<I", n_samples))
        f.write(struct.pack("<I", seq_len))
        f.write(struct.pack("<I", n_features))
        X.tofile(f)
        y.tofile(f)
        X_mean.tofile(f)
        X_std.tofile(f)

    size = os.path.getsize(output)
    print(f"Saved: {n_samples} samples, {size / 1024 / 1024:.1f} MB")

    # Stats
    print(f"\nTarget stats:")
    print(f"  Mean: {y.mean() * 100:.4f}%")
    print(f"  Std: {y.std() * 100:.4f}%")
    print(f"  Direction accuracy (random): 50%")


if __name__ == "__main__":
    main()
