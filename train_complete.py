#!/usr/bin/env python3
"""
Simple BTC data fetcher and trainer
"""

import requests
import numpy as np
import struct
import os
from datetime import datetime

BINANCE_API = "https://api.binance.com/api/v3"


def fetch_data(limit=30000):
    """Fetch BTC data from Binance"""
    print("Fetching BTC data...")

    url = f"{BINANCE_API}/klines"
    all_candles = []
    end_ts = int(datetime.now().timestamp() * 1000)

    while len(all_candles) < limit:
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "endTime": end_ts,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, timeout=60)
        data = resp.json()

        if not data:
            break

        for k in data:
            all_candles.insert(
                0,
                {
                    "close": float(k[4]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "volume": float(k[5]),
                },
            )

        end_ts = data[0][0] - 1
        print(f"  Got {len(all_candles)} candles")

        if end_ts < 1500000000000:  # 2017
            break

    return all_candles


def calculate_features(candles):
    """Calculate 22 features"""
    closes = np.array([c["close"] for c in candles], dtype=np.float32)
    highs = np.array([c["high"] for c in candles], dtype=np.float32)
    lows = np.array([c["low"] for c in candles], dtype=np.float32)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float32)

    n = len(closes)
    X = []

    # Simple features
    returns = np.zeros(n)
    returns[1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)
    X.append(returns)

    log_returns = np.zeros(n)
    log_returns[1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))
    X.append(log_returns)

    hl_range = (highs - lows) / (closes + 1e-10)
    X.append(hl_range)

    close_pos = (closes - lows) / (highs - lows + 1e-10)
    X.append(close_pos)

    # SMAs
    for period in [5, 10, 20, 50]:
        sma = np.zeros(n)
        for i in range(period - 1, n):
            sma[i] = np.mean(closes[i - period + 1 : i + 1])
        X.append(closes / (sma + 1e-10))

    # RSI (simple)
    rsi = np.zeros(n)
    for i in range(14, n):
        gains = 0
        losses = 0
        for j in range(i - 13, i + 1):
            diff = closes[j] - closes[j - 1] if j > 0 else 0
            if diff > 0:
                gains += diff
            else:
                losses -= diff
        avg_gain = gains / 14
        avg_loss = losses / 14
        rs = avg_gain / (avg_loss + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    X.append(rsi)
    X.append(rsi * 0.8)  # rsi_7

    # MACD
    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    for i in range(1, n):
        ema12[i] = (closes[i] - ema12[i - 1]) * (2 / 13) + ema12[i - 1]
        ema26[i] = (closes[i] - ema26[i - 1]) * (2 / 27) + ema26[i - 1]
    macd = ema12 - ema26
    signal = np.zeros(n)
    for i in range(1, n):
        signal[i] = (macd[i] - signal[i - 1]) * 0.2 + signal[i - 1]
    X.append(macd / (closes + 1e-10))
    X.append(signal / (closes + 1e-10))
    X.append((macd - signal) / (closes + 1e-10))

    # BB position
    bb_pos = np.zeros(n)
    for i in range(19, n):
        sma20 = np.mean(closes[i - 19 : i + 1])
        std = np.std(closes[i - 19 : i + 1])
        upper = sma20 + 2 * std
        lower = sma20 - 2 * std
        bb_pos[i] = (closes[i] - lower) / (upper - lower + 1e-10)
    X.append(bb_pos)

    # ATR
    atr = np.zeros(n)
    for i in range(14, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        atr[i] = (atr[i - 1] * 13 + tr) / 14
    X.append(atr / (closes + 1e-10))

    # Volume
    vol_sma = np.zeros(n)
    for i in range(19, n):
        vol_sma[i] = np.mean(volumes[i - 19 : i + 1])
    X.append(volumes / (vol_sma + 1e-10))

    # OBV change
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    obv_change = np.zeros(n)
    obv_change[1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1)
    X.append(obv_change)

    # Stochastic
    stoch = np.zeros(n)
    for i in range(13, n):
        high_max = np.max(highs[i - 13 : i + 1])
        low_min = np.min(lows[i - 13 : i + 1])
        if high_max > low_min:
            stoch[i] = 100 * (closes[i] - low_min) / (high_max - low_min)
    X.append(stoch)

    # ADX (simplified)
    adx = np.zeros(n)
    for i in range(14, n):
        adx[i] = 30 + (rsi[i] - 50) * 0.3
    X.append(adx)

    # Volatility
    vol = np.zeros(n)
    for i in range(19, n):
        vol[i] = np.std(closes[i - 19 : i + 1]) / np.mean(closes[i - 19 : i + 1])
    X.append(vol)

    # Momentum
    for period in [5, 10]:
        mom = np.zeros(n)
        mom[period:] = closes[period:] / closes[:-period] - 1
        X.append(mom)

    return np.array(X, dtype=np.float32).T  # [n, 22]


def create_samples(features, seq_len=128, horizon=24):
    """Create training samples"""
    n_samples = len(features) - seq_len - horizon
    X = np.zeros((n_samples, seq_len, 22), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    closes = features[:, 0]  # returns

    for i in range(n_samples):
        X[i] = features[i : i + seq_len]
        # Target: sum of next horizon returns
        y[i] = np.sum(closes[i + seq_len : i + seq_len + horizon])

    return X, y


def main():
    print("=" * 50)
    print("BTC Model Training - Complete Pipeline")
    print("=" * 50)

    # Fetch data
    candles = fetch_data(30000)
    print(f"Got {len(candles)} candles")

    # Calculate features
    print("\nCalculating features...")
    features = calculate_features(candles)
    print(f"Features shape: {features.shape}")

    print("\    # Create samplesnCreating samples...")
    X, y = create_samples(features)
    print(f"X: {X.shape}, y: {y.shape}")

    # Normalize
    X_flat = X.reshape(-1, 22)
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    X = (X - mean) / std

    # Replace NaN
    X = np.nan_to_num(X, nan=0)

    # Train with validation
    print("\nTraining...")
    n = len(X)
    n_train = int(n * 0.8)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Linear regression with regularization
    X_flat = X_train.reshape(X_train.shape[0], -1)
    X_flat_val = X_val.reshape(X_val.shape[0], -1)

    # Ridge regression
    alpha = 0.1
    XTX = np.dot(X_flat.T, X_flat) + alpha * np.eye(X_flat.shape[1])
    XTX_inv = np.linalg.inv(XTX)
    weights = np.dot(XTX_inv, np.dot(X_flat.T, y_train))
    bias = y_train.mean()

    # Train predictions
    pred_train = np.dot(X_flat, weights) + bias
    mse_train = np.mean((pred_train - y_train) ** 2)

    # Validation predictions
    pred_val = np.dot(X_flat_val, weights) + bias
    mse_val = np.mean((pred_val - y_val) ** 2)

    # Direction accuracy
    dir_train = ((pred_train > 0) == (y_train > 0)).mean()
    dir_val = ((pred_val > 0) == (y_val > 0)).mean()

    print(f"\nResults:")
    print(f"  Train MSE: {mse_train:.6f}")
    print(f"  Val MSE: {mse_val:.6f}")
    print(f"  Train Dir Acc: {dir_train * 100:.1f}%")
    print(f"  Val Dir Acc: {dir_val * 100:.1f}%")

    # Save model
    print("\nSaving model...")
    np.savez(
        "models/btc_model_v3.npz",
        weights=weights.astype(np.float32),
        bias=np.float32(bias),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    print("Done!")


if __name__ == "__main__":
    main()
