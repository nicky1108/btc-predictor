#!/usr/bin/env python3
"""
Simplified BTC Predictor for OpenClaw
使用预训练权重
"""

import struct
import json
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path

# 预训练权重（从最新训练结果 20,000 steps）
PRETRAINED_WEIGHTS = np.array(
    [
        -0.00107110,  # returns
        0.00086322,  # log_returns
        -0.00107972,  # high_low_range
        -0.00297364,  # open_close_range
        -0.00233630,  # ma6_ratio
        -0.00378953,  # ma12_ratio
        -0.00107397,  # ma24_ratio
        0.00059833,  # ma48_ratio
        -0.00035254,  # rsi
        -0.00010074,  # volatility_24h
        -0.00147552,  # volume_ratio
        0.00899673,  # close_position (最重要)
        -0.00014083,  # trend_6h
        0.00880761,  # 4h_returns
        0.00706813,  # 4h_high_low_range
        0.00211928,  # 4h_ma6
        0.00051777,  # 4h_ma12
        -0.00029514,  # 4h_ma24
        -0.00028521,  # 1d_returns
        0.00045947,  # 1d_high_low_range
        0.00449335,  # 1d_ma6
        0.00142000,  # 1d_ma12
    ],
    dtype=np.float32,
)

PRETRAINED_BIAS = 0.00146096


def fetch_latest_data():
    """获取最新BTC数据"""
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(hours=280)

    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000,
    }

    response = requests.get(url, params=params)
    data = response.json()

    candles = []
    for kline in data:
        candles.append(
            {
                "timestamp": datetime.fromtimestamp(kline[0] / 1000),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
            }
        )

    return candles[-256:]


def calculate_features(candles):
    """计算特征"""
    prices = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    features = []
    for i in range(len(candles)):
        c = candles[i]
        current = c["close"]
        price_list = prices[: i + 1]

        # Returns
        returns = (
            (current - candles[i - 1]["close"]) / candles[i - 1]["close"]
            if i > 0
            else 0
        )
        log_returns = (
            np.log(current / candles[i - 1]["close"])
            if i > 0 and candles[i - 1]["close"] > 0
            else 0
        )

        # Price action
        hl_range = (c["high"] - c["low"]) / current if current > 0 else 0
        oc_range = (c["close"] - c["open"]) / current if current > 0 else 0

        # Moving averages
        def ma(prices, window):
            if len(prices) < window:
                return sum(prices) / len(prices) if prices else current
            return sum(prices[-window:]) / window

        ma6 = current / ma(price_list, 6) if ma(price_list, 6) > 0 else 1
        ma12 = current / ma(price_list, 12) if ma(price_list, 12) > 0 else 1
        ma24 = current / ma(price_list, 24) if ma(price_list, 24) > 0 else 1
        ma48 = current / ma(price_list, 48) if ma(price_list, 48) > 0 else 1

        # RSI
        def calc_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            gains, losses = [], []
            for j in range(1, period + 1):
                change = prices[-j] - prices[-j - 1]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(-change)
            avg_gain = sum(gains) / period if gains else 0.0001
            avg_loss = sum(losses) / period if losses else 0.0001
            return 100 - (100 / (1 + avg_gain / avg_loss))

        rsi = calc_rsi(price_list, 14)

        # Volatility
        returns_list = [candles[j].get("r", 0) for j in range(max(0, i - 24), i)]
        if len(returns_list) > 1:
            mean = sum(returns_list) / len(returns_list)
            variance = sum((r - mean) ** 2 for r in returns_list) / len(returns_list)
            volatility = variance**0.5
        else:
            volatility = 0.01

        # Volume
        vol_list = volumes[: i + 1]
        vol_ma = ma(vol_list, 24)
        volume_ratio = c["volume"] / vol_ma if vol_ma > 0 else 1

        # Price position
        hl = c["high"] - c["low"]
        close_pos = (c["close"] - c["low"]) / hl if hl > 0 else 0.5

        # Trend
        trend_6h = (
            (current - candles[max(0, i - 6)]["close"])
            / candles[max(0, i - 6)]["close"]
            if i > 0
            else 0
        )

        feat = [
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
        ]
        features.append(feat)
        c["r"] = returns

    return np.array(features, dtype=np.float32)


def normalize(features):
    """归一化"""
    means = np.array(
        [
            0.000048,
            0.000048,
            0.009259,
            -0.000005,
            1.000068,
            1.000156,
            1.000336,
            1.000685,
            50.751646,
            0.005798,
            1.020091,
            0.516442,
            0.000276,
        ]
    )
    stds = np.array(
        [
            0.007258,
            0.007258,
            0.008982,
            0.007284,
            0.008662,
            0.012945,
            0.019041,
            0.027207,
            16.482281,
            0.004126,
            0.707568,
            0.282538,
            0.017022,
        ]
    )
    return (features - means) / stds


def predict():
    """预测BTC价格"""
    print("📡 Fetching latest BTC data...")
    candles = fetch_latest_data()
    current_price = candles[-1]["close"]

    print(f"💰 Current BTC Price: ${current_price:,.2f}")

    # 计算特征
    print("🔢 Calculating features...")
    features = calculate_features(candles)
    normalized = normalize(features)

    # 预测
    last = normalized[-1, :]
    prediction = np.dot(last, PRETRAINED_WEIGHTS) + PRETRAINED_BIAS
    prediction = np.clip(prediction, -0.3, 0.3)

    future_price = current_price * (1 + prediction)

    # 交易信号
    if prediction > 0.015:
        signal = "STRONG_BUY"
        confidence = min(abs(prediction) * 5000, 100)
    elif prediction > 0:
        signal = "BUY"
        confidence = min(abs(prediction) * 2000, 50)
    elif prediction > -0.015:
        signal = "HOLD"
        confidence = 30
    else:
        signal = "SELL"
        confidence = min(abs(prediction) * 5000, 100)

    result = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "current_price": float(current_price),
        "predicted_return": float(prediction * 100),
        "predicted_price": float(future_price),
        "signal": signal,
        "confidence": float(confidence),
        "model_info": {
            "features": len(PRETRAINED_WEIGHTS),
            "bias": float(PRETRAINED_BIAS),
        },
    }

    # 输出结果
    print("\n" + "=" * 70)
    print("📈 BTC PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nCurrent Price: ${result['current_price']:,.2f}")
    print(f"Predicted Return: {result['predicted_return']:+.3f}%")
    print(f"Predicted Price: ${result['predicted_price']:,.2f}")
    print(f"\nSignal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print("=" * 70)

    print("\nJSON_OUTPUT:")
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    predict()
