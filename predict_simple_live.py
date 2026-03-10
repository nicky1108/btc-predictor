#!/usr/bin/env python3
"""
BTC实时预测 - 使用训练好的序列模型
"""

import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta

# 加载训练好的模型
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "btc_sequence_model.npz")


def load_model():
    """加载训练好的模型"""
    if os.path.exists(MODEL_PATH):
        data = np.load(MODEL_PATH)
        return {
            "weights": data["weights"],
            "bias": data["bias"],
            "time_weights": data["time_weights"],
            "feat_mean": data["feat_mean"],
            "feat_std": data["feat_std"],
        }
    return None


# 尝试加载模型，如果失败则使用默认权重
_model = load_model()

if _model is not None:
    print(f"✓ 已加载训练模型: {MODEL_PATH}")
    PRETRAINED_WEIGHTS = _model["weights"]
    PRETRAINED_BIAS = _model["bias"]
    TIME_WEIGHTS = _model["time_weights"]
    MEANS = _model["feat_mean"]
    STDS = _model["feat_std"]
    USE_SEQUENCE_MODEL = True
else:
    # 回退到旧的权重
    print("⚠️ 未找到训练模型，使用默认权重")
    USE_SEQUENCE_MODEL = False

    PRETRAINED_WEIGHTS = np.array(
        [
            -0.000887,
            0.001630,
            -0.001034,
            -0.003497,
            -0.002416,
            -0.005841,
            -0.000968,
            0.000907,
            0.002064,
            0.000053,
            -0.000921,
            0.009383,
            0.000026,
        ],
        dtype=np.float32,
    )
    PRETRAINED_BIAS = 0.000756
    TIME_WEIGHTS = np.exp(-np.arange(256) / 50)
    TIME_WEIGHTS /= TIME_WEIGHTS.sum()

    MEANS = np.array(
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
    STDS = np.array(
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


def fetch_data():
    """获取1小时数据"""
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(hours=100)

    params = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 100,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
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
        return candles
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def calculate_features(candles):
    """计算13个特征"""
    prices = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    current = candles[-1]["close"]

    # Returns
    returns = (
        (current - candles[-2]["close"]) / candles[-2]["close"]
        if len(candles) > 1
        else 0
    )
    log_returns = (
        np.log(current / candles[-2]["close"])
        if len(candles) > 1 and candles[-2]["close"] > 0
        else 0
    )

    # Price action
    hl_range = (
        (candles[-1]["high"] - candles[-1]["low"]) / current if current > 0 else 0
    )
    oc_range = (
        (candles[-1]["close"] - candles[-1]["open"]) / current if current > 0 else 0
    )

    # Moving averages
    def ma(prices, window):
        if len(prices) < window:
            return sum(prices) / len(prices) if prices else current
        return sum(prices[-window:]) / window

    ma6 = current / ma(prices, 6) if ma(prices, 6) > 0 else 1
    ma12 = current / ma(prices, 12) if ma(prices, 12) > 0 else 1
    ma24 = current / ma(prices, 24) if ma(prices, 24) > 0 else 1
    ma48 = current / ma(prices, 48) if ma(prices, 48) > 0 else 1

    # RSI
    def calc_rsi(prices, period=14):
        if len(prices) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, min(period + 1, len(prices))):
            change = prices[-i] - prices[-i - 1]
            if change > 0:
                gains.append(change)
            else:
                losses.append(-change)
        avg_gain = sum(gains) / period if gains else 0.0001
        avg_loss = sum(losses) / period if losses else 0.0001
        return 100 - (100 / (1 + avg_gain / avg_loss))

    rsi = calc_rsi(prices, 14)

    # Volatility
    returns_list = []
    for i in range(1, min(25, len(candles))):
        r = (candles[-i]["close"] - candles[-i - 1]["close"]) / candles[-i - 1]["close"]
        returns_list.append(r)

    if len(returns_list) > 1:
        mean = sum(returns_list) / len(returns_list)
        variance = sum((r - mean) ** 2 for r in returns_list) / len(returns_list)
        volatility = variance**0.5
    else:
        volatility = 0.01

    # Volume
    vol_ma = ma(volumes, 24)
    volume_ratio = candles[-1]["volume"] / vol_ma if vol_ma > 0 else 1

    # Price position
    hl = candles[-1]["high"] - candles[-1]["low"]
    close_pos = (candles[-1]["close"] - candles[-1]["low"]) / hl if hl > 0 else 0.5

    # Trend
    trend_6h = (
        (current - candles[-min(7, len(candles))]["close"])
        / candles[-min(7, len(candles))]["close"]
        if len(candles) > 6
        else 0
    )

    return [
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


def predict():
    """BTC实时预测"""
    print("=" * 70)
    print("🚀 BTC实时价格预测 (Live)")
    print("=" * 70)
    print()

    # 获取数据
    print("📡 从Binance获取实时数据...")
    candles = fetch_data()
    if not candles:
        print("❌ 无法获取数据")
        return

    current_price = candles[-1]["close"]
    print(f"✅ 获取到 {len(candles)} 条1小时K线")
    print(f"💰 当前BTC价格: ${current_price:,.2f}")
    print()

    # 计算特征
    print("🔢 计算技术指标...")
    features = calculate_features(candles)

    # 显示特征值
    feature_names = [
        "returns",
        "log_returns",
        "hl_range",
        "oc_range",
        "ma6",
        "ma12",
        "ma24",
        "ma48",
        "rsi",
        "volatility",
        "volume_ratio",
        "close_pos",
        "trend_6h",
    ]
    print("📊 当前市场指标:")
    for name, val in zip(feature_names, features):
        if name in ["returns", "log_returns", "trend_6h"]:
            print(f"  {name:15s}: {val * 100:+.3f}%")
        elif name == "rsi":
            print(f"  {name:15s}: {val:.1f}")
        else:
            print(f"  {name:15s}: {val:.4f}")
    print()

    # 归一化和预测
    features_array = np.array(features, dtype=np.float32)

    # 确保是2D数组 [samples, features]
    if len(features_array.shape) == 1:
        features_array = features_array.reshape(1, -1)

    normalized = (features_array - MEANS) / STDS

    if USE_SEQUENCE_MODEL and len(normalized) >= 256:
        # 使用序列模型（时间加权）
        agg = np.zeros(13)
        for t in range(min(256, len(normalized))):
            agg += normalized[-(t + 1), :] * TIME_WEIGHTS[t]
        prediction = float(np.dot(agg, PRETRAINED_WEIGHTS) + PRETRAINED_BIAS)
    else:
        # 使用最后一个时间步
        last_feat = normalized[-1, :]
        prediction = float(np.dot(last_feat, PRETRAINED_WEIGHTS) + PRETRAINED_BIAS)

    prediction = np.clip(prediction, -0.3, 0.3)

    future_price = current_price * (1 + prediction)
    ret = prediction * 100

    # 信号
    if ret > 1.5:
        signal, emoji, conf = "STRONG_BUY", "🚀", 95
    elif ret > 0.5:
        signal, emoji, conf = "BUY", "📈", 75
    elif ret > -0.5:
        signal, emoji, conf = "HOLD", "➡️", 50
    elif ret > -1.5:
        signal, emoji, conf = "SELL", "📉", 75
    else:
        signal, emoji, conf = "STRONG_SELL", "🔴", 95

    # 显示结果
    print("=" * 70)
    print("📈 预测结果")
    print("=" * 70)
    print()
    print(f"  💰 当前价格: ${current_price:,.2f}")
    print(f"  🔮 24h后预测: ${future_price:,.2f}")
    print(f"  📊 预期收益: {ret:+.3f}%")
    print()
    print(f"  {emoji} 交易信号: {signal}")
    print(
        f"  {'🟢' if conf >= 70 else '🟡' if conf >= 50 else '🔴'} 置信度: {conf:.1f}%"
    )
    print()

    # 特征贡献 (使用最后一个时间步)
    last_normalized = normalized[-1, :]
    contributions = [
        (last_normalized[i] * PRETRAINED_WEIGHTS[i] * 100) for i in range(13)
    ]
    top_idx = np.argsort(np.abs(contributions))[-5:][::-1]

    print("📊 主要影响因素:")
    for idx in top_idx:
        name = feature_names[idx]
        contrib = contributions[idx]
        direction = "📈 看涨" if contrib > 0 else "📉 看跌"
        print(f"  {name:15s}: {contrib:+6.3f}% {direction}")
    print()

    # 建议
    print("=" * 70)
    print("💡 交易建议")
    print("=" * 70)
    print()

    if signal in ["STRONG_BUY", "BUY"]:
        print(f"{emoji} {'强烈建议买入！' if signal == 'STRONG_BUY' else '建议买入'}")
        print(f"预期24h上涨 {ret:.2f}%")
        print(f"\n操作建议:")
        print(f"  ✅ 小仓位试探买入")
        print(f"  🛡️ 止损: ${current_price * 0.98:,.2f} (-2%)")
        print(f"  🎯 目标: ${current_price * 1.03:,.2f} (+3%)")
    elif signal == "HOLD":
        print("➡️ 建议观望持有")
        print(f"预测波动较小 ({ret:.2f}%)，建议维持现有仓位")
    else:
        print(f"{emoji} 建议谨慎")
        print(f"预期24h下跌 {abs(ret):.2f}%")

    print()
    print("=" * 70)
    print(f"📅 预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("⚠️ 风险提示: 本预测仅供参考，加密货币市场波动剧烈")
    print("=" * 70)

    # JSON
    result = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "current_price": float(current_price),
        "predicted_price": float(future_price),
        "predicted_return": float(ret),
        "signal": signal,
        "confidence": float(conf),
    }

    print("\n📄 JSON_OUTPUT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predict()
