#!/usr/bin/env python3
"""
BTC实时预测 - 多时间尺度版本
支持1h、4h、1d数据
"""

import numpy as np
import requests
import json
from datetime import datetime, timedelta

# 最新训练权重（22 features）
PRETRAINED_WEIGHTS = np.array(
    [
        -0.00107110,
        0.00086322,
        -0.00107972,
        -0.00297364,
        -0.00233630,
        -0.00378953,
        -0.00107397,
        0.00059833,
        -0.00035254,
        -0.00010074,
        -0.00147552,
        0.00899673,
        -0.00014083,
        0.00880761,
        0.00706813,
        0.00211928,
        0.00051777,
        -0.00029514,
        -0.00028521,
        0.00045947,
        0.00449335,
        0.00142000,
    ],
    dtype=np.float32,
)

PRETRAINED_BIAS = 0.00146096

# 归一化参数
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
        0.000052,
        0.009312,
        1.000085,
        1.000168,
        1.000352,
        0.000218,
        0.035891,
        1.000215,
        1.000428,
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
        0.014521,
        0.018934,
        0.009245,
        0.013872,
        0.021456,
        0.032845,
        0.045231,
        0.011234,
        0.019876,
    ]
)


def fetch_data(interval="1h", limit=100):
    """从Binance获取数据"""
    url = "https://api.binance.com/api/v3/klines"
    end = datetime.now()
    start = end - timedelta(
        hours=limit
        if interval == "1h"
        else limit * 4
        if interval == "4h"
        else limit * 24
    )

    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": limit,
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
        print(f"❌ Error fetching {interval} data: {e}")
        return []


def calculate_features(candles):
    """计算13个基础特征"""
    if not candles:
        return None

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
    print("🚀 BTC实时价格预测")
    print("   基于最新20,000 steps训练模型")
    print("=" * 70)
    print()

    # 获取1h数据
    print("📡 获取1小时数据...")
    candles_1h = fetch_data("1h", 100)
    if not candles_1h:
        print("❌ 无法获取数据")
        return

    current_price = candles_1h[-1]["close"]
    print(f"💰 当前BTC价格: ${current_price:,.2f}")
    print(f"📊 获取到 {len(candles_1h)} 条K线")
    print()

    # 计算1h特征（13个）
    print("🔢 计算技术指标...")
    features_1h = calculate_features(candles_1h)

    # 获取4h数据用于多时间尺度特征
    print("📡 获取4小时数据...")
    candles_4h = fetch_data("4h", 30)
    features_4h = calculate_features(candles_4h) if candles_4h else features_1h[:13]

    # 获取1d数据
    print("📡 获取日线数据...")
    candles_1d = fetch_data("1d", 15)
    features_1d = calculate_features(candles_1d) if candles_1d else features_1h[:13]

    # 组合22个特征（1h: 13个 + 4h: 9个）
    # 取4h的前9个特征（去除重复）
    features_4h_subset = features_4h[:9] if features_4h else features_1h[:9]

    features = features_1h + features_4h_subset

    # 确保只有22个特征
    if len(features) > 22:
        features = features[:22]
    elif len(features) < 22:
        # 如果不足，用1h特征填充
        features = features + features_1h[: 22 - len(features)]

    print(f"✅ 计算完成: {len(features)} 个特征")
    print()

    # 归一化
    features_array = np.array(features, dtype=np.float32)
    normalized = (features_array - MEANS) / STDS

    # 预测
    prediction = np.dot(normalized, PRETRAINED_WEIGHTS) + PRETRAINED_BIAS
    prediction = np.clip(prediction, -0.3, 0.3)

    future_price = current_price * (1 + prediction)
    ret = prediction * 100

    # 交易信号
    if ret > 1.5:
        signal = "STRONG_BUY"
        signal_emoji = "🚀"
        confidence = min(abs(ret) * 30, 95)
    elif ret > 0.5:
        signal = "BUY"
        signal_emoji = "📈"
        confidence = min(abs(ret) * 40, 75)
    elif ret > -0.5:
        signal = "HOLD"
        signal_emoji = "➡️"
        confidence = 50
    elif ret > -1.5:
        signal = "SELL"
        signal_emoji = "📉"
        confidence = min(abs(ret) * 40, 75)
    else:
        signal = "STRONG_SELL"
        signal_emoji = "🔴"
        confidence = min(abs(ret) * 30, 95)

    conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"

    # 显示结果
    print("=" * 70)
    print("📈 预测结果")
    print("=" * 70)
    print()
    print(f"  💰 当前价格: ${current_price:,.2f}")
    print(f"  🔮 24h后预测: ${future_price:,.2f}")
    print(f"  📊 预期收益: {ret:+.3f}%")
    print()
    print(f"  {signal_emoji} 交易信号: {signal}")
    print(f"  {conf_color} 置信度: {confidence:.1f}%")
    print()

    # 特征贡献
    contributions = []
    for i in range(len(PRETRAINED_WEIGHTS)):
        contrib = normalized[i] * PRETRAINED_WEIGHTS[i] * 100
        contributions.append((i, contrib))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    feature_names = [
        "returns",
        "log_returns",
        "hl_range",
        "oc_range",
        "ma6_ratio",
        "ma12_ratio",
        "ma24_ratio",
        "ma48_ratio",
        "rsi",
        "volatility",
        "volume_ratio",
        "close_pos",
        "trend_6h",
        "4h_returns",
        "4h_hl_range",
        "4h_ma6",
        "4h_ma12",
        "4h_ma24",
        "4h_rsi",
        "4h_vol",
        "4h_vol_ratio",
        "4h_close",
    ]

    print("📊 主要影响因素:")
    for idx, contrib in contributions[:5]:
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        direction = "📈 看涨" if contrib > 0 else "📉 看跌"
        print(f"  {name:15s}: {contrib:+6.3f}% {direction}")

    print()

    # 交易建议
    print("=" * 70)
    print("💡 交易建议")
    print("=" * 70)
    print()

    if signal in ["STRONG_BUY", "BUY"]:
        print(
            f"{signal_emoji} {'强烈建议买入！' if signal == 'STRONG_BUY' else '建议买入'}"
        )
        print()
        print(f"理由：模型预测24h上涨 {ret:.2f}%，多时间尺度趋势向上")
        print()
        print("操作建议：")
        print(f"  ✅ {'重仓买入' if signal == 'STRONG_BUY' else '小仓位试探买入'}")
        print(f"  🛡️ 止损: ${current_price * 0.98:,.2f} (-2%)")
        print(f"  🎯 目标: ${current_price * (1 + abs(ret) / 100 * 2):,.2f}")
    elif signal == "HOLD":
        print("➡️ 建议观望持有")
        print(f"市场处于盘整，预测波动较小({ret:.2f}%)")
    else:
        print(f"{signal_emoji} 建议谨慎")
        print(f"模型预测显示下跌风险({ret:.2f}%)")

    print()
    print("=" * 70)
    print(f"📅 预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # JSON输出
    result = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "current_price": float(current_price),
        "predicted_price": float(future_price),
        "predicted_return": float(ret),
        "signal": signal,
        "confidence": float(confidence),
    }

    print("\n📄 JSON_OUTPUT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    predict()
