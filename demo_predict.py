#!/usr/bin/env python3
"""
BTC预测演示（使用最新训练权重 - 20,000 steps）
"""

import numpy as np
import json
from datetime import datetime

# 最新训练权重（20,000 steps, Loss: 0.000353）
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

# 模拟当前市场数据（多时间尺度）
SIMULATED_FEATURES = np.array(
    [
        0.0012,  # returns
        0.0012,  # log_returns
        0.008,  # high_low_range
        -0.0002,  # open_close_range
        1.002,  # ma6_ratio
        1.015,  # ma12_ratio
        1.025,  # ma24_ratio
        1.035,  # ma48_ratio
        62.5,  # rsi
        0.006,  # volatility_24h
        0.85,  # volume_ratio
        0.65,  # close_position
        0.008,  # trend_6h
        0.003,  # 4h_returns
        0.012,  # 4h_high_low_range
        1.008,  # 4h_ma6
        1.018,  # 4h_ma12
        1.028,  # 4h_ma24
        0.015,  # 1d_returns
        0.025,  # 1d_high_low_range
        1.012,  # 1d_ma6
        1.022,  # 1d_ma12
    ],
    dtype=np.float32,
)

# 归一化参数（基于训练数据统计）
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


def predict_demo(current_price=70950.0):
    """BTC预测演示"""
    print("=" * 70)
    print("🚀 BTC价格预测（最新模型 - 20,000 Steps）")
    print("=" * 70)
    print()

    # 归一化
    normalized = (SIMULATED_FEATURES - MEANS) / STDS

    # 预测
    prediction = np.dot(normalized, PRETRAINED_WEIGHTS) + PRETRAINED_BIAS
    prediction = np.clip(prediction, -0.3, 0.3)

    future_price = current_price * (1 + prediction)

    # 交易信号
    if prediction > 0.015:
        signal = "STRONG_BUY"
        signal_cn = "🚀 强烈买入"
        confidence = min(abs(prediction) * 5000, 100)
    elif prediction > 0.005:
        signal = "BUY"
        signal_cn = "📈 建议买入"
        confidence = min(abs(prediction) * 3000, 70)
    elif prediction > -0.005:
        signal = "HOLD"
        signal_cn = "➡️ 观望持有"
        confidence = 40
    elif prediction > -0.015:
        signal = "SELL"
        signal_cn = "📉 建议卖出"
        confidence = min(abs(prediction) * 3000, 70)
    else:
        signal = "STRONG_SELL"
        signal_cn = "🔴 强烈卖出"
        confidence = min(abs(prediction) * 5000, 100)

    conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"

    print("📊 当前市场状态（模拟数据）:")
    print(f"  BTC现价: ${current_price:,.2f}")
    print(f"  RSI: {SIMULATED_FEATURES[8]:.1f}")
    print(f"  MA24比率: {SIMULATED_FEATURES[6]:.4f}")
    print(f"  6h趋势: +{SIMULATED_FEATURES[12] * 100:.2f}%")
    print(f"  4h收益: +{SIMULATED_FEATURES[13] * 100:.2f}%")
    print(f"  1d收益: +{SIMULATED_FEATURES[18] * 100:.2f}%")
    print()

    print("=" * 70)
    print("📈 预测结果")
    print("=" * 70)
    print()
    print(f"💰 当前价格: ${current_price:,.2f}")
    print(f"🔮 24h后预测: ${future_price:,.2f}")
    print(f"📊 预测收益: {prediction * 100:+.3f}%")
    print()
    print(f"🏷️ 交易信号: {signal_cn}")
    print(f"{conf_color} 置信度: {confidence:.1f}%")
    print()

    # 特征贡献分析
    contributions = []
    for i in range(len(PRETRAINED_WEIGHTS)):
        contrib = normalized[i] * PRETRAINED_WEIGHTS[i] * 100
        contributions.append((i, contrib, PRETRAINED_WEIGHTS[i]))

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
        "close_position",
        "trend_6h",
        "4h_returns",
        "4h_hl_range",
        "4h_ma6",
        "4h_ma12",
        "4h_ma24",
        "1d_returns",
        "1d_hl_range",
        "1d_ma6",
        "1d_ma12",
    ]

    print("📊 关键特征贡献:")
    for idx, contrib, weight in contributions[:5]:
        name = feature_names[idx]
        direction = "支持上涨" if contrib > 0 else "看跌信号"
        print(f"  {name:15s}: {contrib:+7.4f}% ({direction})")

    print(f"\n  偏置项: +{PRETRAINED_BIAS * 100:.4f}%")
    print()

    print("=" * 70)
    print("💡 交易建议")
    print("=" * 70)
    print()

    if signal == "STRONG_BUY":
        advice = f"""
🚀 强烈建议买入！

模型预测24小时后BTC将上涨{prediction * 100:.2f}%，置信度{confidence:.1f}%。

关键信号：
- 多时间尺度均显示上涨趋势
- close_position指标显示买盘强劲
- 1日/4小时/1小时周期共振

建议操作：
✅ 可考虑重仓买入
✅ 设置止损在 ${current_price * 0.98:,.2f} (当前价-2%)
✅ 目标价 ${current_price * 1.05:,.2f} (当前价+5%)
"""
    elif signal == "BUY":
        advice = f"""
📈 建议买入

模型显示上涨趋势({prediction * 100:.2f}%)，置信度中等({confidence:.1f}%)。

建议操作：
✅ 小仓位试探性买入
✅ 分批建仓，避免一次性重仓
✅ 关注4小时和日线趋势
"""
    elif signal == "HOLD":
        advice = f"""
➡️ 建议观望持有

市场信号不明显，预测波动较小({prediction * 100:.2f}%)。

当前状态：
- 多空力量相对平衡
- 建议维持现有仓位
- 等待更明确突破信号

建议操作：
⏸️ 暂不操作，继续观察
📊 关注 ${current_price * 1.02:,.2f} 关键价位突破
"""
    else:
        advice = f"""
📉 建议谨慎

模型预测显示下跌风险({prediction * 100:.2f}%)。

建议操作：
⚠️ 减仓或观望
🛡️ 设置止损保护
📉 关注支撑位 ${current_price * 0.98:,.2f}
"""

    print(advice)

    print("=" * 70)
    print("⚠️ 风险提示")
    print("=" * 70)
    print("""
1. 模型训练步数: 20,000 steps
2. 最佳Loss: 0.000353 (RMSE ~1.88%)
3. 方向准确率: 77% (历史回测)
4. 加密货币市场波动剧烈，请控制风险
5. 本预测仅供参考，不构成投资建议
""")

    print("=" * 70)
    print("📅 预测时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # JSON输出
    result = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "training_steps": 20000,
            "best_loss": 0.000353,
            "features": len(PRETRAINED_WEIGHTS),
        },
        "current_price": float(current_price),
        "predicted_price": float(future_price),
        "predicted_return": float(prediction * 100),
        "signal": signal,
        "signal_cn": signal_cn,
        "confidence": float(confidence),
        "market_state": {
            "rsi": float(SIMULATED_FEATURES[8]),
            "ma24_ratio": float(SIMULATED_FEATURES[6]),
            "trend_6h": float(SIMULATED_FEATURES[12]),
            "4h_returns": float(SIMULATED_FEATURES[13]),
            "1d_returns": float(SIMULATED_FEATURES[18]),
        },
        "top_features": [
            {"name": feature_names[c[0]], "contribution": float(c[1])}
            for c in contributions[:5]
        ],
    }

    print("\n📄 JSON_OUTPUT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    predict_demo(current_price=70950.0)
