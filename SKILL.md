---
name: btc-predictor
description: BTC价格预测 - 预测BTC 24小时后的价格走势，提供交易信号和置信度
metadata: {"openclaw": {"emoji": "📈", "requires": {"bins": ["python3"], "env": []}, "homepage": "https://github.com/example/btc-predictor"}}
---

# BTC Price Predictor

BTC价格预测工具，提供实时BTC价格预测和交易建议。

## 功能

- 📊 **实时预测**: 预测BTC 24小时后的价格走势
- 📈 **交易信号**: 提供BUY/SELL/HOLD信号和置信度
- 🔄 **自动更新**: 每天自动更新数据和训练模型

## 使用方法

### 预测BTC价格

```
btc_predictor predict
```

或使用简化版本（推荐）:

```
python3 ~/.openclaw/skills/btc_predictor/predict_simple_live.py
```

### 输出示例

```
======================================================================
📈 BTC PREDICTION RESULTS
======================================================================

Current Price: $70,531.00
Predicted Return: +0.82%
Predicted Price: $71,109.00

Signal: BUY
Confidence: 75.0%
======================================================================
```

## 技术细节

- **数据源**: Binance API (BTC/USDT)
- **模型**: 时间加权序列模型 (13特征, 256时间步)
- **训练**: 基于2017-2026年数据，70,645样本
- **特征**: returns, log_returns, high_low_range, open_close_range, ma6_ratio, ma12_ratio, ma24_ratio, ma48_ratio, rsi, volatility_24h, volume_ratio, close_position, trend_6h
- **模型位置**: `~/.openclaw/skills/btc_predictor/models/btc_sequence_model.npz`

## 注意事项

⚠️ 这是一个实验性项目，**不构成投资建议**。加密货币交易风险极高，请谨慎投资。
