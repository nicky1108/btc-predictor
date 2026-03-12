---
name: btc-predictor
description: BTC价格预测 - 使用双向LSTM + 多时间周期融合模型，预测BTC价格走势，提供交易信号和置信度
metadata: {"openclaw": {"emoji": "📈", "requires": {"bins": ["python3"], "env": []}, "homepage": "https://github.com/nicky1108/btc-predictor"}}
---

# BTC Price Predictor

BTC价格预测工具，使用双向LSTM + 多时间周期融合模型进行实时预测。

## 功能

- 📊 **实时预测**: 预测BTC 6h/24h/48h/168h价格走势
- 📈 **交易信号**: 提供BUY/SELL信号和置信度
- 🔄 **自动更新**: 每天自动更新数据和训练模型
- 🤖 **双向LSTM**: 2层/32维/dropout=0.2
- 📊 **多时间周期**: 融合1h + 4h + 1d数据

## 使用方法

### OpenClaw中调用

在OpenClaw对话中直接说：

```
帮我预测一下BTC明天的价格
```

### 命令行使用

```bash
# 24小时预测 (默认)
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 24

# 6小时预测
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 6

# 48小时预测
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 48
```

### 输出示例

```
Fetching data for 24h prediction...
Calculating features...
Loading model: btc_lstm_h24.pt

==================================================
当前价格: $69,452.02
预测 (24h): $69,730.23 (+0.40%)
信号: BUY
置信度: 56.3% (方向准确率: 55.5%)
==================================================
```
帮我预测一下BTC明天的价格
```

### 命令行使用

```bash
python3 ~/.openclaw/skills/btc_predictor/predict_transformer_v2.py
```

### 输出示例

```
============================================================
BTC Transformer Model (Linear-wrapped)
============================================================

📥 Loading model...
✓ Model loaded
  Architecture: 2 layers, 128 dim, 4 heads
  Parameters: 13 features → linear head

📡 Fetching data...
✓ Got 300 candles

🔮 Running Transformer inference...

============================================================
📈 PREDICTION RESULTS
============================================================

  💰 Current Price: $69,903.76
  🔮 24h Prediction: $66,506.33
  📊 Expected Return: -4.86%

  Signal: STRONG_SELL (confidence: 90%)
```

## 技术细节

- **数据源**: Binance API (BTC/USDT)
- **模型**: Transformer封装线性回归
- **架构**: 2层Transformer, 128维, 4头
- **内部**: 线性回归 + 时间加权聚合
- **训练**: 70,645样本, 方向准确率58.4%
- **特征**: 13个技术指标 (returns, rsi, ma, volatility等)

## 模型位置

- 主模型: `~/.openclaw/skills/btc_predictor/models/btc_model_new.npz`
- 推理脚本: `~/.openclaw/skills/btc_predictor/predict_transformer_v2.py`

## 注意事项

⚠️ 这是一个实验性项目，**不构成投资建议**。加密货币交易风险极高，请谨慎投资。
