# BTC Price Predictor for OpenClaw

使用12层Transformer模型进行BTC价格预测的OpenClaw Skill。

## 🎯 功能特性

- **🤖 12层Transformer模型**: 630万参数的大型时间序列预测模型
- **📊 实时预测**: 预测BTC 24小时后的价格走势
- **📈 交易信号**: 自动生成 BUY/SELL/HOLD 信号和置信度
- **🔄 自动更新**: 每天自动更新数据和模型

## 📁 目录结构

```
btc_predictor/
├── SKILL.md                    # OpenClaw Skill定义
├── mcp_server.py               # MCP服务器 (JSON-RPC)
├── predict_transformer.py       # Transformer模型推理
├── predict_simple_live.py      # 简化版预测
├── models/
│   └── btc_transformer_step1000000.ckpt  # 训练好的Transformer模型
└── data/                       # 数据目录
```

## 🚀 安装

```bash
# 克隆到OpenClaw Skills目录
git clone https://github.com/nicky1108/btc-predictor.git ~/.openclaw/skills/btc_predictor
```

## 💻 使用方法

### OpenClaw中使用

在OpenClaw对话中直接调用：

```
帮我预测一下BTC明天的价格
```

OpenClaw会自动调用`btc_predict`工具返回预测结果。

### 命令行使用

```bash
# 使用Transformer模型预测
python3 ~/.openclaw/skills/btc_predictor/predict_transformer.py
```

输出示例：
```
============================================================
BTC Transformer Prediction (OpenClaw Skill)
============================================================

📥 Loading model...
✓ Model: 12 layers, step=1000000, loss=0.001257

📡 Fetching data...
✓ Got 300 candles

🔮 Running transformer...

============================================================
📈 PREDICTION RESULTS
============================================================

  💰 Current Price: $71,072.54
  🔮 24h Prediction: $71,072.70
  📊 Expected Return: +0.00%

  Signal: HOLD (confidence: 50%)
```

## 🧠 模型信息

| 项目 | 值 |
|------|-----|
| **架构** | 12层Transformer |
| **维度(DIM)** | 256 |
| **隐藏层** | 1024 |
| **注意力头数** | 8 |
| **序列长度** | 256 |
| **参数量** | ~630万 |
| **训练步数** | 1,000,000 |
| **训练损失** | 0.001257 |

### 特征列表 (13个)

1. `returns` - 收益率
2. `log_returns` - 对数收益率
3. `high_low_range` - 最高最低价范围
4. `open_close_range` - 开收盘价范围
5. `ma6_ratio` - 6小时均线比率
6. `ma12_ratio` - 12小时均线比率
7. `ma24_ratio` - 24小时均线比率
8. `ma48_ratio` - 48小时均线比率
9. `rsi` - 相对强弱指数
10. `volatility_24h` - 24小时波动率
11. `volume_ratio` - 成交量比率
12. `close_position` - 收盘位置
13. `trend_6h` - 6小时趋势

## 📊 训练数据

- **数据源**: Binance BTC/USDT 1小时K线
- **时间范围**: 2018-2026
- **样本数**: ~70,000

## 🔧 MCP工具

OpenClaw提供以下MCP工具：

| 工具名 | 描述 |
|--------|------|
| `btc_predict` | BTC价格预测 |
| `btc_trading_advice` | 交易建议 |

## ⚠️ 免责声明

这是一个用于教育和研究的实验性项目。**不构成投资建议**。

- 加密货币交易风险极高，请谨慎投资
- 历史表现不代表未来收益
- 模型预测仅供参考

## 📄 License

MIT License
