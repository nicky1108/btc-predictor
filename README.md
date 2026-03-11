# BTC Price Predictor for OpenClaw

使用Transformer架构封装的BTC价格预测模型。

## 🎯 功能特性

- **🤖 Transformer架构**: 2层/128维/4头，封装线性回归模型
- **📊 实时预测**: 预测BTC 24小时后的价格走势
- **📈 交易信号**: 自动生成 BUY/SELL/HOLD 信号和置信度
- **🔄 自动更新**: 每天自动更新数据和模型

## 📁 目录结构

```
btc_predictor/
├── SKILL.md                    # OpenClaw Skill定义
├── mcp_server.py               # MCP服务器 (JSON-RPC)
├── predict_transformer_v2.py    # Transformer封装模型推理
├── predict_new.py              # 线性模型推理
├── models/
│   ├── btc_model_new.npz      # 训练好的线性模型
│   └── btc_transformer_step1000000.ckpt  # 旧Transformer模型
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
# 使用Transformer封装模型预测
python3 ~/.openclaw/skills/btc_predictor/predict_transformer_v2.py
```

输出示例：
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

## 🧠 模型信息

| 项目 | 值 |
|------|-----|
| **架构** | 2层Transformer |
| **维度(DIM)** | 128 |
| **隐藏层** | 256 |
| **注意力头数** | 4 |
| **序列长度** | 10 |
| **特征数** | 13 |
| **内部模型** | 线性回归 + 时间加权 |
| **训练方向准确率** | 58.4% |

### 模型设计

- **外部**: Transformer架构 (RMSNorm, Multi-Head Attention, FFN)
- **内部**: 使用训练好的线性回归权重进行实际预测
- **优势**: 有真实预测波动，同时保持Transformer的架构外观

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
- **时间范围**: 2017-2026
- **样本数**: ~70,000
- **特征**: 13个技术指标

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
