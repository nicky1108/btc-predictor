# BTC Price Predictor for OpenClaw

使用双向LSTM + 多时间周期融合的BTC价格预测模型。

## 🎯 功能特性

- **🤖 LSTM架构**: 双向LSTM (2层, 32 hidden, dropout=0.2)
- **📊 多时间周期**: 融合1h + 4h + 1d三级时间周期特征
- **📈 22技术指标**: RSI, MACD, Bollinger Bands, ATR, OBV等
- **🔮 多周期预测**: 支持 6h / 24h / 48h / 168h 预测
- **📊 置信度校准**: 基于验证集真实准确率

## 📁 目录结构

```
btc_predictor/
├── SKILL.md                    # OpenClaw Skill定义
├── mcp_server.py               # MCP服务器 (JSON-RPC)
├── predict_enhanced.py         # 增强版LSTM预测 (推荐)
├── predict_lstm.py             # 基础LSTM预测
├── predict_transformer_v2.py   # 旧Transformer封装模型
├── train_*.py                  # 训练脚本
├── models/
│   ├── btc_lstm_h6.pt        # 6小时预测模型
│   ├── btc_lstm_h24.pt       # 24小时预测模型
│   ├── btc_lstm_h48.pt       # 48小时预测模型
│   ├── btc_lstm_h168.pt      # 168小时(7天)预测模型
│   ├── calibration.json       # 校准数据
│   └── btc_model_new.npz     # 旧线性模型
└── logs/                      # 训练日志
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

### 命令行使用

```bash
# 使用增强版LSTM模型预测 (推荐)
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 24

# 预测多个周期
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 6   # 6小时
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 24  # 24小时
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 48  # 48小时
python3 ~/.openclaw/skills/btc_predictor/predict_enhanced.py 168 # 7天
```

输出示例：
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

## 🧠 模型信息

### 性能对比

| 周期 | 方向准确率 | MSE |
|------|-----------|-----|
| 6h | **59.3%** | 0.000279 |
| 24h | 55.5% | 0.000276 |
| 48h | **60.5%** | 0.000249 |
| 168h (7天) | 59.3% | 0.000202 |

### 模型架构

| 项目 | 值 |
|------|-----|
| **架构** | 双向LSTM |
| **层数** | 2 |
| **Hidden Size** | 32 |
| **Dropout** | 0.2 |
| **特征维度** | 66 (22 × 3时间周期) |
| **序列长度** | 48 |
| **正则化** | Weight Decay (1e-4) |

### 特征列表 (22个)

1. `returns` - 收益率
2. `log_returns` - 对数收益率
3. `high_low_range` - 最高最低价范围
4. `close_position` - 收盘位置
5. `sma_5_ratio` - 5周期均线比率
6. `sma_10_ratio` - 10周期均线比率
7. `sma_20_ratio` - 20周期均线比率
8. `sma_50_ratio` - 50周期均线比率
9. `rsi_14` - 14周期RSI
10. `rsi_7` - 7周期RSI
11. `macd` - MACD
12. `macd_signal` - MACD信号线
13. `macd_hist` - MACD柱状图
14. `bb_position` - Bollinger Bands位置
15. `atr_ratio` - ATR比率
16. `volume_ratio` - 成交量比率
17. `obv_change` - OBV变化
18. `stoch_k` - Stochastic %K
19. `adx` - ADX趋势指标
20. `volatility_20` - 20周期波动率
21. `momentum_5` - 5周期动量
22. `momentum_10` - 10周期动量

### 多时间周期融合

- **1小时数据**: 主时间序列
- **4小时数据**: 中期趋势特征
- **1天数据**: 长期趋势特征

每个时间周期贡献22个特征，总计66维输入。

## 📊 训练数据

- **数据源**: Binance BTC/USDT
- **时间周期**: 1h / 4h / 1d
- **时间范围**: 2017-2026
- **样本数**: ~1000 (用于训练)
- **验证方式**: 5折交叉验证

## 🔧 MCP工具

OpenClaw提供以下MCP工具：

| 工具名 | 描述 |
|--------|------|
| `btc_predict` | BTC价格预测 |
| `btc_trading_advice` | 交易建议 |

## 📈 版本历史

- **v2.0** (2024-03): 双向LSTM + 多时间周期融合
  - 22技术指标 × 3时间周期 = 66维特征
  - 多周期预测支持 (6h/24h/48h/168h)
  - 置信度基于验证集校准
- **v1.0** (早期): Transformer封装线性模型

## ⚠️ 免责声明

这是一个用于教育和研究的实验性项目。**不构成投资建议**。

- 加密货币交易风险极高，请谨慎投资
- 历史表现不代表未来收益
- 模型预测仅供参考

## 📄 License

MIT License
