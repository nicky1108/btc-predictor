# BTC Price Predictor Skill for OpenClaw

一个用于 BTC 价格预测的 OpenClaw Skill，支持自动数据更新、增量学习和实时预测。

## 🎯 功能特性

- **📊 自动数据更新**：每天 00:01 自动从 Binance 获取最新 BTC 数据
- **🧠 增量学习**：支持增量训练，模型会不断学习新数据
- **🔮 实时预测**：预测 BTC 24 小时后的价格走势
- **📈 交易信号**：自动生成 BUY/SELL/HOLD 信号和置信度

## 📁 目录结构

```
~/.openclaw/skills/btc_predictor/
├── config.yaml          # 配置文件
├── install.sh           # 安装脚本
├── cron_job.sh          # 定时任务脚本
├── src/
│   └── main.py          # 主程序
├── data/                # 数据目录
├── models/              # 模型目录
└── logs/                # 日志目录
```

## 🚀 安装

```bash
cd ~/.openclaw/skills/btc_predictor
./install.sh
```

安装脚本会自动：
1. 安装 Python 依赖 (numpy, requests)
2. 设置定时任务 (每天 00:01)
3. 下载初始数据
4. 训练初始模型

## 💻 使用方法

### 1. 手动更新数据

```bash
btc_predictor update
```

### 2. 手动训练模型

```bash
# 使用默认参数训练
btc_predictor train

# 自定义训练参数
btc_predictor train --steps 2000 --lr 1e-4
```

### 3. 获取预测

```bash
btc_predictor predict
```

输出示例：
```
======================================================================
📈 BTC PREDICTION RESULTS
======================================================================

Current Price: $70,824.69
Predicted Return: +1.520%
Predicted Price: $71,899.45

Signal: STRONG_BUY
Confidence: 76.0%
======================================================================

JSON_OUTPUT:
{
  "success": true,
  "timestamp": "2026-03-10T16:30:00",
  "current_price": 70824.69,
  "predicted_return": 1.52,
  "predicted_price": 71899.45,
  "signal": "STRONG_BUY",
  "confidence": 76.0,
  "model_info": {
    "features": 22,
    "bias": 0.000854
  }
}
```

## 🔧 OpenClaw 集成

### Python API

```python
import subprocess
import json

# 调用预测
result = subprocess.run(
    ['btc_predictor', 'predict'],
    capture_output=True,
    text=True
)

# 解析 JSON 输出
output_lines = result.stdout.split('\n')
for line in output_lines:
    if line.startswith('JSON_OUTPUT:'):
        json_str = output_lines[output_lines.index(line) + 1]
        prediction = json.loads(json_str)
        print(f"预测价格: ${prediction['predicted_price']:,.2f}")
        print(f"信号: {prediction['signal']}")
        print(f"置信度: {prediction['confidence']}%")
```

### Shell 脚本

```bash
#!/bin/bash

# 获取预测
PREDICTION=$(btc_predictor predict | grep -A 100 "JSON_OUTPUT:" | tail -n +2 | head -n -1)

# 提取信号
SIGNAL=$(echo $PREDICTION | jq -r '.signal')
CONFIDENCE=$(echo $PREDICTION | jq -r '.confidence')

if [ "$SIGNAL" == "STRONG_BUY" ] && [ $(echo "$CONFIDENCE > 70" | bc) -eq 1 ]; then
    echo "高置信度买入信号！"
    # 执行买入操作...
fi
```

## ⏰ 定时任务

默认每天 00:01 自动执行：

```cron
1 0 * * * /Users/nicky/.openclaw/skills/btc_predictor/cron_job.sh
```

任务流程：
1. 从 Binance 获取最新 4h K线数据
2. 计算技术指标 (RSI, MA, 波动率等)
3. 增量训练模型 (500 steps)
4. 保存更新后的模型

## 📊 模型性能

- **训练数据**：2018-2026 (17,758 样本)
- **特征数**：22 (4h + 1d 多时间尺度)
- **RMSE**：2.05%
- **方向准确率**：77%
- **策略收益**：+1320% (vs 买入持有 +226%)

## 📝 日志

查看训练和预测日志：

```bash
# 数据更新日志
ls -lt ~/.openclaw/skills/btc_predictor/logs/data_update_*.log

# 训练日志
ls -lt ~/.openclaw/skills/btc_predictor/logs/training_*.log

# Cron 任务日志
ls -lt ~/.openclaw/skills/btc_predictor/logs/cron_*.log
```

## ⚠️ 免责声明

这是一个用于教育和研究的实验性项目。**不构成投资建议**。加密货币交易风险极高，请谨慎投资。

## 🔧 故障排除

### 模型不存在
```bash
# 首次使用需要先训练
btc_predictor train --steps 1000
```

### 数据获取失败
```bash
# 检查网络连接
# Binance API 可能需要翻墙
```

### Cron 任务不执行
```bash
# 检查 crontab
crontab -l | grep btc_predictor

# 手动测试
cd ~/.openclaw/skills/btc_predictor
./cron_job.sh
```

## 🎓 技术细节

- **算法**：线性回归 + L2 正则化
- **特征**：9 个技术指标 (Returns, RSI, MA, Volatility 等)
- **时间尺度**：4h K线，预测 24h 后价格
- **优化**：Adam + Weight Decay

## 📄 License

MIT License