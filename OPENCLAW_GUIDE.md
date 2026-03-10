# OpenClaw BTC预测器使用指南

## 🚀 快速开始

### 1. 安装Skill

```bash
cd ~/.openclaw/skills/btc_predictor
./install.sh
```

### 2. 配置OpenClaw

确保 `~/.openclaw/mcp.json` 包含以下配置：

```json
{
  "mcpServers": {
    "btc_predictor": {
      "command": "python3",
      "args": [
        "/Users/nicky/.openclaw/skills/btc_predictor/mcp_server.py"
      ],
      "description": "BTC价格预测器"
    }
  }
}
```

## 💬 OpenClaw对话使用

### 场景1：询问BTC价格预测

**你**: 帮我预测一下BTC明天的价格

**OpenClaw** 会自动调用 `btc_predict` 工具：

```
➡️ BTC价格预测

💰 当前价格: $70,824.69
🔮 24小时后: $70,899.45
📊 预测收益: +0.096%

🏷️ 交易信号: HOLD
🟡 置信度: 49.3%

---
📅 更新时间: 2026-03-10T16:30:00
```

### 场景2：询问交易建议

**你**: BTC现在适合买入吗？

**OpenClaw** 会自动调用 `btc_trading_advice` 工具：

```
➡️ 建议观望。市场波动较小(+0.096%)，保持现有仓位。
```

### 场景3：获取市场分析

**你**: 帮我分析下BTC市场

**OpenClaw** 会返回综合分析：

```
📊 BTC市场分析

当前BTC价格位于 $70,824.69，模型预测未来24小时内横盘微弱趋势（约0.1%）。

技术分析显示HOLD信号，建议观望。市场波动较小，建议保持现有仓位。

⚠️ 注意：这是基于历史数据训练的模型预测，不构成投资建议。
```

## 🛠️ 可用工具

| 工具名 | 描述 | 触发关键词 |
|--------|------|-----------|
| `btc_predict` | BTC价格预测 | "预测BTC", "BTC价格", "明天BTC" |
| `btc_trading_advice` | 交易建议 | "买入BTC", "卖出BTC", "交易建议" |

## 📊 理解预测结果

### 交易信号

- **🚀 STRONG_BUY**: 强烈买入 (>1.5%涨幅，置信度>70%)
- **📈 BUY**: 建议买入 (涨幅>0%，置信度中等)
- **➡️ HOLD**: 建议观望 (波动较小)
- **📉 SELL**: 建议卖出 (跌幅>0%)
- **🔴 STRONG_SELL**: 强烈卖出 (跌幅>1.5%)

### 置信度解读

- **🟢 70%+**: 高置信度，模型很有把握
- **🟡 50-70%**: 中等置信度，信号较明确
- **🔴 <50%**: 低置信度，市场信号不明确

## ⚠️ 使用注意事项

1. **这不是投资建议**: 模型基于历史数据训练，不代表未来表现
2. **置信度很重要**: 低置信度(<50%)时建议观望
3. **市场波动**: 加密货币市场剧烈波动，请控制风险
4. **定时更新**: 每天00:01自动更新数据并重新训练

## 🔧 故障排除

### 预测返回错误

```bash
# 检查模型是否存在
ls -lh ~/.openclaw/skills/btc_predictor/models/

# 手动运行预测
python3 ~/.openclaw/skills/btc_predictor/predict.py
```

### MCP工具不显示

```bash
# 重启OpenClaw或重新加载配置
# 检查mcp.json语法
python3 -m json.tool ~/.openclaw/mcp.json
```

### 手动更新数据

```bash
# 手动更新数据和训练
~/.openclaw/skills/btc_predictor/cron_job.sh
```

## 📈 查看日志

```bash
# 查看今天的更新日志
cat ~/.openclaw/skills/btc_predictor/logs/cron_$(date +%Y%m%d).log

# 查看所有日志
ls -lt ~/.openclaw/skills/btc_predictor/logs/
```

## 🎯 高级用法

### 获取原始JSON数据

```bash
# 命令行获取JSON
python3 ~/.openclaw/skills/btc_predictor/openclaw_adapter.py json
```

### 在Python脚本中使用

```python
import subprocess
import json

# 调用预测
result = subprocess.run(
    ['python3', '~/.openclaw/skills/btc_predictor/predict.py'],
    capture_output=True,
    text=True
)

# 解析结果
if 'JSON_OUTPUT:' in result.stdout:
    json_data = result.stdout.split('JSON_OUTPUT:')[1]
    prediction = json.loads(json_data)
    print(f"预测价格: ${prediction['predicted_price']:,.2f}")
    print(f"信号: {prediction['signal']}")
    print(f"置信度: {prediction['confidence']}%")
```

## 📧 支持和反馈

如有问题，请查看日志文件或运行测试：

```bash
# 测试MCP服务器
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | \
  python3 ~/.openclaw/skills/btc_predictor/mcp_server.py

# 测试预测
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"btc_predict"}}' | \
  python3 ~/.openclaw/skills/btc_predictor/mcp_server.py
```

---

**免责声明**: 本Skill仅供教育和研究使用，不构成投资建议。加密货币交易风险极高，请谨慎投资。