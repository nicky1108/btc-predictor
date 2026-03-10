#!/usr/bin/env python3
"""
OpenClaw Adapter for BTC Predictor Skill
在OpenClaw对话中调用BTC预测
"""

import subprocess
import json
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).parent


def predict_btc_price():
    """
    预测BTC价格
    返回格式化的预测结果
    """
    try:
        # 运行预测脚本
        result = subprocess.run(
            ["python3", str(SKILL_DIR / "predict.py")],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        # 解析JSON输出
        output = result.stdout

        # 提取JSON部分
        if "JSON_OUTPUT:" in output:
            json_part = output.split("JSON_OUTPUT:")[1]
            prediction = json.loads(json_part.strip())
            return prediction
        else:
            return {"success": False, "error": "No JSON output found"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def format_prediction(result):
    """
    格式化预测结果为自然语言
    """
    if not result.get("success"):
        return f"❌ 预测失败: {result.get('error', '未知错误')}"

    current = result["current_price"]
    predicted = result["predicted_price"]
    ret = result["predicted_return"]
    signal = result["signal"]
    confidence = result["confidence"]

    # 信号表情
    signal_emoji = {
        "STRONG_BUY": "🚀",
        "BUY": "📈",
        "HOLD": "➡️",
        "SELL": "📉",
        "STRONG_SELL": "🔴",
    }.get(signal, "❓")

    # 置信度颜色
    if confidence >= 70:
        conf_color = "🟢"
    elif confidence >= 50:
        conf_color = "🟡"
    else:
        conf_color = "🔴"

    response = f"""
{signal_emoji} **BTC价格预测**

💰 **当前价格**: ${current:,.2f}
🔮 **24小时后**: ${predicted:,.2f}
📊 **预测收益**: {ret:+.2f}%

🏷️ **交易信号**: {signal}
{conf_color} **置信度**: {confidence:.1f}%

---
📅 更新时间: {result["timestamp"]}
"""

    return response.strip()


def get_trading_advice():
    """
    获取交易建议
    """
    result = predict_btc_price()

    if not result.get("success"):
        return "抱歉，目前无法获取BTC预测。请检查网络连接或稍后重试。"

    signal = result["signal"]
    confidence = result["confidence"]
    ret = result["predicted_return"]

    advice = {
        "STRONG_BUY": f"强烈建议买入！模型预测24小时后上涨{ret:.2f}%，置信度{confidence:.1f}%。",
        "BUY": f"建议买入。预测显示上涨趋势({ret:+.2f}%)，但置信度中等({confidence:.1f}%)。",
        "HOLD": f"建议观望。市场信号不明显，预测波动较小({ret:+.2f}%)，建议持有现有仓位。",
        "SELL": f"建议卖出。预测显示下跌趋势({ret:.2f}%)，建议减仓或做空。",
        "STRONG_SELL": f"强烈建议卖出！模型预测24小时后下跌{ret:.2f}%，置信度{confidence:.1f}%。",
    }.get(signal, "市场信号不明确，建议观望。")

    return advice


def get_market_summary():
    """
    获取市场摘要
    """
    result = predict_btc_price()

    if not result.get("success"):
        return "无法获取市场数据。"

    current = result["current_price"]
    ret = result["predicted_return"]
    signal = result["signal"]

    trend = "上涨" if ret > 0 else "下跌" if ret < 0 else "横盘"
    strength = "强劲" if abs(ret) > 0.02 else "温和" if abs(ret) > 0.01 else "微弱"

    return f"""
📊 **BTC市场分析**

当前BTC价格位于 ${current:,.2f}，模型预测未来24小时内{trend}{strength}趋势（约{abs(ret) * 100:.1f}%）。

技术分析显示{signal.lower().replace("_", " ")}信号，建议{get_trading_advice()}。

⚠️ 注意：这是基于历史数据训练的模型预测，不构成投资建议。加密货币市场波动剧烈，请谨慎决策。
""".strip()


# OpenClaw MCP Tool 定义
MCP_TOOLS = {
    "btc_predict": {
        "description": "预测BTC 24小时后的价格",
        "parameters": {},
        "returns": {
            "current_price": "float",
            "predicted_price": "float",
            "predicted_return": "float",
            "signal": "str (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)",
            "confidence": "float",
        },
    },
    "btc_trading_advice": {
        "description": "获取BTC交易建议",
        "parameters": {},
        "returns": "str",
    },
    "btc_market_summary": {
        "description": "获取BTC市场摘要分析",
        "parameters": {},
        "returns": "str",
    },
}

if __name__ == "__main__":
    # 命令行测试
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "predict":
            result = predict_btc_price()
            print(format_prediction(result))
        elif command == "advice":
            print(get_trading_advice())
        elif command == "summary":
            print(get_market_summary())
        elif command == "json":
            result = predict_btc_price()
            print(json.dumps(result, indent=2))
    else:
        # 默认输出预测
        result = predict_btc_price()
        print(format_prediction(result))
