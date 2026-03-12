#!/usr/bin/env python3
"""
OpenClaw MCP Server for BTC Predictor Skill
支持JSON-RPC协议
"""

import sys
import json
import subprocess
from pathlib import Path

SKILL_DIR = Path(__file__).parent


def predict_btc():
    """运行预测并返回结构化结果"""
    try:
        result = subprocess.run(
            ["python3", str(SKILL_DIR / "predict_enhanced.py"), "24"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            return {"success": False, "error": result.stderr}

        # 提取JSON
        if "JSON_OUTPUT:" in result.stdout:
            json_part = result.stdout.split("JSON_OUTPUT:")[1]
            return json.loads(json_part.strip())
        else:
            return {"success": False, "error": "No JSON output"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def format_markdown(result):
    """格式化为Markdown"""
    if not result.get("success"):
        return f"❌ 预测失败: {result.get('error')}"

    current = result["current_price"]
    predicted = result["predicted_price"]
    ret = result["predicted_return"]
    signal = result["signal"]
    confidence = result["confidence"]

    signal_emoji = {
        "STRONG_BUY": "🚀",
        "BUY": "📈",
        "buy": "📈",
        "HOLD": "➡️",
        "SELL": "📉",
        "sell": "📉",
        "STRONG_SELL": "🔴",
    }.get(signal, "❓")

    conf_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"

    return f"""{signal_emoji} **BTC价格预测**

💰 **当前价格**: ${current:,.2f}
🔮 **24小时后**: ${predicted:,.2f}
📊 **预测收益**: {ret:+.2f}%

🏷️ **交易信号**: {signal}
{conf_color} **置信度**: {confidence:.1f}% (历史准确率: {result.get("direction_accuracy", "N/A")}%)
"""


def handle_initialize(id):
    """MCP初始化握手"""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "btc_predictor", "version": "1.0.0"},
            "capabilities": {"tools": {}},
        },
    }


def handle_tools_list(id):
    """返回可用工具列表"""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "result": {
            "tools": [
                {
                    "name": "btc_predict",
                    "description": "预测BTC 24小时后的价格走势，返回当前价格、预测价格、收益、交易信号和置信度",
                    "inputSchema": {"type": "object", "properties": {}, "required": []},
                },
                {
                    "name": "btc_trading_advice",
                    "description": "基于模型预测提供BTC交易建议（买入/卖出/观望）",
                    "inputSchema": {"type": "object", "properties": {}, "required": []},
                },
            ]
        },
    }


def handle_tools_call(id, name, arguments):
    """处理工具调用"""
    if name == "btc_predict":
        result = predict_btc()
        if result.get("success"):
            content = format_markdown(result)
            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": content}],
                    "isError": False,
                },
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [
                        {"type": "text", "text": f"❌ 预测失败: {result.get('error')}"}
                    ],
                    "isError": True,
                },
            }

    elif name == "btc_trading_advice":
        result = predict_btc()
        if result.get("success"):
            signal = result["signal"]
            confidence = result["confidence"]
            ret = result["predicted_return"]

            advice = {
                "STRONG_BUY": f"🚀 强烈建议买入！预测24h后上涨{ret:.2f}%，置信度{confidence:.1f}%。",
                "BUY": f"📈 建议买入。预测上涨{ret:.2f}%，建议小仓位试探。",
                "HOLD": f"➡️ 建议观望。市场波动较小({ret:.2f}%)，保持现有仓位。",
                "SELL": f"📉 建议卖出。预测下跌{ret:.2f}%，建议减仓。",
                "STRONG_SELL": f"🔴 强烈建议卖出！预测24h后下跌{ret:.2f}%，建议清仓或做空。",
            }.get(signal, "市场信号不明确，建议观望。")

            return {
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{"type": "text", "text": advice}],
                    "isError": False,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": -32601, "message": f"Tool not found: {name}"},
    }


def main():
    """MCP主循环"""
    while True:
        try:
            line = input()
            if not line:
                continue

            request = json.loads(line)
            method = request.get("method")
            id = request.get("id")

            if method == "initialize":
                response = handle_initialize(id)
            elif method == "tools/list":
                response = handle_tools_list(id)
            elif method == "tools/call":
                name = request["params"]["name"]
                arguments = request["params"].get("arguments", {})
                response = handle_tools_call(id, name, arguments)
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            print(json.dumps(response))
            sys.stdout.flush()

        except EOFError:
            break
        except json.JSONDecodeError as e:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": f"Parse error: {e}"},
                    }
                )
            )
            sys.stdout.flush()
        except Exception as e:
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": f"Internal error: {e}"},
                    }
                )
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
