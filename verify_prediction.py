#!/usr/bin/env python3
"""
验证预测准确性 - 获取实时价格并对比
"""

import requests
import json
import os
from datetime import datetime

BINANCE_API = "https://api.binance.com/api/v3"


def get_current_price():
    """获取当前BTC价格"""
    resp = requests.get(f"{BINANCE_API}/ticker/price?symbol=BTCUSDT", timeout=10)
    return float(resp.json()["price"])


def verify_predictions():
    """验证保存的预测"""
    pred_file = (
        "/Users/nicky/.openclaw/skills/btc_predictor/verification/predictions.json"
    )
    result_file = (
        "/Users/nicky/.openclaw/skills/btc_predictor/verification/results.json"
    )

    if not os.path.exists(pred_file):
        print("没有找到预测记录")
        return

    with open(pred_file, "r") as f:
        predictions = json.load(f)

    results = {}
    now = datetime.now()

    print("=" * 50)
    print(f"验证时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    for horizon, pred in predictions.items():
        verify_time = datetime.strptime(pred["verify_time"], "%Y-%m-%d %H:%M:%S %Z")

        if now >= verify_time:
            print(f"\n>>> 验证 {horizon} 预测...")
            current_price = get_current_price()
            predicted_price = pred["predicted_price"]
            original_price = pred["current_price"]

            actual_change = (current_price - original_price) / original_price * 100
            predicted_direction = pred["signal"]
            actual_direction = "BUY" if actual_change > 0 else "SELL"

            direction_correct = predicted_direction == actual_direction

            results[horizon] = {
                "original_price": original_price,
                "predicted_price": predicted_price,
                "current_price": round(current_price, 2),
                "actual_change_pct": round(actual_change, 2),
                "predicted_direction": predicted_direction,
                "actual_direction": actual_direction,
                "direction_correct": direction_correct,
                "verify_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            }

            print(f"  原始价格: ${original_price:.2f}")
            print(f"  预测价格: ${predicted_price:.2f}")
            print(f"  实际价格: ${current_price:.2f}")
            print(f"  实际变化: {actual_change:+.2f}%")
            print(f"  预测方向: {predicted_direction}")
            print(f"  实际方向: {actual_direction}")
            print(f"  ✓ 方向正确!" if direction_correct else f"  ✗ 方向错误!")
        else:
            print(f"\n{horizon} 验证时间未到: {pred['verify_time']}")
            remaining = verify_time - now
            print(
                f"  剩余时间: {remaining.seconds // 3600}h {(remaining.seconds % 3600) // 60}m"
            )

    if results:
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {result_file}")

    return results


if __name__ == "__main__":
    verify_predictions()
