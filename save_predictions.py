#!/usr/bin/env python3
"""
保存预测结果，用于后续验证
"""

import json
from datetime import datetime, timedelta

predictions = {
    "6h": {
        "current_price": 69614.83,
        "predicted_price": 69944.72,
        "predicted_change_pct": 0.47,
        "signal": "BUY",
        "confidence": 60.2,
        "direction_accuracy": 59.3,
        "prediction_time": "2026-03-12 15:25:48 CST",
        "verify_time": "2026-03-12 21:25:48 CST",
    },
    "24h": {
        "current_price": 69600.70,
        "predicted_price": 69825.37,
        "predicted_change_pct": 0.32,
        "signal": "BUY",
        "confidence": 56.1,
        "direction_accuracy": 55.5,
        "prediction_time": "2026-03-12 15:25:48 CST",
        "verify_time": "2026-03-13 15:25:48 CST",
    },
}

with open(
    "/Users/nicky/.openclaw/skills/btc_predictor/verification/predictions.json", "w"
) as f:
    json.dump(predictions, f, indent=2, ensure_ascii=False)

print("预测已保存")
print(f"6h验证时间: {predictions['6h']['verify_time']}")
print(f"24h验证时间: {predictions['24h']['verify_time']}")
