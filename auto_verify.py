#!/usr/bin/env python3
"""
后台自动验证脚本 - 定时检查并验证预测
"""

import time
import subprocess
import os
from datetime import datetime, timedelta

PRED_FILE = "/Users/nicky/.openclaw/skills/btc_predictor/verification/predictions.json"
LOG_FILE = "/Users/nicky/.openclaw/skills/btc_predictor/verification/auto_verify.log"


def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")
    print(msg)


def check_and_verify():
    """检查是否到了验证时间"""
    import json

    if not os.path.exists(PRED_FILE):
        log("没有预测记录，退出")
        return False

    with open(PRED_FILE, "r") as f:
        predictions = json.load(f)

    now = datetime.now()
    verified_any = False

    for horizon, pred in predictions.items():
        verify_time_str = pred["verify_time"]
        verify_time = datetime.strptime(verify_time_str, "%Y-%m-%d %H:%M:%S %Z")

        # 允许5分钟误差
        time_diff = abs((now - verify_time).total_seconds())

        if now >= verify_time and time_diff < 300:
            log(f"开始验证 {horizon}...")
            result = subprocess.run(
                [
                    "python3",
                    "/Users/nicky/.openclaw/skills/btc_predictor/verify_prediction.py",
                ],
                capture_output=True,
                text=True,
            )
            log(result.stdout)
            verified_any = True

    return verified_any


def main():
    log("自动验证服务启动")

    # 验证时间
    verify_6h = datetime(2026, 3, 12, 21, 25, 48)
    verify_24h = datetime(2026, 3, 13, 15, 25, 48)

    while True:
        now = datetime.now()

        # 检查是否到了验证时间 (允许5分钟窗口)
        if now >= verify_6h and (now - verify_6h).seconds < 300:
            log("6h验证时间到达")
            os.system(
                "python3 /Users/nicky/.openclaw/skills/btc_predictor/verify_prediction.py"
            )

        if now >= verify_24h and (now - verify_24h).seconds < 300:
            log("24h验证时间到达")
            os.system(
                "python3 /Users/nicky/.openclaw/skills/btc_predictor/verify_prediction.py"
            )

        # 如果两个时间都过了，退出
        if now > verify_24h:
            log("所有验证完成，退出")
            break

        # 每30秒检查一次
        time.sleep(30)


if __name__ == "__main__":
    main()
