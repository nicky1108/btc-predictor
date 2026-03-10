#!/bin/bash
# BTC Predictor Cron Job
# 每天 00:01 执行数据更新和增量训练

SKILL_DIR="$HOME/.openclaw/skills/btc_predictor"
LOG_FILE="$SKILL_DIR/logs/cron_$(date +%Y%m%d).log"

echo "[$(date)] Starting BTC Predictor scheduled task..." >> "$LOG_FILE"

# 更新数据
cd "$SKILL_DIR"
python3 src/main.py update >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "[$(date)] Data update successful" >> "$LOG_FILE"
    
    # 增量训练
    python3 src/main.py train --steps 500 --lr 1e-4 >> "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[$(date)] Incremental training successful" >> "$LOG_FILE"
    else
        echo "[$(date)] ❌ Training failed" >> "$LOG_FILE"
    fi
else
    echo "[$(date)] ❌ Data update failed" >> "$LOG_FILE"
fi

echo "[$(date)] Task complete" >> "$LOG_FILE"