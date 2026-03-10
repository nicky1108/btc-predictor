#!/bin/bash
# BTC Predictor Skill Installer

echo "🚀 Installing BTC Predictor Skill..."

SKILL_DIR="$HOME/.openclaw/skills/btc_predictor"

# 检查Python依赖
echo "📦 Checking Python dependencies..."
python3 -c "import numpy" 2>/dev/null || pip3 install numpy
python3 -c "import requests" 2>/dev/null || pip3 install requests

# 设置执行权限
chmod +x "$SKILL_DIR/src/main.py"
chmod +x "$SKILL_DIR/cron_job.sh"

# 创建符号链接到PATH
if [ -d "$HOME/.local/bin" ]; then
    ln -sf "$SKILL_DIR/src/main.py" "$HOME/.local/bin/btc_predictor"
    echo "✅ Created command: btc_predictor"
fi

# 设置定时任务 (00:01 每天)
echo "⏰ Setting up cron job..."
CRON_JOB="1 0 * * * $SKILL_DIR/cron_job.sh"

# 检查是否已存在
if crontab -l 2>/dev/null | grep -q "btc_predictor"; then
    echo "⚠️  Cron job already exists"
else
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Cron job added: 00:01 daily"
fi

# 首次数据下载
echo "📥 Downloading initial data (this may take a few minutes)..."
cd "$SKILL_DIR"
python3 src/main.py update

# 首次训练
echo "🧠 Training initial model..."
python3 src/main.py train --steps 1000 --lr 1e-4

echo ""
echo "✅ BTC Predictor Skill installed successfully!"
echo ""
echo "Usage:"
echo "  btc_predictor update    # Update data manually"
echo "  btc_predictor train     # Train model manually"
echo "  btc_predictor predict   # Get prediction"
echo ""
echo "Automatic updates: Every day at 00:01"
echo "Logs: $SKILL_DIR/logs/"