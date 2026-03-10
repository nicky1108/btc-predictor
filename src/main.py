#!/usr/bin/env python3
"""
BTC Price Predictor Skill for OpenClaw
定时数据更新 + 增量学习 + 实时预测
"""

import os
import sys
import json
import struct
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional

# 配置
SKILL_DIR = Path(__file__).parent
DATA_DIR = SKILL_DIR / "data"
MODELS_DIR = SKILL_DIR / "models"
LOGS_DIR = SKILL_DIR / "logs"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Binance API
BINANCE_API = "https://api.binance.com/api/v3/klines"


class BTCDataManager:
    """管理BTC数据，支持增量更新"""

    def __init__(self):
        self.data_1h_path = DATA_DIR / "btc_1h.bin"
        self.data_4h_path = DATA_DIR / "btc_4h.bin"
        self.data_1d_path = DATA_DIR / "btc_1d.bin"

    def log(self, msg: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)

        # 写入日志文件
        log_file = LOGS_DIR / f"data_update_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    def fetch_klines(
        self,
        interval: str,
        limit: int = 1000,
        start_time: datetime = None,
        end_time: datetime = None,
    ) -> List[Dict]:
        """从Binance获取K线数据"""
        params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        try:
            response = requests.get(BINANCE_API, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            candles = []
            for kline in data:
                candles.append(
                    {
                        "timestamp": datetime.fromtimestamp(kline[0] / 1000),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                    }
                )
            return candles
        except Exception as e:
            self.log(f"❌ Error fetching {interval} data: {e}")
            return []

    def load_existing_data(
        self, filepath: Path
    ) -> Tuple[Optional[np.ndarray], int, int, int]:
        """加载已有数据"""
        if not filepath.exists():
            return None, 0, 0, 0

        with open(filepath, "rb") as f:
            n_samples = struct.unpack("<I", f.read(4))[0]
            seq_len = struct.unpack("<I", f.read(4))[0]
            n_features = struct.unpack("<I", f.read(4))[0]
            data = np.fromfile(f, dtype=np.float32)

        return data, n_samples, seq_len, n_features

    def calculate_features(self, candles: List[Dict]) -> np.ndarray:
        """计算技术指标"""
        prices = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]

        features = []
        for i in range(len(candles)):
            c = candles[i]
            current = c["close"]
            price_list = prices[: i + 1]

            # 基础特征
            returns = (
                (current - candles[i - 1]["close"]) / candles[i - 1]["close"]
                if i > 0
                else 0
            )
            hl_range = (c["high"] - c["low"]) / current if current > 0 else 0

            # 移动平均
            def ma(prices, window):
                if len(prices) < window:
                    return sum(prices) / len(prices) if prices else current
                return sum(prices[-window:]) / window

            ma6 = current / ma(price_list, 6) if ma(price_list, 6) > 0 else 1
            ma12 = current / ma(price_list, 12) if ma(price_list, 12) > 0 else 1
            ma24 = current / ma(price_list, 24) if ma(price_list, 24) > 0 else 1

            # RSI
            def calc_rsi(prices, period=14):
                if len(prices) < period + 1:
                    return 50.0
                gains, losses = [], []
                for j in range(1, min(period + 1, len(prices))):
                    change = prices[-j] - prices[-j - 1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(-change)
                avg_gain = sum(gains) / period if gains else 0.0001
                avg_loss = sum(losses) / period if losses else 0.0001
                return 100 - (100 / (1 + avg_gain / avg_loss))

            rsi = calc_rsi(price_list, 14)

            # 波动率
            returns_list = [
                features[j][0] if j < len(features) else 0
                for j in range(max(0, i - 24), i)
            ]
            if len(returns_list) > 1:
                mean = sum(returns_list) / len(returns_list)
                variance = sum((r - mean) ** 2 for r in returns_list) / len(
                    returns_list
                )
                volatility = variance**0.5
            else:
                volatility = 0.01

            # 成交量
            vol_list = volumes[: i + 1]
            vol_ma = ma(vol_list, 24)
            volume_ratio = c["volume"] / vol_ma if vol_ma > 0 else 1

            # 价格位置
            hl = c["high"] - c["low"]
            close_pos = (c["close"] - c["low"]) / hl if hl > 0 else 0.5

            feat = [
                returns,
                hl_range,
                ma6,
                ma12,
                ma24,
                rsi,
                volatility,
                volume_ratio,
                close_pos,
            ]
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征"""
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        stds = np.where(stds < 1e-10, 1, stds)
        return (features - means) / stds

    def create_sequences(
        self, features: np.ndarray, seq_len: int = 64, pred_periods: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """创建训练序列"""
        sequences = []
        targets = []

        # 假设每6个4h周期是24h
        for i in range(seq_len, len(features) - pred_periods):
            seq = features[i - seq_len : i]
            # 目标：预测24h后的收益率（下一个pred_periods的收盘价对比当前价格）
            target_idx = min(i + pred_periods - 1, len(features) - 1)
            # 使用returns作为target（第0个特征就是returns）
            target = features[target_idx, 0] if target_idx < len(features) else 0

            sequences.append(seq)
            targets.append(target)

        return np.array(sequences, dtype=np.float32), np.array(
            targets, dtype=np.float32
        )

    def update_data(self, interval: str = "4h"):
        """增量更新数据"""
        self.log(f"🔄 Starting {interval} data update...")

        # 确定数据文件路径
        if interval == "1h":
            filepath = self.data_1h_path
        elif interval == "4h":
            filepath = self.data_4h_path
        else:
            filepath = self.data_1d_path

        # 检查现有数据
        existing_data, n_samples, seq_len, n_features = self.load_existing_data(
            filepath
        )

        if existing_data is not None:
            self.log(f"📊 Found existing data: {n_samples} samples")
            # 获取最后的时间戳（简化：假设数据是连续的）
            last_fetch_time = datetime.now() - timedelta(days=1)
        else:
            self.log("🆕 No existing data, fetching from 2018...")
            last_fetch_time = datetime(2018, 1, 1)

        # 获取新数据
        end_time = datetime.now()
        new_candles = []

        self.log(f"📡 Fetching {interval} data from {last_fetch_time} to {end_time}...")
        current_start = last_fetch_time

        while current_start < end_time:
            if interval == "1h":
                current_end = min(current_start + timedelta(hours=1000), end_time)
            elif interval == "4h":
                current_end = min(current_start + timedelta(hours=4 * 1000), end_time)
            else:
                current_end = min(current_start + timedelta(days=1000), end_time)

            candles = self.fetch_klines(
                interval, start_time=current_start, end_time=current_end
            )
            if candles:
                new_candles.extend(candles)
                current_start = candles[-1]["timestamp"] + timedelta(
                    hours=1 if interval == "1h" else 4 if interval == "4h" else 24
                )
                self.log(f"  Fetched {len(candles)} candles, total: {len(new_candles)}")
            else:
                break

            import time

            time.sleep(0.1)

        if not new_candles:
            self.log("✅ No new data to update")
            return False

        # 计算特征
        self.log("🔢 Calculating features...")
        features = self.calculate_features(new_candles)
        normalized = self.normalize_features(features)

        # 创建序列
        sequences, targets = self.create_sequences(
            normalized, seq_len=64, pred_periods=6
        )

        if len(sequences) == 0:
            self.log("⚠️  No sequences created")
            return False

        # 保存数据（简化版本，只保存原始特征）
        self.log(f"💾 Saving {len(sequences)} sequences...")

        # 合并新旧数据（简化：直接覆盖）
        n_samples = len(sequences)
        seq_len = sequences.shape[1]
        n_features = sequences.shape[2]

        with open(filepath, "wb") as f:
            f.write(struct.pack("<I", n_samples))
            f.write(struct.pack("<I", seq_len))
            f.write(struct.pack("<I", n_features))

            for i in range(n_samples):
                for t in range(seq_len):
                    for feat in range(n_features):
                        f.write(struct.pack("<f", sequences[i, t, feat]))
                f.write(struct.pack("<f", targets[i]))

        self.log(f"✅ Data update complete: {n_samples} samples saved")
        return True


class BTCModelTrainer:
    """BTC模型训练器"""

    def __init__(self):
        self.model_path = MODELS_DIR / "btc_model.bin"
        self.data_manager = BTCDataManager()

    def log(self, msg: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {msg}"
        print(log_msg)

        log_file = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    def load_model(self) -> Tuple[Optional[np.ndarray], float, int]:
        """加载模型"""
        if not self.model_path.exists():
            self.log("🆕 No existing model, will create new one")
            return None, 0.0, 0

        with open(self.model_path, "rb") as f:
            step = struct.unpack("<i", f.read(4))[0]
            loss = struct.unpack("<f", f.read(4))[0]
            n_feat = struct.unpack("<i", f.read(4))[0]
            weights = np.fromfile(f, dtype=np.float32, count=n_feat)
            bias = struct.unpack("<f", f.read(4))[0]

        self.log(f"📥 Model loaded: step={step}, loss={loss:.6f}, features={n_feat}")
        return weights, bias, n_feat

    def save_model(self, weights: np.ndarray, bias: float, step: int, loss: float):
        """保存模型"""
        with open(self.model_path, "wb") as f:
            f.write(struct.pack("<i", step))
            f.write(struct.pack("<f", loss))
            f.write(struct.pack("<i", len(weights)))
            weights.tofile(f)
            f.write(struct.pack("<f", bias))

        self.log(f"💾 Model saved: step={step}, loss={loss:.6f}")

    def train_incremental(
        self, n_steps: int = 1000, lr: float = 1e-4, weight_decay: float = 1e-5
    ):
        """增量训练"""
        self.log(f"🚀 Starting incremental training...")
        self.log(f"   Steps: {n_steps}, LR: {lr}, Weight Decay: {weight_decay}")

        # 加载现有模型
        weights, bias, n_feat = self.load_model()

        # 加载数据
        data_path = DATA_DIR / "btc_4h.bin"
        if not data_path.exists():
            self.log("❌ No training data available")
            return False

        with open(data_path, "rb") as f:
            n_samples = struct.unpack("<I", f.read(4))[0]
            seq_len = struct.unpack("<I", f.read(4))[0]
            n_features = struct.unpack("<I", f.read(4))[0]

            if weights is None:
                weights = np.zeros(n_features, dtype=np.float32)
                bias = 0.0
                self.log(f"🆕 Initialized new model with {n_features} features")

            # 读取所有数据
            sample_size = seq_len * n_features + 1
            total_floats = n_samples * sample_size
            data = np.fromfile(f, dtype=np.float32, count=total_floats)

        self.log(f"📊 Loaded {n_samples} training samples")

        # 训练
        best_loss = float("inf")
        total_loss = 0

        np.random.seed(42)

        for step in range(n_steps):
            # 随机采样
            idx = np.random.randint(0, n_samples)
            offset = idx * sample_size

            features = data[offset : offset + seq_len * n_features].reshape(
                seq_len, n_features
            )
            target = data[offset + seq_len * n_features]

            # 前向传播
            last = features[-1, :]
            pred = np.dot(last, weights) + bias
            pred = np.clip(pred, -0.3, 0.3)

            # 损失
            diff = pred - target
            loss = diff * diff
            total_loss += loss

            # 反向传播 + L2正则化
            grad = 2.0 * diff
            bias -= lr * grad

            for i in range(n_features):
                l2_penalty = weight_decay * weights[i]
                weights[i] -= lr * (grad * last[i] + l2_penalty)

            # 日志
            if (step + 1) % 100 == 0:
                avg_loss = total_loss / 100
                self.log(
                    f"Step {step + 1}/{n_steps}: loss={avg_loss:.6f}, pred={pred * 100:.4f}%, target={target * 100:.4f}%"
                )

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_model(weights, bias, step + 1, best_loss)

                total_loss = 0

        # 保存最终模型
        self.save_model(weights, bias, n_steps, best_loss)
        self.log(f"✅ Training complete! Best loss: {best_loss:.6f}")

        return True


class BTCPredictor:
    """BTC预测器"""

    def __init__(self):
        self.model_path = MODELS_DIR / "btc_model.bin"
        self.data_manager = BTCDataManager()

    def load_model(self) -> Tuple[Optional[np.ndarray], float, int]:
        """加载模型"""
        if not self.model_path.exists():
            return None, 0.0, 0

        with open(self.model_path, "rb") as f:
            step = struct.unpack("<i", f.read(4))[0]
            loss = struct.unpack("<f", f.read(4))[0]
            n_feat = struct.unpack("<i", f.read(4))[0]
            weights = np.fromfile(f, dtype=np.float32, count=n_feat)
            bias = struct.unpack("<f", f.read(4))[0]

        return weights, bias, n_feat

    def predict(self) -> Dict:
        """预测BTC 24小时后的价格"""
        # 加载模型
        weights, bias, n_feat = self.load_model()

        if weights is None:
            return {
                "success": False,
                "error": "Model not found. Please train the model first.",
            }

        # 获取最新数据
        candles = self.data_manager.fetch_klines("4h", limit=100)

        if not candles:
            return {"success": False, "error": "Failed to fetch latest data"}

        current_price = candles[-1]["close"]

        # 计算特征
        features = self.data_manager.calculate_features(candles)
        normalized = self.data_manager.normalize_features(features)

        # 预测
        last = normalized[-1, :]
        if len(last) < len(weights):
            last = np.pad(last, (0, len(weights) - len(last)))
        elif len(last) > len(weights):
            last = last[: len(weights)]

        prediction = np.dot(last, weights) + bias
        prediction = np.clip(prediction, -0.3, 0.3)

        future_price = current_price * (1 + prediction)

        # 交易信号
        if prediction > 0.015:
            signal = "STRONG_BUY"
            confidence = min(abs(prediction) * 5000, 100)
        elif prediction > 0:
            signal = "BUY"
            confidence = min(abs(prediction) * 2000, 50)
        elif prediction > -0.015:
            signal = "HOLD"
            confidence = 30
        else:
            signal = "SELL"
            confidence = min(abs(prediction) * 5000, 100)

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(current_price),
            "predicted_return": float(prediction * 100),
            "predicted_price": float(future_price),
            "signal": signal,
            "confidence": float(confidence),
            "model_info": {"features": n_feat, "bias": float(bias)},
        }


def main():
    parser = argparse.ArgumentParser(description="BTC Price Predictor Skill")
    parser.add_argument(
        "command", choices=["update", "train", "predict"], help="Command to execute"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    if args.command == "update":
        # 更新数据
        manager = BTCDataManager()
        manager.update_data("4h")

    elif args.command == "train":
        # 训练模型
        trainer = BTCModelTrainer()
        trainer.train_incremental(n_steps=args.steps, lr=args.lr)

    elif args.command == "predict":
        # 预测
        predictor = BTCPredictor()
        result = predictor.predict()

        if result["success"]:
            print("\n" + "=" * 70)
            print("📈 BTC PREDICTION RESULTS")
            print("=" * 70)
            print(f"\nCurrent Price: ${result['current_price']:,.2f}")
            print(f"Predicted Return: {result['predicted_return']:+.3f}%")
            print(f"Predicted Price: ${result['predicted_price']:,.2f}")
            print(f"\nSignal: {result['signal']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print("=" * 70)

            # 输出JSON格式供OpenClaw解析
            print("\nJSON_OUTPUT:")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()
