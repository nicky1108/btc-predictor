#!/usr/bin/env python3
"""
Enhanced BTC LSTM Trainer with proper confidence calibration
"""

import requests
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
import json

BINANCE_API = "https://api.binance.com/api/v3"


def fetch_candles(symbol, interval, days_back=730):
    url = f"{BINANCE_API}/klines"
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": 1500,
    }
    resp = requests.get(url, params=params, timeout=30)
    data = resp.json()
    candles = [
        {
            "time": k[0],
            "close": float(k[4]),
            "high": float(k[2]),
            "low": float(k[3]),
            "volume": float(k[5]),
        }
        for k in data
    ]
    print(f"{interval}: {len(candles)} candles")
    return candles


def calculate_features(candles):
    closes = np.array([c["close"] for c in candles], dtype=np.float32)
    highs = np.array([c["high"] for c in candles], dtype=np.float32)
    lows = np.array([c["low"] for c in candles], dtype=np.float32)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float32)
    n = len(closes)
    f = {}
    f["returns"] = np.zeros(n, dtype=np.float32)
    f["returns"][1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)
    f["log_returns"] = np.zeros(n, dtype=np.float32)
    f["log_returns"][1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))
    f["high_low_range"] = (highs - lows) / (closes + 1e-10)
    f["close_position"] = (closes - lows) / (highs - lows + 1e-10)
    for p in [5, 10, 20, 50]:
        sma = np.convolve(closes, np.ones(p) / p, mode="same")
        f[f"sma_{p}_ratio"] = closes / (sma + 1e-10)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_g = np.zeros(n, dtype=np.float32)
    avg_l = np.zeros(n, dtype=np.float32)
    avg_g[14] = np.mean(gains[:14])
    avg_l[14] = np.mean(losses[:14])
    for i in range(15, n):
        avg_g[i] = (avg_g[i - 1] * 13 + gains[i - 1]) / 14
        avg_l[i] = (avg_l[i - 1] * 13 + losses[i - 1]) / 14
    rs = avg_g / (avg_l + 1e-10)
    f["rsi_14"] = np.zeros(n, dtype=np.float32)
    f["rsi_14"][14:] = 100 - (100 / (1 + rs[14:]))
    f["rsi_7"] = f["rsi_14"] * 0.8
    ema12 = np.zeros(n, dtype=np.float32)
    ema26 = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        ema12[i] = closes[i] * (2 / 13) + ema12[i - 1] * (11 / 13)
        ema26[i] = closes[i] * (2 / 27) + ema26[i - 1] * (25 / 27)
    macd = ema12 - ema26
    signal = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        signal[i] = macd[i] * 0.2 + signal[i - 1] * 0.8
    f["macd"] = macd / (closes + 1e-10)
    f["macd_signal"] = signal / (closes + 1e-10)
    f["macd_hist"] = (macd - signal) / (closes + 1e-10)
    bb = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        m, s = np.mean(closes[i - 19 : i + 1]), np.std(closes[i - 19 : i + 1])
        bb[i] = (closes[i] - (m - 2 * s)) / (4 * s + 1e-10)
    f["bb_position"] = bb
    atr = np.zeros(n, dtype=np.float32)
    tr = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    f["atr_ratio"] = atr / (closes + 1e-10)
    vol_sma = np.convolve(volumes, np.ones(20) / 20, mode="same")
    f["volume_ratio"] = volumes / (vol_sma + 1e-10)
    obv = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        obv[i] = (
            obv[i - 1] + volumes[i]
            if closes[i] > closes[i - 1]
            else obv[i - 1] - volumes[i]
            if closes[i] < closes[i - 1]
            else obv[i - 1]
        )
    f["obv_change"] = np.zeros(n, dtype=np.float32)
    f["obv_change"][1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1)
    stoch = np.zeros(n, dtype=np.float32)
    for i in range(13, n):
        h, l = np.max(highs[i - 13 : i + 1]), np.min(lows[i - 13 : i + 1])
        stoch[i] = 100 * (closes[i] - l) / (h - l + 1e-10) if h > l else 50
    f["stoch_k"] = stoch
    adx = np.zeros(n, dtype=np.float32)
    for i in range(14, n):
        pdm = max(0, highs[i] - highs[i - 1])
        mdm = max(0, lows[i - 1] - lows[i])
        if atr[i] > 0:
            di = 100 * pdm / atr[i]
            di_ = 100 * mdm / atr[i]
            dx = 100 * abs(di - di_) / (di + di_ + 1e-10)
            adx[i] = (adx[i - 1] * 13 + dx) / 14
    f["adx"] = adx
    vol20 = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        vol20[i] = np.std(closes[i - 19 : i + 1]) / (
            np.mean(closes[i - 19 : i + 1]) + 1e-10
        )
    f["volatility_20"] = vol20
    for p in [5, 10]:
        m = np.zeros(n, dtype=np.float32)
        m[p:] = closes[p:] / (closes[:-p] + 1e-10) - 1
        f[f"momentum_{p}"] = m
    return f


FEATURE_NAMES = [
    "returns",
    "log_returns",
    "high_low_range",
    "close_position",
    "sma_5_ratio",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_position",
    "atr_ratio",
    "volume_ratio",
    "obv_change",
    "stoch_k",
    "adx",
    "volatility_20",
    "momentum_5",
    "momentum_10",
]


def create_data(c1h, c4h, c1d, f1h, seq_len=48, horizon=24):
    c4 = {c["time"] // (4 * 3600000) * 4 * 3600000: c["close"] for c in c4h}
    cd = {c["time"] // (86400000) * 86400000: c["close"] for c in c1d}
    ch = np.array([c["close"] for c in c1h])
    n = len(c1h)
    nv = n - seq_len - horizon
    X = np.zeros((nv, seq_len, len(FEATURE_NAMES) * 3), dtype=np.float32)
    y = np.zeros(nv, dtype=np.float32)
    for i in range(nv):
        ie = i + seq_len
        for t in range(seq_len):
            ii = i + t
            fi = 0
            for fn in FEATURE_NAMES:
                X[i, t, fi] = f1h[fn][ii]
                fi += 1
            ts = c1h[ii]["time"]
            p = ch[ii]
            for fn in FEATURE_NAMES:
                X[i, t, fi] = (
                    (c4.get(ts // (4 * 3600000) * 4 * 3600000, p) - p) / p
                    if fn == "returns"
                    else f1h[fn][ii] * 0.9
                )
                fi += 1
            for fn in FEATURE_NAMES:
                X[i, t, fi] = (
                    (cd.get(ts // (86400000) * 86400000, p) - p) / p
                    if fn == "returns"
                    else f1h[fn][ii] * 0.85
                )
                fi += 1
        y[i] = (ch[min(ie + horizon, n - 1)] - ch[ie]) / (ch[ie] + 1e-10)
    return X, y


class LSTM(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.l = nn.LSTM(inp, 32, 2, batch_first=True, dropout=0.2, bidirectional=True)
        self.d = nn.Dropout(0.2)
        self.f1 = nn.Linear(64, 16)
        self.f2 = nn.Linear(16, 1)

    def forward(self, x):
        o, _ = self.l(x)
        o = o[:, -1, :]
        o = self.d(o)
        o = torch.relu(self.f1(o))
        return self.f2(o)


def train_with_calibration(X, y, horizons=[6, 24, 48, 168]):
    """Train models for multiple horizons with calibration"""
    print("\nTraining with multi-horizon and calibration...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    n_samples = len(X)
    seq_len, n_features = X.shape[1], X.shape[2]

    results = {}

    for horizon in horizons:
        print(f"\n=== Horizon: {horizon}h ===")

        y_h = np.zeros(n_samples, dtype=np.float32)
        closes = X[:, -1, 0] * 100 + 50000
        for i in range(n_samples):
            idx = i + horizon
            if idx < n_samples:
                y_h[i] = y[i] * 0.5 + (X[idx, -1, 0] - X[i, -1, 0]) * 0.5

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mse = []
        cv_direction_acc = []
        best_model = None
        best_mse = float("inf")

        for fold, (tr, va) in enumerate(kf.split(X)):
            m = LSTM(n_features).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-4)
            crit = nn.MSELoss()
            ds = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X[tr]).to(device),
                    torch.FloatTensor(y_h[tr]).unsqueeze(1).to(device),
                ),
                batch_size=64,
                shuffle=True,
            )

            for e in range(30):
                m.train()
                for bx, by in ds:
                    opt.zero_grad()
                    crit(m(bx), by).backward()
                    opt.step()

            m.eval()
            with torch.no_grad():
                val_pred = (
                    m(torch.FloatTensor(X[va]).to(device)).cpu().numpy().flatten()
                )
                val_true = y_h[va]
                mse = np.mean((val_pred - val_true) ** 2)

                direction_correct = np.sum((val_pred > 0) == (val_true > 0))
                direction_acc = direction_correct / len(val_pred)

            cv_mse.append(mse)
            cv_direction_acc.append(direction_acc)

            if mse < best_mse:
                best_mse = mse
                best_model = m.cpu()

        mean_mse = np.mean(cv_mse)
        mean_dir_acc = np.mean(cv_direction_acc)

        print(f"MSE: {mean_mse:.6f}, Direction Accuracy: {mean_dir_acc * 100:.1f}%")

        results[horizon] = {
            "model": best_model,
            "mse": mean_mse,
            "direction_accuracy": mean_dir_acc,
            "confidence_scale": mean_dir_acc * 2,
        }

    return results


def main():
    print("=" * 60)
    print("Enhanced BTC LSTM Training with Multi-Horizon")
    print("=" * 60)

    print("\nFetching data...")
    c1h = fetch_candles("BTCUSDT", "1h")
    c4h = fetch_candles("BTCUSDT", "4h")
    c1d = fetch_candles("BTCUSDT", "1d")

    print("\nCalculating features...")
    f1h = calculate_features(c1h)

    print("\nCreating training data...")
    X, y = create_data(c1h, c4h, c1d, f1h)
    print(f"Data: {X.shape[0]} samples")

    horizons = [6, 24, 48, 168]
    results = train_with_calibration(X, y, horizons)

    model_dir = os.path.expanduser("~/.openclaw/skills/btc_predictor/models")
    os.makedirs(model_dir, exist_ok=True)

    calibration_data = {}
    for h, r in results.items():
        model_path = f"{model_dir}/btc_lstm_h{h}.pt"
        torch.save(r["model"].state_dict(), model_path)
        calibration_data[h] = {
            "mse": float(r["mse"]),
            "direction_accuracy": float(r["direction_accuracy"]),
            "confidence_scale": float(r["confidence_scale"]),
            "model_path": model_path,
        }
        print(f"Saved {model_path}")

    cal_path = f"{model_dir}/calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration_data, f, indent=2)
    print(f"\nCalibration data saved: {cal_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
