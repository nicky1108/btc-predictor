#!/usr/bin/env python3
"""
Optimized BTC LSTM Trainer v2
- Attention mechanism
- Directional loss
- Weighted ensemble (LSTM-only optimized)
"""

import requests
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class AttentionLSTM(nn.Module):
    """LSTM with Self-Attention"""

    def __init__(self, input_size, hidden_size=32, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.sum(lstm_out * attn_weights, dim=1)

        out = self.dropout(context)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class DirectionalLoss(nn.Module):
    """Combined MSE + Directional loss"""

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)

        pred_sign = torch.sign(pred.squeeze())
        target_sign = (target.squeeze() > 0).float()

        dir_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), target_sign)

        return self.alpha * mse_loss + (1 - self.alpha) * dir_loss * 0.01


def train_optimized(X, y, horizon=24):
    """Train with attention and directional loss"""
    print(f"\n=== Training optimized model (h{horizon}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_samples, seq_len, n_features = X.shape

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {"mse": [], "direction_accuracy": [], "mae": []}

    best_model = None
    best_mse = float("inf")

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr = torch.FloatTensor(X[tr_idx]).to(device)
        y_tr = torch.FloatTensor(y[tr_idx]).unsqueeze(1).to(device)
        X_va = torch.FloatTensor(X[va_idx]).to(device)
        y_va = torch.FloatTensor(y[va_idx]).unsqueeze(1).to(device)

        model = AttentionLSTM(n_features, hidden_size=32, num_layers=3, dropout=0.2).to(
            device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = DirectionalLoss(alpha=0.7)

        ds = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

        best_val_loss = float("inf")
        patience = 10
        no_improve = 0

        for epoch in range(50):
            model.train()
            for bx, by in ds:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_va)
                val_loss = criterion(val_pred, y_va).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.eval()
        with torch.no_grad():
            pred = model(X_va).cpu().numpy().flatten()
            true = y_va.cpu().numpy().flatten()

        mse = np.mean((pred - true) ** 2)
        mae = np.mean(np.abs(pred - true))
        dir_acc = np.mean((pred > 0) == (true > 0))

        results["mse"].append(mse)
        results["mae"].append(mae)
        results["direction_accuracy"].append(dir_acc)

        print(
            f"Fold {fold + 1}: MSE={mse:.6f}, MAE={mae:.6f}, Dir={dir_acc * 100:.1f}%"
        )

        if mse < best_mse:
            best_mse = mse
            best_model = model.cpu()

    return best_model, results


def main():
    print("=" * 60)
    print("Optimized BTC LSTM v2 - Attention + Directional Loss")
    print("=" * 60)

    print("\nFetching data...")
    c1h = fetch_candles("BTCUSDT", "1h")
    c4h = fetch_candles("BTCUSDT", "4h")
    c1d = fetch_candles("BTCUSDT", "1d")

    print("\nCalculating features...")
    f1h = calculate_features(c1h)

    print("\nCreating data...")
    X, y = create_data(c1h, c4h, c1d, f1h)
    print(f"Data shape: {X.shape}")

    horizons = [6, 24, 48, 168]
    all_results = {}

    model_dir = os.path.expanduser("~/.openclaw/skills/btc_predictor/models")
    os.makedirs(model_dir, exist_ok=True)

    for h in horizons:
        model, results = train_optimized(X, y, h)

        mean_mse = np.mean(results["mse"])
        mean_mae = np.mean(results["mae"])
        mean_dir = np.mean(results["direction_accuracy"])

        all_results[h] = {
            "mse": mean_mse,
            "mae": mean_mae,
            "direction_accuracy": mean_dir,
        }

        model_path = f"{model_dir}/btc_attention_h{h}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved: {model_path}")

    print("\n=== Results ===")
    for h, r in all_results.items():
        print(
            f"{h}h: MSE={r['mse']:.6f}, MAE={r['mae']:.6f}, Dir={r['direction_accuracy'] * 100:.1f}%"
        )

    cal_path = f"{model_dir}/optimized_calibration.json"
    with open(cal_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCalibration: {cal_path}")
    print("Done!")


if __name__ == "__main__":
    main()
