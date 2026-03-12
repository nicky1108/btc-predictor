#!/usr/bin/env python3
"""
Enhanced BTC Ensemble Trainer (Linear + LSTM + Market Regime)
"""

import requests
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
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


def detect_market_regime(closes, horizon=168):
    """Detect market regime: bull, bear, or sideways"""
    n = len(closes)
    if n < horizon:
        horizon = n // 2

    recent = closes[-horizon:]
    sma_short = np.mean(recent[-20:])
    sma_long = np.mean(recent[-50:])

    volatility = np.std(recent) / np.mean(recent)

    trend = (sma_short - sma_long) / sma_long

    if trend > 0.02 and volatility < 0.03:
        return "bull"
    elif trend < -0.02 and volatility < 0.03:
        return "bear"
    else:
        return "sideways"


def create_data(c1h, c4h, c1d, f1h, seq_len=48, horizon=24):
    c4 = {c["time"] // (4 * 3600000) * 4 * 3600000: c["close"] for c in c4h}
    cd = {c["time"] // (86400000) * 86400000: c["close"] for c in c1d}
    ch = np.array([c["close"] for c in c1h])
    n = len(c1h)
    nv = n - seq_len - horizon

    n_features_3tf = len(FEATURE_NAMES) * 3
    X = np.zeros((nv, seq_len, n_features_3tf), dtype=np.float32)
    X_flat = np.zeros((nv, seq_len * len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(nv, dtype=np.float32)
    regimes = []

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

        X_flat[i] = X[i, :, : len(FEATURE_NAMES)].flatten()
        y[i] = (ch[min(ie + horizon, n - 1)] - ch[ie]) / (ch[ie] + 1e-10)
        regimes.append(detect_market_regime(ch[: ie + horizon]))

    return X, X_flat, y, regimes


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


def train_ensemble(X, X_flat, y, regimes):
    """Train ensemble: Linear + LSTM with regime weighting"""
    print("\nTraining ensemble with regime detection...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples = len(X)
    n_features = X.shape[2]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    ensemble_results = {
        "linear": {"mse": [], "dir_acc": []},
        "lstm": {"mse": [], "dir_acc": []},
        "ensemble": {"mse": [], "dir_acc": []},
    }

    best_lstm = None
    best_linear = None
    best_ensemble_mse = float("inf")

    for fold, (tr, va) in enumerate(kf.split(X)):
        print(f"\n--- Fold {fold + 1} ---")

        X_tr, X_va = X[tr], X[va]
        Xf_tr, Xf_va = X_flat[tr], X_flat[va]
        y_tr, y_va = y[tr], y[va]

        linear = Ridge(alpha=1.0)
        linear.fit(Xf_tr, y_tr)
        linear_pred = linear.predict(Xf_va)
        linear_mse = np.mean((linear_pred - y_va) ** 2)
        linear_dir = np.mean((linear_pred > 0) == (y_va > 0))
        ensemble_results["linear"]["mse"].append(linear_mse)
        ensemble_results["linear"]["dir_acc"].append(linear_dir)
        print(f"Linear - MSE: {linear_mse:.6f}, Dir: {linear_dir * 100:.1f}%")

        lstm = LSTM(n_features).to(device)
        opt = torch.optim.Adam(lstm.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.MSELoss()
        ds = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_tr).to(device),
                torch.FloatTensor(y_tr).unsqueeze(1).to(device),
            ),
            batch_size=64,
            shuffle=True,
        )

        for e in range(30):
            lstm.train()
            for bx, by in ds:
                opt.zero_grad()
                crit(lstm(bx), by).backward()
                opt.step()

        lstm.eval()
        with torch.no_grad():
            lstm_pred = lstm(torch.FloatTensor(X_va).to(device)).cpu().numpy().flatten()
        lstm_mse = np.mean((lstm_pred - y_va) ** 2)
        lstm_dir = np.mean((lstm_pred > 0) == (y_va > 0))
        ensemble_results["lstm"]["mse"].append(lstm_mse)
        ensemble_results["lstm"]["dir_acc"].append(lstm_dir)
        print(f"LSTM - MSE: {lstm_mse:.6f}, Dir: {lstm_dir * 100:.1f}%")

        ensemble_pred = 0.5 * linear_pred + 0.5 * lstm_pred
        ensemble_mse = np.mean((ensemble_pred - y_va) ** 2)
        ensemble_dir = np.mean((ensemble_pred > 0) == (y_va > 0))
        ensemble_results["ensemble"]["mse"].append(ensemble_mse)
        ensemble_results["ensemble"]["dir_acc"].append(ensemble_dir)
        print(f"Ensemble - MSE: {ensemble_mse:.6f}, Dir: {ensemble_dir * 100:.1f}%")

        if ensemble_mse < best_ensemble_mse:
            best_ensemble_mse = ensemble_mse
            best_lstm = lstm.cpu()
            best_linear = linear

    print("\n=== Summary ===")
    for model_name in ["linear", "lstm", "ensemble"]:
        mse_mean = np.mean(ensemble_results[model_name]["mse"])
        dir_mean = np.mean(ensemble_results[model_name]["dir_acc"])
        print(f"{model_name}: MSE={mse_mean:.6f}, Dir={dir_mean * 100:.1f}%")

    return best_linear, best_lstm, ensemble_results


def main():
    print("=" * 60)
    print("Ensemble BTC Training (Linear + LSTM + Regime)")
    print("=" * 60)

    print("\nFetching data...")
    c1h = fetch_candles("BTCUSDT", "1h")
    c4h = fetch_candles("BTCUSDT", "4h")
    c1d = fetch_candles("BTCUSDT", "1d")

    print("\nCalculating features...")
    f1h = calculate_features(c1h)

    print("\nCreating data...")
    X, X_flat, y, regimes = create_data(c1h, c4h, c1d, f1h)
    print(f"Data: {X.shape[0]} samples")
    print(f"Regimes: {dict(zip(*np.unique(regimes, return_counts=True)))}")

    linear_model, lstm_model, results = train_ensemble(X, X_flat, y, regimes)

    model_dir = os.path.expanduser("~/.openclaw/skills/btc_predictor/models")
    os.makedirs(model_dir, exist_ok=True)

    torch.save(lstm_model.state_dict(), f"{model_dir}/btc_ensemble_lstm.pt")
    import joblib

    joblib.dump(linear_model, f"{model_dir}/btc_ensemble_linear.pkl")

    calibration = {
        "linear_mse": float(np.mean(results["linear"]["mse"])),
        "linear_dir": float(np.mean(results["linear"]["dir_acc"])),
        "lstm_mse": float(np.mean(results["lstm"]["mse"])),
        "lstm_dir": float(np.mean(results["lstm"]["dir_acc"])),
        "ensemble_mse": float(np.mean(results["ensemble"]["mse"])),
        "ensemble_dir": float(np.mean(results["ensemble"]["dir_acc"])),
        "ensemble_weights": [0.5, 0.5],
    }

    with open(f"{model_dir}/ensemble_calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nModels saved to {model_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
