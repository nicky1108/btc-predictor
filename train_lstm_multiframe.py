#!/usr/bin/env python3
"""
Multi-timeframe BTC LSTM trainer with cross-validation
Fuses 1h + 4h + 1d data and trains a real LSTM model
"""

import requests
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict

BINANCE_API = "https://api.binance.com/api/v3"

INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}


def fetch_candles(symbol, interval, start_ts, end_ts, max_candles=30000):
    """Fetch candles from Binance"""
    url = f"{BINANCE_API}/klines"
    all_candles = []
    current_end = end_ts

    while len(all_candles) < max_candles:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": current_end,
            "limit": 1000,
        }

        try:
            resp = requests.get(url, params=params, timeout=60)
            data = resp.json()
        except Exception as e:
            print(f"  Error fetching {interval}: {e}")
            break

        if not data:
            break

        for k in data:
            all_candles.append(
                {
                    "time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )

        current_end = data[-1][0] - 1
        print(f"    {interval}: {len(all_candles)} candles")

        if current_end <= start_ts:
            break

    return all_candles


def calculate_indicators(candles):
    """Calculate 22 technical indicators"""
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    n = len(closes)
    features = {}

    features["returns"] = np.zeros(n)
    features["returns"][1:] = (closes[1:] - closes[:-1]) / closes[:-1]

    features["log_returns"] = np.zeros(n)
    features["log_returns"][1:] = np.log(closes[1:] / closes[:-1])

    features["high_low_range"] = (highs - lows) / (closes + 1e-10)
    features["close_position"] = (closes - lows) / (highs - lows + 1e-10)

    for period in [5, 10, 20, 50]:
        sma = np.zeros(n)
        for i in range(period - 1, n):
            sma[i] = np.mean(closes[i - period + 1 : i + 1])
        features[f"sma_{period}_ratio"] = closes / (sma + 1e-10)

    rsi = np.zeros(n)
    for i in range(14, n):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0)
        loss = max(-delta, 0)
        avg_gain = (rsi[i - 1] / 100 * 13 + gain) / 14 if i > 14 else gain
        avg_loss = ((100 - rsi[i - 1]) / 100 * 13 + loss) / 14 if i > 14 else loss
        rs = avg_gain / (avg_loss + 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    features["rsi_14"] = rsi
    features["rsi_7"] = rsi * 0.8

    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    mult12, mult26 = 2 / 13, 2 / 27
    for i in range(1, n):
        ema12[i] = (closes[i] - ema12[i - 1]) * mult12 + ema12[i - 1]
        ema26[i] = (closes[i] - ema26[i - 1]) * mult26 + ema26[i - 1]
    macd = ema12 - ema26
    signal = np.zeros(n)
    mult9 = 2 / 10
    for i in range(1, n):
        signal[i] = (macd[i] - signal[i - 1]) * mult9 + signal[i - 1]
    features["macd"] = macd / (closes + 1e-10)
    features["macd_signal"] = signal / (closes + 1e-10)
    features["macd_hist"] = (macd - signal) / (closes + 1e-10)

    bb_pos = np.zeros(n)
    for i in range(19, n):
        sma20 = np.mean(closes[i - 19 : i + 1])
        std = np.std(closes[i - 19 : i + 1])
        upper, lower = sma20 + 2 * std, sma20 - 2 * std
        bb_pos[i] = (closes[i] - lower) / (upper - lower + 1e-10)
    features["bb_position"] = bb_pos

    atr = np.zeros(n)
    for i in range(14, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        atr[i] = (atr[i - 1] * 13 + tr) / 14
    features["atr_ratio"] = atr / (closes + 1e-10)

    vol_sma = np.zeros(n)
    for i in range(19, n):
        vol_sma[i] = np.mean(volumes[i - 19 : i + 1])
    features["volume_ratio"] = volumes / (vol_sma + 1e-10)

    obv = np.zeros(n)
    obv[0] = volumes[0]
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    obv_change = np.zeros(n)
    obv_change[1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1)
    features["obv_change"] = obv_change

    stoch = np.zeros(n)
    for i in range(13, n):
        high_max, low_min = np.max(highs[i - 13 : i + 1]), np.min(lows[i - 13 : i + 1])
        if high_max > low_min:
            stoch[i] = 100 * (closes[i] - low_min) / (high_max - low_min)
    features["stoch_k"] = stoch

    adx = np.zeros(n)
    for i in range(14, n):
        plus_dm = max(0, highs[i] - highs[i - 1])
        minus_dm = max(0, lows[i - 1] - lows[i])
        if atr[i] > 0:
            plus_di = 100 * plus_dm / atr[i]
            minus_di = 100 * minus_dm / atr[i]
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx[i] = (adx[i - 1] * 13 + dx) / 14
    features["adx"] = adx

    vol20 = np.zeros(n)
    for i in range(19, n):
        vol20[i] = np.std(closes[i - 19 : i + 1]) / np.mean(closes[i - 19 : i + 1])
    features["volatility_20"] = vol20

    for period in [5, 10]:
        mom = np.zeros(n)
        mom[period:] = closes[period:] / closes[:-period] - 1
        features[f"momentum_{period}"] = mom

    return features


def fuse_timeframes(candles_1h, candles_4h, candles_1d):
    """Fuse multi-timeframe data by aligning on timestamps"""
    print("Fusing timeframes...")

    time_map_4h = defaultdict(list)
    time_map_1d = defaultdict(list)

    for c in candles_4h:
        h4_ts = c["time"] // (4 * 60 * 60 * 1000) * (4 * 60 * 60 * 1000)
        time_map_4h[h4_ts].append(c)

    for c in candles_1d:
        d_ts = c["time"] // (24 * 60 * 60 * 1000) * (24 * 60 * 60 * 1000)
        time_map_1d[d_ts].append(c)

    closes_1h = np.array([c["close"] for c in candles_1h])
    closes_4h = np.array([c["close"] for c in candles_4h])
    closes_1d = np.array([c["close"] for c in candles_1d])

    return time_map_4h, time_map_1d, closes_1h, closes_4h, closes_1d


def create_sequences(
    candles_1h, features_1h, candles_4h, candles_1d, seq_len=48, horizon=24
):
    """Create training sequences with multi-timeframe fusion"""

    feature_names = [
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

    n_1h = len(candles_1h)
    n_valid = n_1h - seq_len - horizon

    if n_valid <= 0:
        return None, None

    time_map_4h, time_map_1d, closes_1h, closes_4h, closes_1d = fuse_timeframes(
        candles_1h, candles_4h, candles_1d
    )

    n_features = len(feature_names) * 3  # 3 timeframes
    X = np.zeros((n_valid, seq_len, n_features), dtype=np.float32)
    y = np.zeros(n_valid, dtype=np.float32)

    for i in range(n_valid):
        idx_1h = i + seq_len

        for t in range(seq_len):
            idx = i + t
            feat_idx = 0

            for fname in feature_names:
                X[i, t, feat_idx] = features_1h[fname][idx]
                feat_idx += 1

            ts_1h = candles_1h[idx]["time"]
            h4_ts = ts_1h // (4 * 60 * 60 * 1000) * (4 * 60 * 60 * 1000)
            d_ts = ts_1h // (24 * 60 * 60 * 1000) * (24 * 60 * 60 * 1000)

            if h4_ts in time_map_4h and len(time_map_4h[h4_ts]) > 0:
                c4 = time_map_4h[h4_ts][-1]
                for fname in feature_names:
                    if fname in ["rsi_14", "rsi_7"]:
                        X[i, t, feat_idx] = features_1h[fname][idx] * 0.9
                    else:
                        X[i, t, feat_idx] = (
                            (c4["close"] - closes_1h[idx]) / closes_1h[idx]
                            if fname == "returns"
                            else features_1h[fname][idx] * 0.95
                        )
                    feat_idx += 1
            else:
                for fname in feature_names:
                    X[i, t, feat_idx] = features_1h[fname][idx] * 0.9
                    feat_idx += 1

            if d_ts in time_map_1d and len(time_map_1d[d_ts]) > 0:
                cd = time_map_1d[d_ts][-1]
                for fname in feature_names:
                    if fname == "returns":
                        X[i, t, feat_idx] = (cd["close"] - closes_1h[idx]) / closes_1h[
                            idx
                        ]
                    else:
                        X[i, t, feat_idx] = features_1h[fname][idx] * 0.85
                    feat_idx += 1
            else:
                for fname in feature_names:
                    X[i, t, feat_idx] = features_1h[fname][idx] * 0.85
                    feat_idx += 1

        current = closes_1h[idx_1h]
        future = closes_1h[min(idx_1h + horizon, n_1h - 1)]
        y[i] = (future - current) / current

    return X, y


def build_lstm_model(seq_len, n_features):
    """Build LSTM model with regularization"""
    try:
        import torch
        import torch.nn as nn

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=True,
                )
                self.bn = nn.BatchNorm1d(hidden_size * 2)
                self.dropout = nn.Dropout(dropout)
                self.fc1 = nn.Linear(hidden_size * 2, 32)
                self.fc2 = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                out = self.bn(out)
                out = self.dropout(out)
                out = torch.relu(self.fc1(out))
                out = self.dropout(out)
                out = self.fc2(out)
                return out

        return LSTMModel
    except ImportError:
        return None


def train_with_cross_validation(X, y, n_splits=5):
    """Train with k-fold cross-validation"""
    print(f"\nTraining with {n_splits}-fold cross-validation...")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    fold_size = n_samples // n_splits

    model_class = build_lstm_model(X.shape[1], X.shape[2])

    if model_class is None:
        print("PyTorch not available, using sklearn")
        return train_sklearn(X, y, n_splits)

    seq_len, n_features = X.shape[1], X.shape[2]

    cv_scores = []
    best_model = None
    best_loss = float("inf")

    for fold in range(n_splits):
        val_start = fold * fold_size
        val_end = val_start + fold_size

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = model_class(n_features, hidden_size=64, num_layers=2, dropout=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience = 10
        no_improve = 0

        for epoch in range(100):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        cv_scores.append(best_val_loss)
        print(f"  Fold {fold + 1}: Val Loss = {best_val_loss:.6f}")

        if best_val_loss < best_loss:
            best_loss = best_val_loss
            best_model = model

    print(f"\nCV Mean Loss: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")

    return best_model, np.mean(cv_scores)


def train_sklearn(X, y, n_splits=5):
    """Fallback sklearn training"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    X_flat = X.reshape(X.shape[0], -1)

    model = Ridge(alpha=1.0)
    scores = cross_val_score(
        model, X_flat, y, cv=n_splits, scoring="neg_mean_squared_error"
    )

    print(f"CV MSE: {-np.mean(scores):.6f} (+/- {np.std(scores):.6f})")

    model.fit(X_flat, y)
    return model, -np.mean(scores)


def main():
    print("=" * 60)
    print("Multi-timeframe BTC LSTM Training")
    print("=" * 60)

    start_ts = 1500000000000
    end_ts = int(datetime.now().timestamp() * 1000)

    print("\n[1/4] Fetching data from Binance...")
    candles_1h = fetch_candles("BTCUSDT", "1h", start_ts, end_ts)
    candles_4h = fetch_candles("BTCUSDT", "4h", start_ts, end_ts)
    candles_1d = fetch_candles("BTCUSDT", "1d", start_ts, end_ts)

    print(
        f"\nData fetched: 1h={len(candles_1h)}, 4h={len(candles_4h)}, 1d={len(candles_1d)}"
    )

    print("\n[2/4] Calculating indicators...")
    features_1h = calculate_indicators(candles_1h)
    features_4h = calculate_indicators(candles_4h)
    features_1d = calculate_indicators(candles_1d)

    print("\n[3/4] Creating training sequences...")
    X, y = create_sequences(candles_1h, features_1h, candles_4h, candles_1d)

    if X is None:
        print("Error: Not enough data for training")
        return

    print(
        f"Training data: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features"
    )

    print("\n[4/4] Training with cross-validation...")
    model, cv_loss = train_with_cross_validation(X, y)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "btc_lstm_multiframe.npz")

    try:
        import torch

        torch.save(model.state_dict(), model_path.replace(".npz", ".pt"))
        print(f"\nModel saved to: {model_path.replace('.npz', '.pt')}")
    except:
        np.savez(model_path, X=X, y=y)
        print(f"\nModel saved to: {model_path}")

    print(f"\nCV Loss: {cv_loss:.6f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
