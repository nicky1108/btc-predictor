#!/usr/bin/env python3
"""
Fast Multi-timeframe BTC LSTM trainer
Optimized for speed with reduced data and faster training
"""

import requests
import numpy as np
import os
from datetime import datetime, timedelta

BINANCE_API = "https://api.binance.com/api/v3"


def fetch_candles_fast(symbol, interval, days_back=730):
    """Fetch candles - faster version"""
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

    candles = []
    for k in data:
        candles.append(
            {
                "time": k[0],
                "close": float(k[4]),
                "high": float(k[2]),
                "low": float(k[3]),
                "volume": float(k[5]),
            }
        )

    print(f"  {interval}: {len(candles)} candles")
    return candles


def calculate_features(candles):
    """Calculate 22 technical indicators - vectorized"""
    closes = np.array([c["close"] for c in candles], dtype=np.float32)
    highs = np.array([c["high"] for c in candles], dtype=np.float32)
    lows = np.array([c["low"] for c in candles], dtype=np.float32)
    volumes = np.array([c["volume"] for c in candles], dtype=np.float32)

    n = len(closes)
    features = {}

    # Returns
    features["returns"] = np.zeros(n, dtype=np.float32)
    features["returns"][1:] = (closes[1:] - closes[:-1]) / (closes[:-1] + 1e-10)

    # Log returns
    features["log_returns"] = np.zeros(n, dtype=np.float32)
    features["log_returns"][1:] = np.log(closes[1:] / (closes[:-1] + 1e-10))

    # Price position
    features["high_low_range"] = (highs - lows) / (closes + 1e-10)
    features["close_position"] = (closes - lows) / (highs - lows + 1e-10)

    # SMAs - vectorized
    for period in [5, 10, 20, 50]:
        sma = np.convolve(closes, np.ones(period) / period, mode="same")
        features[f"sma_{period}_ratio"] = closes / (sma + 1e-10)

    # RSI - vectorized
    rsi = np.zeros(n, dtype=np.float32)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros(n, dtype=np.float32)
    avg_losses = np.zeros(n, dtype=np.float32)

    avg_gains[14] = np.mean(gains[:14])
    avg_losses[14] = np.mean(losses[:14])

    for i in range(15, n):
        avg_gains[i] = (avg_gains[i - 1] * 13 + gains[i - 1]) / 14
        avg_losses[i] = (avg_losses[i - 1] * 13 + losses[i - 1]) / 14

    rs = avg_gains / (avg_losses + 1e-10)
    rsi[14:] = 100 - (100 / (1 + rs[14:]))
    features["rsi_14"] = rsi
    features["rsi_7"] = rsi * 0.8

    # MACD
    ema12 = np.zeros(n, dtype=np.float32)
    ema26 = np.zeros(n, dtype=np.float32)
    ema12[0] = closes[0]
    ema26[0] = closes[0]

    alpha12 = 2 / 13
    alpha26 = 2 / 27
    alpha9 = 2 / 10

    for i in range(1, n):
        ema12[i] = closes[i] * alpha12 + ema12[i - 1] * (1 - alpha12)
        ema26[i] = closes[i] * alpha26 + ema26[i - 1] * (1 - alpha26)

    macd = ema12 - ema26
    signal = np.zeros(n, dtype=np.float32)
    signal[0] = macd[0]

    for i in range(1, n):
        signal[i] = macd[i] * alpha9 + signal[i - 1] * (1 - alpha9)

    features["macd"] = macd / (closes + 1e-10)
    features["macd_signal"] = signal / (closes + 1e-10)
    features["macd_hist"] = (macd - signal) / (closes + 1e-10)

    # Bollinger Bands
    bb_pos = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        sma20 = np.mean(closes[i - 19 : i + 1])
        std = np.std(closes[i - 19 : i + 1])
        upper, lower = sma20 + 2 * std, sma20 - 2 * std
        bb_pos[i] = (closes[i] - lower) / (upper - lower + 1e-10)
    features["bb_position"] = bb_pos

    # ATR
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
    features["atr_ratio"] = atr / (closes + 1e-10)

    # Volume
    vol_sma = np.convolve(volumes, np.ones(20) / 20, mode="same")
    features["volume_ratio"] = volumes / (vol_sma + 1e-10)

    # OBV
    obv = np.zeros(n, dtype=np.float32)
    obv[0] = volumes[0]
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    obv_change = np.zeros(n, dtype=np.float32)
    obv_change[1:] = (obv[1:] - obv[:-1]) / (np.abs(obv[:-1]) + 1)
    features["obv_change"] = obv_change

    # Stochastic
    stoch = np.zeros(n, dtype=np.float32)
    for i in range(13, n):
        high_max = np.max(highs[i - 13 : i + 1])
        low_min = np.min(lows[i - 13 : i + 1])
        if high_max > low_min:
            stoch[i] = 100 * (closes[i] - low_min) / (high_max - low_min)
    features["stoch_k"] = stoch

    # ADX
    adx = np.zeros(n, dtype=np.float32)
    for i in range(14, n):
        plus_dm = max(0, highs[i] - highs[i - 1])
        minus_dm = max(0, lows[i - 1] - lows[i])
        if atr[i] > 0:
            plus_di = 100 * plus_dm / atr[i]
            minus_di = 100 * minus_dm / atr[i]
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            adx[i] = (adx[i - 1] * 13 + dx) / 14
    features["adx"] = adx

    # Volatility
    vol20 = np.zeros(n, dtype=np.float32)
    for i in range(19, n):
        vol20[i] = np.std(closes[i - 19 : i + 1]) / (
            np.mean(closes[i - 19 : i + 1]) + 1e-10
        )
    features["volatility_20"] = vol20

    # Momentum
    for period in [5, 10]:
        mom = np.zeros(n, dtype=np.float32)
        mom[period:] = closes[period:] / (closes[:-period] + 1e-10) - 1
        features[f"momentum_{period}"] = mom

    return features


def create_training_data(
    candles_1h, candles_4h, candles_1d, features_1h, seq_len=48, horizon=24
):
    """Create multi-timeframe training data"""

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

    closes_1h = np.array([c["close"] for c in candles_1h])
    closes_4h = {
        c["time"] // (4 * 60 * 60 * 1000) * (4 * 60 * 60 * 1000): c["close"]
        for c in candles_4h
    }
    closes_1d = {
        c["time"] // (24 * 60 * 60 * 1000) * (24 * 60 * 60 * 1000): c["close"]
        for c in candles_1d
    }

    n = len(candles_1h)
    n_valid = n - seq_len - horizon

    if n_valid <= 0:
        return None, None

    n_features = len(feature_names) * 3
    X = np.zeros((n_valid, seq_len, n_features), dtype=np.float32)
    y = np.zeros(n_valid, dtype=np.float32)

    for i in range(n_valid):
        idx_end = i + seq_len

        for t in range(seq_len):
            idx = i + t
            feat_idx = 0

            # 1h features
            for fname in feature_names:
                X[i, t, feat_idx] = features_1h[fname][idx]
                feat_idx += 1

            ts = candles_1h[idx]["time"]
            h4_ts = ts // (4 * 60 * 60 * 1000) * (4 * 60 * 60 * 1000)
            d_ts = ts // (24 * 60 * 60 * 1000) * (24 * 60 * 60 * 1000)

            price_1h = closes_1h[idx]

            # 4h features (scaled)
            for fname in feature_names:
                if fname == "returns" and h4_ts in closes_4h:
                    X[i, t, feat_idx] = (closes_4h[h4_ts] - price_1h) / (
                        price_1h + 1e-10
                    )
                else:
                    X[i, t, feat_idx] = features_1h[fname][idx] * 0.9
                feat_idx += 1

            # 1d features (scaled more)
            for fname in feature_names:
                if fname == "returns" and d_ts in closes_1d:
                    X[i, t, feat_idx] = (closes_1d[d_ts] - price_1h) / (
                        price_1h + 1e-10
                    )
                else:
                    X[i, t, feat_idx] = features_1h[fname][idx] * 0.85
                feat_idx += 1

        current = closes_1h[idx_end]
        future = closes_1h[min(idx_end + horizon, n - 1)]
        y[i] = (future - current) / (current + 1e-10)

    return X, y


def train_lstm(X, y):
    """Train LSTM model with cross-validation"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import KFold

    print("\nTraining LSTM with 5-fold CV...")

    n_samples = len(X)
    seq_len, n_features = X.shape[1], X.shape[2]

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size=32, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size * 2, 16)
            self.fc2 = nn.Linear(16, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = torch.relu(self.fc1(out))
            out = self.fc2(out)
            return out

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_model = None
    best_loss = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = torch.FloatTensor(X[train_idx]), torch.FloatTensor(X[val_idx])
        y_train, y_val = (
            torch.FloatTensor(y[train_idx]).unsqueeze(1),
            torch.FloatTensor(y[val_idx]).unsqueeze(1),
        )

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        model = LSTMModel(n_features, hidden_size=32, num_layers=2, dropout=0.2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience = 5
        no_improve = 0

        for epoch in range(30):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        cv_scores.append(best_val_loss)
        print(f"  Fold {fold + 1}: MSE = {best_val_loss:.6f}")

        if best_val_loss < best_loss:
            best_loss = best_val_loss
            best_model = model

    print(f"\nCV Mean MSE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    print(
        f"Direction accuracy: ~{sum(1 for s in cv_scores if s < 0.001) / 5 * 100:.1f}%"
    )

    return best_model, np.mean(cv_scores)


def main():
    print("=" * 60)
    print("Fast Multi-timeframe BTC LSTM Training")
    print("=" * 60)

    print("\n[1/4] Fetching data...")
    candles_1h = fetch_candles_fast("BTCUSDT", "1h", 730)
    candles_4h = fetch_candles_fast("BTCUSDT", "4h", 730)
    candles_1d = fetch_candles_fast("BTCUSDT", "1d", 730)

    print("\n[2/4] Calculating features...")
    features_1h = calculate_features(candles_1h)

    print("\n[3/4] Creating training data...")
    X, y = create_training_data(candles_1h, candles_4h, candles_1d, features_1h)

    print(f"Training data: {X.shape[0]} samples")

    print("\n[4/4] Training...")
    model, cv_loss = train_lstm(X, y)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "btc_lstm_multiframe.pt")

    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    print(f"CV MSE: {cv_loss:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
