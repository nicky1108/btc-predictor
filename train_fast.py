#!/usr/bin/env python3
"""
Fast BTC Model Training - Linear with 22 features
"""

import struct
import numpy as np
import os

DATA_PATH = "/Users/nicky/.openclaw/skills/btc_predictor/btc_training_large.bin"
OUTPUT_PATH = "/Users/nicky/.openclaw/skills/btc_predictor/models/btc_model_v2.npz"

print("=" * 60)
print("BTC Model Training (22 Features)")
print("=" * 60)

# Check if data exists
if not os.path.exists(DATA_PATH):
    print("Data file not found. Please run fetch_large_data.py first.")
    exit(1)

# Load data
print("\nLoading data...")
with open(DATA_PATH, "rb") as f:
    n_samples = struct.unpack("<I", f.read(4))[0]
    seq_len = struct.unpack("<I", f.read(4))[0]
    n_features = struct.unpack("<I", f.read(4))[0]

    print(f"Data: {n_samples} samples, seq={seq_len}, features={n_features}")

    sample_size = seq_len * n_features + 1
    data = np.frombuffer(f.read(), dtype=np.float32)
    data = data[: n_samples * sample_size]

    X = data[:-n_samples].reshape(n_samples, seq_len, n_features)
    y = data[-n_samples:]

# Use last 24 timesteps with exponential weighting
print("\nAggregating sequences...")
time_weights = np.exp(-np.arange(24) / 8)
time_weights = time_weights / time_weights.sum()

X_last = X[:, -24:, :]
X_agg = np.sum(X_last * time_weights.reshape(1, -1, 1), axis=1)

# Stats
feat_mean = X_agg.mean(axis=0)
feat_std = X_agg.std(axis=0) + 1e-8
X_norm = (X_agg - feat_mean) / feat_std

# Train linear model
print("\nTraining...")
XTX = np.dot(X_norm.T, X_norm) + 0.1 * np.eye(n_features)
XTX_inv = np.linalg.inv(XTX)
weights = np.dot(XTX_inv, np.dot(X_norm.T, y)).flatten()
bias = (y - np.dot(X_norm, weights)).mean()

# Evaluate
pred = np.dot(X_norm, weights) + bias
mse = np.mean((pred - y) ** 2)
rmse = np.sqrt(mse)

pred_dir = pred > 0
true_dir = y > 0
dir_acc = (pred_dir == true_dir).mean()

print(f"\nTraining metrics:")
print(f"  MSE: {mse:.6f}")
print(f"  RMSE: {rmse:.6f} ({rmse * 100:.2f}%)")
print(f"  Direction accuracy: {dir_acc * 100:.1f}%")

# Save
print(f"\nSaving to {OUTPUT_PATH}...")
np.savez(
    OUTPUT_PATH,
    weights=weights.astype(np.float32),
    bias=np.float32(bias),
    feat_mean=feat_mean.astype(np.float32),
    feat_std=feat_std.astype(np.float32),
    time_weights=time_weights.astype(np.float32),
)

print("Done!")
