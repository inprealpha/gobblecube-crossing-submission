"""Train the submission model.

The starter baseline used XGBoost for intent and constant velocity for
trajectory. This version keeps the intent model and adds a small ExtraTrees
regressor that predicts per-horizon center residuals over constant velocity.

Writes model.pkl. Run once:

    python baseline.py

This baseline is deliberately weak — it's the bar to beat, not the finish
line.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from predict import HORIZON_KEYS, _constant_velocity_centers, _engineered_features, _trajectory_features

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"


REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]


def row_to_request(row: pd.Series) -> dict:
    return {k: row[k] for k in REQUEST_FIELDS}


def featurize(df: pd.DataFrame, feature_fn=_engineered_features) -> np.ndarray:
    n = len(df)
    sample = feature_fn(row_to_request(df.iloc[0]))
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = feature_fn(row_to_request(df.iloc[i]))
    return X


def trajectory_residual_targets(df: pd.DataFrame) -> np.ndarray:
    """Future-center residuals relative to the constant-velocity baseline."""
    y = np.empty((len(df), len(HORIZON_KEYS) * 2), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        req = row_to_request(row)
        cv_centers, _, _ = _constant_velocity_centers(req)
        for horizon_idx, key in enumerate(HORIZON_KEYS):
            bbox = np.asarray(row[key], dtype=np.float64)
            truth_cx = (bbox[0] + bbox[2]) * 0.5
            truth_cy = (bbox[1] + bbox[3]) * 0.5
            y[i, horizon_idx * 2] = truth_cx - cv_centers[horizon_idx, 0]
            y[i, horizon_idx * 2 + 1] = truth_cy - cv_centers[horizon_idx, 1]
    return y


def mean_ade_from_residuals(pred: np.ndarray, truth: np.ndarray) -> tuple[float, list[float]]:
    pred_h = pred.reshape(len(pred), len(HORIZON_KEYS), 2)
    truth_h = truth.reshape(len(truth), len(HORIZON_KEYS), 2)
    ade_by_horizon = np.hypot(
        pred_h[:, :, 0] - truth_h[:, :, 0],
        pred_h[:, :, 1] - truth_h[:, :, 1],
    ).mean(axis=0)
    return float(ade_by_horizon.mean()), [float(v) for v in ade_by_horizon]


def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    print(f"  positive rates: train {train.will_cross_2s.mean():.3f}, "
          f"dev {dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing intent...")
    t0 = time.time()
    X_train = featurize(train)
    X_dev = featurize(dev)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train.shape}")

    pos_ratio = float(y_train.mean())

    clf = None
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            existing_model = pickle.load(f)
        clf = existing_model.get("intent")
    if clf is not None:
        print("\nReusing existing intent model from model.pkl")
    else:
        print("\nTraining XGBClassifier (no class rebalancing — want calibrated probs)...")
        t0 = time.time()
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            tree_method="hist",
            n_jobs=4,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)
        print(f"  {time.time() - t0:.1f}s")

    dev_probs = clf.predict_proba(X_dev)[:, 1]
    ll = log_loss(y_dev, np.clip(dev_probs, 1e-6, 1 - 1e-6))
    prior_ll = log_loss(y_dev, np.full_like(dev_probs, pos_ratio))
    print(f"\nDev log-loss:  {ll:.4f}  (class-prior baseline {prior_ll:.4f})")

    print("\nFeaturizing trajectory...")
    t0 = time.time()
    X_train_traj = featurize(train, _trajectory_features)
    X_dev_traj = featurize(dev, _trajectory_features)
    y_train_traj = trajectory_residual_targets(train)
    y_dev_traj = trajectory_residual_targets(dev)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train_traj.shape}")

    print("\nTraining trajectory residual regressor...")
    t0 = time.time()
    traj_reg = ExtraTreesRegressor(
        n_estimators=40,
        min_samples_leaf=15,
        random_state=7,
        n_jobs=1,
    )
    traj_reg.fit(X_train_traj, y_train_traj)
    dev_residual_preds = traj_reg.predict(X_dev_traj).astype(np.float32)
    mean_ade, ade_by_horizon = mean_ade_from_residuals(dev_residual_preds, y_dev_traj)
    ade_msg = ", ".join(f"{k}: {v:.1f}" for k, v in zip(HORIZON_KEYS, ade_by_horizon))
    print(f"  {time.time() - t0:.1f}s")
    print(f"\nDev trajectory ADE from residual model: {mean_ade:.2f} px ({ade_msg})")

    print(f"\nSaving model → {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "intent": clf,
            "traj": {
                "kind": "constant_velocity_residual_extratrees",
                "residual_model": traj_reg,
                "feature_count": int(X_train_traj.shape[1]),
            },
        }, f)


if __name__ == "__main__":
    main()
    sys.exit(0)
