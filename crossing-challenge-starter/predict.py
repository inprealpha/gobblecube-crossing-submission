"""Your submission entry point — replace the baseline model here.

Contract (do NOT change the signature):

    predict(request: dict) -> dict

Input: one request dict (keys below). Output: a dict with intent probability
and 4 future bounding boxes.

Request keys (all required):
    ped_id                str       opaque token, stable within the dataset
    frame_w, frame_h      int
    time_of_day, weather, location   str  (may be empty strings)
    ego_available         bool      True when ego speed/yaw history is valid
    bbox_history          list[16][4]  past bboxes at 15 Hz, oldest → current
                                       each bbox is [x1, y1, x2, y2] in pixels
    ego_speed_history     list[16]  past ego speeds (m/s); zeros if unavailable
    ego_yaw_history       list[16]  past yaw rates (rad/s); zeros if unavailable
    requested_at_frame    int       native-30fps frame id of current observation

Output keys (all required):
    intent                float     P(crossing within next 2s), in [0, 1]
    bbox_500ms            list[4]   predicted bbox at +0.5 s
    bbox_1000ms           list[4]   predicted bbox at +1.0 s
    bbox_1500ms           list[4]   predicted bbox at +1.5 s
    bbox_2000ms           list[4]   predicted bbox at +2.0 s

The grader calls predict() once per request, row-by-row. Prediction order
must match input order.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).parent / "model.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]   # at 15 Hz → 0.5, 1.0, 1.5, 2.0 s
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]
CATEGORY_VALUES = {
    "time_of_day": ["", "daytime", "nighttime", "n/a"],
    "weather": ["", "clear", "cloud", "cloudy", "rain", "snow", "n/a"],
    "location": ["", "indoor", "plaza", "street"],
}

_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is None:
        with open(MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
    return _cached_model


def _as_2d(x) -> np.ndarray:
    """Coerce list-of-lists / object-array-of-arrays to (N, 4) float64."""
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _bbox_series(req: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist = _as_2d(req["bbox_history"])  # (16, 4)
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]
    return hist, cx, cy, w, h


def _engineered_features(req: dict) -> np.ndarray:
    """Mirrors baseline.py's feature builder so the XGBoost model sees the same
    layout at inference as at training. Keep this in lock-step with baseline.py.
    """
    _, cx, cy, w, h = _bbox_series(req)
    vx = np.diff(cx)
    vy = np.diff(cy)

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    feats = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx.std() / fw,                            # normalized per frame width
        vy.std() / fh,                            # normalized per frame height
        (h / (w + 1e-6)).mean(),                  # aspect ratio (tallness)
        float(req["ego_available"]),
        ego_s.mean(), ego_s[-1], ego_s.max(),
        ego_y.mean(), ego_y[-1], np.abs(ego_y).max(),
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def _trajectory_features(req: dict) -> np.ndarray:
    """Kinematic features for center-residual trajectory regressors."""
    hist, cx, cy, w, h = _bbox_series(req)

    fw = max(float(req["frame_w"]), 1.0)
    fh = max(float(req["frame_h"]), 1.0)
    dx = np.diff(cx)
    dy = np.diff(cy)
    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)

    feats: list[float] = [
        cx[-1] / fw,
        cy[-1] / fh,
        w[-1] / fw,
        h[-1] / fh,
        (w[-1] * h[-1]) / (fw * fh),
        (cy[-1] + 0.5 * h[-1]) / fh,
        h[-1] / (w[-1] + 1e-6),
    ]
    for n in (1, 2, 4, 8, 15):
        feats.extend([dx[-n:].mean() / fw, dy[-n:].mean() / fh])
    feats.extend([
        (dx[-1] - dx[-4:].mean()) / fw,
        (dy[-1] - dy[-4:].mean()) / fh,
        dx.std() / fw,
        dy.std() / fh,
        (w[-1] - w[-5]) / fw,
        (h[-1] - h[-5]) / fh,
        float(req["ego_available"]),
        ego_s[-1],
        ego_s.mean(),
        ego_s[-1] - ego_s[0],
        np.abs(ego_s).max(),
        ego_y[-1],
        ego_y.mean(),
        ego_y[-1] - ego_y[0],
        np.abs(ego_y).max(),
        np.sin(2 * np.pi * (float(req.get("requested_at_frame", 0)) % 30.0) / 30.0),
        np.cos(2 * np.pi * (float(req.get("requested_at_frame", 0)) % 30.0) / 30.0),
    ])
    for field, values in CATEGORY_VALUES.items():
        value = req.get(field, "")
        feats.extend(1.0 if value == candidate else 0.0 for candidate in values)

    arr = np.asarray(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


def _constant_velocity_centers(req: dict) -> tuple[np.ndarray, float, float]:
    _, cx, cy, w, h = _bbox_series(req)
    w_last = w[-1]
    h_last = h[-1]
    # mean per-frame velocity over last 4 intervals
    vx = float(np.diff(cx[-5:]).mean())
    vy = float(np.diff(cy[-5:]).mean())
    cur_cx, cur_cy = float(cx[-1]), float(cy[-1])

    centers = np.empty((len(HORIZONS_FRAMES), 2), dtype=np.float64)
    for i, horizon in enumerate(HORIZONS_FRAMES):
        centers[i, 0] = cur_cx + vx * horizon
        centers[i, 1] = cur_cy + vy * horizon
    return centers, float(w_last), float(h_last)


def _boxes_from_centers(centers: np.ndarray, w_last: float, h_last: float) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for center, key in zip(centers, HORIZON_KEYS):
        nx, ny = float(center[0]), float(center[1])
        out[key] = [nx - w_last / 2, ny - h_last / 2, nx + w_last / 2, ny + h_last / 2]
    return out


def _constant_velocity_trajectory(req: dict) -> dict[str, list[float]]:
    centers, w_last, h_last = _constant_velocity_centers(req)
    return _boxes_from_centers(centers, w_last, h_last)


def _residual_trajectory(req: dict, traj_model: dict) -> dict[str, list[float]]:
    centers, w_last, h_last = _constant_velocity_centers(req)
    feats = _trajectory_features(req).reshape(1, -1)
    residual_model = traj_model.get("residual_model")
    if residual_model is not None:
        residuals = np.asarray(residual_model.predict(feats)[0], dtype=np.float64).reshape(len(HORIZON_KEYS), 2)
        if not np.isfinite(residuals).all():
            residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
        return _boxes_from_centers(centers + residuals, w_last, h_last)

    regressors = traj_model.get("residual_regressors", [])

    if len(regressors) != len(HORIZON_KEYS) * 2:
        return _boxes_from_centers(centers, w_last, h_last)

    residuals = np.empty((len(HORIZON_KEYS), 2), dtype=np.float64)
    for horizon_idx in range(len(HORIZON_KEYS)):
        residuals[horizon_idx, 0] = float(regressors[horizon_idx * 2].predict(feats)[0])
        residuals[horizon_idx, 1] = float(regressors[horizon_idx * 2 + 1].predict(feats)[0])

    if not np.isfinite(residuals).all():
        residuals = np.nan_to_num(residuals, nan=0.0, posinf=0.0, neginf=0.0)
    return _boxes_from_centers(centers + residuals, w_last, h_last)


def predict(request: dict) -> dict:
    model = _load_model()
    intent_model = model["intent"]
    feats = _engineered_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    intent_prob = float(intent_model.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    traj_model = model.get("traj")
    if traj_model is None:
        out = _constant_velocity_trajectory(request)
    else:
        out = _residual_trajectory(request, traj_model)
    for k in HORIZON_KEYS:
        out[k] = [float(v) if np.isfinite(v) else 0.0 for v in out[k]]
    out["intent"] = intent_prob
    return out
