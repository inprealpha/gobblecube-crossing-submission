"""Root-level submission entry point.

The implementation lives in crossing-challenge-starter/predict.py so the
experiment history remains readable. This wrapper exposes the required
`predict(request: dict) -> dict` function from the repository root.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL_PATH = HERE / "crossing-challenge-starter" / "predict.py"
MODEL_PATH = HERE / "model.pkl"

spec = importlib.util.spec_from_file_location("crossing_predict_impl", IMPL_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Crossing implementation from {IMPL_PATH}")

_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_impl)

# Use the root-level model file for the submission surface.
_impl.MODEL_PATH = MODEL_PATH
_impl._cached_model = None

predict = _impl.predict

# Re-export helpers used by the training script and tests.
HORIZON_KEYS = _impl.HORIZON_KEYS
_constant_velocity_centers = _impl._constant_velocity_centers
_engineered_features = _impl._engineered_features
_intent_features = _impl._intent_features
_trajectory_features = _impl._trajectory_features

__all__ = [
    "predict",
    "HORIZON_KEYS",
    "_constant_velocity_centers",
    "_engineered_features",
    "_intent_features",
    "_trajectory_features",
]
