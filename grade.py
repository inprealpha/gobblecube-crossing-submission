#!/usr/bin/env python
"""Root-level grader entry point for Docker submission mode."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRADE_PATH = HERE / "crossing-challenge-starter" / "grade.py"

spec = importlib.util.spec_from_file_location("crossing_grade_impl", GRADE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Crossing grader from {GRADE_PATH}")

_impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_impl)

if __name__ == "__main__":
    _impl.main(sys.argv)
