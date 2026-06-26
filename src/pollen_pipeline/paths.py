from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
DERIVED_DIR = PROJECT_ROOT / "derived"
RUNS_DIR = PROJECT_ROOT / "runs"
WEIGHTS_DIR = PROJECT_ROOT / "weights"

