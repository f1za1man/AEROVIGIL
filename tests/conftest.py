import os
from pathlib import Path

import pytest

# tests never write to the project's MLflow store (aerovigil.tracking)
os.environ["AEROVIGIL_NO_MLFLOW"] = "1"

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed" / "module03"


@pytest.fixture(scope="session")
def paths():
    p = {
        "root": ROOT,
        "cleaned": PROC / "cleaned",
        "features": PROC / "features",
        "normalized": PROC / "normalized",
        "windows": PROC / "windows",
        "params": PROC / "normalization_parameters.json",
        "splits": ROOT / "data" / "interim" / "module03" / "flight_splits.csv",
        "config": ROOT / "configs" / "preprocessing" / "preprocessing.yaml",
    }
    for key in ("cleaned", "features", "normalized", "windows", "params", "splits"):
        if not p[key].exists():
            pytest.skip(f"Module 03 output missing: {p[key]}")
    return p
