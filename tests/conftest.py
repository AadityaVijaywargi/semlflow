import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    """Isolated working dir with its own params + its own throw-away MLflow DB."""
    shutil.copy(ROOT / "params.yaml", tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    return tmp_path
