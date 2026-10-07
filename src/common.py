"""Shared helpers: params loading, MLflow setup, plotting, model registry."""
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import yaml
from mlflow import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_sha() -> str:
    """Commit hash, so every MLflow run can be traced back to the code version."""
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def setup_mlflow(params: dict) -> None:
    """Use $MLFLOW_TRACKING_URI if set (server / DagsHub), else a local sqlite file."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(params["mlflow"]["experiment"])


def run_tags(kind: str) -> dict:
    return {
        "model_type": kind,
        "git_sha": git_sha(),
        "triggered_by": os.getenv("GITHUB_EVENT_NAME", "local"),
        "ci_run": os.getenv("GITHUB_RUN_ID", "local"),
    }


def log_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    mlflow.log_figure(fig, filename)
    plt.close(fig)


def model_version_for_run(model_name: str, run_id: str):
    """Return the registry version created by a given run (or None)."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    mine = [int(v.version) for v in versions if v.run_id == run_id]
    return max(mine) if mine else None


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
