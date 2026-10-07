"""Stage 3 - deep learning: PyTorch MLP on Digits, tracked with MLflow."""
import json

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.common import (
    ensure_dirs,
    load_params,
    log_confusion_matrix,
    model_version_for_run,
    run_tags,
    setup_mlflow,
)


def _load(path: str):
    df = pd.read_csv(path)
    X = (df.drop(columns="target").values / 16.0).astype("float32")  # pixels are 0..16
    y = df["target"].values.astype("int64")
    return X, y


def build_model(hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(64, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, 10),
    )


def train(params: dict) -> dict:
    ensure_dirs("models", "metrics")
    setup_mlflow(params)
    p = params["dl"]
    name = params["mlflow"]["dl_model_name"]
    seed = params["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr, y_tr = _load("data/processed/digits_train.csv")
    X_te, y_te = _load("data/processed/digits_test.csv")
    Xt_tr, yt_tr = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xt_te = torch.from_numpy(X_te)

    model = build_model(p["hidden_units"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="dl-pytorch-mlp-digits", tags=run_tags("dl")) as run:
        mlflow.log_params({"model": "PyTorch-MLP", **p, "seed": seed})
        for epoch in range(1, p["epochs"] + 1):
            model.train()
            perm = torch.randperm(len(Xt_tr))
            total = 0.0
            for i in range(0, len(perm), p["batch_size"]):
                idx = perm[i : i + p["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(model(Xt_tr[idx]), yt_tr[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            model.eval()
            with torch.no_grad():
                pred = model(Xt_te).argmax(1).numpy()
            # per-epoch metrics -> live learning curves in the MLflow GUI
            mlflow.log_metric("train_loss", total / len(perm), step=epoch)
            mlflow.log_metric("test_accuracy", accuracy_score(y_te, pred), step=epoch)

        metrics = {
            "accuracy": accuracy_score(y_te, pred),
            "f1_macro": f1_score(y_te, pred, average="macro"),
        }
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})
        log_confusion_matrix(y_te, pred, "Digits - PyTorch MLP", "confusion_matrix.png")
        mlflow.log_artifact("params.yaml")
        with torch.no_grad():
            sample_out = model(Xt_te[:5]).numpy()
        mlflow.pytorch.log_model(
            model,
            name="model",
            signature=infer_signature(X_te[:5], sample_out),
            input_example=X_te[:5],
            registered_model_name=name,
        )
        version = model_version_for_run(name, run.info.run_id)

    torch.save(model.state_dict(), "models/dl_model.pt")
    out = {**metrics, "run_id": run.info.run_id, "model_name": name, "model_version": version}
    with open("metrics/dl_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[train_dl] accuracy={metrics['accuracy']:.4f} run_id={run.info.run_id}")
    return out


if __name__ == "__main__":
    train(load_params())
