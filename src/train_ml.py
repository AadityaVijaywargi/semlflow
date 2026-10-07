"""Stage 2 - classic ML: RandomForest on Wine, tracked with MLflow."""
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.common import (
    ensure_dirs,
    load_params,
    log_confusion_matrix,
    model_version_for_run,
    run_tags,
    setup_mlflow,
)


def train(params: dict) -> dict:
    ensure_dirs("models", "metrics")
    setup_mlflow(params)
    p = params["ml"]
    name = params["mlflow"]["ml_model_name"]

    tr = pd.read_csv("data/processed/wine_train.csv")
    te = pd.read_csv("data/processed/wine_test.csv")
    X_tr, y_tr = tr.drop(columns="target"), tr["target"]
    X_te, y_te = te.drop(columns="target"), te["target"]

    with mlflow.start_run(run_name="ml-random-forest-wine", tags=run_tags("ml")) as run:
        mlflow.log_params({"model": "RandomForest", **p, "seed": params["data"]["seed"]})
        model = RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            random_state=params["data"]["seed"],
        ).fit(X_tr, y_tr)

        pred = model.predict(X_te)
        metrics = {
            "accuracy": accuracy_score(y_te, pred),
            "f1_macro": f1_score(y_te, pred, average="macro"),
            "precision_macro": precision_score(y_te, pred, average="macro"),
            "recall_macro": recall_score(y_te, pred, average="macro"),
        }
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_te, pred, "Wine - RandomForest", "confusion_matrix.png")
        mlflow.log_artifact("params.yaml")
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_te, pred),
            input_example=X_te.head(3),
            registered_model_name=name,
            # we trained this model ourselves, so trusting sklearn's Tree type is safe
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        version = model_version_for_run(name, run.info.run_id)

    joblib.dump(model, "models/ml_model.joblib")
    out = {**metrics, "run_id": run.info.run_id, "model_name": name, "model_version": version}
    with open("metrics/ml_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[train_ml] accuracy={metrics['accuracy']:.4f} run_id={run.info.run_id}")
    return out


if __name__ == "__main__":
    train(load_params())
