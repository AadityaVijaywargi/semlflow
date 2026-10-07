import json

import pandas as pd
import pytest

from src.common import load_params
from src.evaluate import evaluate
from src.prepare_data import prepare
from src.train_ml import train as train_ml


def test_prepare_creates_splits(workdir):
    params = load_params()
    prepare(params)
    train = pd.read_csv("data/processed/wine_train.csv")
    test = pd.read_csv("data/processed/wine_test.csv")
    assert len(train) + len(test) == 178
    assert "target" in train.columns
    assert pd.read_csv("data/processed/digits_train.csv").shape[1] == 65


def test_ml_training_meets_threshold(workdir):
    params = load_params()
    prepare(params)
    out = train_ml(params)
    assert out["accuracy"] >= params["thresholds"]["ml_accuracy"]
    assert (workdir / "models" / "ml_model.joblib").exists()
    assert out["model_version"] == 1


def test_dl_training_smoke(workdir):
    pytest.importorskip("torch")
    from src.train_dl import train as train_dl

    params = load_params()
    params["dl"]["epochs"] = 3  # keep unit test fast
    prepare(params)
    out = train_dl(params)
    assert out["accuracy"] > 0.5
    assert (workdir / "models" / "dl_model.pt").exists()


def test_quality_gate_fails_on_bad_model(workdir):
    params = load_params()
    (workdir / "metrics").mkdir()
    for kind, name in (("ml", "a"), ("dl", "b")):
        with open(workdir / "metrics" / f"{kind}_metrics.json", "w") as f:
            json.dump({"accuracy": 0.10, "model_name": name, "model_version": None}, f)
    assert evaluate(params) is False
