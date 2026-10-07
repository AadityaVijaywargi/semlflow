"""Stage 1 - create raw datasets and train/test splits (DVC-tracked outputs)."""
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split

from src.common import ensure_dirs, load_params


def prepare(params: dict) -> None:
    ensure_dirs("data/raw", "data/processed")
    d = params["data"]
    datasets = {
        "wine": load_wine(as_frame=True),      # classic ML  : 178 rows, 13 features
        "digits": load_digits(as_frame=True),  # deep learning: 1797 rows, 64 pixels
    }
    for name, ds in datasets.items():
        df = ds.frame.rename(columns={"target": "target"})
        df.to_csv(f"data/raw/{name}.csv", index=False)
        train, test = train_test_split(
            df, test_size=d["test_size"], random_state=d["seed"], stratify=df["target"]
        )
        train.to_csv(f"data/processed/{name}_train.csv", index=False)
        test.to_csv(f"data/processed/{name}_test.csv", index=False)
        print(f"[prepare] {name}: train={len(train)} test={len(test)}")


if __name__ == "__main__":
    prepare(load_params())
