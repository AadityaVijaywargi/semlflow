"""Stage 4 - quality gate + model promotion.

Fails (exit code 1) if a model is below its threshold, which turns the CI/CD/CT
workflow red. If it passes, the new model version gets the `champion` alias in
the MLflow Model Registry (visible in the GUI under "Models").
"""
import json
import sys

from mlflow import MlflowClient

from src.common import ensure_dirs, load_params, setup_mlflow

CHECKS = (("ml", "ml_accuracy"), ("dl", "dl_accuracy"))


def evaluate(params: dict) -> bool:
    ensure_dirs("reports")
    setup_mlflow(params)
    client = MlflowClient()
    rows, all_ok = [], True

    for kind, key in CHECKS:
        with open(f"metrics/{kind}_metrics.json", encoding="utf-8") as f:
            m = json.load(f)
        threshold = params["thresholds"][key]
        ok = m["accuracy"] >= threshold
        all_ok &= ok
        promoted = False
        if ok and m.get("model_version"):
            try:
                client.set_registered_model_alias(
                    m["model_name"], "champion", str(m["model_version"])
                )
                promoted = True
            except Exception as e:  # registry unavailable -> do not fail the gate
                print(f"[evaluate] could not set alias: {e}")
        rows.append(
            dict(model=m["model_name"], accuracy=round(m["accuracy"], 4), threshold=threshold,
                 passed=ok, version=m.get("model_version"), promoted=promoted)
        )

    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump({"gate_passed": all_ok, "models": rows}, f, indent=2)

    lines = [
        "| model | accuracy | threshold | passed | version | champion |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['accuracy']} | {r['threshold']} | "
            f"{'PASS' if r['passed'] else 'FAIL'} | {r['version']} | "
            f"{'champion' if r['promoted'] else '-'} |"
        )
    with open("reports/summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if evaluate(load_params()) else 1)