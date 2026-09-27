"""Record a completed experiment to an MLflow server (after training completes)."""
import argparse
import json
import os
from pathlib import Path

import mlflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--upload-model", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("japanese-ocr-post-correction")
    with mlflow.start_run(run_name=run_dir.name) as run:
        # Parameters describe the experiment; metrics describe its results.
        mlflow.log_params(report["config"])
        mlflow.log_params({"seed": report["seed"], "best_epoch": report["best_epoch"]})
        for epoch in report["history"]:
            mlflow.log_metrics(
                {name: value for name, value in epoch.items()
                 if name != "epoch" and value is not None},
                step=epoch["epoch"],  # MLflow calls the chart's x-coordinate "step".
            )
        mlflow.log_metrics({
            "baseline_exact_match": report["baseline"]["exact_match"],
            "best_exact_match": report["model"]["exact_match"],
        })
        mlflow.log_artifact(str(run_dir / "report.json"))
        # Weights are large. Upload only when you explicitly request it.
        if args.upload_model:
            mlflow.log_artifacts(str(run_dir / "model"), artifact_path="model")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
