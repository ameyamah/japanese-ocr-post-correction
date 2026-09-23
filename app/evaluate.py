"""Evaluate a saved checkpoint."""
import argparse
from pathlib import Path
import time

import torch

from app.artifacts import write_json
from app.data import load_dataset
from app.metrics import compute_metrics
from app.model import load_model_and_tokenizer, predict



def evaluate(model, tokenizer, rows, training_targets=None):
    records = []
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for row in rows:
        try:
            prediction, error = predict(model, tokenizer, row["src"]), None
        except ValueError as exception:
            prediction, error = None, str(exception)
        records.append({"id": row["id"], "src": row["src"], "tgt": row["tgt"],
                        "prediction": prediction, "error": error})
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    predictions = [r["prediction"] for r in records]
    baseline = compute_metrics(rows, [r["src"] for r in rows])
    candidate = compute_metrics(rows, predictions)
    report = {
        "baseline": baseline, "candidate": candidate,
        "seconds": elapsed, "seconds_per_row": elapsed / len(rows),
    }
    if training_targets is not None:
        report["groups"] = {}
        for name, seen in (("seen", True), ("unseen", False)):
            indices = [i for i, row in enumerate(rows) if (row["tgt"] in training_targets) == seen]
            report["groups"][name] = compute_metrics([rows[i] for i in indices], [predictions[i] for i in indices]) if indices else None
    return report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--training-data", help="Optional training JSON for seen/unseen-name analysis")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        raise ValueError("Evaluate a saved local checkpoint")
    rows = load_dataset(args.data)
    training = load_dataset(args.training_data) if args.training_data else []
    if {r["id"] for r in rows} & {r["id"] for r in training}:
        raise ValueError("Evaluation and training IDs overlap")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    targets = {r["tgt"] for r in training} if training else None
    report, predictions = evaluate(model, tokenizer, rows, targets)
    report["config"] = vars(args)
    write_json(output / "report.json", report)
    write_json(output / "predictions.json", predictions)
    print(report)

if __name__ == "__main__":
    main()
