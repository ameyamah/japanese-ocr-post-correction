"""Generate corrections in batches and compare them with unchanged OCR text."""
import argparse
import json
import time
from pathlib import Path

from torch.utils.data import DataLoader

from app.data import load_dataset
from app.metrics import compute_metrics
from app.model import load_model_and_tokenizer, predict_batch


def evaluate(model, tokenizer, samples, batch_size=16):
    dataloader = DataLoader(
        [sample["src"] for sample in samples], batch_size=batch_size, shuffle=False
    )
    predictions = []
    start = time.perf_counter()
    print(f"Evaluating {len(samples)} samples in {len(dataloader)} batches...", flush=True)
    for batch_number, texts in enumerate(dataloader, start=1):
        predictions.extend(predict_batch(model, tokenizer, texts))
        print(
            f"eval batch {batch_number}/{len(dataloader)} "
            f"({len(predictions)} samples, {time.perf_counter() - start:.1f}s)",
            flush=True,
        )

    metrics = {
        "baseline": compute_metrics(samples, [sample["src"] for sample in samples]),
        "model": compute_metrics(samples, predictions),
        "seconds": time.perf_counter() - start,
    }
    rows = [
        {"id": sample["id"], "src": sample["src"], "tgt": sample["tgt"], "prediction": prediction}
        for sample, prediction in zip(samples, predictions)
    ]
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a saved model directory")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)  # Do not overwrite another experiment.
    samples = load_dataset(args.data)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    metrics, predictions = evaluate(model, tokenizer, samples, args.batch_size)
    report = {"config": vars(args), **metrics}
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
