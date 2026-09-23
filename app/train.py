"""Fine-tune Japanese T5 using minibatches."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from app.artifacts import write_json
from app.data import summarize_dataset, download_dataset, load_dataset
from app.evaluate import evaluate
from app.model import DEFAULT_MODEL_NAME, tokenize_sample, load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--fetch", action="store_true", help="Download/reuse official train and val if paths omitted")
    parser.add_argument("--ocr", choices=("vision", "robota"), default="vision")
    parser.add_argument("--data-root", default="data/japoc")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=200, help="Optimizer updates, not microbatches")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    for key in ("steps", "batch_size", "accumulation", "eval_every", "limit_train", "limit_val"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"{key} must be positive")
    if not 0 < args.learning_rate < 1:
        parser.error("Choose 0 < learning rate < 1")
    if bool(args.train) != bool(args.val) or (args.fetch and args.train):
        parser.error("Supply both --train/--val, or --fetch without those paths")
    if not args.train and not args.fetch:
        parser.error("Supply --train and --val, or explicitly use --fetch")
    return args


def prepare_data(args, tokenizer):
    training, validation = load_dataset(args.train), load_dataset(args.val)
    if {r["id"] for r in training} & {r["id"] for r in validation}:
        raise ValueError("Training and validation IDs overlap")
    training, validation = training[:args.limit_train], validation[:args.limit_val]
    features, accepted, excluded = [], [], []
    for row in training:
        try:
            features.append(tokenize_sample(tokenizer, row))
            accepted.append(row)
        except ValueError as error:
            excluded.append({"id": row["id"], "reason": str(error)})
    if not accepted:
        raise ValueError("No trainable rows after blank/length checks")
    return features, accepted, validation, excluded, summarize_dataset(training)


def iterate_batches(loader):
    while True:
        yield from loader


def train(args: argparse.Namespace, output: Path) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.device)
    if args.precision == "bf16" and (model.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a CUDA device with bf16 support; use fp32 otherwise")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    features, training, validation, excluded, original_audit = prepare_data(args, tokenizer)
    write_json(output / "excluded-training.json", excluded)
    write_json(output / "training-rows.json", training)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(features, batch_size=args.batch_size, shuffle=True,
                        generator=generator, collate_fn=collator)
    batches = iterate_batches(loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    report = {
        "config": vars(args),
        "training_audit": original_audit, "training_rows_used": len(training),
        "validation_audit": summarize_dataset(validation, training),
        "excluded_training_rows": len(excluded),
        "losses": [], "validation_history": [],
    }
    best_rank = None
    for step in range(1, args.steps + 1):
        window = [next(batches) for _ in range(args.accumulation)]
        token_count = sum(int((batch["labels"] != -100).sum()) for batch in window)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        for cpu_batch in window:
            weight = int((cpu_batch["labels"] != -100).sum()) / token_count
            batch = {key: value.to(model.device) for key, value in cpu_batch.items()}
            with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss * weight).backward()
            update_loss += loss.item() * weight
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        report["losses"].append(update_loss)
        print(f"update {step}/{args.steps}: loss={update_loss:.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result, predictions = evaluate(model, tokenizer, validation, {r["tgt"] for r in training})
            metrics = result["candidate"]
            report["validation_history"].append({"step": step, **metrics})
            rank = (metrics["exact_match"], -metrics["damaged"], -metrics["failures"])
            if best_rank is None or rank > best_rank:
                best_rank = rank
                model.save_pretrained(output / "model", safe_serialization=True)
                tokenizer.save_pretrained(output / "model")
                write_json(output / "predictions.json", predictions, replace=True)
                report.update(result)
                report["best_step"] = step
    write_json(output / "report.json", report)
    return report


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.fetch:
        paths = download_dataset(args.data_root, args.ocr)
        args.train, args.val = str(paths["train"]), str(paths["val"])
    report = train(args, output)
    print(json.dumps({"best_step": report["best_step"],
                      "candidate": report["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
