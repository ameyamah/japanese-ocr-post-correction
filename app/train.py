"""Fine-tune T5"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

from app.data import load_dataset
from app.evaluate import evaluate
from app.model import DEFAULT_MODEL_NAME, load_model_and_tokenizer, tokenize_sample


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Train Dataset Path")
    parser.add_argument("--val", required=True, help="Val Dataset Path")
    parser.add_argument("--output", required=True, help="Output Directory Path")
    parser.add_argument("--base-model", default=DEFAULT_MODEL_NAME, help="Base model name (HuggingFace)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    args = parser.parse_args()
    # Catch only settings that would prevent meaningful training.
    counts = [args.epochs, args.batch_size, args.eval_batch_size]
    if min(counts) < 1 or not 0 < args.learning_rate < 1:
        parser.error("Counts must be positive; learning rate must be between 0 and 1")
    return args


def train(args):
    torch.manual_seed(648)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)

    print("Loading model and tokenizer...", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.device)
    if model.device.type == "cuda":
        # Recompute some activations during backward to save GPU memory.
        model.gradient_checkpointing_enable()

    train_samples = load_dataset(args.train)
    val_samples = load_dataset(args.val)
    if {sample["id"] for sample in train_samples} & {sample["id"] for sample in val_samples}:
        raise ValueError("Training and validation IDs overlap; use separate splits")

    features = [tokenize_sample(tokenizer, sample) for sample in train_samples]
    # Different-length sequences need padding before they can form a tensor.
    # -100 tells the loss function to ignore padded target positions.
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train_dataloader = DataLoader(
        features, batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    batches_per_epoch = len(train_dataloader)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0, foreach=False
    )
    history = []
    best_exact_match = -1.0
    report = {"config": vars(args), "seed": 42, "history": history}
    print(f"Training on {model.device}: {batches_per_epoch} batches per epoch", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()  # Evaluation changes the mode. restore it each epoch.
        total_loss = 0.0
        total_tokens = 0
        for batch_number, batch in enumerate(train_dataloader, start=1):
            batch = {name: tensor.to(model.device) for name, tensor in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Training loss is infinite. Stop rather than save bad weights")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

            tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            print(
                f"epoch {epoch}/{args.epochs}, batch {batch_number}/{batches_per_epoch}, "
                f"loss={loss.item():.4f}", flush=True
            )
        # Release the last training batch/gradients before generation.
        optimizer.zero_grad(set_to_none=True)
        del batch, loss

        metrics, predictions = evaluate(model, tokenizer, val_samples, args.eval_batch_size)
        history.append({
            "epoch": epoch, "train_loss": total_loss / total_tokens,
            "train_batches": batches_per_epoch, **metrics["model"],
        })
        print(f"Validation exact match: {metrics['model']['exact_match']:.3f}", flush=True)

        if metrics["model"]["exact_match"] > best_exact_match:
            best_exact_match = metrics["model"]["exact_match"]
            model.save_pretrained(output_dir / "model")
            tokenizer.save_pretrained(output_dir / "model")
            (output_dir / "predictions.json").write_text(
                json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report.update({"best_epoch": epoch, **metrics})
            print(f"Saved selected checkpoint from epoch {epoch}", flush=True)

    # Write report after training finishes.
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Finished. Best epoch: {report['best_epoch']}. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    train(parse_args())
