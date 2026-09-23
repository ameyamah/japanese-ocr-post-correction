"""Evaluation metrics for OCR correction."""
from collections.abc import Sequence

from app.data import OCRSample


# Levenshtein distance between two strings
# using DP to calculate
def levenshtein_distance(source: str, target: str) -> int:
    """Return the minimum character insertions, deletions, and substitutions."""
    previous_row = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, 1):
        current_row = [source_index]
        for target_index, target_character in enumerate(target, 1):
            current_row.append(min(
                current_row[-1] + 1,
                previous_row[target_index] + 1,
                previous_row[target_index - 1] + (source_character != target_character),
            ))
        previous_row = current_row
    return previous_row[-1]


# Calculate metrics
def compute_metrics(
    records: Sequence[OCRSample],
    predictions: Sequence[str | None],
) -> dict[str, int | float | None]:
    """Compute exact match, correction/damage counts, failures, and corpus CER."""
    if not records or len(records) != len(predictions):
        raise ValueError("Need one prediction per record")
    if any(prediction is not None and not isinstance(prediction, str) for prediction in predictions):
        raise ValueError("A prediction must be text or None for a failure")
    originally_correct = [record["src"] == record["tgt"] for record in records]
    predicted_correct = [
        prediction is not None and prediction == record["tgt"]
        for record, prediction in zip(records, predictions)
    ]
    reference_characters = sum(len(record["tgt"]) for record in records)
    character_errors = sum(
        levenshtein_distance(prediction if prediction is not None else "", record["tgt"])
        for record, prediction in zip(records, predictions)
    )
    return {
        "n": len(records),
        "exact_match": sum(predicted_correct) / len(records),
        "fixed": sum(not before and after for before, after in zip(originally_correct, predicted_correct)),
        "damaged": sum(before and not after for before, after in zip(originally_correct, predicted_correct)),
        "failures": sum(prediction is None for prediction in predictions),
        "cer_empty_on_failure": character_errors / reference_characters if reference_characters else None,
    }