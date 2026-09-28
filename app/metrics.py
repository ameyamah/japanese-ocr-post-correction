"""Measure whether correction helps and when it makes mistakes."""
from rapidfuzz.distance import Levenshtein


def compute_metrics(samples, predictions):
    if not samples or len(samples) != len(predictions):
        raise ValueError("Provide one prediction for every sample")

    correct = 0
    fixed = 0
    damaged = 0
    character_errors = 0
    reference_characters = 0
    for sample, prediction in zip(samples, predictions):
        was_correct = sample["src"] == sample["tgt"]
        is_correct = prediction is not None and prediction == sample["tgt"]
        correct += is_correct
        fixed += not was_correct and is_correct
        damaged += was_correct and not is_correct
        character_errors += Levenshtein.distance(prediction or "", sample["tgt"])
        reference_characters += len(sample["tgt"])

    return {
        "samples": len(samples),
        "exact_match": correct / len(samples),
        "cer": character_errors / reference_characters if reference_characters else None,
        "fixed": fixed,
        "damaged": damaged,
        "generation_failures": sum(prediction is None for prediction in predictions),
    }