"""Read the JSON arrays """
import json


def load_dataset(path):
    # src --> input
    # tgt --> ground truth label
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    if not samples:
        raise ValueError(f"Dataset is empty: {path}")
    return samples
