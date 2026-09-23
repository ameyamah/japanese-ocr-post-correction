"""Load OCR examples from JSON and download the public JaPOC dataset."""
import argparse
from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import TypedDict
from urllib.request import urlopen

from app.artifacts import write_bytes, write_json


DATASET_BASE_URL = "https://raw.githubusercontent.com/FastAccounting/ocr_correction_benchmark/main"
MAX_DOWNLOAD_BYTES = 10_000_000 #10 MB limit
DOWNLOAD_TIMEOUT_SECONDS = 30 #30 second timeout


class OCRSample(TypedDict):
    """The dataset's record schema."""
    id: str
    src: str
    tgt: str
    correct: bool


# Validate JSON dataset contents
def validate_records(records: object) -> list[OCRSample]:
    """Validate the dataset schema without filtering or normalizing text."""
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON array of records")
    seen_ids = set()
    for record_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {record_number}: expected an object")
        if not isinstance(record.get("id"), str) or not record["id"].strip():
            raise ValueError(f"Record {record_number}: missing ID")
        if record["id"] in seen_ids:
            raise ValueError(f"Record {record_number}: duplicate ID")
        if not all(isinstance(record.get(field), str) for field in ("src", "tgt")):
            raise ValueError(f"Record {record_number}: src and tgt must be strings")
        if type(record.get("correct")) is not bool or record["correct"] != (record["src"] == record["tgt"]):
            raise ValueError(f"Record {record_number}: wrong correct flag")
        seen_ids.add(record["id"])
    return records


# Load the json dataset
def load_dataset(file_path: str | Path) -> list[OCRSample]:
    """Load and validate a local JSON array."""
    with Path(file_path).open(encoding="utf-8") as file:
        try:
            records = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"{file_path}: expected a JSON array. Convert to JaPOC format and try again.") from error
    return validate_records(records)


# Extract data from JSON file
def parse_upstream_records(content: bytes) -> list[OCRSample]:
    """Decode the publisher's JSONL content line by line."""
    try:
        records = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Expected UTF-8 JSON records from the dataset server") from error
    return validate_records(records)


# Download dataset files.
def download_file(
    url: str,
    destination: str | Path,
    validator: Callable[[bytes], object] | None = None,
) -> Path:
    """Reuse a local file or validate a bounded download before publishing it."""
    destination = Path(destination)
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"Invalid cached file: {destination}")
        content = destination.read_bytes()
    else:
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content = response.read(MAX_DOWNLOAD_BYTES + 1)
    if not content.strip() or len(content) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"Empty or oversized file: {destination}")
    if validator is not None:
        validator(content)
    # write_bytes reuses identical contents and refuses to replace different files.
    write_bytes(destination, content)
    return destination


# Explicit method to download the dataset
def download_dataset(
    data_dir: str | Path = "data/japoc",
    ocr_engine: str = "vision",
    splits: Sequence[str] = ("train", "val"),
) -> dict[str, Path]:
    """Download missing splits and produce ordinary JSON arrays without hashes."""
    if ocr_engine not in ("vision", "robota") or not splits or any(
        split not in ("train", "val", "test") for split in splits
    ):
        raise ValueError("Choose vision/robota and train/val/test splits")
    data_dir = Path(data_dir)
    download_file(f"{DATASET_BASE_URL}/LICENSE", data_dir / "LICENSE")
    dataset_paths = {}
    for split in splits:
        filename = f"{split}_{ocr_engine}_ocr.json"
        source_path = download_file(
            f"{DATASET_BASE_URL}/datasets_companyname/{filename}",
            data_dir / "raw" / filename,
            validator=parse_upstream_records,
        )
        records = parse_upstream_records(source_path.read_bytes())
        output_path = data_dir / ocr_engine / f"{split}.json"
        write_json(output_path, records)
        dataset_paths[split] = output_path
    return dataset_paths


# Audit data
def summarize_dataset(
    records: Sequence[OCRSample],
    training_records: Sequence[OCRSample] = (),
) -> dict[str, int]:
    """Count data-quality issues and overlap without changing the records."""
    training_targets = {record["tgt"] for record in training_records}
    return {
        "rows": len(records),
        "already_correct": sum(record["correct"] for record in records),
        "blank_sources": sum(not record["src"].strip() for record in records),
        "blank_targets": sum(not record["tgt"].strip() for record in records),
        "duplicate_pairs": len(records) - len({(record["src"], record["tgt"]) for record in records}),
        "targets_seen_in_train": sum(record["tgt"] in training_targets for record in records),
    }


# Add support for command line usage
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/japoc")
    parser.add_argument("--ocr", choices=("vision", "robota"), default="vision")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    args = parser.parse_args()
    for split, dataset_path in download_dataset(args.root, args.ocr, args.splits).items():
        print(split, dataset_path, summarize_dataset(load_dataset(dataset_path)))


if __name__ == "__main__":
    main()