# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # get model specifications from huggingface transformers library
# tokenizer = AutoTokenizer.from_pretrained("retrieva-jp/t5-base-medium")
# model = AutoModelForSeq2SeqLM.from_pretrained(
#     "retrieva-jp/t5-base-medium", device_map="auto"
# )


"""Checkpoint selection, tokenization, device placement, and prediction."""
from pathlib import Path
import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding,
    PreTrainedModel, PreTrainedTokenizerBase,
)

from app.data import OCRSample

DEFAULT_MODEL_NAME = "retrieva-jp/t5-base-medium"
MAX_SEQUENCE_LENGTH = 64

# select the best available accelarator for torch
def get_device(name: str = "auto") -> torch.device:
    match name:
        case "cpu":
            return torch.device("cpu")
        case "gpu":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("Select accelarator CUDA is not available.\n" \
                "Falling back to next best available accelarator.")
                return torch.accelarator.get_best_device()            
        case "auto":
            return torch.accelarator.get_best_device()
        case _:
            return torch.accelarator.get_best_device()
        

# Load the model and the tokenizer into pyotrch
def load_model_and_tokenizer(
    model_name_or_path: str | Path = DEFAULT_MODEL_NAME,
    device: str = "auto",
    *,
    device_map: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face model ID or local checkpoint; device_map is for inference."""
    if device_map is not None and device != "auto":
        raise ValueError("Choose either device or device_map, not both")
    model_name_or_path = str(model_name_or_path)
    local_files_only = Path(model_name_or_path).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=False, trust_remote_code=False,
        local_files_only=local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, device_map=device_map, trust_remote_code=False,
        local_files_only=local_files_only,
    )
    if device_map is None:
        model.to(get_device(device))
    return model, tokenizer


# Tokenize any string
def tokenize_text(tokenizer: PreTrainedTokenizerBase, text: str) -> BatchEncoding:
    if not text.strip() or len(text) > 256:
        raise ValueError("Use 1 to 256 nonblank source characters")
    encoded = tokenizer(text, truncation=False)
    if len(encoded["input_ids"]) > MAX_SEQUENCE_LENGTH:
        raise ValueError("Source exceeds token limit")
    return encoded

# tokenize a sample target json
def tokenize_sample(tokenizer: PreTrainedTokenizerBase, sample: OCRSample) -> dict[str, list[int]]:
    encoded = tokenize_text(tokenizer, sample["src"])
    labels = tokenizer(text_target=sample["tgt"], truncation=False)["input_ids"]
    if not sample["tgt"].strip() or len(labels) > MAX_SEQUENCE_LENGTH:
        raise ValueError("Blank or excessively long target")
    return {**encoded, "labels": labels}


# perform inference with the model.
# Get into inference mode and disable gradients for faster computation
@torch.inference_mode()
def predict(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str) -> str:
    encoded = tokenize_text(tokenizer, text)
    batch = {key: torch.tensor([value], device=model.device) for key, value in encoded.items()}
    model.eval()
    generated_ids = model.generate(**batch, max_new_tokens=MAX_SEQUENCE_LENGTH, num_beams=1, do_sample=False)
    if tokenizer.eos_token_id not in generated_ids[0, 1:].tolist():
        raise ValueError("Generation did not finish")
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if not output.strip():
        raise ValueError("Generation returned no text")
    return output
