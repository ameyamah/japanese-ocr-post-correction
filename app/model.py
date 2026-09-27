"""Load Japanese T5, prepare training samples, and generate corrections."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_MODEL_NAME = "retrieva-jp/t5-base-medium"
MAX_LENGTH = 64


def load_model_and_tokenizer(model_name=DEFAULT_MODEL_NAME, device="auto"):
    # Map the model to GPU if available else keep model on CPU.
    # CUDA -> NVIDIA GPU
    if device.lower() == "auto" or device.lower() == "gpu":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


def tokenize_sample(tokenizer, sample):
    # src --> input
    # tgt --> ground truth label
    inputs = tokenizer(sample["src"], max_length=MAX_LENGTH, truncation=True)
    targets = tokenizer(text_target=sample["tgt"], max_length=MAX_LENGTH, truncation=True) #ground truths
    inputs["labels"] = targets["input_ids"]
    # Padding happens later, when the DataLoader forms a batch.
    return inputs


# disable grads for faster performance
@torch.inference_mode()
def predict_batch(model, tokenizer, texts):
    model.eval()  # Disable training behavior such as dropout.
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    # print(inputs)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    generated_ids = model.generate(
        **inputs, max_new_tokens=MAX_LENGTH, num_beams=1, do_sample=False, use_cache=True
    )
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # A decoder that reaches the length cap without EOS has not finished.
    for index, token_ids in enumerate(generated_ids.tolist()):
        if tokenizer.eos_token_id not in token_ids[1:]:
            predictions[index] = None
    return predictions


def predict(model, tokenizer, text):
    # The HTTP API uses the same generation code as evaluation.
    prediction = predict_batch(model, tokenizer, [text])[0]
    if prediction is None or not prediction.strip():
        raise ValueError("The model did not produce a complete, nonblank correction")
    return prediction