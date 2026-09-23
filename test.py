from app.data import load_dataset
from transformers import DataCollatorForSeq2Seq
from app.model import load_model_and_tokenizer, tokenize_sample, predict
model, tokenizer = load_model_and_tokenizer(device = "cpu")
print(model.config.model_type)
print(model.device)
print(tokenizer("会社")["input_ids"])
records = load_dataset("data/japoc/robota/train.json")[:2]
features = [tokenize_sample(tokenizer, record) for record in records]
collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
batch = collator(features)
batch = {name: values.to(model.device) for name, values in batch.items()}
for name, values in batch.items():
    print(name, tuple(values.shape), values.device)

predict(model, tokenizer, "株式会杜テスト")

print(model(**batch).loss.item())