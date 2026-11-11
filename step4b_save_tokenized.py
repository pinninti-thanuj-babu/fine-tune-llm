import json

from transformers import AutoTokenizer

from step4_prepare_data import MODEL_NAME, ChatDataset

# For inspection only: training rebuilds this data in memory and does not read these files.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

for split in ["train", "val"]:
    dataset = ChatDataset(f"data/{split}.jsonl", tokenizer)
    out_path = f"data/{split}_tokenized.jsonl"
    with open(out_path, "w") as f:
        for item in dataset:
            row = {
                "tokens": tokenizer.convert_ids_to_tokens(item["input_ids"]),
                "input_ids": item["input_ids"],
                "labels": item["labels"],  # -100 = ignored, otherwise the token being graded
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(dataset)} examples -> {out_path}")
