import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 128     # our longest example is 122 tokens
IGNORE_INDEX = -100  # PyTorch skips any position whose label is -100
BATCH_SIZE = 8


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def encode_example(messages, tokenizer):
    """Turn one chat example into token IDs plus labels that only cover the assistant's reply."""
    # The PROMPT is everything before the answer: system + user + "<|im_start|>assistant\n".
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    prompt_ids = tokenizer(prompt_text)["input_ids"]
    full_ids = tokenizer(full_text)["input_ids"]
    assert full_ids[:len(prompt_ids)] == prompt_ids, "prompt must be an exact prefix of the full text"
    assert len(full_ids) <= MAX_LENGTH, f"example is {len(full_ids)} tokens, longer than MAX_LENGTH"

    # The template ends with "<|im_end|>\n". We grade up to and including <|im_end|> (so the
    # model learns to stop) but not the newline after it.
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    last_end = len(full_ids) - 1 - full_ids[::-1].index(end_id)

    labels = list(full_ids)
    labels[:len(prompt_ids)] = [IGNORE_INDEX] * len(prompt_ids)                    # mask the prompt
    labels[last_end + 1:] = [IGNORE_INDEX] * (len(full_ids) - last_end - 1)        # mask the trailing newline
    return {"input_ids": full_ids, "labels": labels}


class ChatDataset(Dataset):
    def __init__(self, path, tokenizer):
        self.items = [encode_example(example["messages"], tokenizer) for example in load_jsonl(path)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def make_collate_fn(pad_id, pad_to=None):
    """Builds a function that stacks examples into one rectangular batch, padding as needed.

    pad_to=None pads to the longest example in each batch; pad_to=N pads every batch to exactly N tokens.
    """
    def collate(batch):
        longest = pad_to or max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            n_pad = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * n_pad)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * n_pad)  # 0 = "ignore this padding"
            labels.append(item["labels"] + [IGNORE_INDEX] * n_pad)             # padding is never graded
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
    return collate


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = ChatDataset("data/train.jsonl", tokenizer)
    val_ds = ChatDataset("data/val.jsonl", tokenizer)
    print(f"Encoded {len(train_ds)} training and {len(val_ds)} validation examples")
    print(f"Pad token: {tokenizer.pad_token!r} (id {tokenizer.pad_token_id})")

    # ---- Look at ONE example: what is ignored, what is graded? ----
    item = train_ds[0]
    ids, labels = item["input_ids"], item["labels"]
    ignored = [i for i, label in zip(ids, labels) if label == IGNORE_INDEX]
    graded = [label for label in labels if label != IGNORE_INDEX]
    print(f"\n=== One example: {len(ids)} tokens total ===")
    print(f"IGNORED ({len(ignored)} tokens): {tokenizer.decode(ignored)!r}")
    print(f"GRADED  ({len(graded)} tokens): {tokenizer.decode(graded)!r}")

    # ---- The boundary, token by token ----
    first = next(i for i, label in enumerate(labels) if label != IGNORE_INDEX)
    last = max(i for i, label in enumerate(labels) if label != IGNORE_INDEX)
    print("\n=== Token by token around the two boundaries ===")
    print(f"{'pos':>4}  {'token':<16} {'input_id':>8}  label")
    for pos in list(range(first - 3, first + 3)) + [None] + list(range(last - 1, len(ids))):
        if pos is None:
            print("  ...")
            continue
        token = tokenizer.convert_ids_to_tokens(ids[pos])
        label = "ignored" if labels[pos] == IGNORE_INDEX else labels[pos]
        print(f"{pos:>4}  {token!r:<16} {ids[pos]:>8}  {label}")

    # ---- One real batch ----
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )
    batch = next(iter(loader))
    print("\n=== The first batch ===")
    for name, tensor in batch.items():
        print(f"{name:15} shape {tuple(tensor.shape)}")

    shortest = batch["attention_mask"].sum(dim=1).argmin().item()  # the row with the most padding
    print(f"\nRow {shortest} is the shortest in this batch. Its last 6 positions:")
    print(f"  input_ids      {batch['input_ids'][shortest][-6:].tolist()}")
    print(f"  attention_mask {batch['attention_mask'][shortest][-6:].tolist()}")
    print(f"  labels         {batch['labels'][shortest][-6:].tolist()}")

    # ---- Totals ----
    total = sum(len(x["input_ids"]) for x in train_ds)
    trained = sum(sum(label != IGNORE_INDEX for label in x["labels"]) for x in train_ds)
    print("\n=== Totals for the training set ===")
    print(f"Tokens: {total}, graded tokens: {trained} ({trained / total * 100:.0f}%)")
    print(f"Batches per epoch: {len(loader)} (batch size {BATCH_SIZE})")
