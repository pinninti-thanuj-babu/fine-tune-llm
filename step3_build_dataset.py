import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from loan_facts import FACTS, SYSTEM_PROMPT

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VAL_FRACTION = 0.25  # hold out about a quarter of each fact's questions
SEED = 42
OUT_DIR = Path("data")


def make_example(question, answer):
    """One training example, in the same chat format the model was built on."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


rng = random.Random(SEED)  # fixed seed -> the same split every time
train, val = [], []
seen_questions = set()

for fact in FACTS:
    questions = list(fact["questions"])
    for q in questions:
        assert q not in seen_questions, f"Duplicate question: {q}"
        seen_questions.add(q)

    # Split PER FACT, so every fact appears in training AND has held-out wordings.
    # (The model can't be tested on a fact it never saw, only on new wordings of it.)
    rng.shuffle(questions)
    n_val = max(2, round(len(questions) * VAL_FRACTION))
    val += [make_example(q, fact["answer"]) for q in questions[:n_val]]
    train += [make_example(q, fact["answer"]) for q in questions[n_val:]]

rng.shuffle(train)  # mix the facts up so training doesn't see one topic at a time

OUT_DIR.mkdir(exist_ok=True)
for name, examples in [("train", train), ("val", val)]:
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Facts (topics):    {len(FACTS)}")
print(f"Training examples: {len(train)}   -> data/train.jsonl")
print(f"Validation examples: {len(val)}   -> data/val.jsonl")

# Peek at how long the examples are in tokens; this matters for memory in Step 4.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lengths = []
for example in train + val:
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    lengths.append(len(tokenizer(text)["input_ids"]))
print(f"\nTokens per example: min {min(lengths)}, average {sum(lengths) / len(lengths):.0f}, max {max(lengths)}")

print("\n=== 3 random training examples ===")
for example in rng.sample(train, 3):
    print(f"  Q: {example['messages'][1]['content']}")
    print(f"  A: {example['messages'][2]['content']}\n")

print("=== 3 random validation examples (never trained on) ===")
for example in rng.sample(val, 3):
    print(f"  Q: {example['messages'][1]['content']}")
    print(f"  A: {example['messages'][2]['content']}\n")
