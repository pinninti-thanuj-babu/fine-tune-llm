import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)

text = "What is the capital of India?"
messages = [{"role": "user", "content": text}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

with torch.no_grad():
    logits = model(input_ids).logits[0, -1].float()  # one score per vocabulary entry
probs = torch.softmax(logits, dim=-1)                # scores -> percentages

# Sort every token from most likely to least likely.
sorted_probs, sorted_ids = torch.sort(probs, descending=True)
sorted_probs = sorted_probs.cpu()
sorted_ids = sorted_ids.cpu()
logits = logits.cpu()

print(f"Question: {text}")
print(f"The model produced {probs.shape[0]} probabilities (one per vocabulary entry)")
print(f"They add up to: {probs.sum().item():.4f}")

print("\n=== Top 20 ===")
for rank in range(20):
    token_id = sorted_ids[rank].item()
    print(f"{rank + 1:3}. {tokenizer.decode(token_id)!r:14} id={token_id:<7} {sorted_probs[rank].item() * 100:8.4f}%")

print("\n=== Bottom 5 (least likely) ===")
for rank in range(len(sorted_ids) - 5, len(sorted_ids)):
    token_id = sorted_ids[rank].item()
    print(f"{rank + 1:6}. {tokenizer.decode(token_id)!r:14} id={token_id:<7} {sorted_probs[rank].item() * 100:.10f}%")

print("\n=== How the probability is spread ===")
print(f"Top 1 token holds:    {sorted_probs[:1].sum().item() * 100:6.2f}% of the probability")
print(f"Top 10 tokens hold:   {sorted_probs[:10].sum().item() * 100:6.2f}%")
print(f"Top 100 tokens hold:  {sorted_probs[:100].sum().item() * 100:6.2f}%")
print(f"Tokens with > 1%:     {(probs > 0.01).sum().item()}")
print(f"Tokens with > 0.01%:  {(probs > 0.0001).sum().item()}")
print(f"Tokens with < 0.0001%: {(probs < 0.000001).sum().item()}")

# Save the full list, most likely first.
raw_tokens = tokenizer.convert_ids_to_tokens(list(range(probs.shape[0])))
with open("all_probabilities.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "token_id", "token", "probability", "logit"])
    for rank in range(len(sorted_ids)):
        token_id = sorted_ids[rank].item()
        token = raw_tokens[token_id] or "<unused>"
        writer.writerow([rank + 1, token_id, token, f"{sorted_probs[rank].item():.10f}", f"{logits[token_id].item():.4f}"])

print("\nSaved every probability to all_probabilities.csv")
