import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)

# ---------------------------------------------------------------
# PART 1: words -> tokens -> token IDs
# ---------------------------------------------------------------
text = "What is the capital of India?"
ids = tokenizer(text)["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(ids)

print("=== PART 1: text -> tokens -> IDs ===")
for token, token_id in zip(tokens, ids):
    # "Ġ" is how this tokenizer displays a leading space.
    print(f"  {token!r:12} -> {token_id}")
print(f"Decoded back: {tokenizer.decode(ids)!r}")
print(f"Vocabulary size: {len(tokenizer)} tokens")

# ---------------------------------------------------------------
# PART 2: the chat template (special tokens)
# ---------------------------------------------------------------
messages = [{"role": "user", "content": text}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("\n=== PART 2: chat template ===")
print(prompt)

# ---------------------------------------------------------------
# PART 3: the model's probabilities for the very next token
# ---------------------------------------------------------------
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
print(f"Prompt is {input_ids.shape[1]} tokens long")

with torch.no_grad():
    logits = model(input_ids).logits  # shape: (batch, sequence_length, vocab_size)

# We only care about the LAST position: it predicts the token that comes next.
next_token_logits = logits[0, -1]
probs = torch.softmax(next_token_logits.float(), dim=-1)  # scores -> percentages

print("\n=== PART 3: top 5 predictions for the next token ===")
print(f"Number of scores (one per vocabulary entry): {probs.shape[0]}")
print(f"Probabilities sum to: {probs.sum().item():.4f}")
top = torch.topk(probs, 5)
for p, token_id in zip(top.values, top.indices):
    print(f"  {tokenizer.decode(token_id)!r:12} {p.item() * 100:6.2f}%")

# ---------------------------------------------------------------
# PART 4: the generation loop, written by hand
# ---------------------------------------------------------------
end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

print("\n=== PART 4: generation loop, one round at a time ===")
for round_number in range(1, 41):
    with torch.no_grad():
        logits = model(input_ids).logits
    probs = torch.softmax(logits[0, -1].float(), dim=-1)

    next_id = torch.argmax(probs).item()  # greedy: pick the highest probability
    print(f"Round {round_number:2}: input has {input_ids.shape[1]:3} tokens "
          f"-> picked {tokenizer.decode(next_id)!r:12} ({probs[next_id].item() * 100:5.1f}%)")

    if next_id == end_id:
        print("Model predicted <|im_end|>: it decided it is finished.")
        break

    # Append the picked token to the input, then loop again.
    input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

print("\nFull text so far:")
print(tokenizer.decode(input_ids[0]))
