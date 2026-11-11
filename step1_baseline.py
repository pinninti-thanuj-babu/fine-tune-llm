import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Pick the fastest available device: Apple GPU ("mps") on a Mac, else CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# The tokenizer turns text <-> numbers. The model does the actual prediction.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)

# Count the parameters, so we know what "size" means.
n_params = sum(p.numel() for p in model.parameters())
print(f"Model has {n_params / 1e6:.0f} million parameters")

# Instruct models expect a chat format: a list of role/content messages.
messages = [{"role": "user", "content": "Explain what a database index is."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate: the model predicts one token at a time, up to max_new_tokens.
output = model.generate(**inputs, max_new_tokens=120, do_sample=False)

# Slice off the prompt tokens so we only print what the model wrote.
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\n--- Baseline answer ---")
print(answer)
