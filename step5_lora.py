import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from step4_prepare_data import BATCH_SIZE, IGNORE_INDEX, MODEL_NAME, ChatDataset, make_collate_fn

device = "mps" if torch.backends.mps.is_available() else "cpu"

LORA_CONFIG = LoraConfig(
    r=16,                # rank: how thick the A and B matrices are
    lora_alpha=32,       # the LoRA branch is scaled by alpha / r
    lora_dropout=0.05,   # randomly drop adapter inputs while training
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    return tokenizer, model


def average_loss(model, loader):
    """Average loss per graded token over a whole dataset (the number training tries to lower)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            loss = model(**batch).loss  # already averaged over this batch's graded tokens
            n_tokens = (batch["labels"][:, 1:] != IGNORE_INDEX).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / total_tokens


if __name__ == "__main__":
    tokenizer, model = load_base_model()

    # ---- BEFORE LoRA: one ordinary layer ----
    layer_before = model.model.layers[0].self_attn.q_proj
    print("=== BEFORE: layer 0's q_proj is an ordinary layer ===")
    print(layer_before)
    print(f"Its weight matrix has {layer_before.weight.numel():,} numbers")

    # ---- Attach LoRA ----
    peft_model = get_peft_model(model, LORA_CONFIG)
    peft_model.eval()
    print("\n=== Parameter counts ===")
    peft_model.print_trainable_parameters()

    # ---- AFTER LoRA: the same layer, with an adapter wrapped around it ----
    layer_after = peft_model.base_model.model.model.layers[0].self_attn.q_proj
    print("\n=== AFTER: the same layer now has A and B next to the original ===")
    print(layer_after)

    print("\n=== Which parameters are trainable in layer 0's q_proj? ===")
    for name, param in layer_after.named_parameters():
        print(f"  {name:28} shape {tuple(param.shape)!s:12} trainable={param.requires_grad}")

    # ---- Proof: B starts at zero, so the model is unchanged ----
    lora_a = layer_after.lora_A["default"].weight
    lora_b = layer_after.lora_B["default"].weight
    print("\n=== Starting values ===")
    print(f"A (random):  largest absolute value = {lora_a.abs().max().item():.4f}")
    print(f"B (zeros):   largest absolute value = {lora_b.abs().max().item():.4f}")

    val_ds = ChatDataset("data/val.jsonl", tokenizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=make_collate_fn(tokenizer.pad_token_id))
    batch = {name: tensor.to(device) for name, tensor in next(iter(val_loader)).items()}
    with torch.no_grad():
        with peft_model.disable_adapter():  # original model, adapter switched off
            logits_without = peft_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        logits_with = peft_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    difference = (logits_with.float() - logits_without.float()).abs().max().item()
    print(f"\nLargest difference in output scores, adapter ON vs OFF: {difference}")

    # ---- The number training will try to push down ----
    loss = average_loss(peft_model, val_loader)
    print(f"\nValidation loss BEFORE any training: {loss:.3f}")
