import argparse
import csv
import time

import torch
from peft import get_peft_model
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from step4_prepare_data import BATCH_SIZE, MAX_LENGTH, ChatDataset, make_collate_fn
from step5_lora import LORA_CONFIG, average_loss, device, load_base_model

MAX_GRAD_NORM = 1.0   # gradient clipping: no single update may be bigger than this
WARMUP_FRACTION = 0.1  # ramp the learning rate up over the first 10% of steps
SEED = 0
ADAPTER_DIR = "adapter"
LOG_PATH = "training_log.csv"


def main(epochs, learning_rate):
    torch.manual_seed(SEED)
    tokenizer, base_model = load_base_model()
    model = get_peft_model(base_model, LORA_CONFIG)

    # Pad every batch to the same length. With varying lengths, the Mac GPU's memory pool
    # kept growing until it ran out of memory (around step 90).
    collate = make_collate_fn(tokenizer.pad_token_id, pad_to=MAX_LENGTH)
    train_loader = DataLoader(ChatDataset("data/train.jsonl", tokenizer), batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED), collate_fn=collate)
    val_loader = DataLoader(ChatDataset("data/val.jsonl", tokenizer), batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate)

    # Only the LoRA weights have requires_grad=True, so only they go to the optimizer.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_FRACTION * total_steps), num_training_steps=total_steps
    )
    print(f"{len(train_loader)} steps per epoch x {epochs} epochs = {total_steps} steps, learning rate {learning_rate}")

    log = []  # one row per step: (step, epoch, train_loss, learning_rate, val_loss)
    val_loss = average_loss(model, val_loader)
    print(f"Before training: validation loss {val_loss:.3f}\n")
    log.append((0, 0, "", "", val_loss))

    step = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()  # turns LoRA dropout ON (average_loss switched it off)
        epoch_losses = []
        for batch in train_loader:
            batch = {name: tensor.to(device) for name, tensor in batch.items()}

            loss = model(**batch).loss                                       # 1. forward pass
            loss.backward()                                                  # 2. backward pass: gradients
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            optimizer.step()                                                 # 3. update the LoRA weights
            scheduler.step()                                                 #    and move the learning rate
            optimizer.zero_grad()                                            # 4. reset gradients

            step += 1
            epoch_losses.append(loss.item())
            current_lr = scheduler.get_last_lr()[0]
            log.append((step, epoch, loss.item(), current_lr, ""))
            if step % 5 == 0:
                print(f"  step {step:3}/{total_steps}  train loss {loss.item():.3f}  lr {current_lr:.2e}")

        val_loss = average_loss(model, val_loader)
        log.append((step, epoch, "", "", val_loss))
        mean_train = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch}/{epochs} done | train loss {mean_train:.3f} | validation loss {val_loss:.3f} "
              f"| {time.time() - started:.0f}s elapsed\n")

    model.save_pretrained(ADAPTER_DIR)  # saves ONLY the small LoRA adapter, not the 494M base model
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "epoch", "train_loss", "learning_rate", "val_loss"])
        writer.writerows(log)
    print(f"Saved the adapter to {ADAPTER_DIR}/ and the loss log to {LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    main(args.epochs, args.lr)
