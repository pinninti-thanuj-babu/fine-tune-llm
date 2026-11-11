from peft import PeftModel

from step5_lora import load_base_model
from step7_evaluate import ADAPTER_DIR, generate_answers

HELP = "Type a question and press Enter.  /compare = also show the original model   /quit = exit"

tokenizer, base_model = load_base_model()
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)  # base model + our trained adapter
model.eval()

print("\nZenora Finance customer care (fine-tuned model)")
print(HELP)

compare = False
while True:
    try:
        question = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if not question:
        continue
    if question == "/quit":
        break
    if question == "/compare":
        compare = not compare
        print(f"Compare mode {'ON' if compare else 'OFF'}")
        continue

    # Each question is answered on its own: we trained on single questions, not conversations.
    if compare:
        with model.disable_adapter():  # adapter OFF = the original model
            print(f"Original:   {generate_answers(model, tokenizer, [question])[0]}")
    print(f"Fine-tuned: {generate_answers(model, tokenizer, [question])[0]}")
