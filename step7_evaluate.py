import json

import torch
from peft import PeftModel

from loan_facts import SYSTEM_PROMPT
from step5_lora import device, load_base_model

ADAPTER_DIR = "adapter"

# (category, question, text the answer must contain, or None = just read it)
# Written after training. None of these questions is in train.jsonl or val.jsonl.
CASES = [
    ("New wording of a taught fact", "hey what's the rate if I take a personal loan from you guys?", "10.5%"),
    
    ("New wording of a taught fact", "how many lakhs is the max I can get as a personal loan", "25 lakh"),
    ("New wording of a taught fact", "what's the lowest cibil score you accept for a car loan", "650"),
    ("New wording of a taught fact", "will you charge me if I pay off my personal loan early", "4%"),
    ("New wording of a taught fact", "how do i get in touch with you", "1800-555-0142"),
    ("New wording of a taught fact", "is it okay if my spouse also applies with me on the home loan", "co-applicant"),
    ("Behaviour: never promise approval", "I have all my documents ready so my loan is 100% approved right?", "can't guarantee"),
    ("Behaviour: hand off account questions", "tell me how much I still owe on my loan", "verified agent"),
    ("Behaviour: empathy for hardship", "my shop shut down and I can't pay this month's EMI", "sorry"),
    ("Behaviour: out of scope", "Do you offer gold loans?", "don't have information"),
    ("Behaviour: out of scope", "Who founded Zenora Finance?", "don't have information"),
    ("Trap: fact NOT in our data", "What is the processing fee for a car loan?", "don't have information"),
    ("Trap: fact NOT in our data", "What is the interest rate on a loan against property?", "don't have information"),
    ("Off-topic (just read it)", "What is the capital of India?", None),
    ("Off-topic (just read it)", "Write a Python function that reverses a string.", None),
]


def generate_answers(model, tokenizer, questions, batch_size=16, max_new_tokens=100):
    """Answer each question with greedy decoding, several at a time."""
    tokenizer.padding_side = "left"  # generation continues from the END of the prompt, so pad on the left
    answers = []
    for start in range(0, len(questions), batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True,
            )
            for question in questions[start:start + batch_size]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                # Qwen's default settings add a repetition penalty and sampling knobs;
                # switch them off so we test exactly what training taught.
                repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = inputs["input_ids"].shape[1]
        answers += [tokenizer.decode(row[prompt_length:], skip_special_tokens=True).strip() for row in output]
    return answers


def answer_both(model, tokenizer, questions):
    """(original model's answers, fine-tuned model's answers) for the same questions."""
    with model.disable_adapter():  # adapter OFF = the original model
        before = generate_answers(model, tokenizer, questions)
    after = generate_answers(model, tokenizer, questions)  # adapter ON
    return before, after


def shorten(text, limit=230):
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " ..."


if __name__ == "__main__":
    tokenizer, base_model = load_base_model()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)  # put the saved adapter on top of the base model
    model.eval()

    # ---------------------------------------------------------------
    # PART A: hand-written cases, side by side
    # ---------------------------------------------------------------
    questions = [case[1] for case in CASES]
    before, after = answer_both(model, tokenizer, questions)

    print("=== PART A: hand-written cases (ORIGINAL vs FINE-TUNED) ===")
    passed = {"before": 0, "after": 0}
    checked = 0
    for (category, question, must_contain), old, new in zip(CASES, before, after):
        print(f"\n[{category}]\nQ: {question}")
        for label, answer, key in [("ORIGINAL  ", old, "before"), ("FINE-TUNED", new, "after")]:
            mark = ""
            if must_contain is not None:
                ok = must_contain.lower() in answer.lower()
                passed[key] += ok
                mark = "  ✓" if ok else "  ✗"
            print(f"  {label}: {shorten(answer)}{mark}")
        if must_contain is not None:
            checked += 1
    print(f"\nCases with a check: {checked}")
    print(f"  ORIGINAL model passed   {passed['before']}/{checked}")
    print(f"  FINE-TUNED model passed {passed['after']}/{checked}")

    # ---------------------------------------------------------------
    # PART B: the whole validation set, exact match
    # ---------------------------------------------------------------
    with open("data/val.jsonl") as f:
        val = [json.loads(line)["messages"] for line in f]
    val_questions = [messages[1]["content"] for messages in val]
    expected = [messages[2]["content"] for messages in val]

    before, after = answer_both(model, tokenizer, val_questions)
    exact_before = sum(a.strip() == e.strip() for a, e in zip(before, expected))
    exact_after = sum(a.strip() == e.strip() for a, e in zip(after, expected))

    print("\n=== PART B: the 73 validation questions (new wordings of taught facts) ===")
    print(f"Exact match, ORIGINAL model:   {exact_before}/{len(val)}")
    print(f"Exact match, FINE-TUNED model: {exact_after}/{len(val)}")

    mistakes = [(q, e, a) for q, e, a in zip(val_questions, expected, after) if a.strip() != e.strip()]
    print(f"\nFine-tuned mistakes ({len(mistakes)}):")
    for question, exp, got in mistakes:
        print(f"\n  Q:        {question}\n  EXPECTED: {shorten(exp)}\n  GOT:      {shorten(got)}")
