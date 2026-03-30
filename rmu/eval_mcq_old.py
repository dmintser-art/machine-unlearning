"""
Evaluate a causal LM on MCQ (WMDP/MMLU/MedMCQA) with chat-template prompts
and next-token logits over A/B/C/D at the answer boundary.
"""
import json
import os
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset


LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def _load_wmdp(config="all", split="test"):
    configs = ["wmdp-bio", "wmdp-cyber", "wmdp-chem"] if config == "all" else [config]
    questions, choices, answers = [], [], []
    for cfg in configs:
        ds = load_dataset("cais/wmdp", cfg, split=split)
        for ex in ds:
            questions.append(ex["question"])
            ch = ex["choices"]
            ch_list = list(ch)[:4] if hasattr(ch, "__iter__") and not isinstance(ch, str) else (ch if isinstance(ch, list) else [ch])[:4]
            choices.append(ch_list)
            ans = ex["answer"]
            answers.append(int(ans) if isinstance(ans, (int, np.integer)) else LETTER_TO_IDX.get(str(ans).upper(), 0))
    return questions, choices, answers


def _load_mmlu(config="all", split="test"):
    configs = [config]
    if isinstance(config, str) and config.strip() != "all" and "," in config:
        configs = [c.strip() for c in config.split(",") if c.strip()]

    questions, choices, answers = [], [], []
    for cfg in configs:
        ds = load_dataset("cais/mmlu", cfg, split=split)
        for ex in ds:
            questions.append(ex["question"])
            choices.append(ex["choices"])
            ans = ex["answer"]
            if isinstance(ans, str):
                answers.append(LETTER_TO_IDX.get(ans.upper(), 0))
            else:
                answers.append(int(ans))
    return questions, choices, answers


def _load_medmcqa(split="validation", max_examples=None):
    """MedMCQA: question, opa, opb, opc, opd, cop (0-3). Single-choice only."""
    ds = load_dataset("openlifescienceai/medmcqa", split=split)
    questions, choices, answers = [], [], []
    for i, ex in enumerate(ds):
        if max_examples and i >= max_examples:
            break
        if ex.get("choice_type") == "multi":
            continue
        questions.append(ex["question"])
        choices.append([ex["opa"], ex["opb"], ex["opc"], ex["opd"]])
        cop = ex["cop"]
        if cop in (0, 1, 2, 3):
            answers.append(int(cop))
        elif cop in (1, 2, 3, 4):
            answers.append(int(cop) - 1)
        else:
            answers.append(0)
    return questions, choices, answers


def _load_mcq_json(path):
    """Load MCQ JSON in either dict-of-lists or list-of-objects format."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    questions, choices, answers = [], [], []
    if isinstance(payload, dict):
        q_list = payload.get("questions", [])
        c_list = payload.get("choices", [])
        a_list = payload.get("answers", [])
        n = min(len(q_list), len(c_list), len(a_list))
        for i in range(n):
            q = str(q_list[i])
            ch = c_list[i]
            ans = a_list[i]
            if not isinstance(ch, (list, tuple)):
                continue
            ch4 = list(ch)[:4]
            while len(ch4) < 4:
                ch4.append("")
            if isinstance(ans, str):
                ans_idx = LETTER_TO_IDX.get(ans.upper(), 0)
            else:
                ans_idx = int(ans)
            questions.append(q)
            choices.append(ch4)
            answers.append(ans_idx)
    elif isinstance(payload, list):
        for ex in payload:
            q = str(ex.get("question", ex.get("q", "")))
            ch = ex.get("choices", ex.get("options", []))
            ans = ex.get("answer", ex.get("label", 0))
            if not q or not isinstance(ch, (list, tuple)):
                continue
            ch4 = list(ch)[:4]
            while len(ch4) < 4:
                ch4.append("")
            if isinstance(ans, str):
                ans_idx = LETTER_TO_IDX.get(ans.upper(), 0)
            else:
                ans_idx = int(ans)
            questions.append(q)
            choices.append(ch4)
            answers.append(ans_idx)
    else:
        raise ValueError(f"Unsupported MCQ JSON format in {path!r}: {type(payload).__name__}")

    if not questions:
        raise ValueError(f"No valid MCQ records found in {path!r}.")
    return questions, choices, answers


def _format_mcq_user_prompt(question, choices_list, answer_suffix="Answer:"):
    # Use the prompt style from new_mcq_eval_algorithm.py.
    ch = list(choices_list[:4]) if isinstance(choices_list, (list, tuple)) else [choices_list]
    while len(ch) < 4:
        ch.append("")
    return f"""
The following is a multiple choice question (with answer).

{question}
A. {ch[0]}
B. {ch[1]}
C. {ch[2]}
D. {ch[3]}
{answer_suffix}
""".lstrip("\n")


def _get_choice_token_ids(tokenizer, device):
    choice_ids = []
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not tokenize choice letter {letter!r}")
        choice_ids.append(ids[-1])
    return torch.tensor(choice_ids, dtype=torch.long, device=device)


def _batch_chat_prompts(tokenizer, prompts, device, max_length):
    render_chat = hasattr(tokenizer, "apply_chat_template")
    text_prompts = []
    if render_chat:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except TypeError:
                # Older tokenizer APIs may not expose `tokenize=...`.
                rendered = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
            if isinstance(rendered, str):
                text_prompts.append(rendered)
            elif isinstance(rendered, (list, tuple)):
                # Convert token-id output from legacy APIs back to text for consistent batching.
                text_prompts.append(
                    tokenizer.decode(rendered, skip_special_tokens=False)
                )
            else:
                raise TypeError(
                    f"Unsupported chat template output type: {type(rendered).__name__}"
                )
    else:
        text_prompts = prompts

    tokenized = tokenizer(
        text_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {k: v.to(device) for k, v in tokenized.items()}


def get_mcq_accuracy(model, tokenizer, questions, choices, answers, batch_size=8, max_length=512, device=None):
    """
    Score each MCQ by logits at the next token after `Answer:`,
    choosing among tokens for A/B/C/D.
    """
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    choice_ids = _get_choice_token_ids(tokenizer, device=device)

    correct = 0
    total = len(questions)
    try:
        for i in tqdm(range(0, total, batch_size), desc="MCQ eval", leave=False):
            batch_q = questions[i : i + batch_size]
            batch_c = choices[i : i + batch_size]
            batch_a = answers[i : i + batch_size]
            prompts = [_format_mcq_user_prompt(q, c) for q, c in zip(batch_q, batch_c)]
            inputs = _batch_chat_prompts(
                tokenizer=tokenizer,
                prompts=prompts,
                device=device,
                max_length=max_length,
            )
            with torch.no_grad():
                logits = model(**inputs, use_cache=False).logits

            # Prompts are left-padded, so the last position is the answer boundary.
            last_logits = logits[:, -1, :]
            choice_logits = last_logits.index_select(dim=-1, index=choice_ids)
            preds = choice_logits.argmax(dim=1).cpu().numpy()
            for j, gold in enumerate(batch_a):
                if preds[j] == gold:
                    correct += 1
    finally:
        if was_training:
            model.train()

    return correct / total if total else 0.0


def eval_wmdp(model, tokenizer, config="all", batch_size=8, max_length=512):
    """Evaluate forget performance on WMDP MCQ."""
    questions, choices, answers = _load_wmdp(config=config)
    return get_mcq_accuracy(
        model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length
    )


def eval_mmlu(model, tokenizer, config="all", batch_size=8, max_length=512):
    """Evaluate retain performance on MMLU MCQ."""
    questions, choices, answers = _load_mmlu(config=config, split="test")
    return get_mcq_accuracy(
        model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length
    )


def eval_medmcqa(model, tokenizer, split="validation", batch_size=8, max_length=512, max_examples=2000):
    """Evaluate retain performance on MedMCQA MCQ (optionally subsampled)."""
    questions, choices, answers = _load_medmcqa(split=split, max_examples=max_examples)
    if not questions:
        return 0.0
    return get_mcq_accuracy(
        model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length
    )


def eval_mcq_json(model, tokenizer, path, batch_size=8, max_length=512):
    """Evaluate MCQ from a local JSON file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MCQ JSON not found: {path}")
    questions, choices, answers = _load_mcq_json(path)
    return get_mcq_accuracy(
        model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length
    )
