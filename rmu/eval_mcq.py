"""
Evaluate a causal LM on MCQ (WMDP/MMLU/MedMCQA) with chat-template prompts
and next-token logits over A/B/C/D at the answer boundary.
"""
import json
import os
import re
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import GenerationConfig

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
EXCEPT_RE = re.compile(r"\bexcept\b", flags=re.IGNORECASE)


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

def _normalize_answer(s):
    """Normalize for matching: strip, lower, take first word if multiple."""
    s = s.strip().lower()
    parts = s.split()
    return parts[0] if parts else s

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

def load_mmlu_bio_single_word(split="test", exclude_except_questions=True):
    """
    Load MMLU college_biology + high_school_biology, keep only questions whose
    correct answer (the choice text) is a single word.
    Optionally exclude questions containing the token "EXCEPT" (case-insensitive),
    since free-form without choices is ill-posed for those prompts.
    Returns (questions, choices, answers, correct_texts).
    correct_texts[i] is the string of the correct choice for matching in free-text eval.
    """
    questions, choices, answers = [], [], []
    correct_texts = []
    for config in ("college_biology", "high_school_biology"):
        q, c, a = _load_mmlu(config=config, split=split)
        for i in range(len(q)):
            if exclude_except_questions and EXCEPT_RE.search(str(q[i])):
                continue
            ans_idx = a[i]
            ch = c[i]
            if not isinstance(ch, (list, tuple)) or len(ch) <= ans_idx:
                continue
            correct_str = ch[ans_idx].strip()
            words = correct_str.split()
            if len(words) != 1:
                continue
            questions.append(q[i])
            choices.append(ch)
            answers.append(ans_idx)
            correct_texts.append(correct_str)
    return questions, choices, answers, correct_texts

def eval_mmlu_bio_single_word_free_text(
    model,
    tokenizer,
    max_new_tokens=10,
    max_length=512,
    return_details=False,
    exclude_except_questions=True,
):
    """Free-text (no choices) accuracy on MMLU biology single-word subset.
    If return_details=True, returns (accuracy, list of {question, correct_answer, generated_answer})."""
    questions, _, _, correct_texts = load_mmlu_bio_single_word(
        split="test",
        exclude_except_questions=exclude_except_questions,
    )
    if not questions:
        return (0.0, []) if return_details else 0.0
    return get_free_text_single_word_accuracy(
        model, tokenizer, questions, correct_texts,
        max_new_tokens=max_new_tokens, max_length=max_length,
        return_details=return_details,
    )
    
def get_free_text_single_word_accuracy(
    model, tokenizer, questions, correct_answer_texts, max_new_tokens=10, max_length=512, device=None,
    return_details=False,
):
    """
    Prompt with question only (no choices), generate short answer, compare first word to correct.
    correct_answer_texts[i] is the expected single-word answer string.
    If return_details=True, returns (accuracy, list of {"question", "correct_answer", "generated_answer"}).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    prompt_no_choices = "Answer with a single word.\n\nQuestion: {question}\nAnswer:"
    correct = 0
    total = len(questions)
    details_list = [] if return_details else None
    for i in tqdm(range(total), desc="Free-text eval", leave=False):
        q = questions[i]
        gold_str = correct_answer_texts[i]
        gold = _normalize_answer(gold_str)
        prompt = prompt_no_choices.format(question=q)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            out = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            if hasattr(out, "input_ids"):
                input_ids = out["input_ids"].to(device)
            else:
                input_ids = out.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = tok["input_ids"].to(device)
        # Explicit attention mask so pad=eos does not confuse the model
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        # Greedy decoding; override model defaults to avoid sampling-related warnings
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
        new_tokens = gen[0, input_ids.shape[1] :]
        pred_str = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred_word = _normalize_answer(pred_str)
        if pred_word == gold:
            correct += 1
        if return_details:
            details_list.append({
                "question": q,
                "correct_answer": gold_str,
                "generated_answer": pred_str.strip(),
            })
    acc = correct / total if total else 0.0
    if return_details:
        return acc, details_list
    return acc

def eval_mmlu_bio_single_word_mcq(
    model,
    tokenizer,
    batch_size=8,
    max_length=512,
    exclude_except_questions=True,
):
    """MCQ accuracy on MMLU biology subset where correct answer is a single word."""
    questions, choices, answers, _ = load_mmlu_bio_single_word(
        split="test",
        exclude_except_questions=exclude_except_questions,
    )
    if not questions:
        return 0.0
    return get_mcq_accuracy(
        model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length
    )
    
    
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
    preds = get_mcq_predictions(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        choices=choices,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    correct = 0
    total = len(answers)
    for pred, gold in zip(preds, answers):
        if int(pred) == int(gold):
            correct += 1
    return correct / total if total else 0.0


def get_mcq_predictions(model, tokenizer, questions, choices, batch_size=8, max_length=512, device=None):
    """Return predicted option indices (0..3) for each MCQ question."""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    choice_ids = _get_choice_token_ids(tokenizer, device=device)

    all_preds = []
    total = len(questions)
    try:
        for i in tqdm(range(0, total, batch_size), desc="MCQ pred", leave=False):
            batch_q = questions[i : i + batch_size]
            batch_c = choices[i : i + batch_size]
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
            preds = choice_logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(int(p) for p in preds)
    finally:
        if was_training:
            model.train()
    return all_preds


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
