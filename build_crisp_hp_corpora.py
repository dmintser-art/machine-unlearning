#!/usr/bin/env python3
"""
Build CRISP HP retain and forget corpora from HP MCQ JSON.

Each corpus is a JSON array of text strings (one per item). Format per string:
  "Question: {q}\nA. {c0}\nB. {c1}\nC. {c2}\nD. {c3}\nCorrect answer: {correct_text}"

By default: forget = 70% of items (to unlearn), retain = 30% (to retain), disjoint split.
Override with --forget_ratio and --seed. Output paths: data/crisp_hp_forget.json, data/crisp_hp_retain.json.

Usage:
  python build_crisp_hp_corpora.py --hp_mcq_path data/hp/hp_mcq_unique5.json
  python build_crisp_hp_corpora.py --hp_mcq_path data/hp/hp_mcq_unique5.json --forget_ratio 0.8 -o data
"""
import argparse
import json
import os
import random


def load_hp(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    answers = data.get("answers", [])
    choices = data.get("choices", [])
    n = min(len(questions), len(answers), len(choices))
    return questions[:n], choices[:n], answers[:n]


def format_qa(q, ch, a_idx):
    """One text string per MCQ item (for RMU corpus)."""
    ch = list(ch)[:4] if isinstance(ch, (list, tuple)) else ["", "", "", ""]
    while len(ch) < 4:
        ch.append("")
    correct = ch[a_idx] if 0 <= a_idx < len(ch) else ""
    return (
        f"Question: {q}\n"
        f"A. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\nD. {ch[3]}\n"
        f"Correct answer: {correct}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build CRISP HP forget & retain corpora from HP MCQ")
    parser.add_argument("--hp_mcq_path", type=str, default="data/hp/hp_mcq_unique5.json")
    parser.add_argument("--forget_ratio", type=float, default=0.7, help="Fraction of items for forget set (rest = retain)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--out_dir", type=str, default="data")
    args = parser.parse_args()

    if not os.path.isfile(args.hp_mcq_path):
        parser.error(f"HP MCQ file not found: {args.hp_mcq_path}")

    questions, choices, answers = load_hp(args.hp_mcq_path)
    n = len(questions)
    indices = list(range(n))
    random.seed(args.seed)
    random.shuffle(indices)

    n_forget = max(1, int(n * args.forget_ratio))
    forget_idx = set(indices[:n_forget])
    retain_idx = set(indices[n_forget:])

    forget_texts = []
    for i in forget_idx:
        a = answers[i] if answers[i] in (0, 1, 2, 3) else 0
        forget_texts.append(format_qa(questions[i], choices[i], a))

    retain_texts = []
    for i in retain_idx:
        a = answers[i] if answers[i] in (0, 1, 2, 3) else 0
        retain_texts.append(format_qa(questions[i], choices[i], a))

    os.makedirs(args.out_dir, exist_ok=True)
    forget_path = os.path.join(args.out_dir, "crisp_hp_forget.json")
    retain_path = os.path.join(args.out_dir, "crisp_hp_retain.json")

    with open(forget_path, "w", encoding="utf-8") as f:
        json.dump(forget_texts, f, indent=2, ensure_ascii=False)
    with open(retain_path, "w", encoding="utf-8") as f:
        json.dump(retain_texts, f, indent=2, ensure_ascii=False)

    print(f"Built CRISP HP corpora from {args.hp_mcq_path} (n={n})")
    print(f"  Forget: {len(forget_texts)} texts -> {forget_path}")
    print(f"  Retain: {len(retain_texts)} texts -> {retain_path}")
    print("Use with: --forget_corpora crisp-hp-forget:data/crisp_hp_forget.json --retain_corpora crisp-hp-retain:data/crisp_hp_retain.json")


if __name__ == "__main__":
    main()
