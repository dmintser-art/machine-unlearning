#!/usr/bin/env python3
"""
Filter HP dataset to keep only relatively unique questions.

Modes:
  correct_only (default): each *correct* answer text appears at most max_per_answer times.
  all_options: each of the *four* option texts appears at most max_per_answer times.
  correct_once_distractors_5: correct answer appears at most 1 time (unique); distractors
    (incorrect options) can appear up to max_distractor times (default 5). Compromise for
    more questions while avoiding correct-answer repetition. Use with 5-fold CV split by
    correct answer to avoid leakage.

Usage:
  python filter_hp_unique.py --data_path data/hp/hp_mcq.json --max_per_answer 5 -o data/hp/hp_mcq_unique5.json
  python filter_hp_unique.py --mode correct_once_distractors_5 --max_distractor 5 -o data/hp/hp_mcq_compromise.json
"""
import argparse
import json
import os
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Filter HP to unique questions by answer/option frequency")
    parser.add_argument("--data_path", type=str, default="data/hp/hp_mcq.json", help="Input HP JSON (questions, answers, choices)")
    parser.add_argument("--max_per_answer", type=int, default=5, help="Max times an answer (or each option) may appear")
    parser.add_argument("--mode", type=str, default="correct_only",
                        choices=["correct_only", "all_options", "correct_once_distractors_5"],
                        help="correct_only / all_options / correct_once_distractors_5 (correct once, distractors up to max_distractor).")
    parser.add_argument("--max_distractor", type=int, default=5, help="For correct_once_distractors_5: max times each distractor may appear.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON path (default: input path with _unique{N} suffix)")
    args = parser.parse_args()

    if not os.path.isfile(args.data_path):
        parser.error(f"Input file not found: {args.data_path}. Place your HP JSON (questions, answers, choices) there or pass --data_path /path/to/hp_mcq.json")

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    answers = data.get("answers", [])  # indices 0-3
    choices = data.get("choices", [])

    n = min(len(questions), len(answers), len(choices))
    questions = questions[:n]
    answers = answers[:n]
    choices = choices[:n]

    if args.mode == "correct_only":
        # Count how many times each *correct* answer text appears
        answer_texts = []
        for i in range(n):
            a = answers[i] if answers[i] in (0, 1, 2, 3) else 0
            ch = choices[i] if isinstance(choices[i], list) and len(choices[i]) == 4 else ["", "", "", ""]
            answer_texts.append(ch[a].strip() if a < len(ch) else "")
        count_by_answer = defaultdict(int)
        for t in answer_texts:
            count_by_answer[t] += 1
        keep_mask = [count_by_answer[answer_texts[i]] <= args.max_per_answer for i in range(n)]
    elif args.mode == "all_options":
        # each of the 4 option texts <= max_per_answer
        all_option_texts = []
        for i in range(n):
            ch = choices[i] if isinstance(choices[i], list) and len(choices[i]) == 4 else ["", "", "", ""]
            for j in range(4):
                all_option_texts.append((i, ch[j].strip() if j < len(ch) else ""))
        count_by_option = defaultdict(int)
        for _idx, t in all_option_texts:
            count_by_option[t] += 1
        keep_mask = []
        for i in range(n):
            ch = choices[i] if isinstance(choices[i], list) and len(choices[i]) == 4 else ["", "", "", ""]
            option_texts = [ch[j].strip() if j < len(ch) else "" for j in range(4)]
            if all(count_by_option[t] <= args.max_per_answer for t in option_texts):
                keep_mask.append(True)
            else:
                keep_mask.append(False)
    else:
        # correct_once_distractors_5: correct answer <= 1, each distractor <= max_distractor
        answer_texts = []
        distractor_counts = defaultdict(int)
        for i in range(n):
            a = answers[i] if answers[i] in (0, 1, 2, 3) else 0
            ch = choices[i] if isinstance(choices[i], list) and len(choices[i]) == 4 else ["", "", "", ""]
            correct_t = ch[a].strip() if a < len(ch) else ""
            answer_texts.append(correct_t)
            for j in range(4):
                if j != a:
                    distractor_counts[ch[j].strip() if j < len(ch) else ""] += 1
        count_correct = defaultdict(int)
        for t in answer_texts:
            count_correct[t] += 1
        keep_mask = []
        for i in range(n):
            a = answers[i] if answers[i] in (0, 1, 2, 3) else 0
            ch = choices[i] if isinstance(choices[i], list) and len(choices[i]) == 4 else ["", "", "", ""]
            correct_t = ch[a].strip() if a < len(ch) else ""
            if count_correct[correct_t] > 1:
                keep_mask.append(False)
                continue
            distractors_ok = all(
                distractor_counts[ch[j].strip() if j < len(ch) else ""] <= args.max_distractor
                for j in range(4) if j != a
            )
            keep_mask.append(distractors_ok)

    indices = [i for i in range(n) if keep_mask[i]]

    out_questions = [questions[i] for i in indices]
    out_answers = [answers[i] for i in indices]
    out_choices = [choices[i] for i in indices]

    out = {"questions": out_questions, "answers": out_answers, "choices": out_choices}

    out_path = args.output
    if not out_path:
        base = args.data_path.rsplit(".", 1)[0]
        if args.mode == "all_options":
            suffix = f"unique_all_{args.max_per_answer}"
        elif args.mode == "correct_once_distractors_5":
            suffix = f"correct1_distractors{args.max_distractor}"
        else:
            suffix = f"unique{args.max_per_answer}"
        out_path = f"{base}_{suffix}.json"

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if args.mode == "all_options":
        mode_desc = f"each of 4 options at most {args.max_per_answer}"
    elif args.mode == "correct_once_distractors_5":
        mode_desc = f"correct once, distractors at most {args.max_distractor}"
    else:
        mode_desc = f"correct answer at most {args.max_per_answer}"
    print(f"Original: {n} questions")
    print(f"After filter ({mode_desc}): {len(indices)} questions")
    print(f"Saved to {out_path}")
    return out_path


if __name__ == "__main__":
    main()
