#!/usr/bin/env python3
"""
Report MCQ and free-text accuracy on MMLU biology single-word subset
(high_school_biology + college_biology, questions whose correct answer is one word).

Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct.

Usage:
  python report_qa_mmlu_singleword.py --out_dir outputs/qa_mmlu_singleword_report
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from rmu.eval_mcq import (
    eval_mmlu_bio_single_word_mcq,
    eval_mmlu_bio_single_word_free_text,
    load_mmlu_bio_single_word,
)
from rmu.utils import load_model  # noqa: E402

SINGLEWORD_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def main():
    parser = argparse.ArgumentParser(
        description="MCQ + free-text on MMLU biology single-word subset (Qwen 2.5 7B, Llama 3.1 8B)"
    )
    parser.add_argument("--models", nargs="+", default=SINGLEWORD_MODELS,
                        help="Model IDs (default: Qwen2.5-7B, Llama-3.1-8B)")
    parser.add_argument("--out_dir", type=str, default="outputs/qa_mmlu_singleword_report")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=10,
                        help="Max new tokens for free-text generation")
    parser.add_argument(
        "--include_except_questions",
        action="store_true",
        help="If set, include questions containing 'EXCEPT'. Default excludes them.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exclude_except = not args.include_except_questions
    q, c, a, texts = load_mmlu_bio_single_word(
        split="test",
        exclude_except_questions=exclude_except,
    )
    n_subset = len(q)
    report = {
        "subset": "mmlu_bio_single_word",
        "description": "college_biology + high_school_biology, correct answer is one word",
        "n_examples": n_subset,
        "exclude_except_questions": exclude_except,
        "models": {},
    }
    freeform_details = {}

    for model_id in args.models:
        print("\n" + "=" * 60)
        print(f"Model: {model_id}")
        print("=" * 60)
        try:
            model, tokenizer = load_model(model_id)
            model.eval()
            print("  MMLU-bio single-word MCQ ...")
            acc_mcq = eval_mmlu_bio_single_word_mcq(
                model, tokenizer,
                batch_size=args.batch_size, max_length=args.max_length,
                exclude_except_questions=exclude_except,
            )
            print("  MMLU-bio single-word free-text (no choices) ...")
            acc_free, details_list = eval_mmlu_bio_single_word_free_text(
                model, tokenizer,
                max_new_tokens=args.max_new_tokens, max_length=args.max_length,
                return_details=True,
                exclude_except_questions=exclude_except,
            )
            report["models"][model_id] = {
                "mmlu_bio_singleword_mcq": round(acc_mcq, 4),
                "mmlu_bio_singleword_freeform": round(acc_free, 4),
            }
            freeform_details[model_id] = details_list
            print(f"  Single-word MCQ:     {acc_mcq:.4f}")
            print(f"  Single-word free-text: {acc_free:.4f}")
        except Exception as e:
            report["models"][model_id] = {"error": str(e)}
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    out_path = os.path.join(args.out_dir, "qa_mmlu_singleword_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out_path}")

    details_path = os.path.join(args.out_dir, "freeform_question_answer_details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(freeform_details, f, indent=2, ensure_ascii=False)
    print(f"Free-text details (question, correct_answer, generated_answer) saved to {details_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
