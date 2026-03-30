#!/usr/bin/env python3
"""
Report Q&A performance: run a causal LM on HP MCQ and print accuracy.
Use this to report direct (next-token) MCQ accuracy for Llama, Gemma, or Qwen.

Usage:
  python run_hp_qa_eval.py --model Qwen/Qwen2.5-7B-Instruct --hp_mcq_path data/hp/hp_mcq.json
  python run_hp_qa_eval.py --model meta-llama/Llama-3.1-8B-Instruct --hp_mcq_path data/hp/hp_mcq_unique5.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmu.eval_mcq import eval_hp_mcq
from rmu.utils import load_model


def main():
    parser = argparse.ArgumentParser(description="HP MCQ Q&A accuracy for a causal LM")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model name")
    parser.add_argument("--hp_mcq_path", type=str, default="data/hp/hp_mcq.json", help="Path to HP MCQ JSON")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    if not os.path.isfile(args.hp_mcq_path):
        print(f"Error: HP MCQ file not found: {args.hp_mcq_path}")
        return 1

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    model.eval()

    print(f"Evaluating Q&A on HP MCQ: {args.hp_mcq_path}")
    acc = eval_hp_mcq(
        model, tokenizer,
        path=args.hp_mcq_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"HP Q&A accuracy: {acc:.4f}")
    return 0


if __name__ == "__main__":
    exit(main())
