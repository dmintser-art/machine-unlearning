#!/usr/bin/env python3
"""
Report Q&A performance on WMDP MCQ: run a causal LM and print accuracy.
Use for Llama, Gemma, or Qwen on WMDP (forget-set style eval).

Usage:
  python run_wmdp_qa_eval.py --model meta-llama/Llama-3.1-8B-Instruct --config wmdp-bio
  python run_wmdp_qa_eval.py --model Qwen/Qwen2.5-7B-Instruct --config all
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rmu.eval_mcq import eval_wmdp
from rmu.utils import load_model


def main():
    parser = argparse.ArgumentParser(description="WMDP MCQ Q&A accuracy for a causal LM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model name")
    parser.add_argument("--config", type=str, default="wmdp-bio", help="WMDP config: wmdp-bio, wmdp-cyber, wmdp-chem, or all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    model.eval()

    print(f"Evaluating Q&A on WMDP MCQ (config={args.config})")
    acc = eval_wmdp(
        model, tokenizer,
        config=args.config,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"WMDP Q&A accuracy: {acc:.4f}")
    return 0


if __name__ == "__main__":
    exit(main())
