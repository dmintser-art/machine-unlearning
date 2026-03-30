#!/usr/bin/env python3
"""
Run the same 5-fold performance report (HP + WMDP, embeddings layer 0 + Q&A) for:
  - Llama 3.1 8B (default)
  - Qwen 2.5 7B — Qwen/Qwen2.5-7B-Instruct (text-only)
  - Qwen 3.5 9B — Qwen/Qwen3.5-9B
  - Gemma 3 — google/gemma-3-4b-it
  - Gemma 2 9B — google/gemma-2-9b-it
  - SBERT — sentence-transformers/all-mpnet-base-v2 (HP + WMDP-bio, Cosine only; no Q&A)

Then merge reports into a single JSON for comparison.

Note: Qwen 3.5 requires transformers>=5.2.0 (Python>=3.10). Fallback: --qwen_model Qwen/Qwen2.5-7B-Instruct

Usage:
  python run_all_5fold_reports.py --hp_mcq_path data/hp/hp_mcq_compromise.json --wmdp_config wmdp-bio
  python run_all_5fold_reports.py --skip_llama --skip_qwen35 --skip_gemma3  # Qwen 2.5 only
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Representative models from each collection
MODELS = {
    "llama": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "embedding_backend": "llama",
        "out_dir": "outputs/llama_performance_5fold",
    },
    "qwen25": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "embedding_backend": "qwen",
        "out_dir": "outputs/qwen25_performance_5fold",
    },
    "qwen35": {
        "model": "Qwen/Qwen3.5-9B",
        "embedding_backend": "qwen",
        "out_dir": "outputs/qwen35_performance_5fold",
    },
    "gemma3": {
        "model": "google/gemma-3-4b-it",
        "embedding_backend": "gemma",
        "out_dir": "outputs/gemma3_performance_5fold",
    },
    "gemma2_9b": {
        "model": "google/gemma-2-9b-it",
        "embedding_backend": "gemma",
        "out_dir": "outputs/gemma2_9b_performance_5fold",
    },
    "sbert": {
        "model": "sentence-transformers/all-mpnet-base-v2",
        "embedding_backend": "sbert",
        "out_dir": "outputs/sbert_performance_5fold",
    },
}

REPORT_FILENAME = "performance_5fold_report.json"


def main():
    parser = argparse.ArgumentParser(description="Run 5-fold reports for Llama, Qwen 3.5, Gemma 3 and merge")
    parser.add_argument("--hp_mcq_path", type=str, default="data/hp/hp_mcq_compromise.json")
    parser.add_argument("--wmdp_config", type=str, default="wmdp-bio")
    parser.add_argument("--base_out_dir", type=str, default="outputs",
                        help="Base dir for per-model outputs (e.g. outputs/llama_performance_5fold)")
    parser.add_argument("--skip_llama", action="store_true", help="Skip Llama report")
    parser.add_argument("--skip_qwen25", action="store_true", help="Skip Qwen 2.5 report")
    parser.add_argument("--skip_qwen35", action="store_true", help="Skip Qwen 3.5 report")
    parser.add_argument("--skip_gemma3", action="store_true", help="Skip Gemma 3 report")
    parser.add_argument("--skip_gemma2_9b", action="store_true", help="Skip Gemma 2 9B report")
    parser.add_argument("--skip_sbert", action="store_true", help="Skip SBERT report (HP + WMDP-bio, Cosine only)")
    parser.add_argument("--merged_path", type=str, default=None,
                        help="Path for merged JSON (default: outputs/performance_5fold_all_models.json)")
    parser.add_argument("--qwen_model", type=str, default=None,
                        help="Qwen model (default: Qwen/Qwen3.5-9B). Use Qwen/Qwen2.5-7B-Instruct if transformers<5.2")
    args = parser.parse_args()

    if args.qwen_model is not None:
        MODELS["qwen35"] = {
            "model": args.qwen_model,
            "embedding_backend": "qwen",
            "out_dir": MODELS["qwen35"]["out_dir"],
        }

    reports = {}
    for key, cfg in MODELS.items():
        if key == "llama" and args.skip_llama:
            continue
        if key == "qwen25" and args.skip_qwen25:
            continue
        if key == "qwen35" and args.skip_qwen35:
            continue
        if key == "gemma3" and args.skip_gemma3:
            continue
        if key == "gemma2_9b" and args.skip_gemma2_9b:
            continue
        if key == "sbert" and args.skip_sbert:
            continue
        out_dir = os.path.join(args.base_out_dir, os.path.basename(cfg["out_dir"].rstrip("/")))
        if key == "sbert":
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "report_sbert_performance.py"),
                "--hp_mcq_path", args.hp_mcq_path,
                "--wmdp_config", args.wmdp_config,
                "--sbert_model", cfg["model"],
                "--out_dir", out_dir,
            ]
        else:
            cmd = [
                sys.executable,
                os.path.join(SCRIPT_DIR, "report_llama_performance_5fold.py"),
                "--model", cfg["model"],
                "--embedding_backend", cfg["embedding_backend"],
                "--hp_mcq_path", args.hp_mcq_path,
                "--wmdp_config", args.wmdp_config,
                "--out_dir", out_dir,
            ]
        print("\n" + "=" * 60)
        print(f"Running report for {key}: {cfg['model']}")
        if key == "qwen35":
            print("  (Qwen 3.5; if load fails, use Python 3.10+ and pip install -U 'transformers>=5.2.0' or --qwen_model Qwen/Qwen2.5-7B-Instruct)")
        if key == "sbert":
            print("  (SBERT: HP + WMDP-bio, Cosine only)")
        print("=" * 60)
        r = subprocess.run(cmd, cwd=SCRIPT_DIR)
        if r.returncode != 0:
            print(f"Warning: {key} report failed with return code {r.returncode}")
            reports[key] = {"model": cfg["model"], "error": "run_failed"}
            continue
        report_path = os.path.join(out_dir, REPORT_FILENAME)
        if os.path.isfile(report_path):
            with open(report_path, "r") as f:
                reports[key] = json.load(f)
        else:
            reports[key] = {"model": cfg["model"], "error": "report_not_found"}

    merged = {
        "models": reports,
        "hp_mcq_path": args.hp_mcq_path,
        "wmdp_config": args.wmdp_config,
    }
    out_base = args.base_out_dir if os.path.isdir(args.base_out_dir) else "outputs"
    merged_path = args.merged_path or os.path.join(out_base, "performance_5fold_all_models.json")
    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged report saved to {merged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
