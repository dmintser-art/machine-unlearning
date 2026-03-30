#!/usr/bin/env python3
"""
Systematic report of LM performance on HP (filtered) and WMDP:
  - Embeddings: probe ladder (Cosine, Ridge, Bilinear, MLP) on layer 0 with 5-fold CV and 95% CI.
  - Q&A: direct LM accuracy with 5-fold CV and 95% CI.

Supports Llama, Qwen 3.5 (https://huggingface.co/collections/Qwen/qwen35), and Gemma 3
(https://huggingface.co/collections/google/gemma-3-release). HP uses split-by-correct-answer
folds to avoid data leakage. WMDP uses the same strategy. All results use layer 0 (hidden_state[0]).

Usage:
  python report_llama_performance_5fold.py --hp_mcq_path data/hp/hp_mcq_compromise.json
  python report_llama_performance_5fold.py --model Qwen/Qwen3.5-9B --embedding_backend qwen --out_dir outputs/qwen35_5fold
  python report_llama_performance_5fold.py --model google/gemma-3-4b-it --embedding_backend gemma --out_dir outputs/gemma3_5fold

Note: Qwen 3.5 (e.g. Qwen/Qwen3.5-9B) requires transformers>=5.2.0. If you see "model type qwen3_5 not
recognized", run: pip install -U "transformers>=5.2.0". Or use Qwen 2.5: --model Qwen/Qwen2.5-7B-Instruct.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback

import numpy as np

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from hp_probe_ladder import get_folds_by_correct_answer, load_hp  # noqa: E402
from rmu.eval_mcq import eval_mcq_accuracy, load_wmdp_questions  # noqa: E402
from rmu.utils import load_model  # noqa: E402

N_FOLDS = 5
LAYER_0 = 0
SEED = 42

# t_{0.975, 4} for 5-fold 95% CI
T_95_4 = 2.571

# VLMs / multimodal models load vision + language; text-only Q&A uses a fallback to avoid forward issues.
QA_FALLBACK_FOR_VLM = {
    "Qwen/Qwen3.5-9B": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3.5-4B": "Qwen/Qwen2.5-3B-Instruct",
    # Gemma 3 4B is multimodal; text-only Q&A uses Gemma 2. Use 9B so WMDP Q&A is comparable to other 8B models.
    "google/gemma-3-4b-it": "google/gemma-2-9b-it",
}


def _qa_model_id(model_id):
    """Return model id to use for Q&A. Use text-only fallback for known VLMs."""
    return QA_FALLBACK_FOR_VLM.get(model_id, model_id)


def ci95(accs):
    """Mean and half-width of 95% CI (t-based, df=n_folds-1)."""
    accs = np.asarray(accs, dtype=float)
    k = len(accs)
    if k < 2:
        return float(np.mean(accs)), 0.0
    mean_a = float(np.mean(accs))
    sem = float(np.std(accs, ddof=1) / np.sqrt(k))
    half = T_95_4 * sem
    return mean_a, half


def run_probe_ladder(dataset, layer, n_folds, extra_args, script_dir, cache_dir, out_dir):
    """Run hp_probe_ladder.py for one dataset with 5-fold CV and layer 0."""
    backend = extra_args.get("embedding_backend", "llama")
    model_id = extra_args.get("model_id", extra_args.get("llama_model", "meta-llama/Llama-3.1-8B-Instruct"))
    cmd = [
        sys.executable,
        os.path.join(script_dir, "hp_probe_ladder.py"),
        "--embedding_backend", backend,
        "--dataset", dataset,
        "--qwen_layer", str(layer),
        "--n_folds", str(n_folds),
        "--seed", str(SEED),
        "--out_dir", out_dir,
        "--cache_dir", cache_dir,
    ]
    if backend == "qwen":
        cmd += ["--qwen_model", model_id]
    elif backend == "gemma":
        cmd += ["--gemma_model", model_id]
    else:
        cmd += ["--llama_model", model_id]
    if dataset == "hp":
        cmd += ["--data_path", extra_args["hp_mcq_path"]]
    else:
        cmd += ["--wmdp_config", extra_args.get("wmdp_config", "wmdp-bio")]
    print(f"  Running: hp_probe_ladder.py dataset={dataset} backend={backend} layer={layer} n_folds={n_folds} ...")
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print("  stderr:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        if result.stdout:
            print("  stdout (last 1500 chars):", result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
        raise RuntimeError(f"hp_probe_ladder failed with return code {result.returncode}")
    baseline_path = os.path.join(out_dir, "baseline_accuracies.json")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Expected {baseline_path} after probe ladder")
    with open(baseline_path, "r") as f:
        return json.load(f)


def run_qa_hp_full(model, tokenizer, hp_mcq_path, batch_size=8, max_length=512):
    """Q&A on HP full dataset (no folds; no learning in next-token scoring)."""
    questions, choices, answers = load_hp(hp_mcq_path, log_bad=False)
    if not questions:
        return 0.0, 0.0
    acc = eval_mcq_accuracy(model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length)
    return acc, 0.0


def run_qa_wmdp_full(model, tokenizer, wmdp_config, batch_size=8, max_length=512):
    """Q&A on WMDP full dataset (no folds; no learning in next-token scoring)."""
    questions, choices, answers = load_wmdp_questions(config=wmdp_config, split="test")
    if not questions:
        return 0.0, 0.0
    acc = eval_mcq_accuracy(model, tokenizer, questions, choices, answers, batch_size=batch_size, max_length=max_length)
    return acc, 0.0


def _infer_backend(model_id):
    m = model_id.lower()
    if "qwen" in m:
        return "qwen"
    if "gemma" in m:
        return "gemma"
    return "llama"


def main():
    parser = argparse.ArgumentParser(
        description="LM performance report: HP and WMDP, embeddings (layer 0) + Q&A, 5-fold CV with CI (Llama, Qwen 3.5, Gemma 3)"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model id (e.g. Qwen/Qwen3.5-4B, google/gemma-3-4b-it). Overrides --llama_model.")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Used when --model is not set (Llama report).")
    parser.add_argument("--embedding_backend", type=str, default=None, choices=["llama", "qwen", "gemma"],
                        help="Inferred from --model if not set (qwen/gemma/llama).")
    parser.add_argument("--hp_mcq_path", type=str, default="data/hp/hp_mcq_compromise.json",
                        help="Filtered HP MCQ JSON (e.g. correct_once_distractors_5)")
    parser.add_argument("--wmdp_config", type=str, default="wmdp-bio",
                        help="WMDP config: wmdp-bio, wmdp-cyber, wmdp-chem, or all")
    parser.add_argument("--out_dir", type=str, default="outputs/llama_performance_5fold")
    parser.add_argument("--cache_dir", type=str, default="cache/llama_5fold")
    parser.add_argument("--skip_hp_emb", action="store_true", help="Skip HP embedding (probe) run")
    parser.add_argument("--skip_wmdp_emb", action="store_true", help="Skip WMDP embedding (probe) run")
    parser.add_argument("--skip_hp_qa", action="store_true", help="Skip HP Q&A 5-fold")
    parser.add_argument("--skip_wmdp_qa", action="store_true", help="Skip WMDP Q&A 5-fold")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    model_id = args.model if args.model else args.llama_model
    backend = args.embedding_backend if args.embedding_backend else _infer_backend(model_id)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    extra = {
        "model_id": model_id,
        "embedding_backend": backend,
        "llama_model": model_id,
        "hp_mcq_path": args.hp_mcq_path,
        "wmdp_config": args.wmdp_config,
    }

    qa_model_id = _qa_model_id(model_id)
    report = {
        "model": model_id,
        "qa_model": qa_model_id,  # may differ from model for VLMs (e.g. Qwen 3.5 -> Qwen 2.5 for Q&A)
        "embedding_backend": backend,
        "n_folds": N_FOLDS,
        "layer": LAYER_0,
        "hp": {},
        "wmdp": {"config": args.wmdp_config},
    }
    model, tokenizer = None, None

    # ----- HP: Embeddings (layer 0, 5-fold CV) -----
    print("\n" + "=" * 60)
    print("HP (filtered): Embedding performance (layer 0, 5-fold CV, no leakage)")
    print("=" * 60)
    if not args.skip_hp_emb and os.path.isfile(args.hp_mcq_path):
        hp_emb_dir = os.path.join(args.out_dir, "hp_embedding_layer0")
        os.makedirs(hp_emb_dir, exist_ok=True)
        try:
            bl = run_probe_ladder(
                "hp", LAYER_0, N_FOLDS, extra, SCRIPT_DIR, args.cache_dir, hp_emb_dir
            )
            report["hp"]["embedding"] = {
                "Cosine": round(bl.get("Cosine"), 4),
                "Cosine_ci95": round(bl.get("Cosine_ci95", 0), 4),
                "Ridge": round(bl.get("Ridge"), 4),
                "Ridge_ci95": round(bl.get("Ridge_ci95", 0), 4),
                "Bilinear": round(bl.get("Bilinear"), 4),
                "Bilinear_ci95": round(bl.get("Bilinear_ci95", 0), 4),
                "MLP": round(bl.get("MLP"), 4),
                "MLP_ci95": round(bl.get("MLP_ci95", 0), 4),
            }
            print(f"  Cosine:   {bl['Cosine']:.4f} ± {bl.get('Cosine_ci95', 0):.4f}")
            print(f"  Ridge:    {bl['Ridge']:.4f} ± {bl.get('Ridge_ci95', 0):.4f}")
            print(f"  Bilinear: {bl['Bilinear']:.4f} ± {bl.get('Bilinear_ci95', 0):.4f}")
            print(f"  MLP:      {bl['MLP']:.4f} ± {bl.get('MLP_ci95', 0):.4f}")
        except Exception as e:
            report["hp"]["embedding"] = {"error": str(e)}
            print(f"  Error: {e}")
    elif args.skip_hp_emb:
        print("  Skipped (--skip_hp_emb)")
    else:
        print(f"  Skipped: HP file not found: {args.hp_mcq_path}")

    # ----- HP: Q&A (full data; no folds) -----
    print("\n" + "=" * 60)
    print("HP (filtered): Q&A performance (full data)")
    print("=" * 60)
    if not args.skip_hp_qa and os.path.isfile(args.hp_mcq_path):
        try:
            if qa_model_id != model_id:
                print(f"  Using text-only model for Q&A (VLM fallback): {qa_model_id}")
            if model is None:
                print("  Loading model for Q&A ...")
                model, tokenizer = load_model(qa_model_id)
                model.eval()
            mean_acc, half_ci = run_qa_hp_full(
                model, tokenizer, args.hp_mcq_path,
                batch_size=args.batch_size, max_length=args.max_length,
            )
            report["hp"]["qa"] = {
                "accuracy": round(mean_acc, 4),
                "ci95_half": round(half_ci, 4),
            }
            print(f"  Q&A accuracy: {mean_acc:.4f}")
        except Exception as e:
            report["hp"]["qa"] = {"error": str(e)}
            print(f"  Error: {e}")
            traceback.print_exc()
    elif args.skip_hp_qa:
        print("  Skipped (--skip_hp_qa)")
    else:
        print(f"  Skipped: HP file not found: {args.hp_mcq_path}")

    # ----- WMDP: Embeddings (layer 0, 5-fold CV) -----
    print("\n" + "=" * 60)
    print("WMDP: Embedding performance (layer 0, 5-fold CV, no leakage)")
    print("=" * 60)
    if not args.skip_wmdp_emb:
        wmdp_emb_dir = os.path.join(args.out_dir, "wmdp_embedding_layer0")
        os.makedirs(wmdp_emb_dir, exist_ok=True)
        try:
            bl = run_probe_ladder(
                "wmdp", LAYER_0, N_FOLDS, extra, SCRIPT_DIR, args.cache_dir, wmdp_emb_dir
            )
            report["wmdp"]["embedding"] = {
                "Cosine": round(bl.get("Cosine"), 4),
                "Cosine_ci95": round(bl.get("Cosine_ci95", 0), 4),
                "Ridge": round(bl.get("Ridge"), 4),
                "Ridge_ci95": round(bl.get("Ridge_ci95", 0), 4),
                "Bilinear": round(bl.get("Bilinear"), 4),
                "Bilinear_ci95": round(bl.get("Bilinear_ci95", 0), 4),
                "MLP": round(bl.get("MLP"), 4),
                "MLP_ci95": round(bl.get("MLP_ci95", 0), 4),
            }
            print(f"  Cosine:   {bl['Cosine']:.4f} ± {bl.get('Cosine_ci95', 0):.4f}")
            print(f"  Ridge:    {bl['Ridge']:.4f} ± {bl.get('Ridge_ci95', 0):.4f}")
            print(f"  Bilinear: {bl['Bilinear']:.4f} ± {bl.get('Bilinear_ci95', 0):.4f}")
            print(f"  MLP:      {bl['MLP']:.4f} ± {bl.get('MLP_ci95', 0):.4f}")
        except Exception as e:
            report["wmdp"]["embedding"] = {"error": str(e)}
            print(f"  Error: {e}")
    else:
        print("  Skipped (--skip_wmdp_emb)")

    # ----- WMDP: Q&A (full data; no folds) -----
    print("\n" + "=" * 60)
    print("WMDP: Q&A performance (full data)")
    print("=" * 60)
    if not args.skip_wmdp_qa:
        try:
            if model is None:
                if qa_model_id != model_id:
                    print(f"  Using text-only model for Q&A (VLM fallback): {qa_model_id}")
                print("  Loading model for Q&A ...")
                model, tokenizer = load_model(qa_model_id)
                model.eval()
            mean_acc, half_ci = run_qa_wmdp_full(
                model, tokenizer, args.wmdp_config,
                batch_size=args.batch_size, max_length=args.max_length,
            )
            report["wmdp"]["qa"] = {
                "accuracy": round(mean_acc, 4),
                "ci95_half": round(half_ci, 4),
            }
            print(f"  Q&A accuracy: {mean_acc:.4f}")
        except Exception as e:
            report["wmdp"]["qa"] = {"error": str(e)}
            print(f"  Error: {e}")
            traceback.print_exc()
    else:
        print("  Skipped (--skip_wmdp_qa)")

    # ----- Save and print summary -----
    report_path = os.path.join(args.out_dir, "performance_5fold_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n" + "=" * 60)
    print("Summary (all metrics: mean ± 95% CI half-width, 5-fold CV, layer 0)")
    print("=" * 60)
    print("\nHP (filtered):")
    if report["hp"].get("embedding") and "error" not in report["hp"]["embedding"]:
        e = report["hp"]["embedding"]
        print(f"  Embedding: Cosine {e['Cosine']:.4f}±{e['Cosine_ci95']:.4f}  Ridge {e['Ridge']:.4f}±{e['Ridge_ci95']:.4f}  Bilinear {e['Bilinear']:.4f}±{e['Bilinear_ci95']:.4f}  MLP {e['MLP']:.4f}±{e['MLP_ci95']:.4f}")
    if report["hp"].get("qa") and "error" not in report["hp"]["qa"]:
        print(f"  Q&A:       {report['hp']['qa']['accuracy']:.4f} ± {report['hp']['qa']['ci95_half']:.4f}")
    print("\nWMDP:")
    if report["wmdp"].get("embedding") and "error" not in report["wmdp"]["embedding"]:
        e = report["wmdp"]["embedding"]
        print(f"  Embedding: Cosine {e['Cosine']:.4f}±{e['Cosine_ci95']:.4f}  Ridge {e['Ridge']:.4f}±{e['Ridge_ci95']:.4f}  Bilinear {e['Bilinear']:.4f}±{e['Bilinear_ci95']:.4f}  MLP {e['MLP']:.4f}±{e['MLP_ci95']:.4f}")
    if report["wmdp"].get("qa") and "error" not in report["wmdp"]["qa"]:
        print(f"  Q&A:       {report['wmdp']['qa']['accuracy']:.4f} ± {report['wmdp']['qa']['ci95_half']:.4f}")
    print(f"\nReport saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
