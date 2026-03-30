#!/usr/bin/env python3
"""
Run SBERT on HP and WMDP-bio: embedding only, Cosine baseline only (no Ridge/Bilinear/MLP, no Q&A).
Uses the final SBERT sentence embedding (not a first layer). Outputs a report with only Cosine.
Requires: pip install sentence-transformers

Usage:
  python report_sbert_performance.py --hp_mcq_path data/hp/hp_mcq_compromise.json
  python report_sbert_performance.py --sbert_model sentence-transformers/all-mpnet-base-v2 --out_dir outputs/sbert_performance_5fold
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
N_FOLDS = 5
SEED = 42


def run_probe_ladder_sbert(dataset, script_dir, cache_dir, out_dir, hp_mcq_path, wmdp_config, sbert_model, n_folds, seed):
    """Run hp_probe_ladder.py with embedding_backend=sbert for one dataset. Returns baseline_accuracies dict."""
    cmd = [
        sys.executable,
        os.path.join(script_dir, "hp_probe_ladder.py"),
        "--embedding_backend", "sbert",
        "--sbert_model", sbert_model,
        "--dataset", dataset,
        "--n_folds", str(n_folds),
        "--seed", str(seed),
        "--out_dir", out_dir,
        "--cache_dir", cache_dir,
        "--normalize_sbert",
    ]
    if dataset == "hp":
        cmd += ["--data_path", hp_mcq_path]
    else:
        cmd += ["--wmdp_config", wmdp_config]
    print(f"  Running: hp_probe_ladder.py dataset={dataset} backend=sbert n_folds={n_folds} ...")
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stderr:
            print("  stderr:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        raise RuntimeError(f"hp_probe_ladder failed with return code {result.returncode}")
    baseline_path = os.path.join(out_dir, "baseline_accuracies.json")
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(f"Expected {baseline_path} after probe ladder")
    with open(baseline_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="SBERT performance on HP and WMDP-bio: Cosine only (sentence embedding, no layer 0)."
    )
    parser.add_argument("--hp_mcq_path", type=str, default="data/hp/hp_mcq_compromise.json")
    parser.add_argument("--wmdp_config", type=str, default="wmdp-bio")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--out_dir", type=str, default="outputs/sbert_performance_5fold")
    parser.add_argument("--cache_dir", type=str, default="cache/sbert_5fold")
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    report = {
        "model": args.sbert_model,
        "embedding_backend": "sbert",
        "n_folds": args.n_folds,
        "note": "Cosine only (SBERT sentence embedding); no Ridge/Bilinear/MLP, no Q&A.",
        "hp": {},
        "wmdp": {"config": args.wmdp_config},
    }

    # ----- HP: Cosine only -----
    print("\n" + "=" * 60)
    print("HP (filtered): SBERT Cosine (full-data, no folds)")
    print("=" * 60)
    if os.path.isfile(args.hp_mcq_path):
        try:
            hp_out = os.path.join(args.out_dir, "hp_embedding_layer0")
            os.makedirs(hp_out, exist_ok=True)
            bl = run_probe_ladder_sbert(
                "hp", SCRIPT_DIR, args.cache_dir, hp_out,
                args.hp_mcq_path, args.wmdp_config, args.sbert_model, args.n_folds, args.seed,
            )
            report["hp"]["embedding"] = {
                "Cosine": round(bl["Cosine"], 4),
                "Cosine_ci95": round(bl.get("Cosine_ci95", 0), 4),
            }
            print(f"  Cosine: {bl['Cosine']:.4f}")
        except Exception as e:
            report["hp"]["embedding"] = {"error": str(e)}
            print(f"  Error: {e}")
    else:
        print(f"  Skipped: HP file not found: {args.hp_mcq_path}")

    # ----- WMDP: Cosine only -----
    print("\n" + "=" * 60)
    print(f"WMDP ({args.wmdp_config}): SBERT Cosine (full-data, no folds)")
    print("=" * 60)
    try:
        wmdp_out = os.path.join(args.out_dir, "wmdp_embedding_layer0")
        os.makedirs(wmdp_out, exist_ok=True)
        bl = run_probe_ladder_sbert(
            "wmdp", SCRIPT_DIR, args.cache_dir, wmdp_out,
            args.hp_mcq_path, args.wmdp_config, args.sbert_model, args.n_folds, args.seed,
        )
        report["wmdp"]["embedding"] = {
            "Cosine": round(bl["Cosine"], 4),
            "Cosine_ci95": round(bl.get("Cosine_ci95", 0), 4),
        }
        print(f"  Cosine: {bl['Cosine']:.4f}")
    except Exception as e:
        report["wmdp"]["embedding"] = {"error": str(e)}
        print(f"  Error: {e}")

    report_path = os.path.join(args.out_dir, "performance_5fold_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
