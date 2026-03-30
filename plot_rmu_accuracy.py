#!/usr/bin/env python3
"""
Plot RMU accuracy trajectory with dotted horizontal lines at embedding baseline levels.

Usage:
  python plot_rmu_accuracy.py --eval_steps outputs/rmu_run1/eval_steps.json --baselines outputs/wmdp_baselines/baseline_accuracies.json -o rmu_accuracy_plot.png

  # Without baseline file (no dotted lines):
  python plot_rmu_accuracy.py --eval_steps outputs/rmu_run1/eval_steps.json -o plot.png

  # Baseline dict from CLI:
  python plot_rmu_accuracy.py --eval_steps outputs/rmu_run1/eval_steps.json --baseline_cosine 0.29 --baseline_mlp 0.34 -o plot.png
"""
import argparse
import json
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def load_eval_steps(path):
    with open(path) as f:
        data = json.load(f)
    steps = data.get("steps", data) if isinstance(data, dict) else data
    if not isinstance(steps, list):
        steps = [steps]
    return steps


def load_baselines(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    return {k: float(v) for k, v in d.items() if k in ("Cosine", "Ridge", "Bilinear", "MLP") and isinstance(v, (int, float))}


def plot(eval_steps_path, baselines_path=None, baseline_cosine=None, baseline_ridge=None, baseline_bilinear=None, baseline_mlp=None, out_path="rmu_accuracy_plot.png", title=None):
    steps_log = load_eval_steps(eval_steps_path)
    if not steps_log:
        raise ValueError(f"No steps found in {eval_steps_path}")

    baselines = load_baselines(baselines_path) if baselines_path else {}
    if baseline_cosine is not None:
        baselines["Cosine"] = float(baseline_cosine)
    if baseline_ridge is not None:
        baselines["Ridge"] = float(baseline_ridge)
    if baseline_bilinear is not None:
        baselines["Bilinear"] = float(baseline_bilinear)
    if baseline_mlp is not None:
        baselines["MLP"] = float(baseline_mlp)

    steps = [e["step"] for e in steps_log]
    forget_acc = [e["forget_acc"] for e in steps_log]
    retain_acc = [e.get("retain_acc", e.get("retain_acc_mmlu", 0)) for e in steps_log]
    retain_medmcqa = [e.get("retain_acc_medmcqa") for e in steps_log]
    has_medmcqa = any(x is not None for x in retain_medmcqa)

    if plt is None:
        print("matplotlib not available; dumping data.")
        print("steps", steps)
        print("forget_acc", forget_acc)
        print("retain_acc", retain_acc)
        print("baselines", baselines)
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(steps, forget_acc, label="Forget set (WMDP-bio)", color="C0", marker="o", markersize=4)
    ax.plot(steps, retain_acc, label="Retain (MMLU)", color="C1", marker="s", markersize=4)
    if has_medmcqa:
        medmcqa_vals = [x if x is not None else 0 for x in retain_medmcqa]
        ax.plot(steps, medmcqa_vals, label="Retain (MedMCQA)", color="C2", marker="^", markersize=4)

    # Dotted horizontal lines at embedding baseline accuracies
    colors = {"Cosine": "gray", "Ridge": "green", "Bilinear": "purple", "MLP": "brown"}
    for name, acc in baselines.items():
        ax.axhline(y=acc, color=colors.get(name, "gray"), linestyle="--", linewidth=1, alpha=0.8, label=f"Baseline: {name} ({acc:.2f})")

    ax.set_xlabel("RMU step (batch)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title or "RMU: Forget vs Retain accuracy")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot RMU accuracy with baseline dotted lines")
    ap.add_argument("--eval_steps", type=str, required=True, help="Path to eval_steps.json from RMU run")
    ap.add_argument("--baselines", type=str, default=None, help="Path to baseline_accuracies.json (from hp_probe_ladder on WMDP)")
    ap.add_argument("--baseline_cosine", type=float, default=None)
    ap.add_argument("--baseline_ridge", type=float, default=None)
    ap.add_argument("--baseline_bilinear", type=float, default=None)
    ap.add_argument("--baseline_mlp", type=float, default=None)
    ap.add_argument("-o", "--out", type=str, default="rmu_accuracy_plot.png", help="Output plot path")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()
    plot(
        eval_steps_path=args.eval_steps,
        baselines_path=args.baselines,
        baseline_cosine=args.baseline_cosine,
        baseline_ridge=args.baseline_ridge,
        baseline_bilinear=args.baseline_bilinear,
        baseline_mlp=args.baseline_mlp,
        out_path=args.out,
        title=args.title,
    )


if __name__ == "__main__":
    main()
