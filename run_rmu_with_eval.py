#!/usr/bin/env python3
"""
Run RMU unlearning with checkpoint evaluation and optional W&B logging.

Usage:
  # Single run (official-style hyperparameters)
  python run_rmu_with_eval.py --output_dir outputs/rmu_run1 --max_num_batches 150

  # With W&B
  python run_rmu_with_eval.py --use_wandb --wandb_project rmu-unlearning --output_dir outputs/rmu_wandb

  # Hyperparameter search: use W&B sweep or run multiple jobs with different args:
  python run_rmu_with_eval.py --lr 3e-5 --alpha 800,800 --steering_coeffs 8,8 --output_dir outputs/rmu_sweep1
  python run_rmu_with_eval.py --lr 7e-5 --alpha 1200,1200 --steering_coeffs 6.5,6.5 --output_dir outputs/rmu_sweep2

After running, plot with:
  python plot_rmu_accuracy.py --eval_steps outputs/rmu_run1/eval_steps.json --baselines outputs/wmdp_baselines/baseline_accuracies.json
"""
import argparse
import sys
import os

# Allow running as script from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rmu.utils import set_seed, load_model, get_data
from rmu.unlearn import run_rmu_with_eval, get_args


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(v)


def _normalize_eval_flags(args):
    for name in (
        "wmdp_only_eval",
        "skip_mmlu_eval",
        "skip_medmcqa_eval",
        "skip_baseline_eval",
        "stop_on_target",
        "save_target_checkpoint",
    ):
        if hasattr(args, name):
            setattr(args, name, _as_bool(getattr(args, name)))
    if getattr(args, "wmdp_only_eval", False):
        args.skip_mmlu_eval = True
        args.skip_medmcqa_eval = True


def _normalize_corpora(args):
    for name in ("forget_corpora", "retain_corpora"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if isinstance(value, str):
            setattr(args, name, [s.strip() for s in value.split(",") if s.strip()])


def main():
    args = get_args()
    _normalize_eval_flags(args)
    _normalize_corpora(args)
    set_seed(args.seed)

    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            args.use_wandb = True
            # Apply sweep config overrides (when running as wandb agent)
            if hasattr(wandb, "config") and wandb.config:
                for k, v in dict(wandb.config).items():
                    if hasattr(args, k):
                        setattr(args, k, v)
                # Re-parse list params if they were overwritten as strings
                if isinstance(getattr(args, "alpha", None), str):
                    args.alpha = [float(c) for c in args.alpha.split(",")]
                if isinstance(getattr(args, "steering_coeff_list", None), str):
                    args.steering_coeff_list = [float(c) for c in args.steering_coeff_list.split(",")]
                elif isinstance(getattr(args, "steering_coeffs", None), str):
                    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
                if isinstance(getattr(args, "layer_ids", None), str):
                    args.layer_ids = [int(x) for x in args.layer_ids.split(",")]
                if isinstance(getattr(args, "param_ids", None), str):
                    args.param_ids = [int(x) for x in args.param_ids.split(",") if str(x).strip() != ""]
                if isinstance(getattr(args, "param_names", None), str):
                    args.param_names = [x.strip() for x in args.param_names.split(",") if x.strip()]
                _normalize_eval_flags(args)
                _normalize_corpora(args)
        except Exception as e:
            print(f"W&B init failed: {e}. Continuing without W&B.")
            args.use_wandb = False

    print("Loading model (frozen + trainable copy)...")
    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)
    print("Loading forget & retain corpora (requires HF_TOKEN for WMDP corpora)...")
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    print(f"Forget batches: {[len(f) for f in forget_data_list]}, Retain batches: {[len(r) for r in retain_data_list]}")

    run_rmu_with_eval(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )


if __name__ == "__main__":
    main()
