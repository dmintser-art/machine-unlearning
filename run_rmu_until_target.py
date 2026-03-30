#!/usr/bin/env python3
"""
Run RMU with explicit parameters (no sweep), retrying until target metrics are reached.

This is an additive mode and does not modify the behavior of run_rmu_with_eval.py.
It launches run_rmu_with_eval.py as-is, checks eval_steps.json, and:
1) stops retries when target is met,
2) keeps only the matched checkpoint,
3) optionally runs report_qa_mmlu_singleword.py on that checkpoint.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def _load_eval_steps(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("steps", [])


def _retain_value(entry: Dict, retain_metric: str) -> Optional[float]:
    if retain_metric == "retain_acc_medmcqa":
        return entry.get("retain_acc_medmcqa")
    # Backward compatible for older logs that used retain_acc only.
    return entry.get("retain_acc_mmlu", entry.get("retain_acc"))


def _find_first_target_hit(
    steps: List[Dict],
    target_forget_max: Optional[float],
    target_retain_min: Optional[float],
    retain_metric: str,
) -> Optional[Tuple[int, Dict]]:
    for entry in steps:
        step = int(entry.get("step", 0))
        if step <= 0:
            continue
        forget = entry.get("forget_acc")
        retain = _retain_value(entry, retain_metric)
        forget_ok = target_forget_max is None or (forget is not None and forget <= target_forget_max)
        retain_ok = target_retain_min is None or (retain is not None and retain >= target_retain_min)
        if forget_ok and retain_ok:
            return step, entry
    return None


def _prune_checkpoints(run_dir: str, keep_step: Optional[int]) -> None:
    keep_name = f"checkpoint-{keep_step}" if keep_step is not None else None
    for name in os.listdir(run_dir):
        if not name.startswith("checkpoint-"):
            continue
        if keep_name is not None and name == keep_name:
            continue
        path = os.path.join(run_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def _build_attempt_cmd(
    attempt_dir: str,
    attempt_batches: int,
    checkpoint_interval: int,
    target_forget_max: Optional[float],
    target_retain_min: Optional[float],
    target_retain_metric: str,
    passthrough_args: List[str],
) -> List[str]:
    cmd = [
        sys.executable,
        "run_rmu_with_eval.py",
        "--output_dir", attempt_dir,
        "--max_num_batches", str(attempt_batches),
        "--checkpoint_interval", str(checkpoint_interval),
        "--save_checkpoints",
    ]
    if target_forget_max is not None:
        cmd.extend(["--target_forget_max", str(target_forget_max)])
    if target_retain_min is not None:
        cmd.extend(["--target_retain_min", str(target_retain_min)])
    if target_forget_max is not None or target_retain_min is not None:
        cmd.append("--stop_on_target")
        metric = "medmcqa" if target_retain_metric == "retain_acc_medmcqa" else "mmlu"
        cmd.extend(["--target_retain_metric", metric])
    cmd.extend(passthrough_args)
    return cmd


def _sanitize_passthrough_args(args: List[str]) -> List[str]:
    """Drop flags controlled by target mode wrapper to avoid accidental overrides."""
    blocked = {"--output_dir", "--max_num_batches", "--checkpoint_interval", "--save_checkpoints"}
    out: List[str] = []
    skip_next = False
    for i, tok in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if tok in blocked:
            if tok != "--save_checkpoints":
                skip_next = True
            continue
        # Also handle --flag=value style.
        key = tok.split("=", 1)[0]
        if key in blocked:
            continue
        out.append(tok)
    return out


def _run_singleword_report(
    model_path: str,
    out_dir: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> None:
    cmd = [
        sys.executable,
        "report_qa_mmlu_singleword.py",
        "--models", model_path,
        "--out_dir", out_dir,
        "--batch_size", str(batch_size),
        "--max_length", str(max_length),
        "--max_new_tokens", str(max_new_tokens),
    ]
    print("Running single-word QA report:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Additional RMU mode: run with explicit params, retry until target metrics are hit."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for attempts/results.")
    parser.add_argument("--attempt_batches", type=int, default=200, help="Max RMU batches per attempt.")
    parser.add_argument("--max_retries", type=int, default=0, help="Retries after first attempt (total attempts = 1 + max_retries).")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Checkpoint/eval interval for each attempt.")
    parser.add_argument("--target_forget_max", type=float, default=None, help="Target: forget_acc must be <= this.")
    parser.add_argument("--target_retain_min", type=float, default=None, help="Target: retain metric must be >= this.")
    parser.add_argument(
        "--target_retain_metric",
        type=str,
        default="retain_acc_mmlu",
        choices=["retain_acc_mmlu", "retain_acc_medmcqa"],
        help="Retain metric used for target_retain_min.",
    )
    parser.add_argument("--run_singleword_report", action="store_true", help="Run report_qa_mmlu_singleword.py on target-hit checkpoint.")
    parser.add_argument("--singleword_report_out_dir", type=str, default="", help="Optional report output dir.")
    parser.add_argument("--singleword_report_batch_size", type=int, default=8)
    parser.add_argument("--singleword_report_max_length", type=int, default=512)
    parser.add_argument("--singleword_report_max_new_tokens", type=int, default=10)
    parser.add_argument(
        "--keep_failed_attempt_checkpoints",
        action="store_true",
        help="If set, keep all checkpoints from failed attempts. Default: prune them.",
    )

    args, passthrough_args = parser.parse_known_args()
    passthrough_args = _sanitize_passthrough_args(passthrough_args)

    if args.target_forget_max is None and args.target_retain_min is None:
        raise ValueError("Set at least one target: --target_forget_max and/or --target_retain_min.")

    os.makedirs(args.output_dir, exist_ok=True)
    total_attempts = 1 + max(0, args.max_retries)

    summary = {
        "success": False,
        "target_forget_max": args.target_forget_max,
        "target_retain_min": args.target_retain_min,
        "target_retain_metric": args.target_retain_metric,
        "attempt_batches": args.attempt_batches,
        "max_retries": args.max_retries,
        "attempts": [],
    }

    best_model_path = None
    best_hit_entry = None
    best_attempt_dir = None

    for attempt_idx in range(1, total_attempts + 1):
        attempt_dir = os.path.join(args.output_dir, f"attempt_{attempt_idx:02d}")
        os.makedirs(attempt_dir, exist_ok=True)

        cmd = _build_attempt_cmd(
            attempt_dir=attempt_dir,
            attempt_batches=args.attempt_batches,
            checkpoint_interval=args.checkpoint_interval,
            target_forget_max=args.target_forget_max,
            target_retain_min=args.target_retain_min,
            target_retain_metric=args.target_retain_metric,
            passthrough_args=passthrough_args,
        )
        print(f"\n=== Attempt {attempt_idx}/{total_attempts} ===")
        print("Running:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=True)

        eval_path = os.path.join(attempt_dir, "eval_steps.json")
        if not os.path.exists(eval_path):
            raise FileNotFoundError(f"Missing eval log: {eval_path}")
        steps = _load_eval_steps(eval_path)
        hit = _find_first_target_hit(
            steps=steps,
            target_forget_max=args.target_forget_max,
            target_retain_min=args.target_retain_min,
            retain_metric=args.target_retain_metric,
        )

        attempt_record = {
            "attempt": attempt_idx,
            "attempt_dir": attempt_dir,
            "target_hit": False,
        }
        if hit is not None:
            hit_step, hit_entry = hit
            target_ckpt = os.path.join(attempt_dir, f"checkpoint-{hit_step}")
            if not os.path.isdir(target_ckpt):
                raise FileNotFoundError(
                    f"Target checkpoint not found ({target_ckpt}). "
                    "Make sure checkpoint_interval aligns with evaluated steps."
                )
            _prune_checkpoints(attempt_dir, keep_step=hit_step)
            attempt_record["target_hit"] = True
            attempt_record["target_hit_step"] = hit_step
            attempt_record["target_hit_entry"] = hit_entry
            attempt_record["target_checkpoint_path"] = target_ckpt
            summary["attempts"].append(attempt_record)

            best_model_path = target_ckpt
            best_hit_entry = hit_entry
            best_attempt_dir = attempt_dir
            summary["success"] = True
            break

        summary["attempts"].append(attempt_record)
        if not args.keep_failed_attempt_checkpoints:
            _prune_checkpoints(attempt_dir, keep_step=None)

    if summary["success"]:
        summary["selected_attempt_dir"] = best_attempt_dir
        summary["selected_checkpoint_path"] = best_model_path
        summary["selected_metrics"] = best_hit_entry
        print("\nTarget reached.")
        print(f"  checkpoint: {best_model_path}")
        print(f"  metrics: {best_hit_entry}")

        if args.run_singleword_report and best_model_path is not None:
            report_out_dir = args.singleword_report_out_dir or os.path.join(best_attempt_dir, "qa_mmlu_singleword_report")
            _run_singleword_report(
                model_path=best_model_path,
                out_dir=report_out_dir,
                batch_size=args.singleword_report_batch_size,
                max_length=args.singleword_report_max_length,
                max_new_tokens=args.singleword_report_max_new_tokens,
            )
            summary["singleword_report_out_dir"] = report_out_dir
    else:
        print("\nTarget was not reached within retry budget.")

    summary_path = os.path.join(args.output_dir, "target_mode_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
