#!/usr/bin/env python3
"""
Plot multi-project tradeoff curves from Weights & Biases runs.

Example:
  python plot_wandb_pareto.py \
    --entity adversarial-attacks-course \
    --projects rmu-unlearning-llama31-tradeoff,rmu-unlearning-tradeoff,rmu-unlearning-gemma2-9b-tradeoff \
    --output pareto_wmdp_bio.png
"""

import argparse
from math import ceil
import statistics
import hashlib
import json
import os
import time
import math

import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot bucketed tradeoff curves across W&B projects."
    )
    parser.add_argument("--entity", type=str, required=True, help="W&B entity/user/team")
    parser.add_argument(
        "--projects",
        type=str,
        required=True,
        help="Comma-separated W&B project names.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=1,
        help="Number of subplot columns (e.g., 2 for side-by-side with two projects).",
    )
    parser.add_argument(
        "--titles",
        type=str,
        default="",
        help="Optional comma-separated subplot titles matched to --projects order.",
    )
    parser.add_argument(
        "--x_metric",
        type=str,
        default="forget_acc",
        help="Metric for x-axis (minimize). Default: forget_acc",
    )
    parser.add_argument(
        "--y_metric",
        type=str,
        default="retain_acc_mmlu",
        help="Metric for y-axis (maximize). Default: retain_acc_mmlu",
    )
    parser.add_argument(
        "--p_eff_low",
        type=str,
        default="",
        help="Per-project lower x-thresholds, comma-separated in project order.",
    )
    parser.add_argument(
        "--p_eff_high",
        type=str,
        default="",
        help="Per-project upper x-thresholds, comma-separated in project order.",
    )
    parser.add_argument(
        "--best_retain",
        type=str,
        default="",
        help="Per-project horizontal best-retain reference, comma-separated in project order.",
    )
    parser.add_argument(
        "--states",
        type=str,
        default="finished",
        help="Comma-separated run states to include (e.g. finished,crashed).",
    )
    parser.add_argument(
        "--bucket_count",
        type=int,
        default=12,
        help="Number of x-axis buckets for the tradeoff curve.",
    )
    parser.add_argument(
        "--bucket_agg",
        type=str,
        default="mean",
        choices=["mean", "median", "max"],
        help="Aggregation used for y per x bucket.",
    )
    parser.add_argument(
        "--bucket_min",
        type=float,
        default=None,
        help="Optional minimum x-value for bucket edges. If unset, inferred per project.",
    )
    parser.add_argument(
        "--bucket_max",
        type=float,
        default=None,
        help="Optional maximum x-value for bucket edges. If unset, inferred per project.",
    )
    parser.add_argument(
        "--xpad_frac",
        type=float,
        default=0.08,
        help="Fractional padding added to inferred x-limits.",
    )
    parser.add_argument(
        "--x_min_floor",
        type=float,
        default=0.25,
        help="Minimum x-axis lower bound applied to every subplot.",
    )
    parser.add_argument(
        "--x_tick_step",
        type=float,
        default=0.05,
        help="Fixed x-axis tick step (e.g., 0.05 for 5-percentage-point jumps).",
    )
    parser.add_argument(
        "--shared_y_axis",
        action="store_true",
        help="Use the same y-axis range across all populated subplots.",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=None,
        help="Optional global y-axis minimum (used with --shared_y_axis).",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=None,
        help="Optional global y-axis maximum (used with --shared_y_axis).",
    )
    parser.add_argument(
        "--ypad_frac",
        type=float,
        default=0.06,
        help="Fractional padding for inferred shared y-range.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pareto_wandb_projects.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache/wandb_plot_cache",
        help="Directory for cached run summaries.",
    )
    parser.add_argument(
        "--cache_ttl_minutes",
        type=float,
        default=60.0,
        help="Cache freshness window in minutes.",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache and always fetch from W&B.",
    )
    parser.add_argument(
        "--refresh_cache",
        action="store_true",
        help="Force refresh from W&B and overwrite cache.",
    )
    return parser.parse_args()


def is_number(value):
    return isinstance(value, (int, float)) and value == value


def parse_per_project_values(raw, name, n_projects):
    if raw is None or not str(raw).strip():
        return None
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if len(values) != n_projects:
        raise ValueError(
            f"{name} must have exactly {n_projects} values (got {len(values)}). "
            f"Projects are matched by order."
        )
    return values


def parse_per_project_strings(raw, name, n_projects):
    if raw is None or not str(raw).strip():
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != n_projects:
        raise ValueError(
            f"{name} must have exactly {n_projects} values (got {len(values)}). "
            f"Projects are matched by order."
        )
    return values


def _cache_key(entity, project, x_metric, y_metric, states):
    payload = {
        "entity": entity,
        "project": project,
        "x_metric": x_metric,
        "y_metric": y_metric,
        "states": sorted(states) if states else [],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def _cache_path(cache_dir, key):
    return os.path.join(cache_dir, f"{key}.json")


def _load_points_from_cache(path, ttl_seconds):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        fetched_at = float(payload.get("fetched_at", 0))
        points = payload.get("points", [])
        if time.time() - fetched_at > ttl_seconds:
            return None
        return points
    except Exception:
        return None


def _save_points_to_cache(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "fetched_at": time.time(),
        "points": points,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def get_project_points(
    api,
    entity,
    project,
    x_metric,
    y_metric,
    states,
    use_cache=True,
    refresh_cache=False,
    cache_dir=".cache/wandb_plot_cache",
    cache_ttl_seconds=3600.0,
):
    key = _cache_key(entity, project, x_metric, y_metric, states)
    cpath = _cache_path(cache_dir, key)
    if use_cache and not refresh_cache:
        cached_points = _load_points_from_cache(cpath, cache_ttl_seconds)
        if cached_points is not None:
            return cached_points, True

    runs = api.runs(f"{entity}/{project}")
    try:
        total_runs = len(runs)
    except Exception:
        total_runs = None
    points = []
    for run in tqdm(runs, total=total_runs, desc=f"Fetching runs: {project}", leave=False):
        if states and run.state not in states:
            continue
        summary = run.summary or {}
        x = summary.get(x_metric)
        y = summary.get(y_metric)
        if not (is_number(x) and is_number(y)):
            continue
        points.append({"x": float(x), "y": float(y), "name": run.name, "id": run.id})
    if use_cache:
        _save_points_to_cache(cpath, points)
    return points, False


def bucket_curve(points, x_min, x_max, bucket_count, agg):
    if not points:
        return [], [], [], [], []
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    if x_max <= x_min:
        raise ValueError("bucket_max must be larger than bucket_min.")

    width = (x_max - x_min) / bucket_count
    bins = [[] for _ in range(bucket_count)]

    for p in points:
        x = p["x"]
        if x < x_min or x > x_max:
            continue
        idx = int((x - x_min) / width)
        if idx == bucket_count:
            idx = bucket_count - 1
        bins[idx].append(p)

    bx, by, by_min, by_max, bcount = [], [], [], [], []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        x_values = [p["x"] for p in bucket]
        y_values = [p["y"] for p in bucket]
        if agg == "mean":
            y_stat = sum(y_values) / len(y_values)
        elif agg == "median":
            y_stat = statistics.median(y_values)
        else:  # agg == "max"
            y_stat = max(y_values)
        bx.append(sum(x_values) / len(x_values))
        by.append(y_stat)
        by_min.append(min(y_values))
        by_max.append(max(y_values))
        bcount.append(len(bucket))
    return bx, by, by_min, by_max, bcount


def infer_x_range(xs, p_eff_low=None, p_eff_high=None, pad_frac=0.08):
    values = list(xs)
    if p_eff_low is not None:
        values.append(p_eff_low)
    if p_eff_high is not None:
        values.append(p_eff_high)
    if not values:
        return 0.0, 1.0
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0) * 0.1
    pad = span * max(0.0, pad_frac)
    return lo - pad, hi + pad


def infer_y_range(ys, pad_frac=0.06):
    if not ys:
        return 0.0, 1.0
    lo = min(ys)
    hi = max(ys)
    span = hi - lo
    if span <= 0:
        span = max(abs(lo), 1.0) * 0.1
    pad = span * max(0.0, pad_frac)
    return lo - pad, hi + pad


def main():
    args = parse_args()
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]
    if not projects:
        raise ValueError("No valid projects provided.")
    p_eff_low_list = parse_per_project_values(args.p_eff_low, "p_eff_low", len(projects))
    p_eff_high_list = parse_per_project_values(args.p_eff_high, "p_eff_high", len(projects))
    best_retain_list = parse_per_project_values(args.best_retain, "best_retain", len(projects))
    titles_list = parse_per_project_strings(args.titles, "titles", len(projects))

    states = {s.strip() for s in args.states.split(",") if s.strip()}
    api = wandb.Api()
    use_cache = not args.no_cache
    cache_ttl_seconds = max(0.0, args.cache_ttl_minutes * 60.0)

    n = len(projects)
    cols = max(1, min(args.ncols, n))
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.8 * rows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    plotted_axes = []
    all_project_ys = []

    for idx, project in enumerate(tqdm(projects, desc="Projects", leave=True)):
        ax = axes_flat[idx]
        points = get_project_points(
            api=api,
            entity=args.entity,
            project=project,
            x_metric=args.x_metric,
            y_metric=args.y_metric,
            states=states,
            use_cache=use_cache,
            refresh_cache=args.refresh_cache,
            cache_dir=args.cache_dir,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        if isinstance(points, tuple):
            points, from_cache = points
        else:
            from_cache = False
        if not points:
            ax.set_title(f"{project}\n(no valid runs)")
            ax.set_xlabel(args.x_metric)
            ax.set_ylabel(args.y_metric)
            ax.grid(alpha=0.25)
            continue

        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        all_project_ys.extend(ys)
        p_eff_low = p_eff_low_list[idx] if p_eff_low_list is not None else None
        p_eff_high = p_eff_high_list[idx] if p_eff_high_list is not None else None
        if args.bucket_min is not None and args.bucket_max is not None:
            x_min, x_max = args.bucket_min, args.bucket_max
        else:
            x_min, x_max = infer_x_range(
                xs,
                p_eff_low=p_eff_low,
                p_eff_high=p_eff_high,
                pad_frac=args.xpad_frac,
            )
        x_min = max(x_min, args.x_min_floor)
        if x_max <= x_min:
            x_max = x_min + max(0.05, abs(x_min) * 0.05)
        bx, by, by_min, by_max, bcount = bucket_curve(
            points=points,
            x_min=x_min,
            x_max=x_max,
            bucket_count=args.bucket_count,
            agg=args.bucket_agg,
        )

        ax.scatter(xs, ys, s=22, alpha=0.45, label="Runs")
        if bx:
            ax.plot(
                bx,
                by,
                color="tab:red",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="_nolegend_",
            )
        if p_eff_low_list is not None:
            ax.axvline(
                p_eff_low,
                color="tab:orange",
                linestyle="--",
                linewidth=1.8,
                label="Cosine",
            )
        if p_eff_high_list is not None:
            ax.axvline(
                p_eff_high,
                color="tab:purple",
                linestyle="--",
                linewidth=1.8,
                label="Best Probe",
            )
        if best_retain_list is not None:
            best_retain = best_retain_list[idx]
            ax.axhline(best_retain, color="tab:green", linestyle=":", linewidth=1.8, label="baseline")
            all_project_ys.append(best_retain)

        title_base = titles_list[idx] if titles_list is not None else project
        ax.set_title(title_base)
        ax.set_xlabel(args.x_metric)
        ax.set_ylabel(args.y_metric)
        ax.set_xlim(x_min, x_max)
        if args.x_tick_step is not None and args.x_tick_step > 0:
            step = args.x_tick_step
            start_tick = math.ceil(x_min / step) * step
            ticks = []
            t = start_tick
            while t <= x_max + 1e-12:
                ticks.append(round(t, 10))
                t += step
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{v:.2f}" for v in ticks])
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        plotted_axes.append(ax)

    if args.shared_y_axis and plotted_axes:
        if args.y_min is not None and args.y_max is not None:
            y_min, y_max = args.y_min, args.y_max
        elif args.y_min is not None:
            _, inferred_max = infer_y_range(all_project_ys, pad_frac=args.ypad_frac)
            y_min, y_max = args.y_min, inferred_max
        elif args.y_max is not None:
            inferred_min, _ = infer_y_range(all_project_ys, pad_frac=args.ypad_frac)
            y_min, y_max = inferred_min, args.y_max
        else:
            y_min, y_max = infer_y_range(all_project_ys, pad_frac=args.ypad_frac)
        for ax in plotted_axes:
            ax.set_ylim(y_min, y_max)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
