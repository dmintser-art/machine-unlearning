"""
RMU training loop with checkpoint saving and MCQ evaluation on forget (WMDP) and retain (MMLU) sets.
Official hyperparameters (from WMDP repo / paper):
  lr=5e-5, alpha (retain weight), steering_coeffs, batch_size=4, max_num_batches=80,
  layer_id=7, layer_ids=[5,6,7], param_ids=[9] for Qwen2 (mlp.down_proj.weight)
  For multi-model safety, prefer param_names (e.g. mlp.down_proj.weight).
  Zephyr notebook: alpha=1200, steering_coeffs=6.5, max_num_batches=150
"""
import os
import json
import datetime
import argparse
import numpy as np
import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .utils import (
    set_seed,
    load_model,
    get_params,
    forward_with_cache,
    resolve_decoder_layers,
    get_data,
)
from .eval_mcq_old import eval_wmdp, eval_mmlu, eval_medmcqa, eval_mcq_json


def _resolve_model_hidden_size(model):
    """Resolve model hidden size across model families (Qwen/Llama/Gemma/etc.)."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "n_embd"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        for subcfg_name in ("text_config", "language_config", "decoder_config", "llm_config"):
            subcfg = getattr(cfg, subcfg_name, None)
            if subcfg is None:
                continue
            for attr in ("hidden_size", "d_model", "n_embd"):
                value = getattr(subcfg, attr, None)
                if isinstance(value, int) and value > 0:
                    return value

    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return int(emb.weight.shape[-1])
    except Exception:
        pass

    try:
        layers, _ = resolve_decoder_layers(model)
        if len(layers) > 0:
            layer0 = layers[0]
            if hasattr(layer0, "input_layernorm") and hasattr(layer0.input_layernorm, "weight"):
                return int(layer0.input_layernorm.weight.shape[0])
            mlp = getattr(layer0, "mlp", None)
            down_proj = getattr(mlp, "down_proj", None)
            if down_proj is not None and hasattr(down_proj, "weight"):
                return int(down_proj.weight.shape[0])
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve hidden size for model class {model.__class__.__name__}."
    )


def _prepare_batch_inputs(tokenizer, texts, model, max_length):
    """Tokenize a text batch and ensure required fields (e.g., token_type_ids for Gemma3)."""
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_token_type_ids=True,
    )
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if "token_type_ids" not in tokenized and model_type == "gemma3":
        tokenized["token_type_ids"] = torch.zeros_like(tokenized["input_ids"], dtype=torch.long)
    return tokenized


def _get_model_input_device(model):
    """Best-effort device for model inputs under device_map='auto'."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


def run_rmu_with_eval(
    updated_model,
    frozen_model,
    tokenizer,
    forget_data_list,
    retain_data_list,
    args,
):
    """
    Run RMU for max_num_batches.
    Performs a baseline eval at step 0, then every checkpoint_interval batches
    saves checkpoint and evaluates on WMDP (forget) and MMLU/MedMCQA (retain).
    Logs to wandb if enabled.
    """
    try:
        import wandb
        use_wandb = getattr(args, "use_wandb", False)
    except Exception:
        use_wandb = False

    rmu_config = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    print("===== RMU Config =====")
    for k, v in rmu_config.items():
        print(f"  {k}={v}")
    print("=====")

    updated_model = updated_model.train()
    params, selected_params = get_params(
        updated_model,
        args.layer_ids,
        param_ids=args.param_ids,
        param_names=getattr(args, "param_names", None),
    )
    if not params:
        raise ValueError(
            f"No parameters selected for layer_ids={args.layer_ids}, "
            f"param_ids={args.param_ids}, param_names={getattr(args, 'param_names', [])}."
        )
    print("Trainable parameters:")
    for item in selected_params:
        print(
            f"  layer={item['layer_path']}[{item['layer_id']}] param_id={item['param_id']} "
            f"name={item['name']} shape={item['shape']} selected_by={item['selected_by']}"
        )
    if any(item["name"] == "self_attn.o_proj.weight" for item in selected_params):
        print(
            "WARNING: self_attn.o_proj.weight is selected. If you intended Qwen MLP down-proj, "
            "use --param_ids 9."
        )
    optimizer = AdamW(params, lr=args.lr)
    try:
        frozen_layers, frozen_layers_path = resolve_decoder_layers(frozen_model)
        updated_layers, updated_layers_path = resolve_decoder_layers(updated_model)
        if args.layer_id < 0 or args.layer_id >= len(updated_layers):
            raise ValueError(
                f"layer_id={args.layer_id} out of range for updated model "
                f"({updated_layers_path}, num_layers={len(updated_layers)})."
            )
        if args.layer_id < 0 or args.layer_id >= len(frozen_layers):
            raise ValueError(
                f"layer_id={args.layer_id} out of range for frozen model "
                f"({frozen_layers_path}, num_layers={len(frozen_layers)})."
            )
        frozen_module = frozen_layers[args.layer_id]
        updated_module = updated_layers[args.layer_id]
        print(f"Hook module path (frozen): {frozen_layers_path}[{args.layer_id}]")
        print(f"Hook module path (updated): {updated_layers_path}[{args.layer_id}]")
    except Exception as e:
        if getattr(args, "module_str", None):
            print(f"Auto layer resolution failed ({e}); falling back to module_str.")
            frozen_module = eval(
                args.module_str.format(model_name="frozen_model", layer_id=args.layer_id)
            )
            updated_module = eval(
                args.module_str.format(model_name="updated_model", layer_id=args.layer_id)
            )
        else:
            raise

    hidden_size = _resolve_model_hidden_size(updated_model)
    try:
        hook_device = next(updated_module.parameters()).device
    except StopIteration:
        hook_device = next(updated_model.parameters()).device
    try:
        frozen_hook_device = next(frozen_module.parameters()).device
    except StopIteration:
        frozen_hook_device = next(frozen_model.parameters()).device
    updated_input_device = _get_model_input_device(updated_model)
    frozen_input_device = _get_model_input_device(frozen_model)
    print(f"Resolved hidden_size={hidden_size}; control vectors on device={hook_device}.")

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(
            1, 1, hidden_size,
            dtype=updated_model.dtype,
            device=hook_device,
        )
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )
    checkpoint_interval = getattr(args, "checkpoint_interval", max(1, num_batches // 10))
    output_dir = getattr(args, "output_dir", None) or f"models/rmu_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    if use_wandb and output_dir:
        try:
            run_id = wandb.run.id
            output_dir = os.path.join(output_dir, run_id)
        except Exception:
            pass
    os.makedirs(output_dir, exist_ok=True)

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    # Metrics over steps (for plotting)
    steps_log = []  # list of {step, forget_acc, retain_acc, loss, ...}
    stop_on_target = bool(getattr(args, "stop_on_target", False))
    target_forget_max = getattr(args, "target_forget_max", None)
    target_retain_min = getattr(args, "target_retain_min", None)
    target_retain_metric = str(getattr(args, "target_retain_metric", "mmlu")).lower()
    target_hit = False
    target_hit_step = None
    target_hit_metrics = {}
    target_checkpoint_path = None

    if stop_on_target and target_forget_max is None and target_retain_min is None:
        raise ValueError(
            "--stop_on_target requires at least one of --target_forget_max or --target_retain_min."
        )
    if target_retain_metric not in {"mmlu", "medmcqa"}:
        raise ValueError("--target_retain_metric must be 'mmlu' or 'medmcqa'.")
    if stop_on_target and target_retain_min is not None:
        if target_retain_metric == "mmlu" and getattr(args, "skip_mmlu_eval", False):
            raise ValueError(
                "Target retain metric is MMLU, but --skip_mmlu_eval is set. Disable skip or change target metric."
            )
        if target_retain_metric == "medmcqa" and getattr(args, "skip_medmcqa_eval", False):
            raise ValueError(
                "Target retain metric is MedMCQA, but --skip_medmcqa_eval is set. Disable skip or change target metric."
            )

    def evaluate_and_log(step, loss_value=None, phase="checkpoint"):
        forget_eval_source = getattr(args, "forget_eval_source", "wmdp")
        if forget_eval_source == "json":
            forget_acc = eval_mcq_json(
                updated_model,
                tokenizer,
                path=getattr(args, "forget_eval_path"),
                batch_size=getattr(args, "eval_batch_size", 4),
                max_length=getattr(args, "eval_max_length", 512),
            )
            forget_label = f"MCQ-JSON ({getattr(args, 'forget_eval_path')})"
        else:
            forget_acc = eval_wmdp(
                updated_model, tokenizer,
                config=getattr(args, "wmdp_eval_config", "wmdp-bio"),
                batch_size=getattr(args, "eval_batch_size", 4),
                max_length=getattr(args, "eval_max_length", 512),
            )
            forget_label = f"WMDP ({getattr(args, 'wmdp_eval_config', 'wmdp-bio')})"
        mmlu_acc = None
        if not getattr(args, "skip_mmlu_eval", False):
            mmlu_acc = eval_mmlu(
                updated_model, tokenizer,
                config=getattr(args, "mmlu_eval_config", "all"),
                batch_size=getattr(args, "eval_batch_size", 4),
                max_length=getattr(args, "eval_max_length", 512),
            )
        medmcqa_acc = None
        if not getattr(args, "skip_medmcqa_eval", False):
            medmcqa_acc = eval_medmcqa(
                updated_model, tokenizer,
                split=getattr(args, "medmcqa_eval_split", "validation"),
                batch_size=getattr(args, "eval_batch_size", 4),
                max_length=getattr(args, "eval_max_length", 512),
            )
        entry = {
            "step": step,
            "phase": phase,
            "forget_acc": round(float(forget_acc), 4),
            "retain_acc": round(float(mmlu_acc), 4) if mmlu_acc is not None else None,  # backward compat for plot script
            "retain_acc_mmlu": round(float(mmlu_acc), 4) if mmlu_acc is not None else None,
            "retain_acc_medmcqa": round(float(medmcqa_acc), 4) if medmcqa_acc is not None else None,
            "loss": round(float(loss_value), 6) if loss_value is not None else None,
        }
        steps_log.append(entry)
        msg = [f"  [{phase} {step}] {forget_label} (forget)={forget_acc:.4f}"]
        if mmlu_acc is not None:
            msg.append(f"MMLU={mmlu_acc:.4f}")
        else:
            msg.append("MMLU=SKIPPED")
        if medmcqa_acc is not None:
            msg.append(f"MedMCQA={medmcqa_acc:.4f}")
        else:
            msg.append("MedMCQA=SKIPPED")
        print(" | ".join(msg))
        if use_wandb:
            wandb_log = {
                "forget_acc": forget_acc,
                "step": step,
            }
            if mmlu_acc is not None:
                wandb_log["retain_acc_mmlu"] = mmlu_acc
            if medmcqa_acc is not None:
                wandb_log["retain_acc_medmcqa"] = medmcqa_acc
            if loss_value is not None:
                wandb_log["loss"] = float(loss_value)
            wandb.log(wandb_log, step=step)
        return entry

    if getattr(args, "skip_baseline_eval", False):
        print("Skipping baseline evaluation at step=0.")
    else:
        print("Running baseline evaluation before RMU updates (step=0)...")
        evaluate_and_log(step=0, loss_value=None, phase="baseline")

    n_topics = len(forget_data_list)
    for batch_idx in tqdm(range(num_batches), desc="RMU"):
        topic_idx = batch_idx % n_topics
        batch_in_topic = (batch_idx // n_topics) % len(forget_data_list[topic_idx])
        control_vec = control_vectors_list[topic_idx]
        unlearn_batch = forget_data_list[topic_idx][batch_in_topic]
        retain_batch = retain_data_list[topic_idx][batch_in_topic % len(retain_data_list[topic_idx])]

        max_length = 512 if topic_idx == 0 else 768
        unlearn_inputs = _prepare_batch_inputs(
            tokenizer=tokenizer,
            texts=unlearn_batch,
            model=updated_model,
            max_length=max_length,
        )
        unlearn_inputs = {k: v.to(updated_input_device) for k, v in unlearn_inputs.items()}
        updated_forget_activations = forward_with_cache(
            updated_model, unlearn_inputs, module=updated_module, no_grad=False
        )
        # Expand control_vec (1, 1, D) to (B, T, D) to match activations and avoid broadcast warning
        control_vec_expanded = control_vec.expand_as(updated_forget_activations)
        unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, control_vec_expanded)

        retain_inputs = _prepare_batch_inputs(
            tokenizer=tokenizer,
            texts=retain_batch,
            model=updated_model,
            max_length=512,
        )
        retain_inputs_updated = {k: v.to(updated_input_device) for k, v in retain_inputs.items()}
        retain_inputs_frozen = {k: v.to(frozen_input_device) for k, v in retain_inputs.items()}
        updated_retain_activations = forward_with_cache(
            updated_model, retain_inputs_updated, module=updated_module, no_grad=False
        )
        frozen_retain_activations = forward_with_cache(
            frozen_model, retain_inputs_frozen, module=frozen_module, no_grad=True
        ).to(updated_retain_activations.device)
        retain_loss = torch.nn.functional.mse_loss(
            updated_retain_activations, frozen_retain_activations
        )
        retain_loss = retain_loss * args.alpha[topic_idx]

        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step = batch_idx + 1
        if args.verbose:
            print(f"step={step} loss={loss.item():.4g} unlearn={unlearn_loss.item():.4g} retain={retain_loss.item():.4g}")

        # Checkpoint: evaluate (and optionally save model weights)
        if step % checkpoint_interval == 0 or step == num_batches:
            if getattr(args, "save_checkpoints", False):
                ckpt_path = os.path.join(output_dir, f"checkpoint-{step}")
                updated_model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

            # Eval: forget = WMDP-bio MCQ; retain = MMLU + MedMCQA
            entry = evaluate_and_log(step=step, loss_value=loss.item(), phase="checkpoint")
            if stop_on_target:
                retain_value = (
                    entry.get("retain_acc_medmcqa")
                    if target_retain_metric == "medmcqa"
                    else entry.get("retain_acc_mmlu")
                )
                forget_ok = (
                    target_forget_max is None or
                    (entry.get("forget_acc") is not None and entry["forget_acc"] <= target_forget_max)
                )
                retain_ok = (
                    target_retain_min is None or
                    (retain_value is not None and retain_value >= target_retain_min)
                )
                if forget_ok and retain_ok:
                    target_hit = True
                    target_hit_step = step
                    target_hit_metrics = {
                        "forget_acc": entry.get("forget_acc"),
                        "retain_acc_mmlu": entry.get("retain_acc_mmlu"),
                        "retain_acc_medmcqa": entry.get("retain_acc_medmcqa"),
                    }
                    print(
                        f"Target reached at step={step}: forget_acc={entry.get('forget_acc')}, "
                        f"retain({target_retain_metric})={retain_value}"
                    )
                    if getattr(args, "save_target_checkpoint", True):
                        target_checkpoint_path = os.path.join(output_dir, f"target-hit-step-{step}")
                        updated_model.save_pretrained(target_checkpoint_path)
                        tokenizer.save_pretrained(target_checkpoint_path)
                        print(f"Saved target checkpoint to {target_checkpoint_path}")
                    if use_wandb:
                        wandb.log(
                            {
                                "target_hit": 1,
                                "target_hit_step": step,
                                "target_forget_max": target_forget_max,
                                "target_retain_min": target_retain_min,
                            },
                            step=step,
                        )
                    break

    tokenizer.truncation_side = truncation_side
    # Optional final save (disabled by default to avoid large disk usage in sweeps)
    if getattr(args, "save_final_model", False):
        final_path = os.path.join(output_dir, "final")
        updated_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

    # Save steps log for plotting
    log_path = os.path.join(output_dir, "eval_steps.json")
    run_metadata = {
        "stop_on_target": stop_on_target,
        "target_forget_max": target_forget_max,
        "target_retain_min": target_retain_min,
        "target_retain_metric": target_retain_metric,
        "target_hit": target_hit,
        "target_hit_step": target_hit_step,
        "target_hit_metrics": target_hit_metrics,
        "target_checkpoint_path": target_checkpoint_path,
    }
    with open(log_path, "w") as f:
        json.dump({"steps": steps_log, "config": rmu_config, "run_metadata": run_metadata}, f, indent=2)
    print(f"Saved eval trajectory to {log_path}")
    return steps_log, output_dir


def get_args():
    parser = argparse.ArgumentParser(description="RMU with checkpoint evaluation (WMDP forget, MMLU retain)")
    # Model
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--module_str",
        type=str,
        default="{model_name}.model.layers[{layer_id}]",
        help="Optional fallback module expression used only if auto layer resolution fails.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    # Data (forget = WMDP-bio; retain = all MMLU auxiliary by default)
    parser.add_argument(
        "--retain_corpora",
        type=str,
        default="mmlu-auxiliary-all",
        help="Retain corpora: mmlu-auxiliary-all (all 3 configs), or economics-corpus, law-corpus, physics-corpus, or wikitext,wikitext.",
    )
    parser.add_argument(
        "--forget_corpora",
        type=str,
        default="bio-forget-corpus",
        help="Forget corpora: bio-forget-corpus (WMDP-bio only) or bio-forget-corpus,cyber-forget-corpus.",
    )
    # RMU hyperparameters (WMDP paper / Appendix B.4)
    parser.add_argument("--alpha", type=str, default="1200", help="Retain loss weight (one per forget corpus)")
    parser.add_argument("--steering_coeffs", type=str, default="6.5", help="Steering vector scale (one per forget corpus)")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=150)
    parser.add_argument("--layer_id", type=int, default=7)
    parser.add_argument("--layer_ids", type=str, default="5,6,7")
    parser.add_argument(
        "--param_ids",
        type=str,
        default="9",
        help="Comma-separated parameter indices within each decoder layer. "
             "For Qwen2, param_id 9 corresponds to mlp.down_proj.weight (param_id 6 is self_attn.o_proj.weight).",
    )
    parser.add_argument(
        "--param_names",
        type=str,
        default="",
        help="Comma-separated parameter names within each decoder layer, e.g. mlp.down_proj.weight. "
             "If provided, these are selected in addition to param_ids.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    # Checkpoint & eval
    parser.add_argument("--checkpoint_interval", type=int, default=15, help="Evaluate every N batches")
    parser.add_argument("--save_checkpoints", action="store_true", help="If set, save model/tokenizer at each checkpoint interval.")
    parser.add_argument("--save_final_model", action="store_true", help="If set, save model/tokenizer at the end of training.")
    parser.add_argument("--wmdp_eval_config", type=str, default="wmdp-bio", help="WMDP MCQ eval: wmdp-bio (forget set)")
    parser.add_argument(
        "--forget_eval_source",
        type=str,
        default="wmdp",
        choices=["wmdp", "json"],
        help="Forget-set evaluation source: WMDP Hub dataset or local MCQ JSON.",
    )
    parser.add_argument(
        "--forget_eval_path",
        type=str,
        default="",
        help="Path to local MCQ JSON used when --forget_eval_source json.",
    )
    parser.add_argument(
        "--mmlu_eval_config",
        type=str,
        default="all",
        help="MMLU config(s) for retain eval: 'all', one subject, or comma-separated subjects.",
    )
    parser.add_argument("--medmcqa_eval_split", type=str, default="validation", choices=["validation", "test"], help="MedMCQA split for retain eval")
    parser.add_argument("--wmdp_only_eval", action="store_true", help="Evaluate only on WMDP-bio at checkpoints (skip MMLU and MedMCQA).")
    parser.add_argument("--skip_mmlu_eval", action="store_true", help="Skip MMLU checkpoint evaluation.")
    parser.add_argument("--skip_medmcqa_eval", action="store_true", help="Skip MedMCQA checkpoint evaluation.")
    parser.add_argument("--skip_baseline_eval", action="store_true", help="Skip initial step=0 baseline evaluation.")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_length", type=int, default=512)
    parser.add_argument(
        "--stop_on_target",
        action="store_true",
        help="Stop training early when target metric thresholds are met at a checkpoint eval.",
    )
    parser.add_argument(
        "--target_forget_max",
        type=float,
        default=None,
        help="Target threshold: stop when forget_acc <= this value.",
    )
    parser.add_argument(
        "--target_retain_min",
        type=float,
        default=None,
        help="Target threshold: stop when retain metric >= this value.",
    )
    parser.add_argument(
        "--target_retain_metric",
        type=str,
        default="mmlu",
        choices=["mmlu", "medmcqa"],
        help="Retain metric used with --target_retain_min.",
    )
    parser.add_argument(
        "--save_target_checkpoint",
        action="store_true",
        default=True,
        help="Save model/tokenizer to output_dir/target-hit-step-<step> when target is hit.",
    )
    parser.add_argument(
        "--no_save_target_checkpoint",
        dest="save_target_checkpoint",
        action="store_false",
        help="Disable saving target-hit checkpoint.",
    )
    # W&B
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="rmu-unlearning")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    args.retain_corpora = [s.strip() for s in args.retain_corpora.split(",")]
    args.forget_corpora = [s.strip() for s in args.forget_corpora.split(",")]
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(x) for x in args.layer_ids.split(",")]
    args.param_ids = [int(x) for x in args.param_ids.split(",") if str(x).strip() != ""]
    args.param_names = [x.strip() for x in args.param_names.split(",") if x.strip()]
    if args.forget_eval_source == "json" and not args.forget_eval_path:
        raise ValueError("--forget_eval_path is required when --forget_eval_source json")
    if args.wmdp_only_eval:
        args.skip_mmlu_eval = True
        args.skip_medmcqa_eval = True
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
    if getattr(args, "use_wandb", False):
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora,
        args.retain_corpora,
        args.min_len,
        args.max_len,
        args.batch_size,
    )
    run_rmu_with_eval(
        updated_model,
        frozen_model,
        tokenizer,
        forget_data_list,
        retain_data_list,
        args,
    )
