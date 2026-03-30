#!/usr/bin/env python3
"""
Analyze single-word free-form outputs from report_qa_mmlu_singleword.py.

Features:
1) Deterministic auto-judging from saved JSON results.
2) Export ambiguous cases for manual arbitration.
3) Optional regression mining under MCQ constraint:
   base MCQ correct AND RMU MCQ correct AND base free-form correct AND RMU free-form incorrect.
"""
import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")
EXCEPT_RE = re.compile(r"\bexcept\b", flags=re.IGNORECASE)


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _extract_words(s: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(str(s))]


def _qkey(question: str, correct_answer: str) -> str:
    return f"{_norm_text(question)} || {_norm_text(correct_answer)}"


def _norm_model_key(s: str) -> str:
    txt = str(s).strip()
    if not txt:
        return txt
    txt = txt.rstrip("/\\")
    txt = os.path.normpath(txt)
    if txt.startswith("./"):
        txt = txt[2:]
    return txt


def _filter_except_questions(details: List[Dict], exclude_except_questions: bool) -> Tuple[List[Dict], int]:
    if not exclude_except_questions:
        return details, 0
    kept = []
    dropped = 0
    for ex in details:
        q = str(ex.get("question", ""))
        if EXCEPT_RE.search(q):
            dropped += 1
            continue
        kept.append(ex)
    return kept, dropped


def _load_model_details(path: str, model_key: Optional[str]) -> Tuple[str, List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected dict of model->details in {path}")

    if model_key:
        if model_key in payload:
            key = model_key
        else:
            target = _norm_model_key(model_key)
            matches = [k for k in payload.keys() if _norm_model_key(k) == target]
            if len(matches) == 1:
                key = matches[0]
            else:
                raise KeyError(f"Model key {model_key!r} not found in {path}. Available: {list(payload.keys())}")
    else:
        if len(payload) != 1:
            raise ValueError(
                f"{path} has multiple model keys {list(payload.keys())}; pass --*_model_key explicitly."
            )
        key = next(iter(payload))

    details = payload[key]
    if not isinstance(details, list):
        raise ValueError(f"Expected list for model key {key!r} in {path}")
    return key, details


def _compute_summary_from_judged(judged_by_key: Dict[str, Dict]) -> Dict:
    counts = {"correct": 0, "incorrect": 0, "ambiguous": 0, "total": 0}
    for ex in judged_by_key.values():
        st = ex.get("judge_status", "incorrect")
        if st not in counts:
            st = "incorrect"
        counts[st] += 1
        counts["total"] += 1
    decided = counts["correct"] + counts["incorrect"]
    return {
        "counts": counts,
        "auto_accuracy_conservative": (counts["correct"] / counts["total"]) if counts["total"] else 0.0,
        "auto_accuracy_optimistic": ((counts["correct"] + counts["ambiguous"]) / counts["total"]) if counts["total"] else 0.0,
        "auto_accuracy_decided_only": (counts["correct"] / decided) if decided else 0.0,
    }


def _load_manual_rules(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return payload["rules"]
    if isinstance(payload, list):
        return payload
    raise ValueError("manual adjudication file must be a list of rules or {\"rules\": [...]} format.")


def _apply_manual_rules(
    judged_by_key: Dict[str, Dict],
    rules: List[Dict],
    which: str,
    model_label: str,
) -> Dict[str, int]:
    """
    Apply manual overrides in-place.
    Rule fields:
      - question (required)
      - correct_answer (required)
      - decision: correct|incorrect|ambiguous (required)
      - which: base|rmu|both (optional, default both)
      - model_key: exact model key string (optional)
      - generated_answer: exact expected generated answer after normalization (optional)
      - note: optional text
    """
    stats = {"applied": 0, "skipped": 0, "no_match": 0}
    for r in rules:
        rw = str(r.get("which", "both")).lower()
        if rw not in {"base", "rmu", "both"}:
            stats["skipped"] += 1
            continue
        if rw != "both" and rw != which:
            continue

        rm = r.get("model_key")
        if rm is not None and str(rm) != str(model_label):
            continue

        q = r.get("question")
        c = r.get("correct_answer")
        d = str(r.get("decision", "")).lower()
        if not q or not c or d not in {"correct", "incorrect", "ambiguous"}:
            stats["skipped"] += 1
            continue

        key = _qkey(str(q), str(c))
        ex = judged_by_key.get(key)
        if ex is None:
            stats["no_match"] += 1
            continue

        rg = r.get("generated_answer")
        if rg is not None:
            if _norm_text(str(ex.get("generated_answer", ""))) != _norm_text(str(rg)):
                stats["no_match"] += 1
                continue

        ex["judge_status"] = d
        ex["judge_reason"] = "manual_override"
        if r.get("note"):
            ex["judge_note"] = str(r.get("note"))
        stats["applied"] += 1
    return stats


def _judge_entry(correct_answer: str, generated_answer: str) -> Dict:
    gold_words = _extract_words(correct_answer)
    pred_words = _extract_words(generated_answer)
    gold = gold_words[0] if gold_words else ""

    if not gold:
        return {"status": "ambiguous", "reason": "missing_gold_word", "gold": gold, "pred_first": ""}
    if not pred_words:
        return {"status": "incorrect", "reason": "empty_prediction", "gold": gold, "pred_first": ""}

    pred_first = pred_words[0]
    if pred_first == gold:
        return {"status": "correct", "reason": "first_word_match", "gold": gold, "pred_first": pred_first}

    # Single-word wrong answer can be auto-marked incorrect.
    if len(pred_words) == 1:
        return {"status": "incorrect", "reason": "single_word_mismatch", "gold": gold, "pred_first": pred_first}

    # Multi-word responses are often explanations; keep for arbitration.
    if gold in pred_words:
        return {"status": "ambiguous", "reason": "gold_present_not_first", "gold": gold, "pred_first": pred_first}
    return {"status": "ambiguous", "reason": "multi_word_mismatch", "gold": gold, "pred_first": pred_first}


def _judge_details(details: List[Dict], model_label: str) -> Tuple[Dict[str, Dict], Dict]:
    judged_by_key: Dict[str, Dict] = {}
    for ex in details:
        q = ex.get("question", "")
        c = ex.get("correct_answer", "")
        g = ex.get("generated_answer", "")
        key = _qkey(q, c)
        verdict = _judge_entry(c, g)
        judged = {
            "model": model_label,
            "question": q,
            "correct_answer": c,
            "generated_answer": g,
            "judge_status": verdict["status"],
            "judge_reason": verdict["reason"],
            "gold_word": verdict["gold"],
            "pred_first_word": verdict["pred_first"],
        }
        judged_by_key[key] = judged
    summary = _compute_summary_from_judged(judged_by_key)
    return judged_by_key, summary


def _build_mcq_correct_map(
    model_path: str,
    batch_size: int,
    max_length: int,
    exclude_except_questions: bool,
) -> Dict[str, bool]:
    from rmu.eval_mcq import load_mmlu_bio_single_word, get_mcq_predictions  # lazy import
    from rmu.utils import load_model  # lazy import

    print(f"Loading model for MCQ correctness: {model_path}")
    model, tok = load_model(model_path)
    q, choices, answers, correct_texts = load_mmlu_bio_single_word(
        split="test",
        exclude_except_questions=exclude_except_questions,
    )
    preds = get_mcq_predictions(
        model=model,
        tokenizer=tok,
        questions=q,
        choices=choices,
        batch_size=batch_size,
        max_length=max_length,
    )
    out = {}
    for qi, ai, ci, pi in zip(q, answers, correct_texts, preds):
        out[_qkey(qi, ci)] = int(pi) == int(ai)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge single-word free-form outputs and mine RMU regressions.")
    parser.add_argument("--base_freeform_json", type=str, required=True)
    parser.add_argument("--rmu_freeform_json", type=str, required=True)
    parser.add_argument("--base_model_key", type=str, default="", help="Model key inside base JSON.")
    parser.add_argument("--rmu_model_key", type=str, default="", help="Model key inside RMU JSON.")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--compute_mcq_filter", action="store_true", help="Compute per-question MCQ correctness by loading models.")
    parser.add_argument("--base_model_path", type=str, default="", help="HF ID or local path for base model (used with --compute_mcq_filter).")
    parser.add_argument("--rmu_model_path", type=str, default="", help="HF ID or local path for RMU model (used with --compute_mcq_filter).")
    parser.add_argument("--mcq_batch_size", type=int, default=8)
    parser.add_argument("--mcq_max_length", type=int, default=512)
    parser.add_argument(
        "--manual_adjudication_json",
        type=str,
        default="",
        help="Optional JSON file with manual override rules.",
    )
    parser.add_argument(
        "--include_except_questions",
        action="store_true",
        help="If set, include questions containing 'EXCEPT'. Default excludes them.",
    )
    args = parser.parse_args()
    exclude_except = not args.include_except_questions

    os.makedirs(args.out_dir, exist_ok=True)

    base_key, base_details = _load_model_details(args.base_freeform_json, args.base_model_key or None)
    rmu_key, rmu_details = _load_model_details(args.rmu_freeform_json, args.rmu_model_key or None)
    base_details, base_except_dropped = _filter_except_questions(base_details, exclude_except)
    rmu_details, rmu_except_dropped = _filter_except_questions(rmu_details, exclude_except)

    base_judged, base_summary = _judge_details(base_details, model_label=base_key)
    rmu_judged, rmu_summary = _judge_details(rmu_details, model_label=rmu_key)

    manual_stats = {}
    if args.manual_adjudication_json:
        rules = _load_manual_rules(args.manual_adjudication_json)
        base_stats = _apply_manual_rules(base_judged, rules, which="base", model_label=base_key)
        rmu_stats = _apply_manual_rules(rmu_judged, rules, which="rmu", model_label=rmu_key)
        base_summary = _compute_summary_from_judged(base_judged)
        rmu_summary = _compute_summary_from_judged(rmu_judged)
        manual_stats = {"base": base_stats, "rmu": rmu_stats, "rules_count": len(rules)}

    common_keys = sorted(set(base_judged.keys()) & set(rmu_judged.keys()))
    ambiguous_cases = []
    for key in common_keys:
        b = base_judged[key]
        r = rmu_judged[key]
        if b["judge_status"] == "ambiguous":
            ambiguous_cases.append({"which": "base", **b})
        if r["judge_status"] == "ambiguous":
            ambiguous_cases.append({"which": "rmu", **r})

    regressions = []
    rmu_mcq_correct_freeform_incorrect = []
    mcq_info = {"enabled": bool(args.compute_mcq_filter)}
    if args.compute_mcq_filter:
        base_model_path = args.base_model_path or base_key
        rmu_model_path = args.rmu_model_path or rmu_key
        base_mcq_ok = _build_mcq_correct_map(
            base_model_path,
            args.mcq_batch_size,
            args.mcq_max_length,
            exclude_except_questions=exclude_except,
        )
        rmu_mcq_ok = _build_mcq_correct_map(
            rmu_model_path,
            args.mcq_batch_size,
            args.mcq_max_length,
            exclude_except_questions=exclude_except,
        )
        for key in common_keys:
            b = base_judged[key]
            r = rmu_judged[key]
            if (
                base_mcq_ok.get(key, False)
                and rmu_mcq_ok.get(key, False)
                and b["judge_status"] == "correct"
                and r["judge_status"] == "incorrect"
            ):
                regressions.append(
                    {
                        "question": b["question"],
                        "correct_answer": b["correct_answer"],
                        "base_generated_answer": b["generated_answer"],
                        "rmu_generated_answer": r["generated_answer"],
                        "base_judge_reason": b["judge_reason"],
                        "rmu_judge_reason": r["judge_reason"],
                        "base_mcq_correct": True,
                        "rmu_mcq_correct": True,
                    }
                )
        for key, r in rmu_judged.items():
            if rmu_mcq_ok.get(key, False) and r["judge_status"] == "incorrect":
                rmu_mcq_correct_freeform_incorrect.append(
                    {
                        "question": r["question"],
                        "correct_answer": r["correct_answer"],
                        "rmu_generated_answer": r["generated_answer"],
                        "rmu_judge_reason": r["judge_reason"],
                        "rmu_mcq_correct": True,
                    }
                )
        mcq_info["base_model_path"] = base_model_path
        mcq_info["rmu_model_path"] = rmu_model_path
        mcq_info["base_mcq_covered"] = len(base_mcq_ok)
        mcq_info["rmu_mcq_covered"] = len(rmu_mcq_ok)

    summary = {
        "base_model_key": base_key,
        "rmu_model_key": rmu_key,
        "exclude_except_questions": exclude_except,
        "base_except_questions_dropped": base_except_dropped,
        "rmu_except_questions_dropped": rmu_except_dropped,
        "base_summary": base_summary,
        "rmu_summary": rmu_summary,
        "common_question_count": len(common_keys),
        "ambiguous_case_count": len(ambiguous_cases),
        "regression_count": len(regressions),
        "rmu_mcq_correct_freeform_incorrect_count": len(rmu_mcq_correct_freeform_incorrect),
        "mcq_filter": mcq_info,
        "manual_adjudication": manual_stats,
    }

    with open(os.path.join(args.out_dir, "judged_accuracy_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(args.out_dir, "ambiguous_cases.jsonl"), "w", encoding="utf-8") as f:
        for rec in ambiguous_cases:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(os.path.join(args.out_dir, "regression_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(regressions, f, indent=2, ensure_ascii=False)

    with open(
        os.path.join(args.out_dir, "rmu_mcq_correct_freeform_incorrect.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(rmu_mcq_correct_freeform_incorrect, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.out_dir, "judged_examples_by_model.json"), "w", encoding="utf-8") as f:
        out = {
            "base": [base_judged[k] for k in common_keys],
            "rmu": [rmu_judged[k] for k in common_keys],
        }
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote summary to {os.path.join(args.out_dir, 'judged_accuracy_summary.json')}")
    print(f"Wrote ambiguous cases to {os.path.join(args.out_dir, 'ambiguous_cases.jsonl')}")
    print(f"Wrote regression candidates to {os.path.join(args.out_dir, 'regression_candidates.json')}")
    print(
        "Wrote RMU MCQ-correct/free-form-incorrect cases to "
        f"{os.path.join(args.out_dir, 'rmu_mcq_correct_freeform_incorrect.json')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
