# Beyond Random Guessing: Representation-Level Evaluation of Machine Unlearning

This repository implements **RMU (Representation Misdirection for Unlearning)** with checkpoint evaluation, embedding-based probe baselines, and MCQ/free-text evaluation across multiple LLMs. Evaluated on **WMDP-bio**, **HP (Harry Potter)**, **MMLU**, and **MedMCQA** benchmarks.

Supported models: **Qwen 2.5 7B**, **Llama 3.1 8B**, **Gemma 2 9B**, **Gemma 3 4B**, **Qwen 3.5 9B** (with SBERT as an additional embedding baseline).

---

## Setup

### Requirements

- Python 3.8+ (Python 3.10+ for Qwen 3.5)
- PyTorch, transformers, datasets, sentence-transformers, scikit-learn, tqdm, numpy

```bash
pip install torch transformers datasets sentence-transformers scikit-learn tqdm numpy accelerate
```

### Hugging Face Authentication

Several datasets and models are gated. Accept terms and log in:

```bash
huggingface-cli login
```

Required access:
- [cais/wmdp-bio-forget-corpus](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus)
- [cais/wmdp-mmlu-auxiliary-corpora](https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora)
- [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Optional: `pip install wandb` for W&B sweep logging.

---

## Project Structure

### Core Package: `rmu/`

| File | Description |
|------|-------------|
| `rmu/utils.py` | Model loading, activation caching, data loading (WMDP forget/retain corpora, wikitext, CRISP HP). |
| `rmu/unlearn.py` | RMU training loop with checkpoint saving and MCQ evaluation on forget/retain sets. |
| `rmu/eval_mcq.py` | MCQ evaluation (WMDP, MMLU, MedMCQA, HP) via chat-template prompts and next-token logits over A/B/C/D. Includes MMLU biology single-word subset loading, free-text generation evaluation. |
| `rmu/eval_mcq_old.py` | Legacy MCQ evaluator kept for comparison. |

### RMU Training and Sweeps

| File | Description |
|------|-------------|
| `run_rmu_with_eval.py` | CLI entrypoint for RMU training with checkpoint evaluation and optional W&B logging. |
| `run_rmu_until_target.py` | Wrapper that retries RMU runs until target forget/retain metrics are reached; prunes failed checkpoints. |
| `wandb_sweep_tradeoff_only_bio.yaml` | W&B sweep config for Qwen 2.5 RMU on WMDP-bio with biology-only MMLU retain subsets. |
| `wandb_sweep_llama31_tradeoff_only_bio.yaml` | Same sweep config adapted for Llama 3.1. |

### Evaluation Scripts

| File | Description |
|------|-------------|
| `run_hp_qa_eval.py` | Direct LM accuracy on HP MCQ for a given model. |
| `run_wmdp_qa_eval.py` | Direct LM accuracy on WMDP MCQ for a given model. |
| `new_mcq_eval_algorithm.py` | Standalone WMDP MCQ evaluation with detailed debug output. |
| `debug_qa_hp.py` | Debug HP MCQ scoring (prompts, logits, per-example output). |

### Reporting Scripts

| File | Description |
|------|-------------|
| `report_qa_mmlu_singleword.py` | Evaluate MCQ and free-text accuracy on MMLU biology single-word subset for Qwen 2.5 7B and Llama 3.1 8B. Outputs per-example (question, correct_answer, generated_answer) details. |
| `report_llama_performance_5fold.py` | 5-fold cross-validated performance report (embedding probes + direct Q&A) for a single model on HP and WMDP. |
| `run_all_5fold_reports.py` | Orchestrates 5-fold reports across multiple models (Llama, Qwen, Gemma, SBERT) and merges results. |
| `report_sbert_performance.py` | SBERT embedding-only Cosine/Ridge/Bilinear/MLP baseline report on HP and WMDP-bio. |

### Embedding Probe Ladder

| File | Description |
|------|-------------|
| `hp_probe_ladder.py` | Embedding probe ladder (Cosine, Ridge, Bilinear, MLP) on HP / MMLU / WMDP. Supports SBERT, Qwen, Llama, and Gemma embedding backends with hidden-layer extraction. Includes 5-fold CV with per-answer grouping to prevent leakage. |

### Analysis and Visualization

| File | Description |
|------|-------------|
| `analyze_singleword_freeform.py` | Post-process free-form generation outputs: auto-judging, ambiguous case export, regression mining (base vs RMU). Supports manual adjudication rules. |
| `manual_adjudication.json` | Manual override rules for ambiguous free-form answer judgments. |
| `plot_rmu_accuracy.py` | Plot RMU forget/retain accuracy over training steps with optional embedding baseline dotted lines. Supports kneecap (tradeoff) plots. |
| `plot_wandb_pareto.py` | Aggregate multiple W&B sweep projects into Pareto/tradeoff plots. |

### Data Preparation

| File | Description |
|------|-------------|
| `build_crisp_hp_corpora.py` | Build CRISP-style HP forget and retain text corpora from HP MCQ JSON. |
| `filter_hp_unique.py` | Filter HP MCQ dataset so each correct answer appears at most N times. |

---

## Usage

### 1. Embedding Baselines (Probe Ladder)

Compute baseline accuracies (Cosine, Ridge, Bilinear, MLP) on the forget set using first-layer embeddings:

```bash
# WMDP-bio with Qwen embeddings
python hp_probe_ladder.py --dataset wmdp --wmdp_config wmdp-bio --embedding_backend qwen --qwen_layer 0 --out_dir outputs/wmdp_baselines --cache_dir cache/wmdp

# HP with Llama embeddings, 5-fold CV
python hp_probe_ladder.py --dataset hp --data_path data/hp/hp_mcq_compromise.json --embedding_backend llama --qwen_layer 0 --n_folds 5 --out_dir outputs/llama_hp_5fold --cache_dir cache/hp_llama

# SBERT baseline
python hp_probe_ladder.py --dataset wmdp --wmdp_config wmdp-bio --embedding_backend sbert --normalize_sbert --out_dir outputs/sbert_wmdp
```

### 2. RMU Training

```bash
# Single WMDP-bio RMU run (Qwen)
python run_rmu_with_eval.py --output_dir outputs/rmu_run1 --max_num_batches 150 --checkpoint_interval 15

# HP RMU with CRISP corpora (Llama)
python run_rmu_with_eval.py \
  --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
  --eval_forget_mode hp \
  --hp_mcq_path data/hp/hp_mcq_compromise.json \
  --forget_corpora "crisp-hp-forget:data/crisp_hp_forget.json" \
  --retain_corpora "crisp-hp-retain:data/crisp_hp_retain.json" \
  --output_dir outputs/rmu_hp_llama

# RMU with auto-retry until target metrics
python run_rmu_until_target.py --target_forget_max 0.30 --target_retain_min 0.60 --max_retries 5 --output_dir outputs/rmu_target
```

### 3. MMLU Biology Single-Word Evaluation

Evaluate MCQ and free-text (no choices) accuracy on the subset of MMLU biology questions whose correct answer is a single word:

```bash
python report_qa_mmlu_singleword.py --out_dir outputs/qa_mmlu_singleword_report
```

Outputs:
- `qa_mmlu_singleword_report.json` -- per-model MCQ and free-text accuracy
- `freeform_question_answer_details.json` -- per-example question, correct_answer, generated_answer

### 4. 5-Fold Performance Reports

```bash
# All models (Llama, Qwen 2.5, Gemma 2, SBERT)
python run_all_5fold_reports.py --hp_mcq_path data/hp/hp_mcq_compromise.json --wmdp_config wmdp-bio

# Single model
python report_llama_performance_5fold.py --model meta-llama/Llama-3.1-8B-Instruct --embedding_backend llama --hp_mcq_path data/hp/hp_mcq_compromise.json --n_folds 5 --out_dir outputs/llama_performance_5fold
```

### 5. Post-Hoc Free-Form Analysis

Compare base vs RMU model free-form outputs, auto-judge, export ambiguous cases:

```bash
python analyze_singleword_freeform.py \
  --base_freeform_json outputs/base_freeform_details.json \
  --rmu_freeform_json outputs/rmu_freeform_details.json \
  --out_dir outputs/analysis \
  --manual_adjudication_json manual_adjudication.json
```

### 6. Plotting

```bash
# Accuracy over RMU steps with embedding baseline dotted lines
python plot_rmu_accuracy.py --eval_steps outputs/rmu_run1/eval_steps.json --baselines outputs/wmdp_baselines/baseline_accuracies.json -o rmu_accuracy_plot.png

# Kneecap (forget vs retain tradeoff) plot
python plot_rmu_accuracy.py --eval_steps outputs/rmu_run1/eval_steps.json --kneecap -o kneecap_plot.png

# Pareto plot across W&B sweep projects
python plot_wandb_pareto.py --entity your-entity --projects project1,project2 --output pareto.png
```

---

## Data

Data files are not included in this repository. Required datasets:

| Dataset | Source | Usage |
|---------|--------|-------|
| MMLU (college_biology, high_school_biology) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu/) | Retain evaluation, single-word free-text eval |
| WMDP-bio | [cais/wmdp](https://huggingface.co/datasets/cais/wmdp) | Forget set MCQ evaluation |
| WMDP-bio forget corpus | [cais/wmdp-bio-forget-corpus](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus) (gated) | RMU forget training data |
| MMLU auxiliary corpora | [cais/wmdp-mmlu-auxiliary-corpora](https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora) (gated) | RMU retain training data |
| MedMCQA | [openlifescienceai/medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa) | Retain evaluation |
| HP MCQ | Local JSON (`data/hp/hp_mcq.json`) | Harry Potter forget set evaluation |

HP data preparation:
```bash
python filter_hp_unique.py --data_path data/hp/hp_mcq.json --max_per_answer 5 -o data/hp/hp_mcq_unique5.json
python build_crisp_hp_corpora.py --hp_mcq_path data/hp/hp_mcq_unique5.json
```

---

## RMU Defaults

| Parameter | Default |
|-----------|---------|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Forget corpus | `bio-forget-corpus` (WMDP-bio) |
| Retain corpus | `mmlu-auxiliary-all` |
| alpha (retain weight) | 1200 |
| steering_coeffs | 6.5 |
| Learning rate | 5e-5 |
| layer_id | 7 |
| layer_ids | 5, 6, 7 |
| max_num_batches | 150 |

---

## Qwen 3.5 Note

Qwen 3.5 (`Qwen/Qwen3.5-9B`) requires **transformers >= 5.2.0** and **Python >= 3.10**. Use a separate conda environment:

```bash
conda create -n py310-qwen python=3.10 -y
conda activate py310-qwen
pip install torch "transformers>=5.2.0" datasets accelerate
```

---

## License

MIT
