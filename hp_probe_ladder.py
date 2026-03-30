import os
import json
import argparse
import random
import hashlib
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def md5_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in s)


def l2_normalize(x: np.ndarray, axis=-1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def sanitize_embeddings(x: np.ndarray, name: str):
    if not np.isfinite(x).all():
        n_nan = int(np.isnan(x).sum())
        n_inf = int(np.isinf(x).sum())
        print(f"[WARN] {name} had NaN/Inf (nan={n_nan}, inf={n_inf}) -> replacing with 0")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

# -------------------------
# Data
# -------------------------
def load_hp(path: str, log_bad: bool = True):
    """
    Minimal version: keep original checks, but drop invalid rows instead of raising.
    """
    with open(path, "r") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    answers = data.get("answers", [])
    choices = data.get("choices", [])

    n = min(len(questions), len(answers), len(choices))
    questions = questions[:n]
    answers = answers[:n]
    choices = choices[:n]

    bad_idxs = []
    for i, ch in enumerate(choices):
        if not (isinstance(ch, list) and len(ch) == 4):
            bad_idxs.append(i)
    for i, a in enumerate(answers):
        if a not in (0, 1, 2, 3):
            bad_idxs.append(i)

    bad_idxs = sorted(set(bad_idxs), reverse=True)
    if log_bad and bad_idxs:
        print(f"[load_hp] Removing {len(bad_idxs)} invalid items (up to 20 shown): {bad_idxs[:20]}")

    for i in bad_idxs:
        questions.pop(i)
        choices.pop(i)
        answers.pop(i)

    return questions, choices, answers


LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def load_mmlu(config: str = "all", split: str = "test"):
    """
    Load MMLU from HuggingFace (cais/mmlu).

    Returns:
        questions: list[str]
        choices:   list[list[str]] (len 4)
        answers:   list[int] labels in {0,1,2,3}
        subjects:  list[str] subject name per question
    """
    ds = load_dataset("cais/mmlu", config, split=split)
    questions = []
    choices = []
    answers = []
    subjects = []

    for ex in ds:
        questions.append(ex["question"])
        choices.append(ex["choices"])
        # answer can be a letter ("A"/"B"/"C"/"D") or already an int label
        ans = ex["answer"]
        if isinstance(ans, str):
            answers.append(LETTER_TO_IDX[ans])
        else:
            # ClassLabel or numeric -> make sure it's 0..3
            answers.append(int(ans))
        subjects.append(ex["subject"])

    return questions, choices, answers, subjects


def load_wmdp(config: str = "all", split: str = "test"):
    """
    Load WMDP (Weapons of Mass Destruction Proxy) MCQ from HuggingFace cais/wmdp.

    Configs: 'wmdp-bio', 'wmdp-cyber', 'wmdp-chem', or 'all' (concatenate all three).

    Returns:
        questions: list[str]
        choices:   list[list[str]] (len 4)
        answers:   list[int] labels in {0,1,2,3}
        subjects:  list[str] e.g. 'wmdp-bio', 'wmdp-cyber', 'wmdp-chem'
    """
    configs = ["wmdp-bio", "wmdp-cyber", "wmdp-chem"] if config == "all" else [config]
    questions, choices, answers, subjects = [], [], [], []
    for cfg in configs:
        ds = load_dataset("cais/wmdp", cfg, split=split, trust_remote_code=True)
        for ex in ds:
            questions.append(ex["question"])
            ch = ex["choices"]
            if hasattr(ch, "__iter__") and not isinstance(ch, str):
                choices.append(list(ch)[:4])
            else:
                choices.append(ch if isinstance(ch, list) else [ch])
            ans = ex["answer"]
            answers.append(int(ans) if isinstance(ans, (int, np.integer)) else LETTER_TO_IDX.get(str(ans).upper(), 0))
            subjects.append(cfg)
    return questions, choices, answers, subjects


def subset_hp(questions, choices, answers, indices):
    indices = list(map(int, indices))
    return {
        "questions": [questions[i] for i in indices],
        "choices": [choices[i] for i in indices],
        "answers": [int(answers[i]) for i in indices],
    }


# -------------------------
# Embeddings + cache
# -------------------------
def embed_texts(model: SentenceTransformer, texts, batch_size: int, normalize: bool):
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return embs.astype(np.float32)


def load_or_compute_embeddings(
    sbert: SentenceTransformer,
    texts,
    cache_path: str,
    batch_size: int,
    normalize: bool,
    force_recompute: bool,
):
    if (not force_recompute) and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            return arr.astype(np.float32)
        print(f"[cache mismatch] {cache_path} has {arr.shape}, expected first dim {len(texts)}. Recomputing...")

    ensure_dir(os.path.dirname(cache_path))
    arr = embed_texts(sbert, texts, batch_size=batch_size, normalize=normalize)
    np.save(cache_path, arr)
    return arr


class QwenEmbedder:
    """Create sentence embeddings from a Qwen 2.5 decoder-only model.

    We extract a specific hidden layer and pool across tokens.
    Pooling can be:
      - mean: masked mean pooling over non-pad tokens
      - last_token: hidden state at the last non-pad token

    This is *not* a dedicated embedding model (unlike SBERT), which is useful
    when you want to study shortcut behavior in raw LLM representations.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_length: int = 512,
        dtype: torch.dtype = torch.float16,
        pooling: str = "mean",
        layer_idx: int = 0,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)
        self.pooling = pooling
        self.layer_idx = layer_idx

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # AutoModel returns hidden states (unlike AutoModelForCausalLM).
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

        # If device_map="auto" places the model on GPU, the tensors should be
        # created on the same device. We'll use the first parameter's device.
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device(device)

    @torch.no_grad()
    def encode(self, texts, batch_size: int = 8) -> np.ndarray:
        all_embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Qwen embedding"):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}

            out = self.model(**toks, output_hidden_states=True)
            # HuggingFace: hidden_states[0] = output of embedding layer (token+position), BEFORE any transformer block.
            # hidden_states[1] = after first transformer block, ..., hidden_states[-1] = after last block.
            # So layer_idx=0 is "Qwen (first layer): mean of all tokens right after the embeddings".
            hidden = out.hidden_states[self.layer_idx]  # [B, T, D]
            attn = toks.get("attention_mask", None)

            if self.pooling == "mean":
                # mean-pool only over non-pad tokens
                if attn is None:
                    emb = hidden.mean(dim=1)
                else:
                    attn_f = attn.unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
                    summed = (hidden * attn_f).sum(dim=1)
                    denom = attn_f.sum(dim=1).clamp(min=1.0)
                    emb = summed / denom

            elif self.pooling == "last_token":
                # take the last non-pad token representation
                if attn is None:
                    emb = hidden[:, -1, :]
                else:
                    lengths = attn.sum(dim=1)  # [B]
                    idx = (lengths - 1).clamp(min=0)  # [B]
                    emb = hidden[torch.arange(hidden.size(0), device=hidden.device), idx, :]

            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            all_embs.append(emb.detach().cpu())

        return torch.cat(all_embs, dim=0).numpy()


def load_or_compute_qwen_embeddings(
    embedder: QwenEmbedder,
    texts,
    cache_path: str,
    batch_size: int,
    force_recompute: bool,
):
    """Cache wrapper for Qwen embeddings (mirrors SBERT cache behavior)."""
    if (not force_recompute) and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            return arr.astype(np.float32)
        print(f"[cache mismatch] {cache_path} has {arr.shape}, expected first dim {len(texts)}. Recomputing...")

    ensure_dir(os.path.dirname(cache_path))
    arr = embedder.encode(texts, batch_size=batch_size).astype(np.float32)
    np.save(cache_path, arr)
    return arr


# -------------------------
# Torch datasets
# -------------------------
class MCQTensorDataset(torch.utils.data.Dataset):
    """
    Holds (q_emb, choices_emb[4,d], label) for each question index.
    """

    def __init__(self, q_embs, c_embs_4, labels, indices):
        self.q = torch.from_numpy(q_embs[indices]).float()  # [N, d]
        self.c = torch.from_numpy(c_embs_4[indices]).float()  # [N, 4, d]
        self.y = torch.from_numpy(np.array(labels, dtype=np.int64)[indices]).long()

    def __len__(self):
        return self.q.shape[0]

    def __getitem__(self, i):
        return self.q[i], self.c[i], self.y[i]


# -------------------------
# Probes
# -------------------------
def eval_scores_to_acc(scores: np.ndarray, gold: np.ndarray) -> float:
    # scores: [N,4]
    pred = np.argmax(scores, axis=1)
    return float(accuracy_score(gold, pred))


def per_subject_accuracy(preds, gold_labels, subjects, test_idxs):
    """
    preds:      np.ndarray [n_test] of predicted labels
    gold_labels:list[int] length n (full dataset)
    subjects:   list[str] length n (full dataset)
    test_idxs:  np.ndarray of indices into gold_labels/subjects

    Returns: dict subject -> (acc, correct, total)
    """
    correct = defaultdict(int)
    total = defaultdict(int)

    for pred, idx in zip(preds, test_idxs):
        subj = subjects[idx]
        total[subj] += 1
        if pred == gold_labels[idx]:
            correct[subj] += 1

    stats = {}
    for subj in total:
        acc = correct[subj] / total[subj]
        stats[subj] = (acc, correct[subj], total[subj])
    return stats


def ridge_probe(q_embs, c_embs_4, answers, train_idxs, test_idxs, alpha, return_preds: bool = False):
    """
    Ridge regression probe mapping question embedding -> embedding of correct answer.

    If return_preds=True: returns (acc, preds, reg).
    Otherwise returns acc only.
    """
    answers_arr = np.array(answers, dtype=np.int64)
    n = len(answers_arr)

    # Correct answer embeddings
    correct_embs = c_embs_4[np.arange(n), answers_arr]  # [N, d]

    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(q_embs[train_idxs], correct_embs[train_idxs])

    # Predict embeddings for all questions
    pred_embs = reg.predict(q_embs)  # [N, d]
    pred_embs = l2_normalize(pred_embs, axis=-1)

    cand = c_embs_4  # [N,4,d]
    cand = l2_normalize(cand, axis=-1)

    scores = np.einsum("nkd,nd->nk", cand, pred_embs)  # cosine
    preds = scores.argmax(axis=1)

    acc = (preds[test_idxs] == answers_arr[test_idxs]).mean()
    if return_preds:
        return acc, preds[test_idxs], reg
    return acc


class BilinearProbe(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W)

    def forward(self, q, cand):
        """
        q: [B,d], cand: [B,4,d]
        scores: [B,4] where score_j = (qW)·a_j
        """
        qW = q @ self.W  # [B,d]
        scores = (cand * qW.unsqueeze(1)).sum(dim=-1)  # [B,4]
        return scores


class MLPScorer(nn.Module):
    def __init__(self, d: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = 4 * d  # [q; a; |q-a|; q*a]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, q, cand):
        """
        q: [B,d], cand: [B,4,d]
        returns scores [B,4]
        """
        q4 = q.unsqueeze(1).expand(-1, 4, -1)  # [B,4,d]
        feats = torch.cat([q4, cand, (q4 - cand).abs(), q4 * cand], dim=-1)  # [B,4,4d]
        B = feats.shape[0]
        scores = self.net(feats.view(B * 4, -1)).view(B, 4)  # [B,4]
        return scores


def train_mcq_model(
    model: nn.Module,
    train_loader,
    test_loader,
    device: str,
    lr: float,
    weight_decay: float,
    epochs: int,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_acc = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for q, c, y in train_loader:
            q, c, y = q.to(device), c.to(device), y.to(device)
            scores = model(q, c)  # [B,4]
            loss = F.cross_entropy(scores, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

        # eval
        model.eval()
        all_scores = []
        all_gold = []
        with torch.no_grad():
            for q, c, y in test_loader:
                q, c = q.to(device), c.to(device)
                scores = model(q, c).detach().cpu().numpy()
                all_scores.append(scores)
                all_gold.append(y.numpy())

        all_scores = np.concatenate(all_scores, axis=0)
        all_gold = np.concatenate(all_gold, axis=0)
        acc = eval_scores_to_acc(all_scores, all_gold)

        avg_loss = total_loss / max(n_batches, 1)
        print(f"epoch {ep:03d} | loss {avg_loss:.4f} | test_acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc


def get_preds_mcq_model(model, data_loader, device):
    """
    Run a trained MCQ model over a DataLoader and return predictions as np.array.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        for q, cand, y in data_loader:
            q = q.to(device)
            cand = cand.to(device)
            scores = model(q, cand)  # [B, 4]
            batch_preds = scores.argmax(dim=-1).cpu().numpy()
            all_preds.append(batch_preds)
    return np.concatenate(all_preds, axis=0)


def cosine_baseline(q_embs, c_embs_4, answers, test_idxs, return_preds: bool = False):
    """
    Simple cosine-similarity baseline.

    If return_preds=True: returns (acc, preds) where preds is [n_test].
    Otherwise returns acc only.
    """
    q = q_embs
    cand = c_embs_4
    # cosine since q and cand are already L2-normalized
    scores = np.einsum("nd,nkd->nk", q, cand)  # [N,4]
    answers_arr = np.array(answers, dtype=np.int64)
    preds = scores.argmax(axis=1)
    acc = (preds[test_idxs] == answers_arr[test_idxs]).mean()
    if return_preds:
        return acc, preds[test_idxs]
    return acc


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/hp/hp_mcq.json")
    parser.add_argument("--sbert_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument(
        "--embedding_backend",
        type=str,
        default="sbert",
        choices=["hp", "mmlu", "sbert", "qwen"] if False else ["sbert", "qwen"],
        help="Which embedding backend to use: 'sbert' or 'qwen' (Qwen hidden-state pooling).",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id for Qwen embeddings (used when --embedding_backend=qwen).",
    )
    parser.add_argument(
        "--qwen_max_length",
        type=int,
        default=512,
        help="Max token length for Qwen embedding inputs (used when --embedding_backend=qwen).",
    )
    parser.add_argument(
        "--qwen_batch_size_embed",
        type=int,
        default=8,
        help="Batch size for Qwen embedding extraction (used when --embedding_backend=qwen).",
    )
    parser.add_argument(
        "--qwen_pooling",
        type=str,
        default="mean",
        choices=["mean", "last_token"],
        help="Pooling for Qwen embeddings: mean or last_token (last non-pad token).",
    )
    parser.add_argument(
        "--qwen_layer",
        type=int,
        default=0,
        help="Qwen hidden layer: 0=embedding output only (before any transformer block), 1=after 1st block, -1=last layer.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size_embed", type=int, default=64)
    parser.add_argument("--normalize_sbert", action="store_true",
                        help="Normalize SBERT embeddings at encode-time (recommended).")
    parser.add_argument("--dataset", type=str, default="hp", choices=["hp", "mmlu", "wmdp"],
                        help="Which dataset: 'hp', 'mmlu', or 'wmdp' (forget set for unlearning).",)
    parser.add_argument("--mmlu_config", type=str, default="all",
                        help="MMLU config name (subject or 'all'), e.g. 'all', 'college_medicine', etc.",)
    parser.add_argument("--wmdp_config", type=str, default="all",
                        help="WMDP config: 'all', 'wmdp-bio', 'wmdp-cyber', or 'wmdp-chem'.",)

    # caching + outputs
    parser.add_argument("--cache_dir", type=str, default="cache/sbert_hp")
    parser.add_argument("--out_dir", type=str, default="outputs/hp_splits")
    parser.add_argument("--force_recompute", action="store_true")

    # ridge
    parser.add_argument("--ridge_alpha", type=float, default=10.0)

    # bilinear training
    parser.add_argument("--bilinear_lr", type=float, default=1e-2)
    parser.add_argument("--bilinear_wd", type=float, default=1e-2)
    parser.add_argument("--bilinear_epochs", type=int, default=80)

    # mlp training
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_wd", type=float, default=1e-2)
    parser.add_argument("--mlp_epochs", type=int, default=60)

    # torch loaders
    parser.add_argument("--batch_size_probe", type=int, default=128)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.cache_dir)
    ensure_dir(args.out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # Load + split
    if args.dataset == "hp":
        print(f"Loading HP dataset from {args.data_path}")
        questions, choices, answers = load_hp(args.data_path, log_bad=True)
        subjects = None
    elif args.dataset == "mmlu":
        print(f"Loading MMLU from HuggingFace: config='{args.mmlu_config}', split='test'")
        questions, choices, answers, subjects = load_mmlu(
            config=args.mmlu_config,
            split="test",
        )
    elif args.dataset == "wmdp":
        print(f"Loading WMDP from HuggingFace: config='{args.wmdp_config}', split='test'")
        questions, choices, answers, subjects = load_wmdp(
            config=args.wmdp_config,
            split="test",
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    n = len(questions)
    print(f"Loaded {n} items.")

    idxs = np.arange(n)
    answers_arr = np.array(answers, dtype=np.int64)

    try:
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=answers_arr,
        )
    except ValueError:
        # fallback if stratify fails
        train_idxs, test_idxs = train_test_split(
            idxs,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=None,
        )

    train_idxs = np.array(train_idxs, dtype=np.int64)
    test_idxs = np.array(test_idxs, dtype=np.int64)

    print(f"Loaded {n} questions")
    print(f"Train questions: {len(train_idxs)} | Test questions: {len(test_idxs)}")

    # Save unembedded splits for later Qwen eval (HP only)
    if args.dataset == "hp":
        ds_hash10 = md5_file(args.data_path)[:10]
        split_tag = f"hp_{safe_name(os.path.basename(args.data_path))}_{ds_hash10}_seed{args.seed}_test{args.test_size:.3f}"
        train_json_path = os.path.join(args.out_dir, f"{split_tag}_TRAIN.json")
        test_json_path = os.path.join(args.out_dir, f"{split_tag}_TEST.json")
        meta_path = os.path.join(args.out_dir, f"{split_tag}_META.json")

        save_json(subset_hp(questions, choices, answers, train_idxs), train_json_path)
        save_json(subset_hp(questions, choices, answers, test_idxs), test_json_path)
        save_json(
            {
                "data_path": args.data_path,
                "data_md5_10": ds_hash10,
                "seed": args.seed,
                "test_size": args.test_size,
                "n_total": int(n),
                "n_train": int(len(train_idxs)),
                "n_test": int(len(test_idxs)),
                "train_indices": train_idxs.tolist(),
                "test_indices": test_idxs.tolist(),
                "sbert_model": args.sbert_model,
                "normalize_sbert": bool(args.normalize_sbert),
            },
            meta_path,
        )
        print("\nSaved splits:")
        print(f"  Train (unembedded): {train_json_path}")
        print(f"  Test  (unembedded): {test_json_path}")
        print(f"  Meta:               {meta_path}")
    elif args.dataset == "mmlu":
        ds_hash10 = f"mmlu_{safe_name(args.mmlu_config)}"
    elif args.dataset == "wmdp":
        ds_hash10 = f"wmdp_{safe_name(args.wmdp_config)}"
    else:
        ds_hash10 = "unknown"

    # Embeddings (cached)
    print(f"\nEmbedding backend: {args.embedding_backend}")

    if args.embedding_backend == "sbert":
        print(f"Loading SBERT: {args.sbert_model}")
        embedder = SentenceTransformer(args.sbert_model)
        model_tag = "sbert_" + safe_name(args.sbert_model)
        norm_tag = "norm1" if args.normalize_sbert else "norm0"
    else:
        print(f"Loading Qwen embedder: {args.qwen_model} (pooling={args.qwen_pooling}, layer={args.qwen_layer})")
        embedder = QwenEmbedder(
            model_name=args.qwen_model,
            device=device,
            max_length=args.qwen_max_length,
            pooling=args.qwen_pooling,
            layer_idx=args.qwen_layer,
        )
        model_tag = "qwen_" + safe_name(args.qwen_model)
        # normalize_sbert doesn't apply; we still L2-normalize below.
        layer_str = f"L{args.qwen_layer}" if args.qwen_layer >= 0 else "Llast"
        norm_tag = f"pool_{args.qwen_pooling}_{layer_str}_norm_post"

    q_cache = os.path.join(args.cache_dir, f"{ds_hash10}_{model_tag}_{norm_tag}_Q.npy")
    c_cache = os.path.join(args.cache_dir, f"{ds_hash10}_{model_tag}_{norm_tag}_C_flat.npy")

    print("\nEmbedding questions (with cache)...")
    if args.embedding_backend == "sbert":
        q_embs = load_or_compute_embeddings(
            embedder, questions, q_cache,
            batch_size=args.batch_size_embed,
            normalize=args.normalize_sbert,
            force_recompute=args.force_recompute,
        )
    else:
        q_embs = load_or_compute_qwen_embeddings(
            embedder,
            questions,
            q_cache,
            batch_size=args.qwen_batch_size_embed,
            force_recompute=args.force_recompute,
        )

    print("\nEmbedding choices (with cache)...")
    flat_choices = [choices[i][j] for i in range(n) for j in range(4)]
    if args.embedding_backend == "sbert":
        c_embs_flat = load_or_compute_embeddings(
            embedder, flat_choices, c_cache,
            batch_size=args.batch_size_embed,
            normalize=args.normalize_sbert,
            force_recompute=args.force_recompute,
        )
    else:
        c_embs_flat = load_or_compute_qwen_embeddings(
            embedder,
            flat_choices,
            c_cache,
            batch_size=args.qwen_batch_size_embed,
            force_recompute=args.force_recompute,
        )

    d = c_embs_flat.shape[-1]
    c_embs_4 = c_embs_flat.reshape(n, 4, d).astype(np.float32)

    # To make the ladder comparable, we score with cosine-like similarity for Ridge,
    # and we optionally normalize inputs for torch probes too (helps stability).
    q_embs_n = l2_normalize(q_embs, axis=-1).astype(np.float32)
    c_embs_4_n = l2_normalize(c_embs_4, axis=-1).astype(np.float32)
    q_embs_n = sanitize_embeddings(q_embs_n, "q_embs_n")
    c_embs_4_n = sanitize_embeddings(c_embs_4_n, "c_embs_4_n")

    print("\n=== Cosine baseline: argmax cos(q, option) ===")
    if args.dataset in ("mmlu", "wmdp"):
        # also get predictions per test item (for per-subject breakdown)
        cos_acc, cos_preds = cosine_baseline(
            q_embs_n, c_embs_4_n, answers, test_idxs, return_preds=True
        )
    else:
        cos_acc = cosine_baseline(
            q_embs_n, c_embs_4_n, answers, test_idxs, return_preds=False
        )
        cos_preds = None
    print(f"Cosine MCQ accuracy: {cos_acc:.4f}")

    # -------------------------
    # 1) Ridge mapping probe
    # -------------------------
    print("\n=== Ridge probe (q -> correct a embedding) ===")
    if args.dataset in ("mmlu", "wmdp"):
        ridge_acc, ridge_preds, ridge_reg = ridge_probe(
            q_embs_n,
            c_embs_4_n,
            answers,
            train_idxs,
            test_idxs,
            alpha=args.ridge_alpha,
            return_preds=True,
        )
    else:
        ridge_acc = ridge_probe(
            q_embs_n,
            c_embs_4_n,
            answers,
            train_idxs,
            test_idxs,
            alpha=args.ridge_alpha,
            return_preds=False,
        )
        ridge_preds = None
        ridge_reg = None
    print(f"Ridge MCQ accuracy: {ridge_acc:.4f}")

    # Prepare loaders for torch probes (bilinear, mlp)
    train_ds = MCQTensorDataset(q_embs_n, c_embs_4_n, answers, train_idxs)
    test_ds = MCQTensorDataset(q_embs_n, c_embs_4_n, answers, test_idxs)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size_probe, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size_probe, shuffle=False, drop_last=False
    )

    # -------------------------
    # 2) Bilinear probe
    # -------------------------
    print("\n=== Bilinear probe: s(q,a) = (qW)·a ===")
    bilinear = BilinearProbe(d=d)
    bilinear_acc = train_mcq_model(
        bilinear,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=args.bilinear_lr,
        weight_decay=args.bilinear_wd,
        epochs=args.bilinear_epochs,
    )
    print(f"Bilinear best test MCQ accuracy: {bilinear_acc:.4f}")

    # -------------------------
    # 3) MLP scorer probe
    # -------------------------
    print("\n=== MLP scorer probe on pair features ===")
    mlp = MLPScorer(d=d, hidden=args.mlp_hidden, dropout=args.mlp_dropout)
    mlp_acc = train_mcq_model(
        mlp,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=args.mlp_lr,
        weight_decay=args.mlp_wd,
        epochs=args.mlp_epochs,
    )
    print(f"MLP best test MCQ accuracy: {mlp_acc:.4f}")

    # -------------------------------------------------
    # Per-subject (niche) accuracies for MMLU only
    # -------------------------------------------------
    if args.dataset in ("mmlu", "wmdp") and subjects is not None:
        print("\n=== Per-subject (niche) accuracies (test portion) ===")

        # Cosine & Ridge: we already collected preds for test_idxs

        cos_stats = per_subject_accuracy(cos_preds, answers, subjects, test_idxs)
        ridge_stats = per_subject_accuracy(ridge_preds, answers, subjects, test_idxs)

        # Bilinear & MLP: get predictions via the test_loader
        bilinear_test_preds = get_preds_mcq_model(bilinear, test_loader, device)
        mlp_test_preds = get_preds_mcq_model(mlp, test_loader, device)

        bilinear_stats = per_subject_accuracy(bilinear_test_preds, answers, subjects, test_idxs)
        mlp_stats = per_subject_accuracy(mlp_test_preds, answers, subjects, test_idxs)

        all_subjects = sorted(cos_stats.keys())
        print(f"{'Subject':40s}  {'Cos':>6s}  {'Ridge':>6s}  {'Bil':>6s}  {'MLP':>6s}  (n)")
        for subj in all_subjects:
            c_acc, c_ok, c_tot = cos_stats[subj]
            r_acc, _, _ = ridge_stats[subj]
            b_acc, _, _ = bilinear_stats[subj]
            m_acc, _, _ = mlp_stats[subj]
            print(
                f"{subj:40s}  "
                f"{c_acc*100:6.1f}  "
                f"{r_acc*100:6.1f}  "
                f"{b_acc*100:6.1f}  "
                f"{m_acc*100:6.1f}  "
                f"({c_tot})"
            )

    print("\n=== Summary ===")
    print(f"Cosine:   {cos_acc:.4f}")
    print(f"Ridge:    {ridge_acc:.4f}")
    print(f"Bilinear: {bilinear_acc:.4f}")
    print(f"MLP:      {mlp_acc:.4f}")

    # Save baseline accuracies for RMU plot (dotted lines)
    baseline_summary = {
        "dataset": args.dataset,
        "Cosine": round(float(cos_acc), 4),
        "Ridge": round(float(ridge_acc), 4),
        "Bilinear": round(float(bilinear_acc), 4),
        "MLP": round(float(mlp_acc), 4),
    }
    baseline_path = os.path.join(args.out_dir, "baseline_accuracies.json")
    save_json(baseline_summary, baseline_path)
    print(f"\nBaseline accuracies saved to {baseline_path} (for RMU plot dotted lines).")

    print("\nCaches:")
    print(f"  Q: {q_cache}")
    print(f"  C: {c_cache}")


if __name__ == "__main__":
    main()
