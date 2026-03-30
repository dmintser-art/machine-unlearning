"""
RMU utilities: model loading, activation caching, data loading.
Adapted from https://github.com/centerforaisafety/WMDP
"""
import os
import random
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_with_cache(model, inputs, module, no_grad=True):
    """Run forward and return the output of the given module (e.g. a layer)."""
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)
    model_inputs = dict(inputs)
    # For sharded multi-GPU models, KV cache updates may span devices and crash.
    # RMU only needs activations from a single forward, so disable cache.
    model_inputs.setdefault("use_cache", False)
    if no_grad:
        with torch.no_grad():
            _ = model(**model_inputs)
    else:
        _ = model(**model_inputs)
    hook_handle.remove()
    return cache[0]


def resolve_decoder_layers(model):
    """Return decoder layer ModuleList and its attribute path for supported CausalLM wrappers."""
    candidates = (
        ("model.layers", lambda m: m.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
    )
    for path, getter in candidates:
        try:
            layers = getter(model)
            _ = len(layers)
            return layers, path
        except Exception:
            continue
    raise ValueError(
        "Could not resolve decoder layers. Tried: model.layers, model.language_model.layers "
        f"(model class: {model.__class__.__name__})."
    )


def get_params(model, layer_ids, param_ids=None, param_names=None):
    """Get trainable parameters for the specified layers by index and/or name."""
    if param_ids is None:
        param_ids = []
    if param_names is None:
        param_names = []
    param_id_set = set(param_ids)
    param_name_set = {n.strip() for n in param_names if n and n.strip()}

    layers, layers_path = resolve_decoder_layers(model)

    params = []
    selected = []
    for layer_id in layer_ids:
        if layer_id < 0 or layer_id >= len(layers):
            raise ValueError(
                f"layer_id={layer_id} is out of range for {layers_path} (num_layers={len(layers)})."
            )
        layer = layers[layer_id]
        for i, (name, p) in enumerate(layer.named_parameters()):
            match_id = i in param_id_set
            match_name = name in param_name_set
            if match_id or match_name:
                params.append(p)
                selected.append(
                    {
                        "layer_id": layer_id,
                        "param_id": i,
                        "name": name,
                        "shape": tuple(p.shape),
                        "layer_path": layers_path,
                        "selected_by": (
                            "id+name" if (match_id and match_name) else
                            "id" if match_id else
                            "name"
                        ),
                    }
                )
    return params, selected


def load_model(model_name_or_path, device_map="auto"):
    """Load causal LM and tokenizer."""
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer


def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    """Load forget and retain corpora (list of batch lists of text)."""

    # WMDP MMLU auxiliary has configs: economics-corpus, law-corpus, physics-corpus
    MMLU_AUXILIARY_CONFIGS = {"economics-corpus", "law-corpus", "physics-corpus"}
    TEXT_KEYS = ("text", "content", "body", "paragraph", "passage")

    def append_text(data, text):
        if text is None:
            return
        txt = " ".join(str(text).split())
        if len(txt) <= min_len:
            return
        if max_len and len(txt) > max_len:
            for i in range(0, len(txt), max_len):
                chunk = txt[i : i + max_len]
                if len(chunk) > min_len:
                    data.append(chunk)
        else:
            data.append(txt)

    def append_dataset_text(data, dataset):
        for row in dataset:
            if isinstance(row, str):
                append_text(data, row)
                continue
            if not isinstance(row, dict):
                append_text(data, row)
                continue
            text = None
            for key in TEXT_KEYS:
                val = row.get(key)
                if isinstance(val, str):
                    text = val
                    break
            if text is None:
                for val in row.values():
                    if isinstance(val, str):
                        text = val
                        break
            append_text(data, text)

    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            append_dataset_text(data, raw_data)
        elif name in {"hp-forget-corpus", "whp-forget-corpus"}:
            dataset = load_dataset("WutYee/HarryPotter_books_1to7", split="train")
            append_dataset_text(data, dataset)
        elif name in {"hp-retain-book-corpus", "hp-retain-corpus", "whp-retain-book-corpus", "whp-retain-corpus"}:
            dataset = load_dataset("Blackroot/Tiny-Open-Domain-Books", split="train")
            append_dataset_text(data, dataset)
        elif name in {"hp-retain-wiki-corpus", "whp-retain-wiki-corpus"}:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            append_dataset_text(data, dataset)
        elif name == "mmlu-auxiliary-all":
            # Load all three MMLU auxiliary configs and combine into one retain corpus
            token = os.getenv("HF_TOKEN")
            if not token:
                try:
                    from huggingface_hub import get_token
                    token = get_token()
                except Exception:
                    pass
            if not token:
                raise RuntimeError(
                    "WMDP MMLU auxiliary is gated. Accept terms at https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora and run huggingface-cli login (or set HF_TOKEN)."
                )
            for config in MMLU_AUXILIARY_CONFIGS:
                try:
                    dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", config, split="train", token=token)
                except Exception as e:
                    if "gated" in str(e).lower() or "authenticated" in str(e).lower():
                        raise RuntimeError(
                            "Dataset is gated. Accept the terms at https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora and log in."
                        ) from e
                    raise
                append_dataset_text(data, dataset)
        elif name in MMLU_AUXILIARY_CONFIGS:
            # cais/wmdp-mmlu-auxiliary-corpora requires config: economics-corpus, law-corpus, or physics-corpus
            token = os.getenv("HF_TOKEN")
            if not token:
                try:
                    from huggingface_hub import get_token
                    token = get_token()
                except Exception:
                    pass
            if not token:
                raise RuntimeError(
                    "WMDP MMLU auxiliary is gated. Accept terms at https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora and run huggingface-cli login (or set HF_TOKEN)."
                )
            try:
                dataset = load_dataset("cais/wmdp-mmlu-auxiliary-corpora", name, split="train", token=token)
            except Exception as e:
                if "gated" in str(e).lower() or "authenticated" in str(e).lower():
                    raise RuntimeError(
                        "Dataset is gated. Accept the terms at https://huggingface.co/datasets/cais/wmdp-mmlu-auxiliary-corpora and log in."
                    ) from e
                raise
            append_dataset_text(data, dataset)
        else:
            # Forget corpora: bio-forget-corpus, cyber-forget-corpus (cais/wmdp-bio-forget-corpus, etc.)
            token = os.getenv("HF_TOKEN")
            if not token:
                try:
                    from huggingface_hub import get_token
                    token = get_token()
                except Exception:
                    pass
            if not token:
                raise RuntimeError(
                    "WMDP forget corpora are gated. Accept terms at https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus and run huggingface-cli login (or set HF_TOKEN)."
                )
            try:
                dataset = load_dataset(f"cais/wmdp-{name}", split="train", token=token)
            except Exception as e:
                if "gated" in str(e).lower() or "authenticated" in str(e).lower():
                    raise RuntimeError(
                        "Dataset is gated. Accept the terms in your browser:\n"
                        f"  https://huggingface.co/datasets/cais/wmdp-{name}\n"
                        "Then ensure you're logged in: huggingface-cli login (or set HF_TOKEN)."
                    ) from e
                raise
            append_dataset_text(data, dataset)
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora],
    )
