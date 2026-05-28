# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.0.0",
#   "torch>=2.3.0",
#   "tqdm>=4.66.0",
#   "transformers>=4.45.0",
#   "accelerate>=0.30.0",
# ]
# ///

"""Per-layer k=2 silhouette analysis.

For a single model, extract per-layer target-word embeddings (orig and
artificial sentences), then for each (system_id, semantic_label, layer,
embed_type) compute the KMeans(k=2) silhouette score — exactly as
run_semantic_label_k2_metrics.py does for the final-layer embedding, but
sliced by hidden-state layer.

The script processes one system at a time and discards activations between
systems, so peak memory is bounded by the largest system × n_layers ×
hidden_dim, which keeps Qwen2.5-32B (64 layers x 5120 dim) under a few GB.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import overabundance_common as common


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    s = " ".join(s.split())
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return s


def _meaning_match_key(text: Any) -> str:
    s = _normalize_text(text)
    if not s:
        return s
    s = re.sub(r"\.\s*\(I used[^)]*\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\.\s*I used.*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _system_id(lemma: str, msps: str) -> str:
    return f"{lemma}__{msps}"


def _load_conditioning(path: str) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    df = pd.read_csv(path, encoding="latin-1")
    by_system: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        lemma = _normalize_text(row.get("lemma"))
        msps = _normalize_text(row.get("msps"))
        by_system[(lemma, msps)].append({
            "lemma": lemma,
            "msps": msps,
            "form_1": _normalize_text(row.get("form_1")),
            "form_2": _normalize_text(row.get("form_2")),
            "def": _normalize_text(row.get("def")),
            "def_key": _meaning_match_key(row.get("def")),
            "cond_type": _normalize_text(row.get("cond_type")),
            "p_eq": row.get("p_eq"),
            "p_cat": row.get("p_cat"),
        })
    return by_system


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(X) < 3 or unique_labels.size < 2 or unique_labels.size >= len(X):
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def _compute_k2_sil(X: np.ndarray, seed: int) -> float:
    if len(X) < 3:
        return float("nan")
    try:
        labels = KMeans(n_clusters=2, random_state=seed, n_init="auto").fit_predict(X)
    except Exception:
        return float("nan")
    if np.unique(labels).size < 2:
        return float("nan")
    return _safe_silhouette(X, labels)


def _layer_embeddings(model, tokenizer, sentence: str, word: str, device) -> Optional[List[np.ndarray]]:
    """Forward pass with output_hidden_states; returns one vector per layer.

    Each vector is the mean of last_hidden_state values at the target word's
    token positions, for each layer (index 0 = embeddings; 1..L = transformer
    layers).
    """
    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    encoding_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    located = common._hf_locate_target_in_encoding(tokenizer, encoding_tokens, word)
    if located is None:
        return None
    word_start, word_tokens = located

    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (L+1) tensors

    out: List[np.ndarray] = []
    for layer_t in hidden_states:
        # layer_t shape: (1, seq, hidden)
        word_slice = layer_t[0, word_start : word_start + len(word_tokens)]
        vec = word_slice.mean(dim=0).float().cpu().numpy()
        out.append(vec)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True, help="HuggingFace model id, e.g. answerdotai/ModernBERT-base.")
    p.add_argument("--output-tsv", default=None, help="Output TSV path. Default per_layer_k2_silhouette_<slug>.tsv.")
    p.add_argument("--conditioning-by-meaning-path", default="conditioning_by_meaning.csv")
    p.add_argument("--device", default=None, help="cpu|mps|cuda. Auto-detect if not given.")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-records", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    common.setup_environment()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model_name} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    slug = common.model_slug(args.model_name)
    output_tsv = args.output_tsv or f"per_layer_k2_silhouette_{slug}.tsv"

    cond_by_system = _load_conditioning(args.conditioning_by_meaning_path)

    records = common.load_records(max_records=args.max_records)
    # Group records by (lemma, mps)
    by_system: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_system[(_normalize_text(r["lexeme"]), _normalize_text(r["mps"]))].append(r)

    n_layers_total: Optional[int] = None
    rows: List[Dict[str, Any]] = []

    fieldnames = [
        "model", "embed_type", "layer", "lemma", "msps", "system_id",
        "semantic_label", "form_1", "form_2", "cond_type", "n_tokens", "k2_sil",
    ]
    write_header = True
    if os.path.exists(output_tsv):
        os.remove(output_tsv)

    for (lemma, msps), sys_records in tqdm(sorted(by_system.items()), desc=f"systems ({slug})"):
        cond_rows = cond_by_system.get((lemma, msps), [])
        if not cond_rows:
            continue

        # Per-record per-layer embeddings: orig[record_idx][layer] and art[record_idx][layer]
        per_rec_orig: List[Optional[List[np.ndarray]]] = []
        per_rec_art: List[Optional[List[np.ndarray]]] = []
        record_meaning_key: List[str] = []
        record_forms: List[str] = []

        for rec in sys_records:
            word = rec["target_word"]
            orig = _layer_embeddings(model, tokenizer, rec["orig_sentence"], word, device)
            art = _layer_embeddings(model, tokenizer, rec["artificial_sentence"], word, device)
            per_rec_orig.append(orig)
            per_rec_art.append(art)
            record_meaning_key.append(_meaning_match_key(rec.get("meaning")))
            record_forms.append(_normalize_text(rec.get("original_form")))

        if not per_rec_orig:
            continue

        # Determine number of layers from first non-None record
        n_layers = next((len(o) for o in per_rec_orig if o is not None), None)
        if n_layers is None:
            continue
        if n_layers_total is None:
            n_layers_total = n_layers

        sid = _system_id(lemma, msps)

        # For each conditioning row (semantic_label/meaning), filter records whose
        # meaning matches the cond_row's def and whose form is form_1 or form_2.
        for cond_row in cond_rows:
            form_1, form_2 = cond_row["form_1"], cond_row["form_2"]
            if not form_1 or not form_2:
                continue
            mkey = cond_row["def_key"]

            # Find matching record indices
            matching: List[int] = [
                i for i, key in enumerate(record_meaning_key)
                if key == mkey and record_forms[i] in (form_1, form_2)
            ]
            if len(matching) < 3:
                continue

            for layer_idx in range(n_layers):
                # Collect orig vectors
                orig_vecs = [per_rec_orig[i][layer_idx] for i in matching
                             if per_rec_orig[i] is not None and per_rec_art[i] is not None]
                art_vecs = [per_rec_art[i][layer_idx] for i in matching
                            if per_rec_orig[i] is not None and per_rec_art[i] is not None]
                if len(orig_vecs) < 3:
                    continue

                X_orig = np.stack(orig_vecs)
                X_delta = X_orig - np.stack(art_vecs)

                for embed_type, X in (("orig", X_orig), ("delta", X_delta)):
                    sil = _compute_k2_sil(X, seed=args.seed)
                    rows.append({
                        "model": slug,
                        "embed_type": embed_type,
                        "layer": layer_idx,
                        "lemma": lemma,
                        "msps": msps,
                        "system_id": sid,
                        "semantic_label": cond_row["def"][:80],
                        "form_1": form_1,
                        "form_2": form_2,
                        "cond_type": cond_row["cond_type"],
                        "n_tokens": len(orig_vecs),
                        "k2_sil": sil,
                    })

        # Stream-write rows for this system, then clear
        if rows:
            mode = "w" if write_header else "a"
            with open(output_tsv, mode, encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                if write_header:
                    writer.writeheader()
                    write_header = False
                for row in rows:
                    writer.writerow(row)
            rows = []

    print(f"Wrote {output_tsv} (layers={n_layers_total}).")


if __name__ == "__main__":
    main()
