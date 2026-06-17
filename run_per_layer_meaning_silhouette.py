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

"""Genuine per-LAYER *by-meaning* silhouette analysis.

This is the layer-sliced counterpart to ``run_system_clustering_metrics.py``.
For every hidden-state layer of a model it clusters tokens **by their manually
tagged meaning** (the semantic-label partition) and reports the silhouette of
that partition -- the same metric behind ``clustering_metrics_summary_*``, just
computed one layer at a time instead of over the all-layers concatenation.

Unlike ``run_per_head_meaning_silhouette.py`` (which slices the cached
final-layer embedding into attention heads), this script RE-RUNS the model with
``output_hidden_states=True`` to obtain true per-layer representations -- the
per-layer embeddings are not stored in the cache. Token metadata (lemma, mps,
meaning_index, forms) is read from the cache so that the by-meaning labeling and
aggregation -- imported verbatim from ``run_system_clustering_metrics`` -- match
the headline analysis exactly.

For each layer we feed in:
    embedding_type="orig"   ->  hidden_states[layer] at the target word
    embedding_type="delta"  ->  orig hidden_states[layer] - artificial hidden_states[layer]

This is distinct from ``run_per_layer_silhouette.py``, which computes an
unsupervised within-meaning k=2 split (mirroring
``run_semantic_label_k2_metrics.py``) and is not comparable to by-meaning
clustering.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import overabundance_common as common
import run_system_clustering_metrics as scm
from run_per_layer_silhouette import _layer_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genuine per-layer by-meaning silhouette.")
    p.add_argument("--model-name", required=True, help="HuggingFace model id, e.g. answerdotai/ModernBERT-base.")
    p.add_argument(
        "--cache-path",
        default=None,
        help="Cache file for token metadata. Default head_embed_cache_<slug>.jsonl.",
    )
    p.add_argument("--embedding-types", default="orig,delta", help="Subset of {orig,delta}.")
    p.add_argument(
        "--label-types",
        default="raw,conditioned-exclude,conditioned-cat-only",
        help="By-meaning label types (parallel to the headline analysis).",
    )
    p.add_argument("--conditioning-by-meaning-path", default="conditioning_by_meaning.csv")
    p.add_argument("--conditioning-by-form-path", default="conditioning_by_form.csv")
    p.add_argument("--cond-counts-path", default="df_cond_types.csv")
    p.add_argument("--pair-p-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=".")
    p.add_argument(
        "--slug",
        default=None,
        help="Override the output/model slug (default: derived from --model-name). "
        "Use to match cache naming, e.g. bert_base_uncased.",
    )
    p.add_argument("--device", default=None, help="cpu|mps|cuda. Auto-detect if not given.")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--max-records", type=int, default=0, help="Cap records (smoke test).")
    p.add_argument("--max-layers", type=int, default=0, help="Cap layers scored (0 = all).")
    return p.parse_args()


def _resolve_device(arg: Optional[str]) -> str:
    if arg:
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_metadata_frame(records: List[Dict[str, Any]], model: str, cache_path: str) -> pd.DataFrame:
    """Metadata columns matching scm._extract_tokens_from_cache, plus row_pos."""
    rows: List[Dict[str, Any]] = []
    for pos, rec in enumerate(records):
        lemma = scm._normalize_text(rec.get("lexeme"))
        msps = scm._normalize_text(rec.get("mps"))
        if not lemma or not msps:
            continue
        meaning_id_raw = rec.get("meaning_index")
        if meaning_id_raw is None or (isinstance(meaning_id_raw, float) and np.isnan(meaning_id_raw)):
            continue
        rows.append(
            {
                "row_pos": pos,
                "model": model,
                "cache_path": cache_path,
                "cache_key": rec.get("cache_key", rec.get("ID")),
                "token_id": rec.get("cache_key", rec.get("ID")),
                "lemma": lemma,
                "msps": msps,
                "system_id": scm._system_id(lemma, msps),
                "form": scm._normalize_text(rec.get("original_form")),
                "other_form": scm._normalize_text(rec.get("partner_form")),
                "meaning": scm._normalize_text(rec.get("meaning")),
                "meaning_id": str(meaning_id_raw),
                "sentence": rec.get("orig_sentence"),
                "embedding_type": "perlayer",  # placeholder; real value set per layer
            }
        )
    return pd.DataFrame(rows)


def _score_layer(
    df_annot: pd.DataFrame,
    df_systems: pd.DataFrame,
    per_rec_orig: List[Optional[np.ndarray]],
    per_rec_art: List[Optional[np.ndarray]],
    *,
    layer: int,
    embedding_type: str,
    label_types: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    def _vec(pos: int) -> Optional[np.ndarray]:
        o = per_rec_orig[pos]
        if o is None or layer >= o.shape[0]:
            return None
        if embedding_type == "orig":
            return o[layer]
        a = per_rec_art[pos]
        if a is None or layer >= a.shape[0]:
            return None
        return o[layer] - a[layer]

    df = df_annot.copy()
    df["embedding"] = [_vec(int(p)) for p in df["row_pos"].to_numpy()]
    df = df[df["embedding"].notna()].copy()

    metrics_rows: List[Dict[str, Any]] = []
    systems_by_id = {str(r["system_id"]): r for r in df_systems.to_dict("records")}
    for system_id, df_sys in df.groupby("system_id", sort=True):
        sys_row = systems_by_id.get(str(system_id), {"system_id": system_id})
        for label_type in label_types:
            label_col = scm.LABEL_COLUMN_BY_TYPE[label_type]
            validity_col = scm.VALIDITY_COLUMN_BY_TYPE[label_type]
            df_eval = df_sys[df_sys[validity_col]].copy()
            df_eval = df_eval[df_eval[label_col].notna()].copy()

            base = dict(sys_row)
            base.update(
                {
                    "embedding_type": embedding_type,
                    "layer": layer,
                    "label_type": label_type,
                    "label_column": label_col,
                    "n_samples_eval": int(len(df_eval)),
                }
            )
            if df_eval.empty:
                base["silhouette_cosine"] = float("nan")
            else:
                base.update(
                    scm._analyze_labels(
                        df_eval,
                        label_col=label_col,
                        label_type=label_type,
                        seed=seed,
                        standardize=False,
                        normalize_l2=False,
                        pca_components=0,
                    )
                )
            metrics_rows.append(base)
    return metrics_rows


def main() -> None:
    args = parse_args()
    common.setup_environment()

    embedding_types = [t.strip() for t in args.embedding_types.split(",") if t.strip()]
    for t in embedding_types:
        if t not in {"orig", "delta"}:
            raise SystemExit(f"Unsupported embedding type: {t}")
    need_art = "delta" in embedding_types
    label_types = scm._resolve_label_types(args.label_types)

    slug = args.slug or common.model_slug(args.model_name)
    cache_path = args.cache_path or f"head_embed_cache_{slug}.jsonl"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    # Transformer-XL is deprecated in transformers: its tokenizer is gated behind
    # TRUST_REMOTE_CODE and its forward pass needs a type_as monkeypatch (mirrors
    # run_transfoxl.py).
    is_transfoxl = "transfo-xl" in args.model_name.lower()
    if is_transfoxl:
        os.environ.setdefault("TRUST_REMOTE_CODE", "True")
        common.patch_transfoxl_torch_type_as(torch)

    device = _resolve_device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    print(f"Loading model {args.model_name} (device={device}, dtype={args.dtype})")
    tok_kwargs = {} if is_transfoxl else {"trust_remote_code": True}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tok_kwargs)
    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    print(f"Reading token metadata from {cache_path}")
    cache = common.load_cache(cache_path)
    records = list(cache.values())
    if args.max_records:
        records = records[: args.max_records]

    cond_meaning = scm._load_conditioning_by_meaning(args.conditioning_by_meaning_path)
    pair_tests = scm._load_pair_tests(args.conditioning_by_form_path, alpha=args.pair_p_alpha)
    cond_counts = scm._load_cond_counts(args.cond_counts_path)

    df_meta = _build_metadata_frame(records, slug, cache_path)
    if df_meta.empty:
        raise RuntimeError("No usable tokens after metadata filtering.")
    df_annot, df_systems = scm._annotate_tokens(
        df_meta, cond_meaning=cond_meaning, pair_tests=pair_tests, cond_counts=cond_counts
    )

    # Only forward-pass the records that survive metadata filtering.
    needed_positions = sorted(int(p) for p in df_annot["row_pos"].unique())
    per_rec_orig: List[Optional[np.ndarray]] = [None] * len(records)
    per_rec_art: List[Optional[np.ndarray]] = [None] * len(records)

    n_skipped = 0
    n_layers = 0
    for pos in tqdm(needed_positions, desc=f"forward ({slug})", unit="tok"):
        rec = records[pos]
        orig_word = scm._normalize_text(rec.get("original_form")) or rec.get("target_word")
        orig = _layer_embeddings(model, tokenizer, rec.get("orig_sentence"), orig_word, device)
        if orig is None:
            n_skipped += 1
            continue
        per_rec_orig[pos] = np.stack(orig)
        if n_layers == 0:
            n_layers = per_rec_orig[pos].shape[0]
        if need_art:
            art_word = scm._normalize_text(rec.get("partner_form")) or rec.get("target_word")
            art = _layer_embeddings(model, tokenizer, rec.get("artificial_sentence"), art_word, device)
            if art is not None and np.stack(art).shape[0] == n_layers:
                per_rec_art[pos] = np.stack(art)

    if args.max_layers and args.max_layers < n_layers:
        n_layers = args.max_layers
    print(f"tokens={len(needed_positions)} skipped={n_skipped} layers={n_layers}")

    all_metrics: List[Dict[str, Any]] = []
    for embedding_type in embedding_types:
        for layer in tqdm(range(n_layers), desc=f"score {embedding_type}", unit="layer"):
            all_metrics.extend(
                _score_layer(
                    df_annot, df_systems, per_rec_orig, per_rec_art,
                    layer=layer, embedding_type=embedding_type,
                    label_types=label_types, seed=args.seed,
                )
            )
    metrics_df = pd.DataFrame(all_metrics)

    summary_parts: List[pd.DataFrame] = []
    for (embed_type, layer), sub in metrics_df.groupby(["embedding_type", "layer"], sort=True):
        agg = scm._aggregate_summary(sub)
        agg.insert(0, "layer", layer)
        agg.insert(0, "embedding_type_layer", embed_type)
        summary_parts.append(agg)
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"per_layer_meaning_metrics_{slug}.tsv")
    summary_path = os.path.join(args.output_dir, f"per_layer_meaning_summary_{slug}.tsv")
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    print(f"saved {metrics_path}")
    print(f"saved {summary_path}")

    if not summary_df.empty:
        view = summary_df[summary_df["pair_significance_bucket"] == "all"]
        for embed_type in embedding_types:
            ev = view[view["embedding_type_layer"] == embed_type]
            if ev.empty:
                continue
            pivot = ev.pivot_table(index="layer", columns="label_type", values="silhouette_cosine_mean")
            print(f"\n[{embed_type}] macro silhouette_cosine by layer:")
            print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
