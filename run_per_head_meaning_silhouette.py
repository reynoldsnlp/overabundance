# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.0.0",
#   "tqdm>=4.66.0",
# ]
# ///

"""Per-attention-head *by-meaning* silhouette analysis.

IMPORTANT: the vectors stored as ``orig_head_embeddings`` in the
``head_embed_cache_*.jsonl`` files are NOT per-layer hidden states. They are the
``num_attention_heads`` equal-size slices of the model's *final-layer* word
embedding (see ``overabundance_common.hf_get_embedding_with_heads``). So this
script clusters tokens **by their manually tagged meaning** within each
attention-head subspace of the final layer -- it asks "does any single head's
slice separate the senses?" For genuine per-LAYER analysis, see
``run_per_layer_meaning_silhouette.py``, which re-runs the model with
``output_hidden_states=True``.

It reuses the labeling, metric, and aggregation functions from
``run_system_clustering_metrics`` so the numbers are comparable to the headline
results (whose "orig" source is the concatenation of all of these head slices,
i.e. the full final-layer embedding). The ``layer`` column below is really the
attention-head index. Per head we feed in:

    embedding_type="orig"   ->  orig_head_embeddings[head]
    embedding_type="delta"  ->  orig_head_embeddings[head] - art_head_embeddings[head]
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import overabundance_common as common
import run_system_clustering_metrics as scm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-pattern",
        default="head_embed_cache_*.jsonl",
        help="Glob for cache files. Pass specific files to limit which models run.",
    )
    p.add_argument(
        "--embedding-types",
        default="orig,delta",
        help="Comma-separated subset of {orig,delta} to score per layer.",
    )
    p.add_argument(
        "--label-types",
        default="raw,conditioned-exclude,conditioned-cat-only",
        help="Comma-separated by-meaning label types (parallel to the headline analysis).",
    )
    p.add_argument("--conditioning-by-meaning-path", default="conditioning_by_meaning.csv")
    p.add_argument("--conditioning-by-form-path", default="conditioning_by_form.csv")
    p.add_argument("--cond-counts-path", default="df_cond_types.csv")
    p.add_argument("--pair-p-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=".")
    p.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Cap layers scored per model (0 = all). Useful for very deep models.",
    )
    return p.parse_args()


def _build_metadata_frame(records: List[Dict[str, Any]], model: str, cache_path: str) -> pd.DataFrame:
    """Mirror the metadata columns of ``scm._extract_tokens_from_cache`` (sans embedding).

    Adds ``row_pos`` so each surviving row can later be matched back to its
    cache record to pull a specific layer's vector.
    """
    rows: List[Dict[str, Any]] = []
    for pos, rec in enumerate(records):
        lemma = scm._normalize_text(rec.get("lexeme"))
        msps = scm._normalize_text(rec.get("mps"))
        if not lemma or not msps:
            continue
        meaning_id_raw = rec.get("meaning_index")
        if meaning_id_raw is None or (isinstance(meaning_id_raw, float) and np.isnan(meaning_id_raw)):
            continue
        if not common.cache_has_contextual_fields(rec):
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
                # Placeholder; the real per-layer value is set in _score_layer.
                # _annotate_tokens only needs the column to exist.
                "embedding_type": "perlayer",
            }
        )
    return pd.DataFrame(rows)


def _layer_vector(rec: Dict[str, Any], layer: int, embedding_type: str) -> Optional[np.ndarray]:
    orig_heads = common._as_np_heads(rec.get("orig_head_embeddings"))
    if orig_heads is None or layer >= len(orig_heads):
        return None
    if embedding_type == "orig":
        return np.asarray(orig_heads[layer], dtype=float)
    art_heads = common._as_np_heads(rec.get("art_head_embeddings"))
    if art_heads is None or layer >= len(art_heads):
        return None
    return np.asarray(orig_heads[layer], dtype=float) - np.asarray(art_heads[layer], dtype=float)


def _score_layer(
    df_annot: pd.DataFrame,
    df_systems: pd.DataFrame,
    records: List[Dict[str, Any]],
    *,
    layer: int,
    embedding_type: str,
    label_types: List[str],
    seed: int,
) -> List[Dict[str, Any]]:
    """Attach the chosen layer's vectors and run the per-system by-meaning metrics."""
    vecs = [_layer_vector(records[int(p)], layer, embedding_type) for p in df_annot["row_pos"].to_numpy()]
    df = df_annot.copy()
    df["embedding"] = vecs
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

    label_types = scm._resolve_label_types(args.label_types)
    embedding_types = [t.strip() for t in args.embedding_types.split(",") if t.strip()]
    for t in embedding_types:
        if t not in {"orig", "delta"}:
            raise SystemExit(f"Unsupported embedding type: {t}")

    cache_paths = sorted(glob.glob(args.cache_pattern))
    if not cache_paths:
        raise FileNotFoundError(f"No cache files matched: {args.cache_pattern}")

    cond_meaning = scm._load_conditioning_by_meaning(args.conditioning_by_meaning_path)
    pair_tests = scm._load_pair_tests(args.conditioning_by_form_path, alpha=args.pair_p_alpha)
    cond_counts = scm._load_cond_counts(args.cond_counts_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for cache_path in cache_paths:
        model = scm._model_stub_from_cache_path(cache_path)
        print(f"\n=== {model} ({os.path.basename(cache_path)}) ===")
        cache = common.load_cache(cache_path)
        records = list(cache.values())

        df_meta = _build_metadata_frame(records, model, cache_path)
        if df_meta.empty:
            print("  no usable tokens; skipping.")
            continue

        # Labeling is embedding-agnostic: annotate once, reuse for every layer.
        df_annot, df_systems = scm._annotate_tokens(
            df_meta,
            cond_meaning=cond_meaning,
            pair_tests=pair_tests,
            cond_counts=cond_counts,
        )

        # Layer count from the first record carrying head embeddings.
        n_layers = 0
        for rec in records:
            heads = common._as_np_heads(rec.get("orig_head_embeddings"))
            if heads is not None:
                n_layers = len(heads)
                break
        if args.max_layers and args.max_layers < n_layers:
            n_layers = args.max_layers
        print(f"  tokens={len(df_annot)}  systems={df_systems['system_id'].nunique()}  layers={n_layers}")

        all_metrics: List[Dict[str, Any]] = []
        for embedding_type in embedding_types:
            for layer in tqdm(range(n_layers), desc=f"  {embedding_type}", unit="layer"):
                all_metrics.extend(
                    _score_layer(
                        df_annot,
                        df_systems,
                        records,
                        layer=layer,
                        embedding_type=embedding_type,
                        label_types=label_types,
                        seed=args.seed,
                    )
                )

        metrics_df = pd.DataFrame(all_metrics)

        # Macro aggregation per (embedding_type, layer): reuse the headline aggregator.
        summary_parts: List[pd.DataFrame] = []
        for (embed_type, layer), sub in metrics_df.groupby(["embedding_type", "layer"], sort=True):
            agg = scm._aggregate_summary(sub)
            agg.insert(0, "layer", layer)
            agg.insert(0, "embedding_type_layer", embed_type)
            summary_parts.append(agg)
        summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

        metrics_path = os.path.join(args.output_dir, f"per_head_meaning_metrics_{model}.tsv")
        summary_path = os.path.join(args.output_dir, f"per_head_meaning_summary_{model}.tsv")
        metrics_df.to_csv(metrics_path, sep="\t", index=False)
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"  saved {metrics_path}")
        print(f"  saved {summary_path}")

        # Console preview: macro silhouette (cosine, 'all' bucket) by layer x label_type.
        if not summary_df.empty:
            view = summary_df[summary_df["pair_significance_bucket"] == "all"]
            for embed_type in embedding_types:
                ev = view[view["embedding_type_layer"] == embed_type]
                if ev.empty:
                    continue
                pivot = ev.pivot_table(
                    index="layer", columns="label_type", values="silhouette_cosine_mean"
                )
                print(f"\n  [{embed_type}] macro silhouette_cosine by layer:")
                print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
