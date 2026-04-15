# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.7.0",
#   "tqdm>=4.66.0",
# ]
# ///

"""Compute per-meaning k=2 clustering metrics within a single cache file.

Each output row corresponds to one matched `lemma x msps x semantic_label`
slice. The script evaluates both `orig` and `delta` embeddings across one or
more cache files and writes the requested conditioning columns plus k=2 and
form-based dispersion metrics.

Variance columns are scalar within-cluster variances measured as the mean
squared Euclidean distance from each point to its cluster centroid.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from run_system_clustering_metrics import (
    _extract_tokens_from_cache,
    _load_conditioning_by_meaning,
    _match_conditioning_rows_to_meanings,
    _normalize_text,
    _parse_csv_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-path",
        type=str,
        default="",
        help="Optional path to one head_embed_cache*.jsonl file.",
    )
    parser.add_argument(
        "--cache-pattern",
        type=str,
        default="head_embed_cache*.jsonl",
        help="Glob pattern for cache files when --cache-path is not provided.",
    )
    parser.add_argument(
        "--conditioning-by-meaning-path",
        type=str,
        default="conditioning_by_meaning.csv",
        help="CSV with per-meaning corpus conditioning annotations.",
    )
    parser.add_argument(
        "--lemma",
        type=str,
        default="",
        help="Optional comma-separated lemma filter.",
    )
    parser.add_argument(
        "--msps",
        type=str,
        default="",
        help="Optional comma-separated slot filter.",
    )
    parser.add_argument(
        "--system-id",
        type=str,
        default="",
        help="Optional comma-separated system_id filter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for k=2 KMeans.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="",
        help="Optional TSV path. Defaults to clustering_semantic_label_k2_<model>.tsv.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress indicators.",
    )
    return parser.parse_args()


def _model_stub_from_cache_path(cache_path: str) -> str:
    return os.path.basename(cache_path).replace("head_embed_cache_", "").replace(".jsonl", "")


def _default_output_path(cache_path: str) -> str:
    model_stub = _model_stub_from_cache_path(cache_path)
    return f"clustering_semantic_label_k2_{model_stub}.tsv"


def _default_multi_output_path(cache_paths: List[str]) -> str:
    if len(cache_paths) == 1:
        return _default_output_path(cache_paths[0])
    return "clustering_semantic_label_k2_multi.tsv"


def _build_actual_meanings(df_sys: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    actual_meanings: Dict[str, Dict[str, str]] = {}
    for meaning_id, meaning in (
        df_sys[["meaning_id", "meaning"]].drop_duplicates().sort_values(["meaning_id", "meaning"]).itertuples(index=False)
    ):
        meaning_text = _normalize_text(meaning)
        if not meaning_text:
            continue
        meaning_id_text = _normalize_text(meaning_id)
        existing = actual_meanings.get(meaning_text)
        if existing is not None and existing["meaning_id"] != meaning_id_text:
            continue
        actual_meanings[meaning_text] = {"meaning_id": meaning_id_text, "meaning": meaning_text}
    return actual_meanings


def _cluster_variance(X: np.ndarray) -> float:
    if X.size == 0:
        return float("nan")
    center = np.mean(X, axis=0)
    sq_dists = np.sum((X - center) ** 2, axis=1)
    return float(np.mean(sq_dists))


def _safe_mean(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(np.mean([a, b]))


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = np.unique(labels)
    if len(X) < 3 or unique_labels.size < 2 or unique_labels.size >= len(X):
        return float("nan")
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def _compute_k2_metrics(X: np.ndarray, *, seed: int) -> Dict[str, float]:
    out = {
        "k2_sil": float("nan"),
        "k2_var_1": float("nan"),
        "k2_var_2": float("nan"),
        "k2_var_mean": float("nan"),
    }
    if len(X) < 2:
        return out

    try:
        labels = KMeans(n_clusters=2, random_state=seed, n_init="auto").fit_predict(X)
    except Exception:
        return out

    label_values = sorted(np.unique(labels).tolist())
    if label_values:
        out["k2_var_1"] = _cluster_variance(X[labels == label_values[0]])
    if len(label_values) >= 2:
        out["k2_var_2"] = _cluster_variance(X[labels == label_values[1]])
        out["k2_var_mean"] = _safe_mean(out["k2_var_1"], out["k2_var_2"])
        out["k2_sil"] = _safe_silhouette(X, labels)
    return out


def _compute_form_metrics(X: np.ndarray, forms: np.ndarray, *, form_1: str, form_2: str) -> Dict[str, float]:
    out = {
        "form_sil": float("nan"),
        "form_var_1": float("nan"),
        "form_var_2": float("nan"),
        "form_var_mean": float("nan"),
        "all_var": float("nan"),
    }

    keep_mask = np.isin(forms, [form_1, form_2])
    if not np.any(keep_mask):
        return out

    X_kept = X[keep_mask]
    forms_kept = forms[keep_mask]
    out["all_var"] = _cluster_variance(X_kept)

    form_1_mask = forms_kept == form_1
    form_2_mask = forms_kept == form_2
    if np.any(form_1_mask):
        out["form_var_1"] = _cluster_variance(X_kept[form_1_mask])
    if np.any(form_2_mask):
        out["form_var_2"] = _cluster_variance(X_kept[form_2_mask])
    if np.any(form_1_mask) and np.any(form_2_mask):
        out["form_var_mean"] = _safe_mean(out["form_var_1"], out["form_var_2"])
        out["form_sil"] = _safe_silhouette(X_kept, forms_kept)
    return out


def _compute_rows_for_embedding_type(
    *,
    cache_path: str,
    embed_type: str,
    cond_rows_by_system: Dict[str, List[Dict[str, Any]]],
    lemma_filter: Optional[set[str]],
    msps_filter: Optional[set[str]],
    system_filter: Optional[set[str]],
    seed: int,
    show_progress: bool,
) -> List[Dict[str, Any]]:
    model = _model_stub_from_cache_path(cache_path)
    if show_progress:
        print(f"Loading {embed_type} embeddings from {os.path.basename(cache_path)}...")
    df_tokens = _extract_tokens_from_cache(
        cache_path,
        embedding_type=embed_type,
        lemma_filter=lemma_filter,
        msps_filter=msps_filter,
        system_filter=system_filter,
    )
    if df_tokens.empty:
        if show_progress:
            print(f"No usable {embed_type} embeddings found.")
        return []

    rows: List[Dict[str, Any]] = []
    grouped_systems = list(df_tokens.groupby("system_id", sort=True))
    if show_progress:
        print(f"Scoring {len(grouped_systems)} systems for {embed_type} embeddings...")
    iterator = grouped_systems
    if show_progress:
        iterator = tqdm(grouped_systems, desc=f"{embed_type} systems", unit="system")

    for system_id, df_sys in iterator:
        cond_rows = cond_rows_by_system.get(str(system_id), [])
        if not cond_rows:
            continue

        actual_meanings = _build_actual_meanings(df_sys)
        matched = _match_conditioning_rows_to_meanings(actual_meanings, cond_rows)
        if not matched:
            continue

        for semantic_label, cond_row in sorted(matched.items(), key=lambda item: str(item[0])):
            form_1 = _normalize_text(cond_row.get("form_1"))
            form_2 = _normalize_text(cond_row.get("form_2"))
            if not form_1 or not form_2:
                continue

            df_meaning = df_sys[df_sys["meaning_id"].astype(str) == str(semantic_label)].copy()
            if df_meaning.empty:
                continue

            df_meaning = df_meaning[df_meaning["form"].isin([form_1, form_2])].copy()
            if df_meaning.empty:
                continue

            X = np.stack(df_meaning["embedding"].to_numpy())
            forms = df_meaning["form"].astype(str).to_numpy()

            row = {
                "model": model,
                "embed_type": embed_type,
                "lemma": _normalize_text(cond_row.get("lemma")),
                "form_1": form_1,
                "form_1_freq": cond_row.get("form_1_freq"),
                "form_2": form_2,
                "form_2_freq": cond_row.get("form_2_freq"),
                "msps": _normalize_text(cond_row.get("msps")),
                "semantic_label": str(semantic_label),
                "semantic_def": _normalize_text(cond_row.get("def")),
                "chisq_eq": cond_row.get("chisq_eq"),
                "p_eq": cond_row.get("p_eq"),
                "p_cat": cond_row.get("p_cat"),
                "cond_type": _normalize_text(cond_row.get("cond_type")),
            }
            row.update(_compute_k2_metrics(X, seed=seed))
            row.update(_compute_form_metrics(X, forms, form_1=form_1, form_2=form_2))
            rows.append(row)

    return rows


def main() -> None:
    args = parse_args()
    if args.cache_path:
        cache_paths = [args.cache_path]
    else:
        cache_paths = sorted(glob.glob(args.cache_pattern))

    if not cache_paths:
        if args.cache_path:
            raise FileNotFoundError(f"Cache file not found: {args.cache_path}")
        raise FileNotFoundError(f"No cache files matched: {args.cache_pattern}")

    cond_meaning = _load_conditioning_by_meaning(args.conditioning_by_meaning_path)
    cond_rows_by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in cond_meaning.to_dict("records"):
        cond_rows_by_system[str(row["system_id"])].append(row)

    lemma_filter = _parse_csv_list(args.lemma)
    msps_filter = _parse_csv_list(args.msps)
    system_filter = _parse_csv_list(args.system_id)

    output_rows: List[Dict[str, Any]] = []
    if not args.no_progress:
        print(f"Found {len(cache_paths)} cache file(s).")

    cache_iterator = cache_paths
    if not args.no_progress and len(cache_paths) > 1:
        cache_iterator = tqdm(cache_paths, desc="cache files", unit="cache")

    for cache_path in cache_iterator:
        for embed_type in ["orig", "delta"]:
            output_rows.extend(
                _compute_rows_for_embedding_type(
                    cache_path=cache_path,
                    embed_type=embed_type,
                    cond_rows_by_system=cond_rows_by_system,
                    lemma_filter=lemma_filter,
                    msps_filter=msps_filter,
                    system_filter=system_filter,
                    seed=args.seed,
                    show_progress=not args.no_progress,
                )
            )

    if not output_rows:
        raise RuntimeError("No matched semantic-label slices were found.")

    out_df = pd.DataFrame(output_rows)
    sort_cols = ["model", "embed_type", "lemma", "msps", "semantic_label", "semantic_def"]
    out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    output_path = args.output_path or _default_multi_output_path(cache_paths)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()