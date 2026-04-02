# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.0.0",
#   "tqdm>=4.66.0",
# ]
# ///

"""Compute clustering geometry metrics within lemma x msps systems.

This runner is the lightweight geometry-focused clustering report. It only
supports `orig` and `delta` embeddings, never pools across slots or lexemes,
and uses canonical label types such as `raw` and `conditioned-exclude`.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from run_system_clustering_metrics import (
    CANONICAL_LABEL_TYPES,
    LABEL_COLUMN_BY_TYPE,
    VALIDITY_COLUMN_BY_TYPE,
    _annotate_tokens,
    _extract_tokens_from_cache,
    _load_cond_counts,
    _load_conditioning_by_meaning,
    _load_pair_tests,
    _model_tag,
    _parse_csv_list,
    _resolve_label_types,
    _safe_float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-pattern",
        type=str,
        default="head_embed_cache_*.jsonl",
        help="Glob pattern for cache files.",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["orig", "delta"],
        default="orig",
        help="Clustering embedding type.",
    )
    parser.add_argument(
        "--embedding-source",
        dest="embedding_type",
        choices=["orig", "delta"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--label-types",
        type=str,
        default="raw,conditioned-exclude",
        help=(
            "Comma-separated label types to score. Supported values: "
            + ", ".join(CANONICAL_LABEL_TYPES)
            + ", all_conditioned, all."
        ),
    )
    parser.add_argument(
        "--conditioning-by-meaning-path",
        type=str,
        default="conditioning_by_meaning.csv",
        help="CSV with per-meaning corpus conditioning annotations.",
    )
    parser.add_argument(
        "--conditioning-by-form-path",
        type=str,
        default="conditioning_by_form.csv",
        help="CSV with per-system omnibus conditioning tests.",
    )
    parser.add_argument(
        "--cond-counts-path",
        type=str,
        default="df_cond_types.csv",
        help="CSV with per-system counts of no_cond/prob/cat meanings.",
    )
    parser.add_argument(
        "--pair-p-alpha",
        type=float,
        default=0.05,
        help="Threshold used to mark a system-level conditioning test as significant.",
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
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Number of bootstrap samples per system for silhouette CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output TSVs.",
    )
    return parser.parse_args()


def _intra_inter_ratio(X: np.ndarray, y: np.ndarray, metric: str) -> float:
    d = pairwise_distances(X, metric=metric)
    intra = []
    inter = []
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                intra.append(d[i, j])
            else:
                inter.append(d[i, j])
    if not intra or not inter:
        return float("nan")
    return float(np.mean(intra) / np.mean(inter))


def _dunn_index(X: np.ndarray, y: np.ndarray, metric: str) -> float:
    labels = sorted(set(y.tolist()))
    d = pairwise_distances(X, metric=metric)

    max_intra = 0.0
    min_inter = float("inf")

    for lab in labels:
        idx = np.where(y == lab)[0]
        if len(idx) >= 2:
            sub = d[np.ix_(idx, idx)]
            max_intra = max(max_intra, float(np.max(sub)))

    for i, lab_a in enumerate(labels):
        idx_a = np.where(y == lab_a)[0]
        for lab_b in labels[i + 1 :]:
            idx_b = np.where(y == lab_b)[0]
            sub = d[np.ix_(idx_a, idx_b)]
            min_inter = min(min_inter, float(np.min(sub)))

    if max_intra <= 0.0 or min_inter == float("inf"):
        return float("nan")
    return float(min_inter / max_intra)


def _centroid_separation(X: np.ndarray, y: np.ndarray, metric: str) -> float:
    labels = sorted(set(y.tolist()))
    centroids = []
    for lab in labels:
        idx = np.where(y == lab)[0]
        centroids.append(np.mean(X[idx], axis=0))
    if len(centroids) < 2:
        return float("nan")
    cd = pairwise_distances(np.stack(centroids), metric=metric)
    vals = cd[np.triu_indices_from(cd, k=1)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def _nearest_centroid_loo_accuracy(X: np.ndarray, y: np.ndarray, metric: str) -> float:
    labels = sorted(set(y.tolist()))
    n = len(y)
    correct = 0
    used = 0

    for i in range(n):
        xi = X[i]
        yi = y[i]
        centroids = {}
        for lab in labels:
            idx = np.where(y == lab)[0]
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
            centroids[lab] = np.mean(X[idx], axis=0)

        if not centroids:
            continue

        labs = list(centroids.keys())
        mat = np.stack([centroids[lab] for lab in labs])
        dist = pairwise_distances(xi.reshape(1, -1), mat, metric=metric)[0]
        pred = labs[int(np.argmin(dist))]
        used += 1
        if pred == yi:
            correct += 1

    if used == 0:
        return float("nan")
    return float(correct / used)


def _label_entropy(y: np.ndarray) -> float:
    counts = Counter(y.tolist())
    n = len(y)
    probs = np.array([count / n for count in counts.values()], dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(-(probs * np.log2(probs)).sum())


def _bootstrap_silhouette(
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    vals = []
    n = len(y)
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            continue
        try:
            vals.append(float(silhouette_score(Xb, yb, metric=metric)))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return lo, hi


def _per_system_metrics(
    df_eval: pd.DataFrame,
    *,
    label_col: str,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    y = df_eval[label_col].astype(str).to_numpy()
    X = np.stack(df_eval["embedding"].to_numpy())

    label_counts = Counter(y.tolist())
    singleton_count = sum(1 for _, v in label_counts.items() if v == 1)
    sparse_count = sum(1 for _, v in label_counts.items() if v < 3)

    out = {
        "n_samples": int(len(y)),
        "n_labels": int(len(label_counts)),
        "label_entropy": _label_entropy(y),
        "label_max_count": int(max(label_counts.values())) if label_counts else 0,
        "label_min_count": int(min(label_counts.values())) if label_counts else 0,
        "label_singleton_fraction": float(singleton_count / len(label_counts)) if label_counts else float("nan"),
        "label_sparse_fraction": float(sparse_count / len(label_counts)) if label_counts else float("nan"),
    }

    if len(label_counts) < 2:
        out.update(
            {
                "silhouette": float("nan"),
                "silhouette_ci_low": float("nan"),
                "silhouette_ci_high": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan"),
                "dunn": float("nan"),
                "intra_inter_ratio": float("nan"),
                "centroid_separation": float("nan"),
                "nearest_centroid_loo_acc": float("nan"),
            }
        )
        return out

    try:
        out["silhouette"] = float(silhouette_score(X, y, metric=metric))
    except Exception:
        out["silhouette"] = float("nan")

    ci_lo, ci_hi = _bootstrap_silhouette(X, y, metric=metric, n_samples=bootstrap_samples, rng=rng)
    out["silhouette_ci_low"] = ci_lo
    out["silhouette_ci_high"] = ci_hi

    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, y))
    except Exception:
        out["davies_bouldin"] = float("nan")

    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, y))
    except Exception:
        out["calinski_harabasz"] = float("nan")

    out["dunn"] = _dunn_index(X, y, metric=metric)
    out["intra_inter_ratio"] = _intra_inter_ratio(X, y, metric=metric)
    out["centroid_separation"] = _centroid_separation(X, y, metric=metric)
    out["nearest_centroid_loo_acc"] = _nearest_centroid_loo_accuracy(X, y, metric=metric)
    return out


def _aggregate(sub: pd.DataFrame, value_cols: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    macro = {"aggregation": "macro"}
    for col in value_cols:
        macro[col] = _safe_float(sub[col].mean())
    rows.append(macro)

    weights = sub["n_samples"].to_numpy(dtype=float)
    micro = {"aggregation": "micro", "total_samples": int(np.sum(weights))}
    for col in value_cols:
        vals = sub[col].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            micro[col] = float("nan")
            continue
        w = weights[mask]
        v = vals[mask]
        if np.sum(w) <= 0:
            micro[col] = float("nan")
        else:
            micro[col] = float(np.average(v, weights=w))
    rows.append(micro)
    return rows


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    label_types = _resolve_label_types(args.label_types)

    lemma_filter = _parse_csv_list(args.lemma)
    msps_filter = _parse_csv_list(args.msps)
    system_filter = _parse_csv_list(args.system_id)

    cache_paths = sorted(glob.glob(args.cache_pattern))
    if not cache_paths:
        raise FileNotFoundError(f"No cache files matched: {args.cache_pattern}")

    cond_meaning = _load_conditioning_by_meaning(args.conditioning_by_meaning_path)
    pair_tests = _load_pair_tests(args.conditioning_by_form_path, alpha=args.pair_p_alpha)
    cond_counts = _load_cond_counts(args.cond_counts_path)

    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    iterator = cache_paths
    if not args.no_progress:
        iterator = tqdm(cache_paths, desc="Cache Files")

    for cache_path in iterator:
        df_tokens = _extract_tokens_from_cache(
            cache_path,
            embedding_type=args.embedding_type,
            lemma_filter=lemma_filter,
            msps_filter=msps_filter,
            system_filter=system_filter,
        )
        if df_tokens.empty:
            continue

        df_tokens, df_systems = _annotate_tokens(
            df_tokens,
            cond_meaning=cond_meaning,
            pair_tests=pair_tests,
            cond_counts=cond_counts,
        )

        for _, sys_row in df_systems.iterrows():
            system_id = str(sys_row["system_id"])
            df_sys = df_tokens[df_tokens["system_id"] == system_id]

            for label_type in label_types:
                label_col = LABEL_COLUMN_BY_TYPE[label_type]
                validity_col = VALIDITY_COLUMN_BY_TYPE[label_type]
                df_eval = df_sys[df_sys[validity_col]].copy()
                if label_type.startswith("conditioned-"):
                    df_eval = df_eval[df_eval[label_col].notna()].copy()
                if df_eval.empty:
                    continue

                for distance in ["cosine", "euclidean"]:
                    row = sys_row.to_dict()
                    row.update(
                        {
                            "embedding_type": args.embedding_type,
                            "label_type": label_type,
                            "label_column": label_col,
                            "distance": distance,
                        }
                    )
                    row.update(
                        _per_system_metrics(
                            df_eval,
                            label_col=label_col,
                            metric=distance,
                            bootstrap_samples=args.bootstrap_samples,
                            rng=rng,
                        )
                    )
                    metric_rows.append(row)

    if not metric_rows:
        raise RuntimeError("No usable records found for metrics.")

    metrics_df = pd.DataFrame(metric_rows)

    value_cols = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "dunn",
        "intra_inter_ratio",
        "centroid_separation",
        "nearest_centroid_loo_acc",
    ]
    for (model, label_type, distance), sub in metrics_df.groupby(["model", "label_type", "distance"]):
        for bucket_name, bucket_df in [
            ("all", sub),
            ("pair_significant", sub[sub["pair_significant"] == True]),
            ("pair_not_significant", sub[sub["pair_significant"] == False]),
        ]:
            if bucket_df.empty:
                continue
            for agg in _aggregate(bucket_df, value_cols):
                row = dict(agg)
                row["model"] = model
                row["embedding_type"] = str(bucket_df["embedding_type"].iloc[0])
                row["label_type"] = label_type
                row["distance"] = distance
                row["pair_significance_bucket"] = bucket_name
                row["n_systems"] = int(bucket_df["system_id"].nunique())
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = _model_tag(cache_paths)
    per_system_path = os.path.join(args.output_dir, f"clustering_metrics_per_system_{args.embedding_type}_{model_tag}.tsv")
    summary_path = os.path.join(args.output_dir, f"clustering_metrics_summary_{args.embedding_type}_{model_tag}.tsv")

    metrics_df.sort_values(["model", "system_id", "label_type", "distance"]).to_csv(per_system_path, sep="\t", index=False)
    summary_df.sort_values(["model", "label_type", "distance", "pair_significance_bucket", "aggregation"]).to_csv(
        summary_path,
        sep="\t",
        index=False,
    )

    print(f"Saved {per_system_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
