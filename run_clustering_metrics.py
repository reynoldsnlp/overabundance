# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.0.0",
#   "tqdm>=4.66.0",
#   "matplotlib>=3.8.0",
# ]
# ///

"""Compute label-based clustering metrics for cached head embeddings.

This script does NOT run a clustering algorithm. It treats meaning_index labels
as fixed cluster labels and evaluates cohesion/separation per lexeme.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import multiprocessing as mp
import os
import re
import time
from collections import Counter
import difflib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

import overabundance_common as common


_WORKER_CACHE_BY_PATH: Dict[str, Dict[Any, Dict[str, Any]]] = {}
_WORKER_HUMAN_COND_CACHE: Dict[tuple[str, str], Dict[str, Any]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-pattern",
        type=str,
        default="head_embed_cache_*.jsonl",
        help="Glob pattern for cache files.",
    )
    parser.add_argument(
        "--embedding-source",
        choices=["delta", "delta_from_raw", "head", "head_delta", "head_delta_from_raw", "orig", "art", "orig_head", "art_head"],
        default="delta",
        help="Vector source: delta, raw contextual, or head-based variants.",
    )
    parser.add_argument(
        "--head-indices",
        type=str,
        default="",
        help="Comma-separated head indices for --embedding-source=head (default: all heads).",
    )
    parser.add_argument(
        "--head-sweep",
        action="store_true",
        help="Evaluate multiple head selections in one run (ignored for --embedding-source=delta).",
    )
    parser.add_argument(
        "--head-combo-sizes",
        type=str,
        default="1,2",
        help="Comma-separated combination sizes for --head-sweep (e.g. '1,2').",
    )
    parser.add_argument(
        "--include-all-heads",
        action="store_true",
        help="When --head-sweep is set, also evaluate pooled all-heads selection.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=200,
        help="Number of bootstrap samples per lexeme for silhouette CI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--global-label-scope",
        choices=["lexeme_prefixed", "meaning_only"],
        default="lexeme_prefixed",
        help="Label scope for pooled-global metrics across lexemes.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes for head-config evaluation.",
    )
    parser.add_argument(
        "--no-incremental-charts",
        action="store_true",
        help="Disable periodic progress chart rendering.",
    )
    parser.add_argument(
        "--incremental-chart-every",
        type=int,
        default=10,
        help="Refresh progress chart every N completed head-config tasks.",
    )
    parser.add_argument(
        "--chart-output-dir",
        type=str,
        default="docs/clustering",
        help="Directory for incremental progress charts.",
    )
    parser.add_argument(
        "--heartbeat-file",
        type=str,
        default="clustering_metrics_heartbeat.log",
        help="Append-only worker heartbeat log for live progress monitoring.",
    )
    parser.add_argument(
        "--heartbeat-every",
        type=int,
        default=5,
        help="Write worker heartbeat every N lexemes per task.",
    )
    parser.add_argument(
        "--label-scheme",
        choices=["raw_meaning", "human_expected", "both"],
        default="both",
        help="Ground-truth label regime to score against.",
    )
    parser.add_argument(
        "--conditioning-by-meaning-path",
        type=str,
        default="conditioning_by_meaning.csv",
        help="CSV with per-meaning human conditioning tests.",
    )
    parser.add_argument(
        "--conditioning-by-form-path",
        type=str,
        default="conditioning_by_form.csv",
        help="CSV with per-lemma/per-mps omnibus human conditioning tests.",
    )
    parser.add_argument(
        "--human-p-alpha",
        type=float,
        default=0.05,
        help="Significance threshold used to convert human p-values into expected cluster groupings.",
    )
    parser.add_argument(
        "--skip-global",
        action="store_true",
        help="Skip pooled-global clustering metrics and only compute per-lexeme results.",
    )
    return parser.parse_args()


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _parse_combo_sizes(raw: str) -> List[int]:
    out: List[int] = []
    for piece in (raw or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(int(piece))
    return sorted(set(x for x in out if x > 0))


def _infer_head_count(cache_paths: List[str]) -> Optional[int]:
    for cache_path in cache_paths:
        cache = common.load_cache(cache_path)
        for rec in cache.values():
            heads = rec.get("orig_head_embeddings")
            if isinstance(heads, list) and heads:
                return len(heads)
    return None


def _head_label(embedding_source: str, head_indices: Optional[List[int]]) -> str:
    if embedding_source not in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}:
        return ""
    if head_indices is None:
        return "all"
    return ",".join(str(i) for i in head_indices)


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
    s = re.sub(r'\.\s*\(I used[^)]*\)\s*$', "", s, flags=re.IGNORECASE)
    s = re.sub(r'\.\s*I used.*$', "", s, flags=re.IGNORECASE)
    return s.strip()


def _load_human_conditioning(by_form_path: str, by_meaning_path: str) -> Dict[str, Any]:
    cache_key = (by_form_path, by_meaning_path)
    cached = _WORKER_HUMAN_COND_CACHE.get(cache_key)
    if cached is not None:
        return cached

    overall_p: Dict[tuple[str, str], float] = {}
    meaning_rows: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    with open(by_form_path, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (_normalize_text(row.get("lemma")), _normalize_text(row.get("msps")))
            overall_p[key] = _safe_float(row.get("p_value"))

    tmp: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    with open(by_meaning_path, newline="", encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lemma = _normalize_text(row.get("lemma"))
            mps = _normalize_text(row.get("msps"))
            if not lemma or not mps:
                continue
            out = dict(row)
            out["lemma"] = lemma
            out["msps"] = mps
            out["def_norm"] = _normalize_text(row.get("def"))
            out["def_match_key"] = _meaning_match_key(row.get("def"))
            out["p_eq"] = _safe_float(row.get("p_eq"))
            out["p_cat"] = _safe_float(row.get("p_cat"))
            out["form_1_freq"] = _safe_float(row.get("form_1_freq"))
            out["form_2_freq"] = _safe_float(row.get("form_2_freq"))
            tmp.setdefault((lemma, mps), []).append(out)

    meaning_rows = tmp
    cached = {"overall_p": overall_p, "meaning_rows": meaning_rows}
    _WORKER_HUMAN_COND_CACHE[cache_key] = cached
    return cached


def _conditioning_bucket(row: Dict[str, Any]) -> str:
    cond_type = str(row.get("cond_type", "")).strip() or "other"
    f1 = _safe_float(row.get("form_1_freq"))
    f2 = _safe_float(row.get("form_2_freq"))
    dominant = _normalize_text(row.get("form_1")) if f1 >= f2 else _normalize_text(row.get("form_2"))
    if cond_type == "no_cond":
        return "no_cond"
    if dominant:
        return f"{cond_type}:{dominant}"
    return cond_type


def _match_human_rows_to_meanings(
    actual_meanings: Dict[str, Dict[str, str]],
    conditioning_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    matched: Dict[str, Dict[str, Any]] = {}
    used_actual: set[str] = set()
    actual_by_match_key: Dict[str, List[Dict[str, str]]] = {}

    for actual in actual_meanings.values():
        match_key = _meaning_match_key(actual["meaning"])
        actual_by_match_key.setdefault(match_key, []).append(actual)

    for cond_row in conditioning_rows:
        key = cond_row["def_norm"]
        actual = actual_meanings.get(key)
        if actual is None:
            continue
        mid = actual["meaning_index"]
        matched[mid] = cond_row
        used_actual.add(key)

    for cond_row in conditioning_rows:
        if cond_row in matched.values():
            continue
        match_key = cond_row.get("def_match_key", cond_row["def_norm"])
        candidates = [
            actual
            for actual in actual_by_match_key.get(match_key, [])
            if actual["meaning"] not in used_actual
        ]
        if len(candidates) != 1:
            continue
        actual = candidates[0]
        mid = actual["meaning_index"]
        matched[mid] = cond_row
        used_actual.add(actual["meaning"])

    remaining_actual = [k for k in actual_meanings.keys() if k not in used_actual]
    for cond_row in conditioning_rows:
        if cond_row in matched.values():
            continue
        best_key = None
        best_score = 0.0
        for actual_key in remaining_actual:
            score = difflib.SequenceMatcher(
                None,
                cond_row.get("def_match_key", cond_row["def_norm"]),
                _meaning_match_key(actual_key),
            ).ratio()
            if score > best_score:
                best_score = score
                best_key = actual_key
        if best_key is not None and best_score >= 0.88:
            mid = actual_meanings[best_key]["meaning_index"]
            matched[mid] = cond_row
            remaining_actual = [k for k in remaining_actual if k != best_key]

    return matched


def _human_expected_labels(
    rows: List[Dict[str, Any]],
    *,
    human_cond: Dict[str, Any],
    alpha: float,
) -> List[str]:
    labels = ["human::other"] * len(rows)
    by_lexeme_mps: Dict[tuple[str, str], List[int]] = {}
    for idx, row in enumerate(rows):
        lemma = _normalize_text(row.get("lexeme"))
        mps = _normalize_text(row.get("mps"))
        by_lexeme_mps.setdefault((lemma, mps), []).append(idx)

    for (lemma, mps), idxs in by_lexeme_mps.items():
        if not idxs:
            continue
        overall_p = human_cond["overall_p"].get((lemma, mps), float("nan"))
        if not np.isfinite(overall_p) or overall_p >= alpha:
            for idx in idxs:
                labels[idx] = f"{mps}::merged"
            continue

        actual_meanings: Dict[str, Dict[str, str]] = {}
        for idx in idxs:
            meaning = _normalize_text(rows[idx].get("meaning"))
            mid = _normalize_text(rows[idx].get("meaning_index"))
            if meaning and mid:
                actual_meanings.setdefault(meaning, {"meaning_index": mid, "meaning": meaning})

        conditioning_rows = human_cond["meaning_rows"].get((lemma, mps), [])
        matched = _match_human_rows_to_meanings(actual_meanings, conditioning_rows)

        bucket_by_mid: Dict[str, str] = {}
        bucket_counts: Counter[str] = Counter()
        for mid, cond_row in matched.items():
            bucket = _conditioning_bucket(cond_row)
            bucket_by_mid[mid] = bucket
            bucket_counts[bucket] += 1

        default_bucket = bucket_counts.most_common(1)[0][0] if bucket_counts else "other"
        for idx in idxs:
            mid = _normalize_text(rows[idx].get("meaning_index"))
            bucket = bucket_by_mid.get(mid, default_bucket)
            labels[idx] = f"{mps}::{bucket}"

    return labels


def _resolve_head_configs(args: argparse.Namespace, cache_paths: List[str], explicit: Optional[List[int]]) -> List[Optional[List[int]]]:
    is_head_source = args.embedding_source in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}
    if not is_head_source:
        return [None]

    if not args.head_sweep:
        return [explicit]

    n_heads = _infer_head_count(cache_paths)
    if n_heads is None:
        return []

    combo_sizes = _parse_combo_sizes(args.head_combo_sizes)
    if not combo_sizes:
        combo_sizes = [1, 2]

    configs: List[Optional[List[int]]] = []
    if args.include_all_heads:
        configs.append(None)

    for k in combo_sizes:
        if k > n_heads:
            continue
        for combo in itertools.combinations(range(n_heads), k):
            configs.append(list(combo))

    return configs


def _model_tag_from_cache_paths(cache_paths: List[str]) -> str:
    models = [
        os.path.basename(p)
        .replace("head_embed_cache_", "")
        .replace("delta_cache_", "")
        .replace(".jsonl", "")
        for p in cache_paths
    ]
    uniq = sorted(set(models))
    if len(uniq) == 1:
        return uniq[0]
    return "multi"


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
    c = Counter(y.tolist())
    n = len(y)
    probs = np.array([v / n for v in c.values()], dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(-(probs * np.log2(probs)).sum())


def _bootstrap_silhouette(X: np.ndarray, y: np.ndarray, metric: str, n_samples: int, rng: np.random.Generator):
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


def _resolve_label_schemes(raw: str) -> List[str]:
    if raw == "both":
        return ["raw_meaning", "human_expected"]
    return [raw]


def _per_lexeme_metrics(
    rows: List[Dict[str, Any]],
    *,
    metric: str,
    bootstrap_samples: int,
    rng: np.random.Generator,
    label_values: Optional[List[str]] = None,
) -> Dict[str, Any]:
    X = np.stack([r["vector"] for r in rows])
    if label_values is None:
        y = np.array([str(r["meaning_index"]) for r in rows])
    else:
        y = np.array([str(x) for x in label_values])

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
        "token_mismatch_rate": float(np.mean([bool(r.get("token_mismatch", False)) for r in rows])),
        "missing_meaning_count": int(sum(1 for r in rows if r.get("missing_meaning", False))),
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


def _aggregate(df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
    out_rows = []

    macro = {"aggregation": "macro"}
    for c in value_cols:
        macro[c] = _safe_float(df[c].mean())
    out_rows.append(macro)

    weights = df["n_samples"].to_numpy(dtype=float)
    weight_sum = float(np.sum(weights))
    micro = {"aggregation": "micro"}
    for c in value_cols:
        vals = df[c].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            micro[c] = float("nan")
            continue
        w = weights[mask]
        v = vals[mask]
        if np.sum(w) <= 0:
            micro[c] = float("nan")
        else:
            micro[c] = float(np.average(v, weights=w))
    micro["total_samples"] = int(weight_sum)
    out_rows.append(micro)

    return pd.DataFrame(out_rows)


def _load_cache_for_worker(cache_path: str) -> Dict[Any, Dict[str, Any]]:
    cached = _WORKER_CACHE_BY_PATH.get(cache_path)
    if cached is None:
        cached = common.load_cache(cache_path)
        _WORKER_CACHE_BY_PATH[cache_path] = cached
    return cached


def _append_rows(path: str, rows: List[Dict[str, Any]], *, header_written: bool) -> bool:
    if not rows:
        return header_written
    df = pd.DataFrame(rows)
    mode = "a" if header_written else "w"
    df.to_csv(path, sep="\t", index=False, mode=mode, header=not header_written)
    return True


def _write_progress_chart(aggregate_path: str, out_path: str) -> None:
    if not os.path.exists(aggregate_path):
        return
    df = pd.read_csv(aggregate_path, sep="\t")
    if df.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    sub = df[(df["aggregation"] == "micro") & (df["distance"] == "cosine")].copy()
    if sub.empty:
        return
    sub["silhouette"] = pd.to_numeric(sub["silhouette"], errors="coerce")
    sub = sub[np.isfinite(sub["silhouette"])]
    if sub.empty:
        return

    best = sub.groupby("model", as_index=False)["silhouette"].max().sort_values("silhouette", ascending=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(best["model"], best["silhouette"], color="#1f77b4")
    ax.set_title("Best Micro-Cosine Silhouette So Far")
    ax.set_ylabel("silhouette")
    ax.set_xlabel("model")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _append_heartbeat(path: str, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _worker_eval_task(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = int(task["task_id"])
    cache_path = str(task["cache_path"])
    model_stub = str(task["model_stub"])
    embedding_source = str(task["embedding_source"])
    head_indices = task["head_indices"]
    bootstrap_samples = int(task["bootstrap_samples"])
    seed = int(task["seed"])
    global_label_scope = str(task["global_label_scope"])
    skip_global = bool(task.get("skip_global", False))
    label_schemes = _resolve_label_schemes(str(task.get("label_scheme", "both")))
    conditioning_by_meaning_path = str(task.get("conditioning_by_meaning_path", "conditioning_by_meaning.csv"))
    conditioning_by_form_path = str(task.get("conditioning_by_form_path", "conditioning_by_form.csv"))
    human_p_alpha = float(task.get("human_p_alpha", 0.05))
    heartbeat_file = str(task.get("heartbeat_file", "")).strip()
    heartbeat_every = max(1, int(task.get("heartbeat_every", 5)))

    rng = np.random.default_rng(seed + task_id * 9973)
    cache = _load_cache_for_worker(cache_path)
    human_cond = _load_human_conditioning(conditioning_by_form_path, conditioning_by_meaning_path)

    if heartbeat_file:
        _append_heartbeat(
            heartbeat_file,
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} START task={task_id} model={model_stub} heads={_head_label(embedding_source, head_indices)}",
        )

    by_lexeme: Dict[str, List[Dict[str, Any]]] = {}
    all_points: List[Dict[str, Any]] = []
    for rec in cache.values():
        lexeme = rec.get("lexeme")
        if not isinstance(lexeme, str) or not lexeme:
            continue

        vec = common.select_record_embedding(
            rec,
            embedding_source=embedding_source,
            head_indices=head_indices,
        )
        if vec is None:
            continue

        meaning_index = rec.get("meaning_index")
        if meaning_index is None or (isinstance(meaning_index, float) and np.isnan(meaning_index)):
            continue

        token_mismatch = False
        oc = rec.get("orig_token_count")
        ac = rec.get("art_token_count")
        if isinstance(oc, int) and isinstance(ac, int):
            token_mismatch = oc != ac

        point = {
            "vector": vec,
            "meaning_index": str(meaning_index),
            "meaning": _normalize_text(rec.get("meaning")),
            "lexeme": lexeme,
            "mps": _normalize_text(rec.get("mps")),
            "token_mismatch": token_mismatch,
            "missing_meaning": False,
        }
        by_lexeme.setdefault(lexeme, []).append(point)
        all_points.append(point)

    per_rows: List[Dict[str, Any]] = []
    head_label = _head_label(embedding_source, head_indices)

    lex_items = sorted(by_lexeme.items())
    for lex_i, (lexeme, rows) in enumerate(lex_items, start=1):
        label_values_by_scheme = {
            "raw_meaning": None,
            "human_expected": _human_expected_labels(rows, human_cond=human_cond, alpha=human_p_alpha),
        }
        for label_scheme in label_schemes:
            label_values = label_values_by_scheme[label_scheme]
            for dist in ["cosine", "euclidean"]:
                met = _per_lexeme_metrics(
                    rows,
                    metric=dist,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                    label_values=label_values,
                )
                met["model"] = model_stub
                met["lexeme"] = lexeme
                met["distance"] = dist
                met["embedding_source"] = embedding_source
                met["head_indices"] = head_label
                met["condition"] = "per_lexeme"
                met["global_label_scope"] = "meaning_only"
                met["label_scheme"] = label_scheme
                per_rows.append(met)

        if heartbeat_file and (lex_i == 1 or lex_i == len(lex_items) or (lex_i % heartbeat_every) == 0):
            _append_heartbeat(
                heartbeat_file,
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} TASK task={task_id} model={model_stub} heads={head_label} lexeme={lex_i}/{len(lex_items)}",
            )

    global_rows: List[Dict[str, Any]] = []
    if not skip_global:
        if heartbeat_file:
            _append_heartbeat(
                heartbeat_file,
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} GLOBAL_START task={task_id} model={model_stub} heads={head_label}",
            )
        human_global_labels = _human_expected_labels(all_points, human_cond=human_cond, alpha=human_p_alpha)
        for label_scheme in label_schemes:
            if label_scheme == "raw_meaning":
                base_labels = [str(r["meaning_index"]) for r in all_points]
            else:
                base_labels = human_global_labels

            if global_label_scope == "lexeme_prefixed":
                labels = [f"{r['lexeme']}::{lab}" for r, lab in zip(all_points, base_labels)]
            else:
                labels = base_labels

            for dist in ["cosine", "euclidean"]:
                met = _per_lexeme_metrics(
                    all_points,
                    metric=dist,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                    label_values=labels,
                )
                met["model"] = model_stub
                met["lexeme"] = "__ALL__"
                met["distance"] = dist
                met["embedding_source"] = embedding_source
                met["head_indices"] = head_label
                met["condition"] = "global_pooled"
                met["global_label_scope"] = global_label_scope
                met["label_scheme"] = label_scheme
                global_rows.append(met)
                if heartbeat_file:
                    _append_heartbeat(
                        heartbeat_file,
                        (
                            f"{time.strftime('%Y-%m-%d %H:%M:%S')} GLOBAL_STEP task={task_id} "
                            f"model={model_stub} heads={head_label} label_scheme={label_scheme} distance={dist}"
                        ),
                    )

    metric_cols = [
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "dunn",
        "intra_inter_ratio",
        "centroid_separation",
        "nearest_centroid_loo_acc",
    ]

    aggregate_rows: List[Dict[str, Any]] = []
    if per_rows:
        per_df = pd.DataFrame(per_rows)
        for (distance, label_scheme), sub in per_df.groupby(["distance", "label_scheme"]):
            agg = _aggregate(sub, metric_cols)
            for _, row in agg.iterrows():
                out = row.to_dict()
                out["distance"] = distance
                out["model"] = model_stub
                out["head_indices"] = head_label
                out["embedding_source"] = embedding_source
                out["label_scheme"] = label_scheme
                aggregate_rows.append(out)

    result = {
        "task_id": task_id,
        "model_stub": model_stub,
        "head_label": head_label,
        "per_rows": per_rows,
        "global_rows": global_rows,
        "aggregate_rows": aggregate_rows,
        "n_lexemes": len(by_lexeme),
        "n_points": len(all_points),
    }

    if heartbeat_file:
        _append_heartbeat(
            heartbeat_file,
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} DONE task={task_id} model={model_stub} heads={head_label} lexemes={len(by_lexeme)} points={len(all_points)}",
        )

    return result


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    explicit_head_indices = common.parse_head_indices(args.head_indices)

    cache_paths = sorted(glob.glob(args.cache_pattern))
    if not cache_paths:
        print(f"No cache files matched: {args.cache_pattern}")
        return

    head_configs = _resolve_head_configs(args, cache_paths, explicit_head_indices)
    is_head_source = args.embedding_source in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}
    if is_head_source and not head_configs:
        print("No head configurations available. Ensure caches include raw head embeddings.")
        return
    if is_head_source:
        print(f"Evaluating {len(head_configs)} head configuration(s).")

    model_tag = _model_tag_from_cache_paths(cache_paths)
    use_progress = not args.no_progress

    suffix = args.embedding_source
    if is_head_source:
        if args.head_sweep:
            suffix = "head_sweep"
        elif explicit_head_indices:
            suffix = f"head_{'-'.join(str(i) for i in explicit_head_indices)}"

    per_lexeme_path = f"clustering_metrics_per_lexeme_{suffix}_{model_tag}.tsv"
    global_path = f"clustering_metrics_global_{suffix}_{model_tag}.tsv"
    aggregate_path = f"clustering_metrics_aggregate_{suffix}_{model_tag}.tsv"

    for path in [per_lexeme_path, global_path, aggregate_path]:
        if os.path.exists(path):
            os.remove(path)
    if args.heartbeat_file:
        with open(args.heartbeat_file, "w", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} RUN_START tasks_pending=0\n")

    tasks: List[Dict[str, Any]] = []
    task_id = 0
    for cache_path in cache_paths:
        model_stub = (
            os.path.basename(cache_path)
            .replace("head_embed_cache_", "")
            .replace("delta_cache_", "")
            .replace(".jsonl", "")
        )
        for head_indices in head_configs:
            tasks.append(
                {
                    "task_id": task_id,
                    "cache_path": cache_path,
                    "model_stub": model_stub,
                    "embedding_source": args.embedding_source,
                    "head_indices": head_indices,
                    "bootstrap_samples": args.bootstrap_samples,
                    "seed": args.seed,
                    "global_label_scope": args.global_label_scope,
                    "label_scheme": args.label_scheme,
                    "conditioning_by_meaning_path": args.conditioning_by_meaning_path,
                    "conditioning_by_form_path": args.conditioning_by_form_path,
                    "human_p_alpha": args.human_p_alpha,
                    "skip_global": args.skip_global,
                    "heartbeat_file": args.heartbeat_file,
                    "heartbeat_every": args.heartbeat_every,
                }
            )
            task_id += 1

    if not tasks:
        print("No tasks to run.")
        return

    print(f"Running {len(tasks)} head-config task(s) across {len(cache_paths)} cache file(s) with {args.workers} worker(s).")
    if args.heartbeat_file:
        with open(args.heartbeat_file, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} RUN_CONFIG tasks_total={len(tasks)} workers={args.workers}\n")
        print(f"Heartbeat log: {args.heartbeat_file}")

    per_written = False
    global_written = False
    agg_written = False
    completed = 0
    charts_enabled = not args.no_incremental_charts
    progress_chart_path = os.path.join(args.chart_output_dir, f"clustering_progress_{suffix}_{model_tag}.png")

    with mp.Pool(processes=max(1, args.workers)) as pool:
        iterator = pool.imap_unordered(_worker_eval_task, tasks, chunksize=1)
        pbar = tqdm(total=len(tasks), desc="Head-Config Tasks", disable=not use_progress)
        for result in iterator:
            per_written = _append_rows(per_lexeme_path, result.get("per_rows", []), header_written=per_written)
            global_written = _append_rows(global_path, result.get("global_rows", []), header_written=global_written)
            agg_written = _append_rows(aggregate_path, result.get("aggregate_rows", []), header_written=agg_written)

            completed += 1
            pbar.update(1)
            if use_progress:
                pbar.set_postfix(
                    model=result.get("model_stub", "?"),
                    heads=result.get("head_label", "?"),
                    lex=result.get("n_lexemes", 0),
                    pts=result.get("n_points", 0),
                )

            if charts_enabled and agg_written and (
                completed == 1
                or completed == len(tasks)
                or completed % max(1, args.incremental_chart_every) == 0
            ):
                _write_progress_chart(aggregate_path, progress_chart_path)
        pbar.close()

    if not (per_written or global_written or agg_written):
        print("No usable records found for metrics.")
        return

    if per_written:
        print(f"Saved {per_lexeme_path}")
    if global_written:
        print(f"Saved {global_path}")
    if agg_written:
        print(f"Saved {aggregate_path}")
    if charts_enabled and os.path.exists(progress_chart_path):
        print(f"Saved {progress_chart_path}")


if __name__ == "__main__":
    main()
