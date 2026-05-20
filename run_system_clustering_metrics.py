# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.7.0",
# ]
# ///

"""Evaluate clustering within lemma x msps systems.

This script treats `lemma x msps` as the only analysis unit. It does not pool
past and participle together, and it only supports clustering embeddings from
`orig` or `delta`.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import Normalizer, StandardScaler

import overabundance_common as common


CANONICAL_LABEL_TYPES = (
    "raw",
    "form",
    "conditioned-keep",
    "conditioned-collapse",
    "conditioned-exclude",
    "conditioned-cat-only",
)

LABEL_COLUMN_BY_TYPE = {
    "raw": "raw_label",
    "form": "form_label",
    "conditioned-keep": "conditioned_keep_label",
    "conditioned-collapse": "conditioned_collapse_label",
    "conditioned-exclude": "conditioned_exclude_label",
    "conditioned-cat-only": "cat_only_label",
}

VALIDITY_COLUMN_BY_TYPE = {
    "raw": "raw_ok",
    "form": "form_ok",
    "conditioned-keep": "conditioned_ok",
    "conditioned-collapse": "conditioned_ok",
    "conditioned-exclude": "conditioned_ok",
    "conditioned-cat-only": "conditioned_ok",
}


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
        "--token-source",
        dest="embedding_type",
        choices=["orig", "delta"],
        help=argparse.SUPPRESS,
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
        help="Optional comma-separated system_id filter (for example hang__pst).",
    )
    parser.add_argument(
        "--label-types",
        type=str,
        default="raw,form,conditioned-exclude,conditioned-cat-only",
        help=(
            "Comma-separated label types to score. Supported values: "
            "raw, form, conditioned-keep, conditioned-collapse, conditioned-exclude, "
            "conditioned-cat-only, cat-only, "
            "all_conditioned, all."
        ),
    )
    parser.add_argument(
        "--pair-p-alpha",
        type=float,
        default=0.05,
        help="Threshold used to mark a system-level conditioning test as significant.",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize vectors before KMeans.",
    )
    parser.add_argument(
        "--normalize-l2",
        action="store_true",
        help="L2-normalize vectors before KMeans.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=0,
        help="Optional PCA dimension for KMeans preprocessing (0 disables PCA).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output TSVs.",
    )
    return parser.parse_args()


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


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


def _parse_csv_list(raw: str) -> Optional[set[str]]:
    vals = {_normalize_text(piece) for piece in (raw or "").split(",") if _normalize_text(piece)}
    return vals or None


def _resolve_label_types(raw: str) -> List[str]:
    alias = {
        "semantic": "raw",
        "joint_keep": "conditioned-keep",
        "joint_collapse": "conditioned-collapse",
        "joint_exclude": "conditioned-exclude",
        "cat_only": "conditioned-cat-only",
        "cat-only": "conditioned-cat-only",
        "conditioned_cat_only": "conditioned-cat-only",
    }
    conditioned = [
        "conditioned-keep",
        "conditioned-collapse",
        "conditioned-exclude",
        "conditioned-cat-only",
    ]
    ordered: List[str] = []
    for piece in (raw or "").split(","):
        value = alias.get(piece.strip().lower(), piece.strip().lower())
        if not value:
            continue
        if value == "all":
            for item in CANONICAL_LABEL_TYPES:
                if item not in ordered:
                    ordered.append(item)
            continue
        if value == "all_conditioned":
            for item in conditioned:
                if item not in ordered:
                    ordered.append(item)
            continue
        if value not in CANONICAL_LABEL_TYPES:
            raise ValueError(f"Unsupported label type: {value}")
        if value not in ordered:
            ordered.append(value)
    if not ordered:
        ordered = ["raw", "form", "conditioned-exclude", "conditioned-cat-only"]
    return ordered


def _system_id(lemma: str, msps: str) -> str:
    return f"{lemma}__{msps}"


def _model_stub_from_cache_path(cache_path: str) -> str:
    return os.path.basename(cache_path).replace("head_embed_cache_", "").replace(".jsonl", "")


def _model_tag(cache_paths: List[str]) -> str:
    if len(cache_paths) == 1:
        return common.model_slug(_model_stub_from_cache_path(cache_paths[0]))
    return "multi"


def _load_conditioning_by_meaning(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df["lemma"] = df["lemma"].map(_normalize_text)
    df["msps"] = df["msps"].map(_normalize_text)
    df["form_1"] = df["form_1"].map(_normalize_text)
    df["form_2"] = df["form_2"].map(_normalize_text)
    df["cond_type"] = df["cond_type"].map(_normalize_text)
    df["def"] = df["def"].map(_normalize_text)
    df["def_match_key"] = df["def"].map(_meaning_match_key)
    df["form_1_freq"] = df["form_1_freq"].map(_safe_float)
    df["form_2_freq"] = df["form_2_freq"].map(_safe_float)
    df["p_eq"] = df["p_eq"].map(_safe_float)
    df["p_cat"] = df["p_cat"].map(_safe_float)
    df["preferred_form"] = np.where(
        df["form_1_freq"].fillna(float("-inf")) >= df["form_2_freq"].fillna(float("-inf")),
        df["form_1"],
        df["form_2"],
    )
    df["system_id"] = df.apply(lambda row: _system_id(str(row["lemma"]), str(row["msps"])), axis=1)
    return df


def _load_pair_tests(path: str, alpha: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df["lemma"] = df["lemma"].map(_normalize_text)
    df["msps"] = df["msps"].map(_normalize_text)
    df["p_value"] = df["p_value"].map(_safe_float)
    df["test_type"] = df["test_type"].map(_normalize_text)
    df["system_id"] = df.apply(lambda row: _system_id(str(row["lemma"]), str(row["msps"])), axis=1)
    df["pair_significant"] = df["p_value"].apply(lambda x: bool(np.isfinite(x) and x < alpha))
    return df


def _load_cond_counts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df["lemma"] = df["lemma"].map(_normalize_text)
    df["msps"] = df["msps"].map(_normalize_text)
    df["system_id"] = df.apply(lambda row: _system_id(str(row["lemma"]), str(row["msps"])), axis=1)
    for col in ["no_cond", "prob", "cat"]:
        df[col] = df[col].fillna(0).astype(int)
    return df


def _expected_form_pair(cond_rows: List[Dict[str, Any]]) -> Tuple[Optional[Tuple[str, str]], str]:
    pair_set = set()
    for row in cond_rows:
        f1 = _normalize_text(row.get("form_1"))
        f2 = _normalize_text(row.get("form_2"))
        if not f1 or not f2:
            continue
        pair_set.add(tuple(sorted((f1, f2))))

    if not pair_set:
        return None, "missing_conditioning_forms"
    if len(pair_set) > 1:
        joined = "|".join("/".join(pair) for pair in sorted(pair_set))
        return None, f"inconsistent_conditioning_forms:{joined}"

    pair = next(iter(pair_set))
    if len(pair) != 2:
        return None, "invalid_conditioning_form_pair"
    return pair, ""


def _match_conditioning_rows_to_meanings(
    actual_meanings: Dict[str, Dict[str, str]],
    conditioning_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    matched: Dict[str, Dict[str, Any]] = {}
    used_actual: set[str] = set()
    actual_by_match_key: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for actual in actual_meanings.values():
        actual_by_match_key[_meaning_match_key(actual["meaning"])].append(actual)

    for cond_row in conditioning_rows:
        key = _normalize_text(cond_row.get("def"))
        actual = actual_meanings.get(key)
        if actual is None:
            continue
        matched[actual["meaning_id"]] = cond_row
        used_actual.add(key)

    for cond_row in conditioning_rows:
        if cond_row in matched.values():
            continue
        match_key = str(cond_row.get("def_match_key", ""))
        candidates = [
            actual
            for actual in actual_by_match_key.get(match_key, [])
            if actual["meaning"] not in used_actual
        ]
        if len(candidates) != 1:
            continue
        actual = candidates[0]
        matched[actual["meaning_id"]] = cond_row
        used_actual.add(actual["meaning"])

    remaining_actual = [text for text in actual_meanings.keys() if text not in used_actual]
    for cond_row in conditioning_rows:
        if cond_row in matched.values():
            continue
        best_key = None
        best_score = 0.0
        for actual_key in remaining_actual:
            score = difflib.SequenceMatcher(
                None,
                str(cond_row.get("def_match_key", _normalize_text(cond_row.get("def")))),
                _meaning_match_key(actual_key),
            ).ratio()
            if score > best_score:
                best_score = score
                best_key = actual_key
        if best_key is not None and best_score >= 0.88:
            actual = actual_meanings[best_key]
            matched[actual["meaning_id"]] = cond_row
            remaining_actual = [text for text in remaining_actual if text != best_key]

    return matched


def _label_entropy(y: np.ndarray) -> float:
    counts = Counter(y.tolist())
    n = len(y)
    if n == 0 or not counts:
        return float("nan")
    probs = np.array([count / n for count in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def _mean_within_between(X: np.ndarray, y: np.ndarray, metric: str) -> Tuple[float, float, float]:
    d = pairwise_distances(X, metric=metric)
    within: List[float] = []
    between: List[float] = []
    n = len(y)
    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                within.append(float(d[i, j]))
            else:
                between.append(float(d[i, j]))
    within_mean = float(np.mean(within)) if within else float("nan")
    between_mean = float(np.mean(between)) if between else float("nan")
    if not np.isfinite(within_mean) or not np.isfinite(between_mean) or between_mean == 0.0:
        ratio = float("nan")
    else:
        ratio = float(within_mean / between_mean)
    return within_mean, between_mean, ratio


def _nearest_centroid_loo_accuracy(X: np.ndarray, y: np.ndarray, metric: str) -> float:
    labels = sorted(set(y.tolist()))
    if len(labels) < 2:
        return float("nan")

    correct = 0
    used = 0
    for i in range(len(y)):
        centroids = {}
        for lab in labels:
            idx = np.where(y == lab)[0]
            idx = idx[idx != i]
            if len(idx) == 0:
                continue
            centroids[lab] = np.mean(X[idx], axis=0)
        if len(centroids) < 2:
            continue
        centroid_labels = list(centroids.keys())
        centroid_matrix = np.stack([centroids[lab] for lab in centroid_labels])
        dist = pairwise_distances(X[i].reshape(1, -1), centroid_matrix, metric=metric)[0]
        pred = centroid_labels[int(np.argmin(dist))]
        used += 1
        if pred == y[i]:
            correct += 1
    if used == 0:
        return float("nan")
    return float(correct / used)


def _preprocess_for_kmeans(
    X: np.ndarray,
    *,
    standardize: bool,
    normalize_l2: bool,
    pca_components: int,
) -> Tuple[np.ndarray, int]:
    out = X.astype(float, copy=True)
    if standardize:
        out = StandardScaler().fit_transform(out)
    used_components = 0
    if pca_components > 0:
        max_components = min(out.shape[0], out.shape[1])
        if max_components >= 1:
            used_components = min(int(pca_components), int(max_components))
            if used_components > 0:
                out = PCA(n_components=used_components, random_state=0).fit_transform(out)
    if normalize_l2:
        out = Normalizer(norm="l2").fit_transform(out)
    return out, used_components


def _analyze_labels(
    df_eval: pd.DataFrame,
    *,
    label_col: str,
    label_type: str,
    seed: int,
    standardize: bool,
    normalize_l2: bool,
    pca_components: int,
) -> Dict[str, Any]:
    y = df_eval[label_col].astype(str).to_numpy()
    X = np.stack(df_eval["embedding"].to_numpy())
    label_counts = Counter(y.tolist())
    singleton_count = sum(1 for count in label_counts.values() if count == 1)
    sparse_count = sum(1 for count in label_counts.values() if count < 3)

    out: Dict[str, Any] = {
        "label_type": label_type,
        "label_column": label_col,
        "n_samples_eval": int(len(y)),
        "n_labels": int(len(label_counts)),
        "label_entropy": _label_entropy(y),
        "label_max_count": int(max(label_counts.values())) if label_counts else 0,
        "label_min_count": int(min(label_counts.values())) if label_counts else 0,
        "label_singleton_fraction": float(singleton_count / len(label_counts)) if label_counts else float("nan"),
        "label_sparse_fraction": float(sparse_count / len(label_counts)) if label_counts else float("nan"),
        "silhouette_cosine": float("nan"),
        "silhouette_euclidean": float("nan"),
        "within_cosine": float("nan"),
        "between_cosine": float("nan"),
        "within_between_ratio_cosine": float("nan"),
        "within_euclidean": float("nan"),
        "between_euclidean": float("nan"),
        "within_between_ratio_euclidean": float("nan"),
        "nearest_centroid_loo_acc_cosine": float("nan"),
        "nearest_centroid_loo_acc_euclidean": float("nan"),
        "kmeans_ari": float("nan"),
        "kmeans_ami": float("nan"),
        "kmeans_pca_components_used": 0,
    }

    if len(label_counts) < 2:
        return out

    for metric in ["cosine", "euclidean"]:
        try:
            out[f"silhouette_{metric}"] = float(silhouette_score(X, y, metric=metric))
        except Exception:
            out[f"silhouette_{metric}"] = float("nan")

        within_mean, between_mean, ratio = _mean_within_between(X, y, metric=metric)
        out[f"within_{metric}"] = within_mean
        out[f"between_{metric}"] = between_mean
        out[f"within_between_ratio_{metric}"] = ratio
        out[f"nearest_centroid_loo_acc_{metric}"] = _nearest_centroid_loo_accuracy(X, y, metric=metric)

    try:
        X_cluster, used_components = _preprocess_for_kmeans(
            X,
            standardize=standardize,
            normalize_l2=normalize_l2,
            pca_components=pca_components,
        )
        if len(label_counts) <= len(X_cluster):
            pred = KMeans(n_clusters=len(label_counts), random_state=seed, n_init="auto").fit_predict(X_cluster)
            out["kmeans_ari"] = float(adjusted_rand_score(y, pred))
            out["kmeans_ami"] = float(adjusted_mutual_info_score(y, pred))
            out["kmeans_pca_components_used"] = int(used_components)
    except Exception:
        pass

    return out


def _make_conditioned_label(meaning_id: str, form: str, cond_type: str, strategy: str) -> Optional[str]:
    cond_type = _normalize_text(cond_type)
    if not meaning_id or not form or not cond_type:
        return None
    if strategy in {"cat-only", "conditioned-cat-only"}:
        if cond_type != "cat":
            return None
        return f"{meaning_id}__{form}"
    if cond_type == "no_cond":
        return meaning_id
    if cond_type == "prob" and strategy == "exclude":
        return None
    if cond_type == "prob" and strategy == "collapse":
        return meaning_id
    return f"{meaning_id}__{form}"


def _extract_tokens_from_cache(
    cache_path: str,
    *,
    embedding_type: str,
    lemma_filter: Optional[set[str]],
    msps_filter: Optional[set[str]],
    system_filter: Optional[set[str]],
) -> pd.DataFrame:
    cache = common.load_cache(cache_path)
    model = _model_stub_from_cache_path(cache_path)
    rows: List[Dict[str, Any]] = []

    for rec in cache.values():
        lemma = _normalize_text(rec.get("lexeme"))
        msps = _normalize_text(rec.get("mps"))
        if not lemma or not msps:
            continue

        system_id = _system_id(lemma, msps)
        if lemma_filter is not None and lemma not in lemma_filter:
            continue
        if msps_filter is not None and msps not in msps_filter:
            continue
        if system_filter is not None and system_id not in system_filter:
            continue

        vector = common.select_record_embedding(rec, embedding_source=embedding_type)
        if vector is None:
            continue

        meaning_id_raw = rec.get("meaning_index")
        if meaning_id_raw is None or (isinstance(meaning_id_raw, float) and np.isnan(meaning_id_raw)):
            continue

        rows.append(
            {
                "model": model,
                "cache_path": cache_path,
                "cache_key": rec.get("cache_key", rec.get("ID")),
                "token_id": rec.get("cache_key", rec.get("ID")),
                "lemma": lemma,
                "msps": msps,
                "system_id": system_id,
                "form": _normalize_text(rec.get("original_form")),
                "other_form": _normalize_text(rec.get("partner_form")),
                "meaning": _normalize_text(rec.get("meaning")),
                "meaning_id": str(meaning_id_raw),
                "sentence": rec.get("orig_sentence"),
                "embedding_type": embedding_type,
                "embedding": np.asarray(vector, dtype=float),
            }
        )

    return pd.DataFrame(rows)


def _annotate_tokens(
    df_tokens: pd.DataFrame,
    *,
    cond_meaning: pd.DataFrame,
    pair_tests: pd.DataFrame,
    cond_counts: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_tokens.copy()
    df["raw_label"] = df["meaning_id"].astype(str)
    df["form_label"] = df["form"].astype(str)
    df["joint_label"] = df.apply(
        lambda row: f"{row['meaning_id']}__{row['form']}" if row["meaning_id"] and row["form"] else "",
        axis=1,
    )
    df["conditioned_keep_label"] = None
    df["conditioned_collapse_label"] = None
    df["conditioned_exclude_label"] = None
    df["cat_only_label"] = None

    for col in [
        "expected_form_1",
        "expected_form_2",
        "conditioning_def",
        "cond_type",
        "cond_form_1",
        "cond_form_2",
        "preferred_form",
        "validation_notes",
    ]:
        df[col] = ""

    for col in ["form_valid", "has_conditioning", "raw_ok", "form_ok", "conditioned_ok", "system_has_conditioning_pair"]:
        df[col] = False

    pair_by_system = {row["system_id"]: row for row in pair_tests.to_dict("records")}
    counts_by_system = {row["system_id"]: row for row in cond_counts.to_dict("records")}
    cond_rows_by_system = defaultdict(list)
    for row in cond_meaning.to_dict("records"):
        cond_rows_by_system[str(row["system_id"])].append(row)

    system_rows: List[Dict[str, Any]] = []

    for system_id, sub in df.groupby("system_id", sort=True):
        idxs = sub.index.tolist()
        cond_rows = cond_rows_by_system.get(system_id, [])
        pair_row = pair_by_system.get(system_id, {})
        counts_row = counts_by_system.get(system_id, {})

        expected_pair, pair_note = _expected_form_pair(cond_rows)
        expected_forms = set(expected_pair) if expected_pair is not None else set()
        expected_form_1 = expected_pair[0] if expected_pair is not None else ""
        expected_form_2 = expected_pair[1] if expected_pair is not None else ""

        actual_meanings: Dict[str, Dict[str, str]] = {}
        duplicate_meaning_texts = 0
        for meaning_id, meaning in (
            sub[["meaning_id", "meaning"]].drop_duplicates().sort_values(["meaning_id", "meaning"]).itertuples(index=False)
        ):
            if not meaning:
                continue
            existing = actual_meanings.get(meaning)
            if existing is not None and existing["meaning_id"] != meaning_id:
                duplicate_meaning_texts += 1
                continue
            actual_meanings[meaning] = {"meaning_id": meaning_id, "meaning": meaning}

        matched = _match_conditioning_rows_to_meanings(actual_meanings, cond_rows)
        matched_counts = Counter(_normalize_text(row.get("cond_type")) for row in matched.values())
        observed_forms = sorted({_normalize_text(x) for x in sub["form"].tolist() if _normalize_text(x)})

        system_notes: List[str] = []
        if pair_note:
            system_notes.append(pair_note)
        if duplicate_meaning_texts:
            system_notes.append(f"duplicate_meaning_texts={duplicate_meaning_texts}")
        if not cond_rows:
            system_notes.append("missing_conditioning_rows")
        if not pair_row:
            system_notes.append("missing_pair_test")
        if not counts_row:
            system_notes.append("missing_cond_counts")

        for idx in idxs:
            meaning_id = str(df.at[idx, "meaning_id"])
            form = _normalize_text(df.at[idx, "form"])
            note_parts: List[str] = []

            df.at[idx, "expected_form_1"] = expected_form_1
            df.at[idx, "expected_form_2"] = expected_form_2
            df.at[idx, "system_has_conditioning_pair"] = expected_pair is not None

            form_valid = bool(form) and (not expected_forms or form in expected_forms)
            if not form_valid:
                note_parts.append("unexpected_form")
            df.at[idx, "form_valid"] = form_valid

            cond_row = matched.get(meaning_id)
            if cond_row is None:
                note_parts.append("unmatched_conditioning_meaning")
            else:
                cond_type = _normalize_text(cond_row.get("cond_type"))
                df.at[idx, "has_conditioning"] = True
                df.at[idx, "conditioning_def"] = _normalize_text(cond_row.get("def"))
                df.at[idx, "cond_type"] = cond_type
                df.at[idx, "cond_form_1"] = _normalize_text(cond_row.get("form_1"))
                df.at[idx, "cond_form_2"] = _normalize_text(cond_row.get("form_2"))
                df.at[idx, "preferred_form"] = _normalize_text(cond_row.get("preferred_form"))
                df.at[idx, "conditioned_keep_label"] = _make_conditioned_label(meaning_id, form, cond_type, "keep")
                df.at[idx, "conditioned_collapse_label"] = _make_conditioned_label(meaning_id, form, cond_type, "collapse")
                df.at[idx, "conditioned_exclude_label"] = _make_conditioned_label(meaning_id, form, cond_type, "exclude")
                df.at[idx, "cat_only_label"] = _make_conditioned_label(
                    meaning_id,
                    form,
                    cond_type,
                    "conditioned-cat-only",
                )

            df.at[idx, "raw_ok"] = bool(meaning_id) and form_valid
            df.at[idx, "form_ok"] = bool(form) and form_valid
            df.at[idx, "conditioned_ok"] = bool(df.at[idx, "has_conditioning"]) and form_valid
            df.at[idx, "validation_notes"] = ";".join(note_parts)

        system_rows.append(
            {
                "model": str(sub["model"].iloc[0]),
                "cache_path": str(sub["cache_path"].iloc[0]),
                "embedding_type": str(sub["embedding_type"].iloc[0]),
                "system_id": system_id,
                "lemma": str(sub["lemma"].iloc[0]),
                "msps": str(sub["msps"].iloc[0]),
                "n_tokens_total_system": int(len(sub)),
                "n_tokens_with_conditioning": int(df.loc[idxs, "has_conditioning"].sum()),
                "n_tokens_unmatched_conditioning": int((~df.loc[idxs, "has_conditioning"]).sum()),
                "n_tokens_unexpected_form": int((~df.loc[idxs, "form_valid"]).sum()),
                "n_meanings_total_system": int(sub["meaning_id"].nunique()),
                "n_meanings_with_conditioning": int(len(matched)),
                "n_forms_observed_system": int(len(observed_forms)),
                "observed_forms": ",".join(observed_forms),
                "expected_form_1": expected_form_1,
                "expected_form_2": expected_form_2,
                "pair_p_value": _safe_float(pair_row.get("p_value")),
                "pair_test_type": _normalize_text(pair_row.get("test_type")),
                "pair_significant": bool(pair_row.get("pair_significant", False)),
                "pair_attested_meanings": int(pair_row.get("attested_meanings", 0) or 0),
                "cond_no_cond_count_expected": int(counts_row.get("no_cond", 0) or 0),
                "cond_prob_count_expected": int(counts_row.get("prob", 0) or 0),
                "cond_cat_count_expected": int(counts_row.get("cat", 0) or 0),
                "cond_no_cond_count_matched": int(matched_counts.get("no_cond", 0)),
                "cond_prob_count_matched": int(matched_counts.get("prob", 0)),
                "cond_cat_count_matched": int(matched_counts.get("cat", 0)),
                "system_notes": ";".join(system_notes),
            }
        )

    return df, pd.DataFrame(system_rows)


def _aggregate_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "silhouette_cosine",
        "silhouette_euclidean",
        "nearest_centroid_loo_acc_cosine",
        "nearest_centroid_loo_acc_euclidean",
        "within_between_ratio_cosine",
        "within_between_ratio_euclidean",
        "kmeans_ari",
        "kmeans_ami",
    ]

    rows: List[Dict[str, Any]] = []
    for model, model_df in metrics_df.groupby("model", dropna=False):
        for label_type, sub in model_df.groupby("label_type", dropna=False):
            for sig_bucket, sig_df in [
                ("all", sub),
                ("pair_significant", sub[sub["pair_significant"] == True]),
                ("pair_not_significant", sub[sub["pair_significant"] == False]),
            ]:
                if sig_df.empty:
                    continue
                row: Dict[str, Any] = {
                    "model": model,
                    "embedding_type": str(sig_df["embedding_type"].iloc[0]),
                    "label_type": label_type,
                    "pair_significance_bucket": sig_bucket,
                    "n_systems": int(sig_df["system_id"].nunique()),
                    "n_rows": int(len(sig_df)),
                    "n_samples_eval_total": int(sig_df["n_samples_eval"].sum()),
                    "n_samples_eval_mean": _safe_float(sig_df["n_samples_eval"].mean()),
                    "n_labels_mean": _safe_float(sig_df["n_labels"].mean()),
                }
                weights = sig_df["n_samples_eval"].to_numpy(dtype=float)
                for col in metric_cols:
                    vals = sig_df[col].to_numpy(dtype=float)
                    finite = np.isfinite(vals)
                    row[f"{col}_mean"] = float(np.mean(vals[finite])) if np.any(finite) else float("nan")
                    if np.any(finite) and np.any(weights[finite] > 0):
                        row[f"{col}_weighted_mean"] = float(np.average(vals[finite], weights=weights[finite]))
                    else:
                        row[f"{col}_weighted_mean"] = float("nan")
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

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

    all_metrics: List[pd.DataFrame] = []
    all_tokens: List[pd.DataFrame] = []

    for cache_path in cache_paths:
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

        metrics_rows: List[Dict[str, Any]] = []
        for _, sys_row in df_systems.iterrows():
            system_id = str(sys_row["system_id"])
            df_sys = df_tokens[df_tokens["system_id"] == system_id]

            for label_type in label_types:
                label_col = LABEL_COLUMN_BY_TYPE[label_type]
                validity_col = VALIDITY_COLUMN_BY_TYPE[label_type]
                df_eval = df_sys[df_sys[validity_col]].copy()
                df_eval = df_eval[df_eval[label_col].notna()].copy()

                base = sys_row.to_dict()
                base.update(
                    {
                        "label_type": label_type,
                        "label_column": label_col,
                        "n_samples_eval": int(len(df_eval)),
                    }
                )

                if df_eval.empty:
                    base.update(
                        {
                            "n_labels": 0,
                            "label_entropy": float("nan"),
                            "label_max_count": 0,
                            "label_min_count": 0,
                            "label_singleton_fraction": float("nan"),
                            "label_sparse_fraction": float("nan"),
                            "silhouette_cosine": float("nan"),
                            "silhouette_euclidean": float("nan"),
                            "within_cosine": float("nan"),
                            "between_cosine": float("nan"),
                            "within_between_ratio_cosine": float("nan"),
                            "within_euclidean": float("nan"),
                            "between_euclidean": float("nan"),
                            "within_between_ratio_euclidean": float("nan"),
                            "nearest_centroid_loo_acc_cosine": float("nan"),
                            "nearest_centroid_loo_acc_euclidean": float("nan"),
                            "kmeans_ari": float("nan"),
                            "kmeans_ami": float("nan"),
                            "kmeans_pca_components_used": 0,
                        }
                    )
                else:
                    base.update(
                        _analyze_labels(
                            df_eval,
                            label_col=label_col,
                            label_type=label_type,
                            seed=args.seed,
                            standardize=args.standardize,
                            normalize_l2=args.normalize_l2,
                            pca_components=args.pca_components,
                        )
                    )
                metrics_rows.append(base)

        all_metrics.append(pd.DataFrame(metrics_rows))

        token_out = df_tokens.drop(columns=["embedding"]).copy()
        all_tokens.append(token_out)

    if not all_metrics:
        raise RuntimeError("No systems remained after filtering.")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    tokens_df = pd.concat(all_tokens, ignore_index=True)
    summary_df = _aggregate_summary(metrics_df)

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = _model_tag(cache_paths)

    metrics_path = os.path.join(args.output_dir, f"clustering_system_metrics_{args.embedding_type}_{model_tag}.tsv")
    tokens_path = os.path.join(args.output_dir, f"clustering_system_tokens_{args.embedding_type}_{model_tag}.tsv")
    summary_path = os.path.join(args.output_dir, f"clustering_system_summary_{args.embedding_type}_{model_tag}.tsv")

    metrics_df.sort_values(["model", "system_id", "label_type"]).to_csv(metrics_path, sep="\t", index=False)
    tokens_df.sort_values(["model", "system_id", "token_id"]).to_csv(tokens_path, sep="\t", index=False)
    summary_df.sort_values(["model", "label_type", "pair_significance_bucket"]).to_csv(summary_path, sep="\t", index=False)

    print(f"Saved {metrics_path}")
    print(f"Saved {tokens_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
