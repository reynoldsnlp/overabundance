# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.8.0",
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scikit-learn>=1.0.0",
#   "tqdm>=4.66.0",
# ]
# ///

"""K-means elbow analysis (silhouette-based) per lexeme x mps system.

For each (lexeme, mps) system in the chosen embedding cache, sweeps k in
[K_MIN, K_MAX], fits KMeans, scores with silhouette, and records the optimal
k (argmax silhouette). Writes a TSV of results and one PNG grid of elbow
charts per optimal-k bucket.

Runs the full matrix of (embedding_type x cond_bucket) by default, where
cond_bucket is one of {all, no_cond, prob, cat}. When a category bucket is
selected, only tokens whose corresponding meaning is labelled with that
cond_type (per conditioning_by_meaning.csv) are kept.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import overabundance_common as common


K_MIN = 2
K_MAX = 10

COND_BUCKETS = ["all", "no_cond", "prob", "cat"]


def _system_id(lemma: str, msps: str) -> str:
    return f"{lemma}__{msps}"


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


def _load_cond_lookup(path: str) -> Dict[Tuple[str, str, str], str]:
    if not os.path.exists(path):
        print(f"[warn] conditioning file not found: {path} — per-category buckets will be empty.")
        return {}
    df = pd.read_csv(path, encoding="latin-1")
    lookup: Dict[Tuple[str, str, str], str] = {}
    for _, row in df.iterrows():
        key = (
            _normalize_text(row.get("lemma")),
            _normalize_text(row.get("msps")),
            _meaning_match_key(row.get("def")),
        )
        lookup[key] = _normalize_text(row.get("cond_type"))
    return lookup


def _collect_systems(
    cache_path: str,
    embedding_type: str,
    cond_lookup: Dict[Tuple[str, str, str], str],
    cond_filter: Optional[str],
) -> Dict[str, Dict]:
    cache = common.load_cache(cache_path)
    buckets: Dict[str, Dict] = defaultdict(lambda: {"lemma": "", "msps": "", "X": []})
    for rec in cache.values():
        lemma = _normalize_text(rec.get("lexeme"))
        msps = _normalize_text(rec.get("mps"))
        if not lemma or not msps:
            continue

        if cond_filter is not None:
            key = (lemma, msps, _meaning_match_key(rec.get("meaning")))
            if cond_lookup.get(key) != cond_filter:
                continue

        vec = common.select_record_embedding(rec, embedding_source=embedding_type)
        if vec is None:
            continue
        sid = _system_id(lemma, msps)
        buckets[sid]["lemma"] = lemma
        buckets[sid]["msps"] = msps
        buckets[sid]["X"].append(np.asarray(vec, dtype=float))
    return buckets


def _analyze_system(X: np.ndarray) -> Dict[int, Dict[str, float]]:
    n = X.shape[0]
    max_k = min(K_MAX, n - 1)
    results: Dict[int, Dict[str, float]] = {}
    for k in range(K_MIN, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        sil = float(silhouette_score(X, labels))
        results[k] = {"silhouette": sil, "inertia": float(km.inertia_)}
    return results


def _plot_grid(systems: List[Dict], out_path: str, optimal_k: int, subtitle: str = "") -> None:
    n = len(systems)
    side = max(2, math.ceil(math.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(3.2 * side, 2.6 * side), squeeze=False)

    for idx, sysinfo in enumerate(systems):
        ax = axes[idx // side][idx % side]
        ks = sysinfo["ks"]
        sils = sysinfo["sils"]
        ax.plot(ks, sils, marker="o", linewidth=1.2)
        best = sysinfo["best_k"]
        best_sil = sils[ks.index(best)]
        ax.axvline(best, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.scatter([best], [best_sil], color="red", zorder=5, s=30)
        ax.set_title(f"{sysinfo['system_id']} (n={sysinfo['n']})", fontsize=8)
        ax.set_xlabel("k", fontsize=7)
        ax.set_ylabel("silhouette", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(n, side * side):
        axes[j // side][j % side].axis("off")

    suptitle = f"Silhouette elbow — optimal k = {optimal_k} (n_systems = {n})"
    if subtitle:
        suptitle += f" — {subtitle}"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-path",
        default="head_embed_cache_answerdotai_ModernBERT-base.jsonl",
        help="Path to the head-embedding cache jsonl.",
    )
    p.add_argument(
        "--embedding-type",
        choices=["orig", "delta", "both"],
        default="both",
    )
    p.add_argument(
        "--cond-bucket",
        choices=COND_BUCKETS + ["matrix"],
        default="matrix",
        help="Which cond_type bucket to run. 'matrix' runs all four (all + per-category).",
    )
    p.add_argument(
        "--conditioning-by-meaning-path",
        default="conditioning_by_meaning.csv",
        help="CSV mapping (lemma, msps, def) -> cond_type.",
    )
    p.add_argument(
        "--index-root",
        default="docs/elbow_charts",
        help="Parent directory; index.html is regenerated here.",
    )
    return p.parse_args()


def _run_one(
    cache_path: str,
    embedding_type: str,
    cond_lookup: Dict[Tuple[str, str, str], str],
    cond_bucket: str,
    output_tsv: str,
    output_dir: str,
) -> List[int]:
    os.makedirs(output_dir, exist_ok=True)
    cond_filter = None if cond_bucket == "all" else cond_bucket

    buckets = _collect_systems(cache_path, embedding_type, cond_lookup, cond_filter)

    rows: List[Dict] = []
    grids: Dict[int, List[Dict]] = defaultdict(list)

    desc = f"systems ({embedding_type}/{cond_bucket})"
    for sid, bucket in tqdm(sorted(buckets.items()), desc=desc):
        X = np.stack(bucket["X"]) if bucket["X"] else np.zeros((0, 0))
        n = X.shape[0]
        if n < 3:
            rows.append({
                "system_id": sid,
                "lemma": bucket["lemma"],
                "msps": bucket["msps"],
                "cond_bucket": cond_bucket,
                "n_tokens": n,
                "optimal_k": "",
                "best_silhouette": "",
                "k_range": "",
                "silhouettes": "",
                "inertias": "",
                "note": "too few tokens",
            })
            continue

        per_k = _analyze_system(X)
        if not per_k:
            rows.append({
                "system_id": sid,
                "lemma": bucket["lemma"],
                "msps": bucket["msps"],
                "cond_bucket": cond_bucket,
                "n_tokens": n,
                "optimal_k": "",
                "best_silhouette": "",
                "k_range": "",
                "silhouettes": "",
                "inertias": "",
                "note": "no valid k",
            })
            continue

        ks = sorted(per_k.keys())
        sils = [per_k[k]["silhouette"] for k in ks]
        inertias = [per_k[k]["inertia"] for k in ks]
        best_idx = int(np.argmax(sils))
        best_k = ks[best_idx]
        best_sil = sils[best_idx]

        rows.append({
            "system_id": sid,
            "lemma": bucket["lemma"],
            "msps": bucket["msps"],
            "cond_bucket": cond_bucket,
            "n_tokens": n,
            "optimal_k": best_k,
            "best_silhouette": best_sil,
            "k_range": f"{ks[0]}-{ks[-1]}",
            "silhouettes": ",".join(f"{v:.6f}" for v in sils),
            "inertias": ",".join(f"{v:.6f}" for v in inertias),
            "note": "",
        })

        grids[best_k].append({
            "system_id": sid,
            "n": n,
            "ks": ks,
            "sils": sils,
            "best_k": best_k,
        })

    df = pd.DataFrame(rows).sort_values(["optimal_k", "system_id"], na_position="last")
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Wrote {output_tsv} ({len(df)} systems)")

    subtitle = f"{embedding_type} / cond_type={cond_bucket}"
    ks_written: List[int] = []
    for k, systems in sorted(grids.items()):
        systems_sorted = sorted(systems, key=lambda s: s["system_id"])
        out_path = os.path.join(output_dir, f"elbow_grid_k{k}.png")
        _plot_grid(systems_sorted, out_path, k, subtitle=subtitle)
        print(f"  k={k}: {len(systems_sorted)} systems -> {out_path}")
        ks_written.append(k)
    return ks_written


def _cleanup_stale_pngs(index_root: str, emb_types: List[str]) -> None:
    """Remove old top-level PNGs at docs/elbow_charts/<emb>/elbow_grid_k*.png
    that predate the cond_bucket subdir layout."""
    for emb in emb_types:
        emb_dir = os.path.join(index_root, emb)
        if not os.path.isdir(emb_dir):
            continue
        for name in os.listdir(emb_dir):
            full = os.path.join(emb_dir, name)
            if os.path.isfile(full) and name.startswith("elbow_grid_k") and name.endswith(".png"):
                os.remove(full)
                print(f"  removed stale {full}")


def _write_index(
    index_root: str,
    cache_path: str,
    runs: Dict[str, Dict[str, List[int]]],
) -> None:
    os.makedirs(index_root, exist_ok=True)
    out_path = os.path.join(index_root, "index.html")
    lines: List[str] = [
        "<html>",
        "<head><title>K-means elbow / silhouette analysis</title></head>",
        "<body>",
        "<h1>K-means elbow / silhouette analysis</h1>",
        "<p>One PNG per optimal-k bucket. Each subplot shows silhouette vs k for a",
        "single lexeme&times;mps system; the red dashed line marks the optimal k",
        "(argmax silhouette). Source: <code>run_elbow_analysis.py</code> on",
        f"<code>{os.path.basename(cache_path)}</code>",
        f"(k&nbsp;&isin;&nbsp;[{K_MIN},{K_MAX}]).</p>",
        "<p>The <code>all</code> bucket uses every token; the <code>no_cond</code>,",
        "<code>prob</code>, and <code>cat</code> buckets restrict to tokens whose",
        "meaning is labelled with that <code>cond_type</code> in",
        "<code>conditioning_by_meaning.csv</code>.</p>",
    ]
    for emb_type in ("orig", "delta"):
        if emb_type not in runs:
            continue
        lines.append(f"<h2>Embedding type: <code>{emb_type}</code></h2>")
        for bucket in COND_BUCKETS:
            ks = runs[emb_type].get(bucket)
            if ks is None:
                continue
            lines.append(f"<h3>cond_type = <code>{bucket}</code></h3>")
            if not ks:
                lines.append("<p><em>(no systems)</em></p>")
                continue
            lines.append("<ul>")
            for k in ks:
                rel = f"{emb_type}/{bucket}/elbow_grid_k{k}.png"
                lines.append(
                    f'<li><a href="{rel}">{rel}</a> &mdash; systems whose optimal k = {k}</li>'
                )
            lines.append("</ul>")
    lines += ["</body>", "</html>", ""]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    emb_types = ["orig", "delta"] if args.embedding_type == "both" else [args.embedding_type]
    cond_buckets = COND_BUCKETS if args.cond_bucket == "matrix" else [args.cond_bucket]

    cond_lookup = _load_cond_lookup(args.conditioning_by_meaning_path)

    _cleanup_stale_pngs(args.index_root, emb_types)

    runs: Dict[str, Dict[str, List[int]]] = {emb: {} for emb in emb_types}
    for emb_type in emb_types:
        for bucket in cond_buckets:
            output_dir = os.path.join(args.index_root, emb_type, bucket)
            output_tsv = f"elbow_silhouette_per_system_{emb_type}_{bucket}.tsv"
            ks = _run_one(
                args.cache_path, emb_type, cond_lookup, bucket, output_tsv, output_dir,
            )
            runs[emb_type][bucket] = ks

    _write_index(args.index_root, args.cache_path, runs)


if __name__ == "__main__":
    main()
