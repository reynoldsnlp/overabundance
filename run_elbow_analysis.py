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
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import overabundance_common as common


K_MIN = 2
K_MAX = 10


def _system_id(lemma: str, msps: str) -> str:
    return f"{lemma}__{msps}"


def _collect_systems(cache_path: str, embedding_type: str) -> Dict[str, Dict]:
    cache = common.load_cache(cache_path)
    buckets: Dict[str, Dict] = defaultdict(lambda: {"lemma": "", "msps": "", "X": []})
    for rec in cache.values():
        lemma = str(rec.get("lexeme") or "").strip()
        msps = str(rec.get("mps") or "").strip()
        if not lemma or not msps:
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


def _plot_grid(systems: List[Dict], out_path: str, optimal_k: int) -> None:
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

    fig.suptitle(f"Silhouette elbow — optimal k = {optimal_k} (n_systems = {n})", fontsize=11)
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
    p.add_argument("--embedding-type", choices=["orig", "delta"], default="orig")
    p.add_argument("--output-tsv", default="elbow_silhouette_per_system.tsv")
    p.add_argument("--output-dir", default="docs/elbow_charts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    buckets = _collect_systems(args.cache_path, args.embedding_type)

    rows: List[Dict] = []
    grids: Dict[int, List[Dict]] = defaultdict(list)

    for sid, bucket in tqdm(sorted(buckets.items()), desc="systems"):
        X = np.stack(bucket["X"])
        n = X.shape[0]
        if n < 3:
            rows.append({
                "system_id": sid,
                "lemma": bucket["lemma"],
                "msps": bucket["msps"],
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
    df.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"Wrote {args.output_tsv} ({len(df)} systems)")

    for k, systems in sorted(grids.items()):
        systems_sorted = sorted(systems, key=lambda s: s["system_id"])
        out_path = os.path.join(args.output_dir, f"elbow_grid_k{k}.png")
        _plot_grid(systems_sorted, out_path, k)
        print(f"  k={k}: {len(systems_sorted)} systems -> {out_path}")


if __name__ == "__main__":
    main()
