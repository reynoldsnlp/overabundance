# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "matplotlib>=3.8.0",
# ]
# ///

"""Generate PNG visual summaries from clustering sweep TSV outputs."""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import overabundance_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-pattern",
        type=str,
        default="clustering_metrics_aggregate_head_sweep_*.tsv",
        help="Glob pattern for per-lexeme aggregate TSV files.",
    )
    parser.add_argument(
        "--per-lexeme-pattern",
        type=str,
        default="clustering_metrics_per_lexeme_head_sweep_*.tsv",
        help="Glob pattern for per-lexeme metrics TSV files.",
    )
    parser.add_argument(
        "--global-pattern",
        type=str,
        default="clustering_metrics_global_head_sweep_*.tsv",
        help="Glob pattern for pooled-global metrics TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/clustering",
        help="Directory where PNG charts will be written.",
    )
    parser.add_argument(
        "--distance",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric subset to visualize.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["micro", "macro"],
        default="micro",
        help="Aggregation subset to visualize.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top head configs to include in top-config bar charts.",
    )
    parser.add_argument(
        "--skip-index-update",
        action="store_true",
        help="Do not regenerate docs index files after writing charts.",
    )
    return parser.parse_args()


def _clean_model_name(model: str) -> str:
    return model.replace("/", "_")


def _is_head_source(s: str) -> bool:
    return s in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}


def _load_tsv_glob(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        return pd.DataFrame()

    frames = []
    for path in paths:
        df = pd.read_csv(path, sep="\t")
        if "model" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)

    for col in ["silhouette", "davies_bouldin", "nearest_centroid_loo_acc"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["head_indices"] = out.get("head_indices", "").fillna("").astype(str)
    out["distance"] = out.get("distance", "").astype(str)
    if "aggregation" in out.columns:
        out["aggregation"] = out["aggregation"].astype(str)
    else:
        out["aggregation"] = ""
    if "embedding_source" in out.columns:
        out["embedding_source"] = out["embedding_source"].astype(str)
    else:
        out["embedding_source"] = ""
    return out


def _is_single_head(s: str) -> bool:
    return s.isdigit()


def _is_pair_head(s: str) -> bool:
    pieces = [p.strip() for p in s.split(",") if p.strip()]
    return len(pieces) == 2 and all(p.isdigit() for p in pieces)


def _parse_pair(s: str) -> Optional[Tuple[int, int]]:
    if not _is_pair_head(s):
        return None
    a, b = [int(x.strip()) for x in s.split(",")]
    return (a, b) if a <= b else (b, a)


def _combo_size(s: str) -> Optional[int]:
    s = str(s).strip()
    if not s:
        return None
    if s == "all":
        return None
    pieces = [p.strip() for p in s.split(",") if p.strip()]
    if not pieces:
        return None
    if all(p.isdigit() for p in pieces):
        return len(pieces)
    return None


def plot_best_vs_all(df: pd.DataFrame, out_dir: str, filename: str, title: str) -> Optional[str]:
    rows = []
    for model, sub in df.groupby("model"):
        sub = sub[np.isfinite(sub["silhouette"])].copy()
        if sub.empty:
            continue

        all_row = sub[sub["head_indices"] == "all"]
        best_row = sub.loc[sub["silhouette"].idxmax()]
        all_val = float(all_row.iloc[0]["silhouette"]) if not all_row.empty else np.nan

        rows.append(
            {
                "model": model,
                "all_silhouette": all_val,
                "best_silhouette": float(best_row["silhouette"]),
                "best_head_indices": str(best_row["head_indices"]),
            }
        )

    if not rows:
        return None

    plot_df = pd.DataFrame(rows).sort_values("best_silhouette", ascending=False)
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    all_vals = plot_df["all_silhouette"].to_numpy(dtype=float)
    best_vals = plot_df["best_silhouette"].to_numpy(dtype=float)
    finite_all = np.isfinite(all_vals)

    if np.any(finite_all):
        ax.bar(
            x[finite_all] - width / 2,
            all_vals[finite_all],
            width,
            label="all heads",
            color="#4c78a8",
        )

    if np.any(~finite_all):
        # Keep category alignment even when all-head baseline wasn't evaluated.
        ax.bar(
            x[~finite_all] - width / 2,
            np.zeros(np.sum(~finite_all), dtype=float),
            width,
            label="all heads (missing)",
            facecolor="none",
            edgecolor="#999999",
            linewidth=1.0,
        )

    ax.bar(x + width / 2, best_vals, width, label="best config", color="#f58518")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y_best = float(row["best_silhouette"])
        y_all = row["all_silhouette"]
        ax.text(i + width / 2, y_best, f"{y_best:.3f}\n{row['best_head_indices']}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(y_all):
            ax.text(i - width / 2, float(y_all), f"{float(y_all):.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(i - width / 2, 0.0, "N/A", ha="center", va="bottom", fontsize=8, color="#666666")

    ax.set_xticks(x)
    ax.set_xticklabels([_clean_model_name(m) for m in plot_df["model"]], rotation=20, ha="right")
    ax.set_ylabel("Silhouette")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    out_path = os.path.join(out_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_top_configs_per_model(df: pd.DataFrame, out_dir: str, top_n: int, prefix: str) -> List[str]:
    paths: List[str] = []
    for model, sub in df.groupby("model"):
        sub = sub[np.isfinite(sub["silhouette"])].copy()
        if sub.empty:
            continue

        top = sub.sort_values("silhouette", ascending=False).head(top_n)
        fig_h = max(4.5, 0.35 * len(top) + 1.5)
        fig, ax = plt.subplots(figsize=(9.5, fig_h))

        y = np.arange(len(top))
        ax.barh(y, top["silhouette"], color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(top["head_indices"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel("Silhouette")
        ax.set_title(f"Top {len(top)} Head Configs: {_clean_model_name(model)}")
        ax.grid(axis="x", alpha=0.25)

        for yi, val in enumerate(top["silhouette"].to_numpy(dtype=float)):
            ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)

        out_path = os.path.join(out_dir, f"{prefix}_top_configs_{_clean_model_name(model)}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_single_head_heatmap(df: pd.DataFrame, out_dir: str, prefix: str, title_prefix: str) -> List[str]:
    paths: List[str] = []
    for model, sub in df.groupby("model"):
        singles = sub[sub["head_indices"].apply(_is_single_head)].copy()
        if singles.empty:
            continue

        singles["head_idx"] = singles["head_indices"].astype(int)
        singles = singles.sort_values("head_idx")

        values = singles["silhouette"].to_numpy(dtype=float).reshape(1, -1)
        labels = singles["head_idx"].astype(str).tolist()

        fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(labels)), 2.6))
        im = ax.imshow(values, cmap="viridis", aspect="auto")
        ax.set_yticks([0])
        ax.set_yticklabels(["silhouette"])
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("Head Index")
        ax.set_title(f"{title_prefix} Single-Head Silhouette: {_clean_model_name(model)}")

        for j, v in enumerate(values[0]):
            if np.isfinite(v):
                ax.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        out_path = os.path.join(out_dir, f"{prefix}_single_head_heatmap_{_clean_model_name(model)}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_pair_heatmap(df: pd.DataFrame, out_dir: str, prefix: str, title_prefix: str) -> List[str]:
    paths: List[str] = []
    for model, sub in df.groupby("model"):
        pairs = sub[sub["head_indices"].apply(_is_pair_head)].copy()
        if pairs.empty:
            continue

        parsed = pairs["head_indices"].apply(_parse_pair)
        pairs = pairs[parsed.notna()].copy()
        if pairs.empty:
            continue

        pairs["pair"] = parsed[parsed.notna()]
        max_head = max(max(p) for p in pairs["pair"])
        mat = np.full((max_head + 1, max_head + 1), np.nan, dtype=float)

        for _, row in pairs.iterrows():
            i, j = row["pair"]
            mat[i, j] = row["silhouette"]
            mat[j, i] = row["silhouette"]

        fig, ax = plt.subplots(figsize=(7.8, 6.6))
        im = ax.imshow(mat, cmap="viridis", aspect="equal")
        ax.set_xlabel("Head Index")
        ax.set_ylabel("Head Index")
        ax.set_title(f"{title_prefix} Head-Pair Silhouette Matrix: {_clean_model_name(model)}")
        ax.set_xticks(np.arange(max_head + 1))
        ax.set_yticks(np.arange(max_head + 1))

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        out_path = os.path.join(out_dir, f"{prefix}_pair_head_heatmap_{_clean_model_name(model)}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_per_lexeme_violin_all_heads(df: pd.DataFrame, out_dir: str) -> Optional[str]:
    rows = []
    for model, sub in df.groupby("model"):
        s = sub[sub["head_indices"] == "all"]
        vals = s["silhouette"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((model, vals))

    if not rows:
        return None

    labels = [_clean_model_name(m) for m, _ in rows]
    values = [v for _, v in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    parts = ax.violinplot(values, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Silhouette")
    ax.set_title("Per-Lexeme Silhouette Distribution (All-Heads)")
    ax.grid(axis="y", alpha=0.25)

    for i, vals in enumerate(values, start=1):
        mean_v = float(np.mean(vals))
        ax.text(i, mean_v, f"{mean_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, "per_lexeme_violin_all_heads.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_per_lexeme_box_best_head(per_df: pd.DataFrame, global_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    rows = []
    for model, gsub in global_df.groupby("model"):
        gsub = gsub[np.isfinite(gsub["silhouette"])].copy()
        if gsub.empty:
            continue
        best = gsub.loc[gsub["silhouette"].idxmax()]
        head_label = str(best["head_indices"])
        psub = per_df[(per_df["model"] == model) & (per_df["head_indices"] == head_label)]
        vals = psub["silhouette"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((model, head_label, vals))

    if not rows:
        return None

    labels = [f"{_clean_model_name(m)}\nhead={h}" for m, h, _ in rows]
    values = [v for _, _, v in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Silhouette")
    ax.set_title("Per-Lexeme Silhouette Distribution (Best Global Head Config)")
    ax.grid(axis="y", alpha=0.25)

    for i, vals in enumerate(values, start=1):
        med_v = float(np.median(vals))
        ax.text(i, med_v, f"{med_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, "per_lexeme_box_best_global_head.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_combo_size_summary(global_df: pd.DataFrame, out_dir: str) -> Optional[str]:
    work = global_df.copy()
    work["combo_size"] = work["head_indices"].apply(_combo_size)
    work = work[np.isfinite(work["silhouette"]) & work["combo_size"].notna()].copy()
    if work.empty:
        return None

    work["combo_size"] = work["combo_size"].astype(int)
    rows = []
    for model, sub in work.groupby("model"):
        for size, ssub in sub.groupby("combo_size"):
            vals = ssub["silhouette"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append((model, int(size), vals))

    if not rows:
        return None

    rows.sort(key=lambda x: (_clean_model_name(x[0]), x[1]))
    labels = [f"{_clean_model_name(m)}\nk={k}" for m, k, _ in rows]
    values = [v for _, _, v in rows]

    fig, ax = plt.subplots(figsize=(max(10.0, 0.45 * len(labels)), 5.6))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Silhouette")
    ax.set_title("Global Silhouette by Head-Combination Size")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for i, vals in enumerate(values, start=1):
        med_v = float(np.median(vals))
        ax.text(i, med_v, f"{med_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, "global_combo_size_boxplot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    agg_df = _load_tsv_glob(args.aggregate_pattern)
    per_df = _load_tsv_glob(args.per_lexeme_pattern)
    global_df = _load_tsv_glob(args.global_pattern)

    agg_subset = agg_df[(agg_df["distance"] == args.distance) & (agg_df["aggregation"] == args.aggregation)].copy()
    agg_subset = agg_subset[agg_subset["embedding_source"].apply(_is_head_source)].copy()

    per_subset = per_df[per_df["distance"] == args.distance].copy()
    per_subset = per_subset[per_subset["embedding_source"].apply(_is_head_source)].copy()

    global_subset = global_df[global_df["distance"] == args.distance].copy()
    global_subset = global_subset[global_subset["embedding_source"].apply(_is_head_source)].copy()

    if agg_subset.empty and per_subset.empty and global_subset.empty:
        print("No rows matched the selected filters.")
        return

    saved: List[str] = []

    # Global pooled condition (labels isolated by lexeme::meaning_index).
    if not global_subset.empty:
        first = plot_best_vs_all(
            global_subset,
            args.output_dir,
            filename="global_best_vs_all_silhouette.png",
            title="Global Pooled: Best Head Config vs All-Heads Baseline",
        )
        if first is not None:
            saved.append(first)
        saved.extend(plot_top_configs_per_model(global_subset, args.output_dir, top_n=args.top_n, prefix="global"))
        saved.extend(plot_single_head_heatmap(global_subset, args.output_dir, prefix="global", title_prefix="Global"))
        saved.extend(plot_pair_heatmap(global_subset, args.output_dir, prefix="global", title_prefix="Global"))
        cpath = plot_combo_size_summary(global_subset, args.output_dir)
        if cpath is not None:
            saved.append(cpath)

    # Per-lexeme condition (one lexeme at a time).
    if not per_subset.empty:
        vpath = plot_per_lexeme_violin_all_heads(per_subset, args.output_dir)
        if vpath is not None:
            saved.append(vpath)
        if not global_subset.empty:
            bpath = plot_per_lexeme_box_best_head(per_subset, global_subset, args.output_dir)
            if bpath is not None:
                saved.append(bpath)

    # Retain the legacy summary plots from per-lexeme aggregate output.
    if not agg_subset.empty:
        first = plot_best_vs_all(
            agg_subset,
            args.output_dir,
            filename="per_lexeme_aggregate_best_vs_all_silhouette.png",
            title="Per-Lexeme Aggregate: Best Head Config vs All-Heads Baseline",
        )
        if first is not None:
            saved.append(first)
        saved.extend(plot_top_configs_per_model(agg_subset, args.output_dir, top_n=args.top_n, prefix="per_lexeme_aggregate"))

    if not saved:
        print("No charts were generated.")
        return

    print("Saved charts:")
    for p in saved:
        print(f"- {p}")

    if not args.skip_index_update:
        common.generate_index_html(args.output_dir)
        # Keep the docs tree index pages in sync with newly written charts.
        common.update_docs_indexes("docs")
        print(f"Updated index files under docs/ (including {args.output_dir}/index.html)")


if __name__ == "__main__":
    main()
