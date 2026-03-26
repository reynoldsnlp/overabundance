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
        "--label-scheme",
        choices=["raw_meaning", "human_expected"],
        default="raw_meaning",
        help="Filter clustering TSVs to one label scheme. TSVs without this column are treated as raw_meaning.",
    )
    parser.add_argument(
        "--skip-index-update",
        action="store_true",
        help="Do not regenerate docs index files after writing charts.",
    )
    return parser.parse_args()


def _clean_model_name(model: str) -> str:
    return model.replace("/", "_")


def _is_supported_source(s: str) -> bool:
    return s in {"delta", "delta_from_raw", "orig", "art", "head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}


def _is_head_source(s: str) -> bool:
    return s in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}


def _label_scheme_slug(label_scheme: str) -> str:
    return str(label_scheme).strip() or "raw_meaning"


def _label_scheme_title(label_scheme: str) -> str:
    if label_scheme == "human_expected":
        return "Significance-Collapsed Labels"
    return "Raw Meaning Labels"


def _title_with_scheme(title: str, label_scheme: str) -> str:
    return f"{title} ({_label_scheme_title(label_scheme)})"


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
    if "label_scheme" in out.columns:
        out["label_scheme"] = out["label_scheme"].fillna("").replace("", "raw_meaning").astype(str)
    else:
        out["label_scheme"] = "raw_meaning"
    return out


def _filter_subset(
    df: pd.DataFrame,
    *,
    distance: str,
    label_scheme: str,
    aggregation: Optional[str] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "distance" not in out.columns or "embedding_source" not in out.columns or "label_scheme" not in out.columns:
        return pd.DataFrame(columns=out.columns)
    out = out[out["distance"] == distance].copy()
    if aggregation is not None:
        if "aggregation" not in out.columns:
            return pd.DataFrame(columns=out.columns)
        out = out[out["aggregation"] == aggregation].copy()
    out = out[out["embedding_source"].apply(_is_supported_source)].copy()
    out = out[out["label_scheme"] == label_scheme].copy()
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


def _config_label(head_indices: str, embedding_source: str) -> str:
    head_indices = str(head_indices).strip()
    embedding_source = str(embedding_source).strip()
    if head_indices:
        return head_indices
    return embedding_source or "default"


def _has_head_sweep_configs(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    return bool(df["embedding_source"].apply(_is_head_source).any())


def _select_reference_subset(sub: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    all_rows = sub[sub["head_indices"] == "all"].copy()
    if not all_rows.empty:
        return all_rows, "all"
    if sub.empty:
        return sub.copy(), ""
    unique_config_labels = {
        _config_label(row.head_indices, row.embedding_source)
        for row in sub[["head_indices", "embedding_source"]].itertuples(index=False)
    }
    if len(unique_config_labels) == 1:
        return sub.copy(), next(iter(unique_config_labels))
    return pd.DataFrame(columns=sub.columns), ""


def plot_best_vs_all(df: pd.DataFrame, out_dir: str, filename: str, title: str) -> Optional[str]:
    rows = []
    for model, sub in df.groupby("model"):
        sub = sub[np.isfinite(sub["silhouette"])].copy()
        if sub.empty:
            continue

        all_row, baseline_label = _select_reference_subset(sub)
        best_row = sub.loc[sub["silhouette"].idxmax()]
        all_val = float(all_row.iloc[0]["silhouette"]) if not all_row.empty else np.nan

        rows.append(
            {
                "model": model,
                "baseline_silhouette": all_val,
                "baseline_label": baseline_label or "baseline",
                "best_silhouette": float(best_row["silhouette"]),
                "best_config_label": _config_label(best_row["head_indices"], best_row.get("embedding_source", "")),
            }
        )

    if not rows:
        return None

    plot_df = pd.DataFrame(rows).sort_values("best_silhouette", ascending=False)
    x = np.arange(len(plot_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    all_vals = plot_df["baseline_silhouette"].to_numpy(dtype=float)
    best_vals = plot_df["best_silhouette"].to_numpy(dtype=float)
    finite_all = np.isfinite(all_vals)
    baseline_legend = "all heads" if plot_df["baseline_label"].eq("all").all() else "reference config"

    if np.any(finite_all):
        ax.bar(
            x[finite_all] - width / 2,
            all_vals[finite_all],
            width,
            label=baseline_legend,
            color="#4c78a8",
        )

    if np.any(~finite_all):
        # Keep category alignment even when all-head baseline wasn't evaluated.
        ax.bar(
            x[~finite_all] - width / 2,
            np.zeros(np.sum(~finite_all), dtype=float),
            width,
            label=f"{baseline_legend} (missing)",
            facecolor="none",
            edgecolor="#999999",
            linewidth=1.0,
        )

    ax.bar(x + width / 2, best_vals, width, label="best config", color="#f58518")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y_best = float(row["best_silhouette"])
        y_all = row["baseline_silhouette"]
        ax.text(i + width / 2, y_best, f"{y_best:.3f}\n{row['best_config_label']}", ha="center", va="bottom", fontsize=8)
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


def plot_top_configs_per_model(
    df: pd.DataFrame,
    out_dir: str,
    top_n: int,
    prefix: str,
    *,
    label_scheme: str,
) -> List[str]:
    paths: List[str] = []
    for model, sub in df.groupby("model"):
        sub = sub[np.isfinite(sub["silhouette"])].copy()
        if sub.empty:
            continue

        top = sub.sort_values("silhouette", ascending=False).head(top_n)
        y_labels = [
            _config_label(head_indices, embedding_source)
            for head_indices, embedding_source in top[["head_indices", "embedding_source"]].itertuples(index=False)
        ]
        fig_h = max(4.5, 0.35 * len(top) + 1.5)
        fig, ax = plt.subplots(figsize=(9.5, fig_h))

        y = np.arange(len(top))
        ax.barh(y, top["silhouette"], color="#1f77b4")
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Silhouette")
        title_prefix = "Top Configs" if len(set(y_labels)) != 1 or y_labels[0] != top.iloc[0]["embedding_source"] else "Config"
        ax.set_title(_title_with_scheme(f"{title_prefix}: {_clean_model_name(model)}", label_scheme))
        ax.grid(axis="x", alpha=0.25)

        for yi, val in enumerate(top["silhouette"].to_numpy(dtype=float)):
            ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)

        out_path = os.path.join(out_dir, f"{prefix}_top_configs_{_clean_model_name(model)}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_single_head_heatmap(
    df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    title_prefix: str,
    *,
    label_scheme: str,
) -> List[str]:
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
        ax.set_title(_title_with_scheme(f"{title_prefix} Single-Head Silhouette: {_clean_model_name(model)}", label_scheme))

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


def plot_pair_heatmap(
    df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    title_prefix: str,
    *,
    label_scheme: str,
) -> List[str]:
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
        ax.set_title(_title_with_scheme(f"{title_prefix} Head-Pair Silhouette Matrix: {_clean_model_name(model)}", label_scheme))
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


def plot_per_lexeme_violin_all_heads(df: pd.DataFrame, out_dir: str, *, label_scheme: str) -> Optional[str]:
    rows = []
    for model, sub in df.groupby("model"):
        s, config_label = _select_reference_subset(sub)
        vals = s["silhouette"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((model, config_label or "reference", vals))

    if not rows:
        return None

    labels = [
        _clean_model_name(m) if config_label == "all" else f"{_clean_model_name(m)}\n{config_label}"
        for m, config_label, _ in rows
    ]
    values = [v for _, _, v in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    parts = ax.violinplot(values, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.7)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Silhouette")
    title = "Per-Lexeme Silhouette Distribution (Reference Config)"
    if all(config_label == "all" for _, config_label, _ in rows):
        title = "Per-Lexeme Silhouette Distribution (All-Heads)"
    ax.set_title(_title_with_scheme(title, label_scheme))
    ax.grid(axis="y", alpha=0.25)

    for i, vals in enumerate(values, start=1):
        mean_v = float(np.mean(vals))
        ax.text(i, mean_v, f"{mean_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, f"per_lexeme_violin_all_heads_{_label_scheme_slug(label_scheme)}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_per_lexeme_box_best_head(
    per_df: pd.DataFrame,
    global_df: pd.DataFrame,
    out_dir: str,
    *,
    label_scheme: str,
) -> Optional[str]:
    rows = []
    for model, gsub in global_df.groupby("model"):
        gsub = gsub[np.isfinite(gsub["silhouette"])].copy()
        if gsub.empty:
            continue
        best = gsub.loc[gsub["silhouette"].idxmax()]
        head_label = str(best["head_indices"])
        config_label = _config_label(best["head_indices"], best.get("embedding_source", ""))
        psub = per_df[(per_df["model"] == model) & (per_df["head_indices"] == head_label)]
        vals = psub["silhouette"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((model, config_label, vals))

    if not rows:
        return None

    labels = [f"{_clean_model_name(m)}\nconfig={h}" for m, h, _ in rows]
    values = [v for _, _, v in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Silhouette")
    ax.set_title(_title_with_scheme("Per-Lexeme Silhouette Distribution (Best Global Config)", label_scheme))
    ax.grid(axis="y", alpha=0.25)

    for i, vals in enumerate(values, start=1):
        med_v = float(np.median(vals))
        ax.text(i, med_v, f"{med_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, f"per_lexeme_box_best_global_head_{_label_scheme_slug(label_scheme)}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_combo_size_summary(global_df: pd.DataFrame, out_dir: str, *, label_scheme: str) -> Optional[str]:
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
    ax.set_title(_title_with_scheme("Global Silhouette by Head-Combination Size", label_scheme))
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for i, vals in enumerate(values, start=1):
        med_v = float(np.median(vals))
        ax.text(i, med_v, f"{med_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, f"global_combo_size_boxplot_{_label_scheme_slug(label_scheme)}.png")
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

    agg_subset = _filter_subset(agg_df, distance=args.distance, aggregation=args.aggregation, label_scheme=args.label_scheme)
    per_subset = _filter_subset(per_df, distance=args.distance, label_scheme=args.label_scheme)
    global_subset = _filter_subset(global_df, distance=args.distance, label_scheme=args.label_scheme)

    if agg_subset.empty and per_subset.empty and global_subset.empty:
        print(f"No rows matched the selected filters for label_scheme={args.label_scheme}.")
        return

    saved: List[str] = []
    scheme_slug = _label_scheme_slug(args.label_scheme)
    global_has_head_sweep = _has_head_sweep_configs(global_subset)
    agg_has_head_sweep = _has_head_sweep_configs(agg_subset)

    # Global pooled condition under the selected label scheme.
    if not global_subset.empty:
        if global_has_head_sweep:
            first = plot_best_vs_all(
                global_subset,
                args.output_dir,
                filename=f"global_best_vs_all_silhouette_{scheme_slug}.png",
                title=_title_with_scheme("Global Pooled: Best Head Config vs All-Heads Baseline", args.label_scheme),
            )
            if first is not None:
                saved.append(first)
        saved.extend(
            plot_top_configs_per_model(
                global_subset,
                args.output_dir,
                top_n=args.top_n,
                prefix=f"global_{scheme_slug}",
                label_scheme=args.label_scheme,
            )
        )
        if global_has_head_sweep:
            saved.extend(
                plot_single_head_heatmap(
                    global_subset,
                    args.output_dir,
                    prefix=f"global_{scheme_slug}",
                    title_prefix="Global",
                    label_scheme=args.label_scheme,
                )
            )
            saved.extend(
                plot_pair_heatmap(
                    global_subset,
                    args.output_dir,
                    prefix=f"global_{scheme_slug}",
                    title_prefix="Global",
                    label_scheme=args.label_scheme,
                )
            )
            cpath = plot_combo_size_summary(global_subset, args.output_dir, label_scheme=args.label_scheme)
            if cpath is not None:
                saved.append(cpath)

    # Per-lexeme condition (one lexeme at a time).
    if not per_subset.empty:
        vpath = plot_per_lexeme_violin_all_heads(per_subset, args.output_dir, label_scheme=args.label_scheme)
        if vpath is not None:
            saved.append(vpath)
        if not global_subset.empty:
            bpath = plot_per_lexeme_box_best_head(
                per_subset,
                global_subset,
                args.output_dir,
                label_scheme=args.label_scheme,
            )
            if bpath is not None:
                saved.append(bpath)

    # Retain the legacy summary plots from per-lexeme aggregate output.
    if not agg_subset.empty:
        if agg_has_head_sweep:
            first = plot_best_vs_all(
                agg_subset,
                args.output_dir,
                filename=f"per_lexeme_aggregate_best_vs_all_silhouette_{scheme_slug}.png",
                title=_title_with_scheme("Per-Lexeme Aggregate: Best Head Config vs All-Heads Baseline", args.label_scheme),
            )
            if first is not None:
                saved.append(first)
        saved.extend(
            plot_top_configs_per_model(
                agg_subset,
                args.output_dir,
                top_n=args.top_n,
                prefix=f"per_lexeme_aggregate_{scheme_slug}",
                label_scheme=args.label_scheme,
            )
        )

    if not saved:
        print("No charts were generated.")
        return

    print("Saved charts:")
    for p in saved:
        print(f"- {p}")

    if not args.skip_index_update:
        docs_root = common.refresh_docs_indexes_for_path(args.output_dir)
        if docs_root is not None:
            print(f"Updated index files under {docs_root} (including {args.output_dir}/index.html)")
        else:
            print(f"Updated {args.output_dir}/index.html")


if __name__ == "__main__":
    main()
