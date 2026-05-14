# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "matplotlib>=3.8.0",
# ]
# ///

"""Generate clustering charts from per-system TSV outputs."""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import overabundance_common as common


CANONICAL_LABEL_TYPES = (
    "raw",
    "form",
    "conditioned-keep",
    "conditioned-collapse",
    "conditioned-exclude",
    "cat-only",
)


def _resolve_label_type(raw: str) -> str:
    alias = {
        "semantic": "raw",
        "joint_keep": "conditioned-keep",
        "joint_collapse": "conditioned-collapse",
        "joint_exclude": "conditioned-exclude",
        "cat_only": "cat-only",
    }
    value = alias.get(str(raw).strip().lower(), str(raw).strip().lower())
    if value not in CANONICAL_LABEL_TYPES:
        raise ValueError(f"Unsupported label type: {value}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-pattern",
        type=str,
        default="clustering_metrics_per_system_*.tsv",
        help="Glob pattern for per-system metrics TSV files.",
    )
    parser.add_argument(
        "--embedding-type",
        choices=["orig", "delta"],
        default="orig",
        help="Embedding type subset to visualize.",
    )
    parser.add_argument(
        "--label-type",
        type=str,
        default="raw",
        help=(
            "Label type subset to visualize. Supported values: "
            + ", ".join(CANONICAL_LABEL_TYPES)
        ),
    )
    parser.add_argument(
        "--distance",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance subset to visualize.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of best and worst systems to plot per model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where PNG charts will be written. Defaults to docs/clustering_<emb_type>_<label_type>/.",
    )
    parser.add_argument(
        "--skip-index-update",
        action="store_true",
        help="Do not regenerate docs index files after writing charts.",
    )
    return parser.parse_args()


def _label_title(label_type: str) -> str:
    mapping = {
        "raw": "Raw Meaning Labels",
        "form": "Form Labels",
        "conditioned-keep": "Conditioned Labels (Keep Prob)",
        "conditioned-collapse": "Conditioned Labels (Collapse Prob)",
        "conditioned-exclude": "Conditioned Labels (Exclude Prob)",
        "cat-only": "Cat-Only Form-Meaning Labels",
    }
    return mapping.get(label_type, label_type)


def _clean_model_name(model: str) -> str:
    return model.replace("/", "_")


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
    for col in ["silhouette", "nearest_centroid_loo_acc", "n_samples", "n_labels"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["embedding_type"] = out.get("embedding_type", "").astype(str)
    out["label_type"] = out.get("label_type", "").astype(str)
    out["distance"] = out.get("distance", "").astype(str)
    out["system_id"] = out.get("system_id", "").astype(str)
    out["lemma"] = out.get("lemma", "").astype(str)
    out["msps"] = out.get("msps", "").astype(str)
    return out


def _filter_subset(df: pd.DataFrame, *, embedding_type: str, label_type: str, distance: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out[out["embedding_type"] == embedding_type].copy()
    out = out[out["label_type"] == label_type].copy()
    out = out[out["distance"] == distance].copy()
    out = out[np.isfinite(out["silhouette"])].copy()
    return out


def plot_silhouette_distribution(df: pd.DataFrame, out_dir: str, *, label_type: str, distance: str) -> Optional[str]:
    rows = []
    for model, sub in df.groupby("model"):
        vals = sub["silhouette"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((model, vals))

    if not rows:
        return None

    labels = [_clean_model_name(model) for model, _ in rows]
    values = [vals for _, vals in rows]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Silhouette")
    ax.set_title(f"System Silhouette Distribution: {_label_title(label_type)} ({distance})")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)

    for i, vals in enumerate(values, start=1):
        med_v = float(np.median(vals))
        ax.text(i, med_v, f"{med_v:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, f"silhouette_distribution_{distance}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_mean_silhouette(df: pd.DataFrame, out_dir: str, *, label_type: str, distance: str) -> Optional[str]:
    rows = []
    for model, sub in df.groupby("model"):
        vals = sub["silhouette"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append((model, float(np.mean(vals))))

    if not rows:
        return None

    plot_df = pd.DataFrame(rows, columns=["model", "mean_silhouette"]).sort_values("mean_silhouette", ascending=False)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x, plot_df["mean_silhouette"], color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([_clean_model_name(model) for model in plot_df["model"]], rotation=20, ha="right")
    ax.set_ylabel("Mean silhouette")
    ax.set_title(f"Mean System Silhouette: {_label_title(label_type)} ({distance})")
    ax.grid(axis="y", alpha=0.25)

    for i, value in enumerate(plot_df["mean_silhouette"].to_numpy(dtype=float)):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    out_path = os.path.join(out_dir, f"mean_silhouette_{distance}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_ranked_systems(
    df: pd.DataFrame,
    out_dir: str,
    *,
    label_type: str,
    distance: str,
    top_n: int,
    ascending: bool,
) -> List[str]:
    saved: List[str] = []
    suffix = "lowest" if ascending else "top"
    title_prefix = "Lowest" if ascending else "Top"

    for model, sub in df.groupby("model"):
        work = sub.sort_values("silhouette", ascending=ascending).head(top_n).copy()
        if work.empty:
            continue

        labels = [f"{row.lemma}__{row.msps}" for row in work.itertuples(index=False)]
        values = work["silhouette"].to_numpy(dtype=float)
        y = np.arange(len(work))
        fig_h = max(4.5, 0.35 * len(work) + 1.5)

        fig, ax = plt.subplots(figsize=(10, fig_h))
        ax.barh(y, values, color="#f58518" if ascending else "#4c78a8")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Silhouette")
        ax.set_title(f"{title_prefix} Systems: {_clean_model_name(model)} ({_label_title(label_type)}, {distance})")
        ax.grid(axis="x", alpha=0.25)

        for yi, val in enumerate(values):
            ax.text(val, yi, f" {val:.3f}", va="center", ha="left", fontsize=8)

        out_path = os.path.join(out_dir, f"{suffix}_systems_{_clean_model_name(model)}_{distance}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        saved.append(out_path)

    return saved


def main() -> None:
    args = parse_args()
    label_type = _resolve_label_type(args.label_type)

    output_dir = args.output_dir or os.path.join("docs", f"clustering_{args.embedding_type}_{label_type}")
    os.makedirs(output_dir, exist_ok=True)

    metrics_df = _load_tsv_glob(args.metrics_pattern)
    subset = _filter_subset(
        metrics_df,
        embedding_type=args.embedding_type,
        label_type=label_type,
        distance=args.distance,
    )
    if subset.empty:
        print(
            "No rows matched the selected filters for "
            f"embedding_type={args.embedding_type}, label_type={label_type}, distance={args.distance}."
        )
        return

    saved: List[str] = []
    dist_path = plot_silhouette_distribution(subset, output_dir, label_type=label_type, distance=args.distance)
    if dist_path is not None:
        saved.append(dist_path)
    mean_path = plot_mean_silhouette(subset, output_dir, label_type=label_type, distance=args.distance)
    if mean_path is not None:
        saved.append(mean_path)
    saved.extend(
        plot_ranked_systems(
            subset,
            output_dir,
            label_type=label_type,
            distance=args.distance,
            top_n=args.top_n,
            ascending=False,
        )
    )
    saved.extend(
        plot_ranked_systems(
            subset,
            output_dir,
            label_type=label_type,
            distance=args.distance,
            top_n=args.top_n,
            ascending=True,
        )
    )

    if not saved:
        print("No charts were generated.")
        return

    print("Saved charts:")
    for path in saved:
        print(f"- {path}")

    if not args.skip_index_update:
        docs_root = common.refresh_docs_indexes_for_path(output_dir)
        if docs_root is not None:
            print(f"Updated index files under {docs_root} (including {output_dir}/index.html)")
        else:
            print(f"Updated {output_dir}/index.html")


if __name__ == "__main__":
    main()
