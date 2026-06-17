# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.0.0", "matplotlib>=3.7.0"]
# ///

"""Plot genuine per-layer by-meaning silhouette curves for the IMM 2026 slides.

Reads per_layer_meaning_summary_<slug>.tsv (produced by
run_per_layer_meaning_silhouette.py) for each model and draws, for the "orig"
embedding, silhouette-vs-relative-depth curves for the three by-meaning label
types. Relative depth (layer / max_layer) puts models with different layer
counts on a common x-axis.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt
import pandas as pd

MODELS = [
    ("bert_base_uncased", "BERT-base"),
    ("kanishka_GlossBERT", "GlossBERT"),
    ("answerdotai_ModernBERT-base", "ModernBERT-base"),
    ("transfo-xl-wt103", "Transformer-XL"),
]
LABELS = [
    ("raw", "by meaning (all senses)", "#1f77b4"),
    ("conditioned-exclude", "by meaning (non-categorical)", "#2ca02c"),
]
# TSV inputs live at the repo root; only the rendered figure belongs under docs/.
TSV_DIR = "."
OUTDIR = "docs/per_layer_silhouette"
EMBED = "orig"


def _load(slug: str) -> pd.DataFrame | None:
    path = os.path.join(TSV_DIR, f"per_layer_meaning_summary_{slug}.tsv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t")
    return df[(df.pair_significance_bucket == "all") & (df.embedding_type_layer == EMBED)]


def main() -> None:
    panels = [(s, t, _load(s)) for s, t in MODELS]
    panels = [(s, t, d) for s, t, d in panels if d is not None and not d.empty]
    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), sharey=True, squeeze=False
    )
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")

    for idx, (ax, (slug, title, df)) in enumerate(zip(axes, panels)):
        piv = df.pivot_table(index="layer", columns="label_type", values="silhouette_cosine_mean")
        max_layer = piv.index.max()
        depth = piv.index / max_layer if max_layer else piv.index
        for key, lab, color in LABELS:
            if key in piv.columns:
                ax.plot(depth, piv[key], marker="o", ms=3, lw=1.8, color=color, label=lab)
        ax.axhspan(-1, 0.25, color="0.92", zorder=0)  # "no substantial structure" band
        ax.axhline(0, color="0.5", lw=0.8, ls=":")
        ax.set_title(f"{title}  ({int(max_layer)+1} layers)", fontsize=10)
        ax.set_xlabel("relative depth (input → output)")
        ax.set_ylim(-0.65, 0.65)
        ax.grid(alpha=0.25)
        if idx % ncols == 0:
            ax.set_ylabel("macro silhouette (cosine)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Clustering by meaning across layers: genuine-meaning signal (blue/green)\n"
        "rises with depth but stays weak (grey = no substantial structure)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "per_layer_meaning_curves.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
