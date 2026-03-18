# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.8.0",
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "pillow>=10.0.0",
#   "scikit-learn>=1.5.0",
#   "tqdm>=4.0.0",
#   "numba>=0.59.0",
#   "llvmlite>=0.42.0",
#   "umap-learn>=0.5.6",
# ]
# ///

"""Grid search UMAP hyperparameters and render static plot grids.

Outputs (per model):
- docs/grid_search/UMAP/<model-stub>/umap_grid.html
- docs/grid_search/UMAP/<model-stub>/img/*.png

This script uses existing JSONL caches produced by:
- run_bert_base_uncased_legacy.py
- run_modernbert.py
- run_transfoxl.py

It does not load transformer models.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# When run as `uv run scripts/...py`, Python's import root is `scripts/`.
# Add the repo root so we can import `overabundance_common.py`.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

import overabundance_common as common


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    stub: str


MODELS: List[ModelSpec] = [
    ModelSpec(model_name="bert-base-uncased", stub="bert_base_uncased"),
    ModelSpec(model_name="answerdotai/ModernBERT-base", stub=common.model_slug("answerdotai/ModernBERT-base")),
    ModelSpec(model_name="transfo-xl-wt103", stub=common.model_slug("transfo-xl-wt103")),
]


def _load_embed_records(spec: ModelSpec, *, max_records: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    cache_path = f"head_embed_cache_{spec.stub}.jsonl"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Missing cache file {cache_path}. Run the corresponding model runner first to populate it."
        )

    records = common.load_records(max_records=max_records)
    cache = common.load_cache(cache_path)

    embed_records: List[Dict[str, Any]] = []
    missing = 0
    for rec in records:
        key = rec.get("cache_key")
        item = cache.get(key)
        if item is None:
            missing += 1
            continue
        vec = common.select_record_embedding(item, embedding_source="delta")
        if vec is None:
            missing += 1
            continue
        item = dict(item)
        item["delta"] = vec
        embed_records.append(item)

    if not embed_records:
        raise RuntimeError(f"No usable cached vectors found in {cache_path}.")

    return embed_records, missing


def _lexeme_colors(lexemes: List[str]):
    import matplotlib

    cmap = matplotlib.colormaps.get_cmap("tab20")
    uniq = sorted(set(lexemes))
    return {lex: cmap(i % 20) for i, lex in enumerate(uniq)}


def _save_scatter_png(
    out_path: str,
    *,
    xy: np.ndarray,
    lexemes: List[str],
    color_by_lexeme: Dict[str, Any],
    size_px: int = 220,
) -> None:
    import matplotlib.pyplot as plt

    colors = [color_by_lexeme.get(lx, (0.3, 0.3, 0.3, 0.8)) for lx in lexemes]

    dpi = 120
    inches = size_px / dpi
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xy[:, 0], xy[:, 1], s=4, c=colors, alpha=0.7, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _neighbor_grid(n_points: int) -> List[int]:
    max_nn = max(2, n_points // 4)

    # A compact grid that still spans the full range.
    candidates = set()
    for v in [2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]:
        if 2 <= v <= max_nn:
            candidates.add(v)

    # Add a few evenly spaced values to cover the tail.
    for v in np.linspace(2, max_nn, num=10):
        candidates.add(int(round(float(v))))

    out = sorted(x for x in candidates if 2 <= x <= max_nn)
    if max_nn not in out:
        out.append(max_nn)
    return out


def _write_grid_html(
    out_path: str,
    *,
    title: str,
    row_labels: List[str],
    col_labels: List[str],
    cell_imgs: List[List[str]],
    cell_params: List[List[str]],
    note_lines: List[str],
) -> None:
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      .note { color: #444; margin: 8px 0 18px 0; }
      .wrap { overflow: auto; border: 1px solid #eee; padding: 8px; }
      table { border-collapse: collapse; }
      th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
      th { position: sticky; top: 0; background: #fafafa; z-index: 2; }
      .rowhdr { position: sticky; left: 0; background: #fafafa; z-index: 1; }
      .param { font-size: 12px; color: #333; margin-bottom: 4px; white-space: nowrap; }
      img { display: block; width: 220px; height: 220px; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """

    lines = [
        "<html>",
        "<head>",
        f"<title>{esc(title)}</title>",
        css,
        "</head>",
        "<body>",
        f"<h1>{esc(title)}</h1>",
    ]

    if note_lines:
        lines.append("<div class='note'>" + "<br>".join(esc(x) for x in note_lines) + "</div>")

    lines.append("<div class='wrap'>")
    lines.append("<table>")

    lines.append("<tr>")
    lines.append("<th></th>")
    for c in col_labels:
        lines.append(f"<th>{esc(c)}</th>")
    lines.append("</tr>")

    for r_i, rlab in enumerate(row_labels):
        lines.append("<tr>")
        lines.append(f"<th class='rowhdr'>{esc(rlab)}</th>")
        for c_i in range(len(col_labels)):
            img = cell_imgs[r_i][c_i]
            params = cell_params[r_i][c_i]
            if not img:
                lines.append(f"<td><div class='param'><code>{esc(params)}</code></div><em>N/A</em></td>")
            else:
                lines.append(
                    "<td>"
                    + f"<div class='param'><code>{esc(params)}</code></div>"
                    + f"<img src='{esc(img)}' loading='lazy' />"
                    + "</td>"
                )
        lines.append("</tr>")

    lines += ["</table>", "</div>", "</body>", "</html>"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="If set, only use the first N records (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--pca-dims",
        type=int,
        default=50,
        help="Reduce embedding vectors to this many dims with PCA before UMAP (speed).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated list of model stubs to run (e.g. 'bert_base_uncased,transfo-xl-wt103').",
    )
    args = parser.parse_args()

    import umap

    out_root = os.path.join("docs", "grid_search", "UMAP")
    os.makedirs(out_root, exist_ok=True)

    min_dists = [round(x, 2) for x in np.arange(0.0, 1.0, 0.1)] + [0.99]

    only = {s.strip() for s in args.models.split(",") if s.strip()}

    for spec in MODELS:
        if only and spec.stub not in only:
            continue
        embed_records, missing = _load_embed_records(spec, max_records=args.max_records)

        X = np.array([r["delta"] for r in embed_records], dtype=float)
        lexemes = [str(r.get("lexeme", "")) for r in embed_records]
        color_by = _lexeme_colors(lexemes)

        # Speed: PCA pre-reduction
        pca_dims = min(args.pca_dims, X.shape[1])
        Xp = PCA(n_components=pca_dims, random_state=0).fit_transform(X)

        n = Xp.shape[0]
        n_neighbors_values = _neighbor_grid(n)

        model_dir = os.path.join(out_root, spec.stub)
        img_dir = os.path.join(model_dir, "img")
        os.makedirs(img_dir, exist_ok=True)

        row_labels = [f"n_neighbors={k}" for k in n_neighbors_values]
        col_labels = [f"min_dist={d}" for d in min_dists]

        cell_imgs: List[List[str]] = [["" for _ in min_dists] for _ in n_neighbors_values]
        cell_params: List[List[str]] = [["" for _ in min_dists] for _ in n_neighbors_values]

        print(
            f"\n[{spec.stub}] UMAP grid: {len(n_neighbors_values)}x{len(min_dists)} ({n} points; missing={missing}; max_n_neighbors={n//4})"
        )

        for r_i, k in enumerate(tqdm(n_neighbors_values, desc=f"UMAP n_neighbors ({spec.stub})")):
            for c_i, d in enumerate(min_dists):
                params = f"n_neighbors={k}, min_dist={d}"
                cell_params[r_i][c_i] = params

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=int(k),
                    min_dist=float(d),
                    random_state=0,
                    metric="euclidean",
                )
                xy = reducer.fit_transform(Xp)

                img_name = f"umap_k{k}_d{str(d).replace('.', 'p')}.png"
                img_path = os.path.join(img_dir, img_name)
                _save_scatter_png(img_path, xy=xy, lexemes=lexemes, color_by_lexeme=color_by)
                cell_imgs[r_i][c_i] = os.path.join("img", img_name)

        html_path = os.path.join(model_dir, "umap_grid.html")
        _write_grid_html(
            html_path,
            title=f"UMAP grid search: {spec.model_name}",
            row_labels=row_labels,
            col_labels=col_labels,
            cell_imgs=cell_imgs,
            cell_params=cell_params,
            note_lines=[
                f"points: {len(embed_records)} (missing cached vectors: {missing})",
                f"PCA pre-reduction: {pca_dims} dims",
                f"n_neighbors grid spans 2..N/4 (N={n})",
            ],
        )
        print(f"Wrote {html_path}")

        # Index generation (exclude img/)
        common.generate_index_html(model_dir)

    # Index generation (exclude img/)
    common.generate_index_html(os.path.join("docs", "grid_search"))
    common.generate_index_html(os.path.join("docs", "grid_search", "UMAP"))
    common.generate_index_html("docs")


if __name__ == "__main__":
    main()
