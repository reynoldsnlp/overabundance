# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib>=3.8.0",
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "scipy>=1.14.0",
# ]
# ///

"""Per-layer silhouette significance analysis.

Reads one or more `per_layer_k2_silhouette_<model>.tsv` files produced by
run_per_layer_silhouette.py. For each (model, embed_type) slice, tests
whether per-system k2 silhouette differs across layers, identifies the
best layer (argmax of mean silhouette across systems), and runs pairwise
Wilcoxon signed-rank tests vs that best layer with Holm-Bonferroni
correction. Writes one HTML report and per-model line plots.
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import overabundance_common as common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-pattern",
        default="per_layer_k2_silhouette_*.tsv",
        help="Glob for per-layer TSVs.",
    )
    p.add_argument("--output-dir", default=os.path.join("docs", "per_layer_silhouette"))
    return p.parse_args()


def _holm(p_values: List[float]) -> List[float]:
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    adjusted = [0.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(indexed):
        running = max(running, min(1.0, p * (m - rank)))
        adjusted[idx] = running
    return adjusted


def _layer_stats(sub: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for layer, sl in sub.groupby("layer"):
        vals = pd.to_numeric(sl["k2_sil"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rows.append({
            "layer": int(layer),
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan"),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
        })
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def _friedman_across_layers(sub: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Friedman test: each system is a 'subject' with one observation per layer."""
    pivot = sub.pivot_table(
        index=["system_id", "semantic_label", "form_1", "form_2"],
        columns="layer",
        values="k2_sil",
        aggfunc="first",
    )
    pivot = pivot.dropna(how="any")
    if pivot.shape[0] < 3 or pivot.shape[1] < 3:
        return None
    arrays = [pivot[col].to_numpy(dtype=float) for col in sorted(pivot.columns)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.friedmanchisquare(*arrays)
    return {
        "Q": float(result.statistic),
        "df": int(pivot.shape[1] - 1),
        "p": float(result.pvalue),
        "n_systems_paired": int(pivot.shape[0]),
        "n_layers": int(pivot.shape[1]),
    }


def _wilcoxon_vs_best(sub: pd.DataFrame, best_layer: int) -> List[Dict[str, object]]:
    pivot = sub.pivot_table(
        index=["system_id", "semantic_label", "form_1", "form_2"],
        columns="layer",
        values="k2_sil",
        aggfunc="first",
    )
    if best_layer not in pivot.columns:
        return []
    other_layers = sorted(c for c in pivot.columns if c != best_layer)
    raw_rows: List[Dict[str, object]] = []
    raw_ps: List[float] = []
    for layer in other_layers:
        paired = pivot[[best_layer, layer]].dropna(how="any")
        if paired.shape[0] < 3:
            raw_rows.append({"layer": int(layer), "n": int(paired.shape[0]),
                             "W": float("nan"), "p_raw": float("nan")})
            raw_ps.append(float("nan"))
            continue
        x = paired[best_layer].to_numpy(dtype=float)
        y = paired[layer].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                W, p = float(w.statistic), float(w.pvalue)
            except ValueError:
                W, p = float("nan"), float("nan")
        raw_rows.append({"layer": int(layer), "n": int(paired.shape[0]), "W": W, "p_raw": p})
        raw_ps.append(p)

    valid_idx = [i for i, p in enumerate(raw_ps) if np.isfinite(p)]
    holm_inputs = [raw_ps[i] for i in valid_idx]
    holm_out = _holm(holm_inputs)
    for k, row in enumerate(raw_rows):
        if k in valid_idx:
            row["p_holm"] = holm_out[valid_idx.index(k)]
        else:
            row["p_holm"] = float("nan")
    return raw_rows


def _fmt(x: object, places: int = 4) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, float) and not np.isfinite(x):
        return "—"
    if isinstance(x, float):
        return f"{x:.{places}f}"
    return str(x)


def _sig_marker(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _stats_table_html(stats_df: pd.DataFrame, best_layer: int) -> str:
    lines = ['<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-family:monospace;font-size:0.9em">']
    lines.append("<thead><tr>" + "".join(
        f"<th>{h}</th>" for h in ["layer", "n", "mean", "std", "median", "Q25", "Q75"]
    ) + "</tr></thead>")
    lines.append("<tbody>")
    for _, row in stats_df.iterrows():
        bold = ' style="font-weight:bold;background:#ffd"' if int(row["layer"]) == best_layer else ''
        lines.append(
            f"<tr{bold}>"
            f"<td>{int(row['layer'])}</td>"
            f"<td style='text-align:right'>{int(row['n'])}</td>"
            f"<td style='text-align:right'>{_fmt(row['mean'])}</td>"
            f"<td style='text-align:right'>{_fmt(row['std'])}</td>"
            f"<td style='text-align:right'>{_fmt(row['median'])}</td>"
            f"<td style='text-align:right'>{_fmt(row['q25'])}</td>"
            f"<td style='text-align:right'>{_fmt(row['q75'])}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _wilcoxon_table_html(wilc_rows: List[Dict[str, object]]) -> str:
    if not wilc_rows:
        return "<p><em>Not enough paired observations for Wilcoxon.</em></p>"
    lines = ['<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-family:monospace;font-size:0.9em">']
    lines.append("<thead><tr>" + "".join(
        f"<th>{h}</th>" for h in ["layer", "n_paired", "W", "p (raw)", "p (Holm)"]
    ) + "</tr></thead>")
    lines.append("<tbody>")
    for row in wilc_rows:
        p_holm = float(row.get("p_holm", float("nan")))
        lines.append(
            "<tr>"
            f"<td>{int(row['layer'])}</td>"
            f"<td style='text-align:right'>{int(row['n'])}</td>"
            f"<td style='text-align:right'>{_fmt(row['W'], places=2)}</td>"
            f"<td style='text-align:right'>{_fmt(row['p_raw'])}</td>"
            f"<td style='text-align:right'>{_fmt(p_holm)}{_sig_marker(p_holm)}</td>"
            "</tr>"
        )
    lines.append("</tbody></table>")
    return "\n".join(lines)


def _plot_layer_curve(stats_df: pd.DataFrame, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = stats_df["layer"].to_numpy()
    means = stats_df["mean"].to_numpy()
    q25 = stats_df["q25"].to_numpy()
    q75 = stats_df["q75"].to_numpy()
    ax.plot(layers, means, marker="o", color="C0", label="mean")
    ax.fill_between(layers, q25, q75, alpha=0.2, color="C0", label="Q25–Q75")
    best_layer = int(stats_df.loc[stats_df["mean"].idxmax(), "layer"])
    ax.axvline(best_layer, color="red", linestyle="--", linewidth=0.9, alpha=0.7, label=f"best layer = {best_layer}")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean k2_sil across systems")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(args.input_pattern))
    if not paths:
        raise FileNotFoundError(f"No per-layer TSVs matched: {args.input_pattern}")

    os.makedirs(args.output_dir, exist_ok=True)

    dfs = []
    for p in paths:
        df = pd.read_csv(p, sep="\t")
        df["k2_sil"] = pd.to_numeric(df["k2_sil"], errors="coerce")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)

    html_lines: List[str] = [
        "<html>",
        "<head><title>Per-layer silhouette analysis</title></head>",
        "<body style='font-family:sans-serif;max-width:1100px;margin:1em auto'>",
        "<h1>Per-layer k=2 silhouette analysis</h1>",
        f"<p>Generated: {datetime.now().isoformat(timespec='seconds')}.</p>",
        "<p>For each (<i>model</i>, <i>embed_type</i>) slice, the table below shows ",
        "k=2 silhouette descriptive statistics across systems for every hidden-state ",
        "layer (layer 0 = token embeddings; deeper layers are transformer blocks). ",
        "The best layer (highest mean silhouette, highlighted) is then compared to ",
        "every other layer via paired Wilcoxon signed-rank tests across systems, ",
        "with Holm-Bonferroni correction. Friedman omnibus test across all layers ",
        "is reported above the tables. Marker: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001.</p>",
        "<p><b>Caveat for layer 0:</b> layer 0 is the static token-embedding ",
        "lookup, which encodes only token identity. Because the cluster targets ",
        "are tokens of two different surface forms, layer 0 can trivially ",
        "separate them by form, often yielding silhouette &asymp; 1. Interpret as ",
        "a baseline rather than a contextual representation.</p>",
    ]

    models = sorted(full["model"].unique())
    for model in models:
        html_lines.append(f"<h2>Model: <code>{model}</code></h2>")
        for embed_type in sorted(full[full["model"] == model]["embed_type"].unique()):
            sub = full[(full["model"] == model) & (full["embed_type"] == embed_type)].copy()
            sub = sub.dropna(subset=["k2_sil"])
            html_lines.append(f"<h3>embed_type = <code>{embed_type}</code></h3>")
            if sub.empty:
                html_lines.append("<p><em>(no rows)</em></p>")
                continue

            stats_df = _layer_stats(sub)
            if stats_df.empty:
                html_lines.append("<p><em>(no finite silhouettes)</em></p>")
                continue
            best_layer = int(stats_df.loc[stats_df["mean"].idxmax(), "layer"])
            contextual_df = stats_df[stats_df["layer"] > 0]
            best_contextual = int(contextual_df.loc[contextual_df["mean"].idxmax(), "layer"]) if not contextual_df.empty else best_layer

            fried = _friedman_across_layers(sub)
            if fried is not None:
                html_lines.append(
                    f"<p>Friedman across layers: Q = {_fmt(fried['Q'], 3)}, "
                    f"df = {fried['df']}, "
                    f"<b>p = {_fmt(fried['p'])}</b> "
                    f"(n systems with full layer coverage = {fried['n_systems_paired']}, "
                    f"layers = {fried['n_layers']})</p>"
                )
            else:
                html_lines.append("<p>Friedman: insufficient paired data.</p>")

            plot_name = f"layer_curve_{model}_{embed_type}.png"
            plot_path = os.path.join(args.output_dir, plot_name)
            _plot_layer_curve(stats_df, f"{model} / {embed_type}", plot_path)
            html_lines.append(f'<p><img src="{plot_name}" alt="layer curve {model} {embed_type}"></p>')

            html_lines.append(
                f"<p><b>Best layer overall (argmax mean): {best_layer}.</b> "
                f"Best contextual layer (excluding layer 0): {best_contextual}.</p>"
            )
            html_lines.append(_stats_table_html(stats_df, best_layer))

            wilc = _wilcoxon_vs_best(sub, best_layer)
            html_lines.append(f"<h4>Pairwise Wilcoxon vs best layer ({best_layer})</h4>")
            html_lines.append(_wilcoxon_table_html(wilc))

            if best_contextual != best_layer:
                wilc_ctx = _wilcoxon_vs_best(
                    sub[sub["layer"] > 0],
                    best_contextual,
                )
                html_lines.append(
                    f"<h4>Pairwise Wilcoxon vs best contextual layer ({best_contextual}), excluding layer 0</h4>"
                )
                html_lines.append(_wilcoxon_table_html(wilc_ctx))

    html_lines += ["</body>", "</html>", ""]

    out_path = os.path.join(args.output_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
    print(f"Wrote {out_path}")

    docs_root = common._find_docs_root(args.output_dir) or "docs"
    common.update_docs_indexes(docs_root)
    print(f"Updated index files under {docs_root}.")


if __name__ == "__main__":
    main()
