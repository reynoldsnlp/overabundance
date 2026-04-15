# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "plotly>=6.0.0",
#   "scipy>=1.14.0",
# ]
# ///

"""Analyze semantic-label k=2 metrics and generate summary plots.

The script reads `clustering_semantic_label_k2_multi.tsv`, derives a few
comparison features such as k2/form ratios, computes Pearson and Spearman
correlations against `p_eq` and `p_cat`, writes a short markdown report, and
generates interactive HTML visualizations.
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

import overabundance_common as common


CORE_METRICS = [
    "k2_sil",
    "form_sil",
    "k2_form_sil_ratio",
    "k2_var_mean",
    "form_var_mean",
    "k2_form_var_ratio",
    "all_var",
]

VIOLIN_METRICS = [
    "k2_sil",
    "form_sil",
    "k2_form_sil_ratio",
    "k2_var_mean",
    "form_var_mean",
    "k2_form_var_ratio",
]

TARGETS = ["p_eq", "p_cat"]

PRETTY_LABELS = {
    "k2_sil": "k=2 silhouette",
    "form_sil": "Form silhouette",
    "k2_form_sil_ratio": "k=2 / form silhouette ratio",
    "k2_var_mean": "k=2 mean variance",
    "form_var_mean": "Form mean variance",
    "k2_form_var_ratio": "k=2 / form variance ratio",
    "all_var": "All-token variance",
    "p_eq": "p_eq",
    "p_cat": "p_cat",
}

COND_TYPE_ORDER = ["no_cond", "prob", "cat"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="clustering_semantic_label_k2_multi.tsv",
        help="Input TSV produced by run_semantic_label_k2_metrics.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("docs", "semantic_label_k2_analysis"),
        help="Directory for analysis tables, report, and plots.",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=25,
        help="Minimum finite pairs required for a reported correlation.",
    )
    parser.add_argument(
        "--skip-index-update",
        action="store_true",
        help="Do not regenerate docs index files after writing outputs.",
    )
    return parser.parse_args()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(num), np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0.0)
    out[mask] = num[mask] / den[mask]
    return pd.Series(out, index=numerator.index, dtype=float)


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "form_1_freq",
        "form_2_freq",
        "chisq_eq",
        "p_eq",
        "p_cat",
        "k2_sil",
        "k2_var_1",
        "k2_var_2",
        "k2_var_mean",
        "form_sil",
        "form_var_1",
        "form_var_2",
        "form_var_mean",
        "all_var",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["model", "embed_type", "lemma", "msps", "semantic_label", "cond_type"]:
        out[col] = out[col].fillna("").astype(str)

    out["cond_type"] = pd.Categorical(out["cond_type"], categories=COND_TYPE_ORDER, ordered=True)
    out["k2_form_sil_ratio"] = _safe_divide(out["k2_sil"], out["form_sil"])
    out["k2_form_var_ratio"] = _safe_divide(out["k2_var_mean"], out["form_var_mean"])

    epsilon = 1e-6
    out["p_eq_clipped"] = out["p_eq"].clip(lower=epsilon, upper=1.0)
    out["p_cat_clipped"] = out["p_cat"].clip(lower=epsilon, upper=1.0)
    out["neg_log10_p_eq"] = -np.log10(out["p_eq_clipped"])
    out["neg_log10_p_cat"] = -np.log10(out["p_cat_clipped"])
    out["model_embed"] = out["model"] + " | " + out["embed_type"]
    return out


def _correlation_rows(
    df: pd.DataFrame,
    *,
    group_type: str,
    group_label: str,
    min_pairs: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for metric in CORE_METRICS:
        for target in TARGETS:
            subset = df[[metric, target]].replace([np.inf, -np.inf], np.nan).dropna()
            n_pairs = len(subset)
            if n_pairs < min_pairs:
                continue

            x = subset[metric].to_numpy(dtype=float)
            y = subset[target].to_numpy(dtype=float)
            if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pearson = stats.pearsonr(x, y)
                spearman = stats.spearmanr(x, y)

            rows.append(
                {
                    "group_type": group_type,
                    "group_label": group_label,
                    "metric": metric,
                    "target": target,
                    "n_pairs": int(n_pairs),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_rho": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                }
            )
    return rows


def _build_correlation_table(df: pd.DataFrame, *, min_pairs: int) -> pd.DataFrame:
    rows = _correlation_rows(df, group_type="overall", group_label="all", min_pairs=min_pairs)
    for (model, embed_type), sub in df.groupby(["model", "embed_type"], sort=True):
        label = f"{model} | {embed_type}"
        rows.extend(_correlation_rows(sub, group_type="model_embed", group_label=label, min_pairs=min_pairs))
    return pd.DataFrame(rows)


def _build_cond_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["cond_type", "embed_type"], observed=False)[CORE_METRICS]
        .agg(["median", "mean", "count"])
        .reset_index()
    )
    grouped.columns = ["_".join(str(part) for part in col if part) for col in grouped.columns.to_flat_index()]
    return grouped


def _top_overall(corr_df: pd.DataFrame, *, target: str, metric_filter: Iterable[str] | None = None) -> pd.DataFrame:
    subset = corr_df[(corr_df["group_type"] == "overall") & (corr_df["target"] == target)].copy()
    if metric_filter is not None:
        subset = subset[subset["metric"].isin(list(metric_filter))].copy()
    subset["abs_spearman"] = subset["spearman_rho"].abs()
    return subset.sort_values(["abs_spearman", "n_pairs"], ascending=[False, False]).head(3)


def _top_model_embed(corr_df: pd.DataFrame, *, target: str) -> pd.DataFrame:
    subset = corr_df[corr_df["target"] == target].copy()
    subset = subset[subset["group_type"] == "model_embed"].copy()
    subset["abs_spearman"] = subset["spearman_rho"].abs()
    return subset.sort_values(["abs_spearman", "n_pairs"], ascending=[False, False]).head(3)


def _median_by_cond(df: pd.DataFrame, metric: str) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for cond_type, sub in df.groupby("cond_type", observed=False, sort=False):
        vals = pd.to_numeric(sub[metric], errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            continue
        rows.append((str(cond_type), float(np.median(vals))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def _direction_text(metric: str, target: str, rho: float) -> str:
    metric_label = PRETTY_LABELS.get(metric, metric)
    target_label = PRETTY_LABELS.get(target, target)
    if rho < 0:
        return f"Higher {metric_label} tends to line up with lower {target_label}."
    return f"Higher {metric_label} tends to line up with higher {target_label}."


def _write_insights_report(df: pd.DataFrame, corr_df: pd.DataFrame, output_path: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Semantic Label k=2 Analysis",
        "",
        f"Generated: {timestamp}",
        f"Rows analyzed: {len(df)}",
        f"Models: {', '.join(sorted(df['model'].unique().tolist()))}",
        "",
        "## Overall Correlation Highlights",
        "",
    ]

    for target in TARGETS:
        best = _top_overall(corr_df, target=target)
        ratio_best = _top_overall(corr_df, target=target, metric_filter=["k2_form_sil_ratio", "k2_form_var_ratio"])
        if not best.empty:
            row = best.iloc[0]
            lines.append(
                "- Strongest overall Spearman association with "
                f"`{target}`: `{row['metric']}` (rho={row['spearman_rho']:.3f}, "
                f"Pearson={row['pearson_r']:.3f}, n={int(row['n_pairs'])}). "
                + _direction_text(str(row["metric"]), target, float(row["spearman_rho"]))
            )
        if not ratio_best.empty:
            row = ratio_best.iloc[0]
            lines.append(
                "- Strongest ratio-based association with "
                f"`{target}`: `{row['metric']}` (rho={row['spearman_rho']:.3f}, n={int(row['n_pairs'])}). "
                + _direction_text(str(row["metric"]), target, float(row["spearman_rho"]))
            )
        lines.append("")

    lines += ["## Cond_type Contrasts", ""]
    for metric in ["k2_sil", "k2_form_sil_ratio", "k2_var_mean", "k2_form_var_ratio"]:
        medians = _median_by_cond(df, metric)
        if not medians:
            continue
        summary = ", ".join(f"{name}={value:.3f}" for name, value in medians)
        lines.append(f"- Median `{metric}` by `cond_type`: {summary}.")
    lines.append("")

    lines += ["## Strongest Model x Embed Slices", ""]
    for target in TARGETS:
        top_rows = _top_model_embed(corr_df, target=target)
        if top_rows.empty:
            continue
        for row in top_rows.itertuples(index=False):
            lines.append(
                "- "
                f"`{row.group_label}` with `{row.metric}` vs `{row.target}`: "
                f"Spearman rho={row.spearman_rho:.3f}, Pearson r={row.pearson_r:.3f}, n={int(row.n_pairs)}."
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def _write_correlation_heatmap(corr_df: pd.DataFrame, *, method: str, output_path: str) -> None:
    overall = corr_df[corr_df["group_type"] == "overall"].copy()
    value_col = "spearman_rho" if method == "spearman" else "pearson_r"
    heat = overall.pivot(index="metric", columns="target", values=value_col)
    heat = heat.reindex(index=CORE_METRICS, columns=TARGETS)

    z = heat.to_numpy(dtype=float)
    text = np.where(np.isfinite(z), np.round(z, 3).astype(str), "")
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[PRETTY_LABELS.get(col, col) for col in heat.columns.tolist()],
            y=[PRETTY_LABELS.get(idx, idx) for idx in heat.index.tolist()],
            colorscale="RdBu",
            zmid=0.0,
            text=text,
            texttemplate="%{text}",
            colorbar_title=f"{method.title()} corr",
        )
    )
    fig.update_layout(
        title=f"Overall {method.title()} Correlations with p-values",
        width=900,
        height=520,
    )
    fig.write_html(output_path)


def _write_scatter_plots(df: pd.DataFrame, output_dir: str) -> List[str]:
    saved: List[str] = []
    for target in TARGETS:
        target_clipped = f"{target}_clipped"
        for metric in CORE_METRICS:
            plot_df = df[[metric, target, target_clipped, "cond_type", "model", "embed_type", "lemma", "msps", "semantic_label"]].copy()
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric, target_clipped])
            if plot_df.empty:
                continue

            fig = px.scatter(
                plot_df,
                x=metric,
                y=target_clipped,
                color="cond_type",
                symbol="model",
                facet_col="embed_type",
                hover_data=["lemma", "msps", "semantic_label", target],
                opacity=0.7,
                log_y=True,
                title=f"{PRETTY_LABELS.get(metric, metric)} vs {PRETTY_LABELS.get(target, target)}",
                labels={metric: PRETTY_LABELS.get(metric, metric), target_clipped: f"{PRETTY_LABELS.get(target, target)} (log scale)"},
            )
            fig.update_traces(marker={"size": 7, "line": {"width": 0}})
            fig.update_layout(width=1200, height=520)
            out_path = os.path.join(output_dir, f"scatter_{metric}_vs_{target}.html")
            fig.write_html(out_path)
            saved.append(out_path)
    return saved


def _write_violin_plots(df: pd.DataFrame, output_dir: str) -> List[str]:
    saved: List[str] = []
    for metric in VIOLIN_METRICS:
        plot_df = df[[metric, "cond_type", "embed_type", "model"]].copy()
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
        if plot_df.empty:
            continue

        fig = px.violin(
            plot_df,
            x="cond_type",
            y=metric,
            color="embed_type",
            facet_col="model",
            facet_col_wrap=2,
            box=True,
            points="outliers",
            category_orders={"cond_type": COND_TYPE_ORDER},
            title=f"{PRETTY_LABELS.get(metric, metric)} by cond_type",
            labels={metric: PRETTY_LABELS.get(metric, metric), "cond_type": "cond_type"},
        )
        fig.update_layout(width=1300, height=900)
        out_path = os.path.join(output_dir, f"violin_{metric}_by_cond_type.html")
        fig.write_html(out_path)
        saved.append(out_path)
    return saved


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input TSV not found: {args.input_path}")

    print(f"Loading {args.input_path}...")
    df = pd.read_csv(args.input_path, sep="\t")
    df = _prepare_dataframe(df)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Writing analysis outputs to {args.output_dir}...")

    print("Computing correlation tables...")
    corr_df = _build_correlation_table(df, min_pairs=args.min_pairs)
    if corr_df.empty:
        raise RuntimeError("No correlations met the minimum finite-pair threshold.")
    corr_path = os.path.join(args.output_dir, "correlation_summary.tsv")
    corr_df.sort_values(["group_type", "group_label", "target", "metric"]).to_csv(corr_path, sep="\t", index=False)

    print("Computing cond_type summary table...")
    cond_summary = _build_cond_type_summary(df)
    cond_summary_path = os.path.join(args.output_dir, "cond_type_metric_summary.tsv")
    cond_summary.to_csv(cond_summary_path, sep="\t", index=False)

    print("Writing insights report...")
    report_path = os.path.join(args.output_dir, "insights.md")
    _write_insights_report(df, corr_df, report_path)

    print("Writing heatmaps...")
    _write_correlation_heatmap(corr_df, method="spearman", output_path=os.path.join(args.output_dir, "heatmap_spearman.html"))
    _write_correlation_heatmap(corr_df, method="pearson", output_path=os.path.join(args.output_dir, "heatmap_pearson.html"))

    print("Writing scatter plots...")
    scatter_dir = os.path.join(args.output_dir, "scatter")
    os.makedirs(scatter_dir, exist_ok=True)
    _write_scatter_plots(df, scatter_dir)

    print("Writing violin plots...")
    violin_dir = os.path.join(args.output_dir, "violin")
    os.makedirs(violin_dir, exist_ok=True)
    _write_violin_plots(df, violin_dir)

    if not args.skip_index_update:
        docs_root = common._find_docs_root(args.output_dir) or "docs"
        common.update_docs_indexes(docs_root)
        print(f"Updated index files under {docs_root}.")

    print(f"Saved {corr_path}")
    print(f"Saved {cond_summary_path}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()