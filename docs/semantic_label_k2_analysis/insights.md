# Semantic Label k=2 Analysis

Generated: 2026-05-27T21:45:24
Rows analyzed: 1942
Models: answerdotai_ModernBERT-base, bert_base_uncased, kanishka_GlossBERT, transfo-xl-wt103

## Overall Correlation Highlights

- Strongest overall Spearman association with `p_eq`: `k2_form_var_ratio` (rho=-0.291, Pearson=-0.189, n=1774). Higher k=2 / form variance ratio tends to line up with lower p_eq.
- Strongest ratio-based association with `p_eq`: `k2_form_var_ratio` (rho=-0.291, n=1774). Higher k=2 / form variance ratio tends to line up with lower p_eq.

- Strongest overall Spearman association with `p_cat`: `k2_form_sil_ratio` (rho=-0.257, Pearson=-0.002, n=1774). Higher k=2 / form silhouette ratio tends to line up with lower p_cat.
- Strongest ratio-based association with `p_cat`: `k2_form_sil_ratio` (rho=-0.257, n=1774). Higher k=2 / form silhouette ratio tends to line up with lower p_cat.

## Cond_type Contrasts

- Median `k2_sil` by `cond_type`: no_cond=0.253, prob=0.240, cat=0.127.
- Median `k2_form_sil_ratio` by `cond_type`: no_cond=1.000, prob=1.000, cat=0.906.
- Median `k2_var_mean` by `cond_type`: prob=16.871, cat=16.064, no_cond=15.399.
- Median `k2_form_var_ratio` by `cond_type`: cat=1.062, no_cond=1.000, prob=1.000.

## Strongest Model x Embed Slices

- `answerdotai_ModernBERT-base | orig` with `k2_form_var_ratio` vs `p_eq`: Spearman rho=-0.478, Pearson r=-0.272, n=224.
- `answerdotai_ModernBERT-base | delta` with `all_var` vs `p_eq`: Spearman rho=0.438, Pearson r=0.276, n=245.
- `kanishka_GlossBERT | delta` with `k2_form_sil_ratio` vs `p_eq`: Spearman rho=0.373, Pearson r=-0.012, n=224.

- `answerdotai_ModernBERT-base | delta` with `all_var` vs `p_cat`: Spearman rho=-0.491, Pearson r=-0.441, n=245.
- `answerdotai_ModernBERT-base | orig` with `k2_form_var_ratio` vs `p_cat`: Spearman rho=0.446, Pearson r=0.753, n=224.
- `answerdotai_ModernBERT-base | delta` with `k2_form_var_ratio` vs `p_cat`: Spearman rho=0.411, Pearson r=0.736, n=224.
