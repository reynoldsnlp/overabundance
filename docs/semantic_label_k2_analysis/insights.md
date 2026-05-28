# Semantic Label k=2 Analysis

Generated: 2026-05-28T09:29:28
Rows analyzed: 2432
Models: Qwen_Qwen2.5-32B, answerdotai_ModernBERT-base, bert_base_uncased, kanishka_GlossBERT, transfo-xl-wt103

## Overall Correlation Highlights

- Strongest overall Spearman association with `p_eq`: `k2_form_var_ratio` (rho=-0.271, Pearson=-0.180, n=2222). Higher k=2 / form variance ratio tends to line up with lower p_eq.
- Strongest ratio-based association with `p_eq`: `k2_form_var_ratio` (rho=-0.271, n=2222). Higher k=2 / form variance ratio tends to line up with lower p_eq.

- Strongest overall Spearman association with `p_cat`: `k2_form_sil_ratio` (rho=-0.239, Pearson=-0.009, n=2222). Higher k=2 / form silhouette ratio tends to line up with lower p_cat.
- Strongest ratio-based association with `p_cat`: `k2_form_sil_ratio` (rho=-0.239, n=2222). Higher k=2 / form silhouette ratio tends to line up with lower p_cat.

## Cond_type Contrasts

- Median `k2_sil` by `cond_type`: no_cond=0.269, prob=0.259, cat=0.151.
- Median `k2_form_sil_ratio` by `cond_type`: no_cond=1.000, prob=1.000, cat=0.977.
- Median `k2_var_mean` by `cond_type`: prob=39.382, no_cond=38.120, cat=37.894.
- Median `k2_form_var_ratio` by `cond_type`: cat=1.037, no_cond=1.000, prob=1.000.

## Strongest Model x Embed Slices

- `answerdotai_ModernBERT-base | orig` with `k2_form_var_ratio` vs `p_eq`: Spearman rho=-0.478, Pearson r=-0.272, n=224.
- `answerdotai_ModernBERT-base | delta` with `all_var` vs `p_eq`: Spearman rho=0.438, Pearson r=0.276, n=245.
- `kanishka_GlossBERT | delta` with `k2_form_sil_ratio` vs `p_eq`: Spearman rho=0.373, Pearson r=-0.012, n=224.

- `answerdotai_ModernBERT-base | delta` with `all_var` vs `p_cat`: Spearman rho=-0.491, Pearson r=-0.441, n=245.
- `answerdotai_ModernBERT-base | orig` with `k2_form_var_ratio` vs `p_cat`: Spearman rho=0.446, Pearson r=0.753, n=224.
- `answerdotai_ModernBERT-base | delta` with `k2_form_var_ratio` vs `p_cat`: Spearman rho=0.411, Pearson r=0.736, n=224.
