# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "scikit-learn>=1.0.0",
#   "protobuf>=4.0.0",
#   "sacremoses>=0.1.0",
#   "torch>=2.0.0",
#   "tqdm>=4.0.0",
#   "transformers>=4.48.0",
#   "numba>=0.59.0",
#   "llvmlite>=0.42.0",
#   "umap-learn",
# ]
# ///

"""Run the experiment with transfo-xl-wt103 (Transformer-XL).

Notes:
- Transformers marks TransfoXL as deprecated and gates its tokenizer behind TRUST_REMOTE_CODE.
- We also apply a small torch patch for a known deprecated-codepath bug.
"""

import os

import overabundance_common as common


def main() -> None:
    args = common.parse_runner_args()

    common.setup_environment()

    # Recent transformers versions gate TransfoXLTokenizer behind a trust switch
    # because it relies on `pickle.load` for vocab files.
    os.environ.setdefault("TRUST_REMOTE_CODE", "True")

    if args.deps_only:
        common.deps_only_report(extra_imports=["sacremoses", "google.protobuf"])
        return

    model_name = "transfo-xl-wt103"
    slug = common.model_slug(model_name)
    cache_path, use_cache = common.setup_cache(model_name=model_name, cache_slug=slug)
    records = common.load_records(max_records=args.max_records)

    if use_cache:
        cache = common.load_cache(cache_path)
        missing = [r for r in records if r.get("cache_key") not in cache or cache[r.get("cache_key")].get("delta") is None]
        if not missing:
            embed_records = [cache[r["cache_key"]] for r in records]
            common.write_visualizations(embed_records, model_name=model_name, slug=slug, docs_dir="docs")
            common.write_tokens_per_word_summary(embed_records, model_slug=slug, docs_dir="docs")
            common.write_skipped_html([], model_slug=slug, docs_dir="docs")
            common.update_docs_indexes("docs")
            print("Updated docs indexes (cache-only run).")
            return

    import torch
    from transformers import AutoModel, AutoTokenizer

    common.patch_transfoxl_torch_type_as(torch)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def get_embedding(sentence: str, word: str):
        return common.hf_get_embedding(tokenizer, model, sentence, word)

    skipped_rows = []

    def log_skip(rec, reason: str) -> None:
        skipped_rows.append(
            {
                "cache_key": rec.get("cache_key"),
                "ID": rec.get("ID"),
                "lexeme": rec.get("lexeme"),
                "mps": rec.get("mps"),
                "original_form": rec.get("original_form"),
                "partner_form": rec.get("partner_form"),
                "reason": reason,
            }
        )

    embed_records = common.compute_delta_records(
        records,
        get_embedding,
        cache_path=cache_path,
        use_cache=use_cache,
        desc=f"Extracting embeddings ({model_name})",
        get_tokenization=lambda s, w: common.hf_target_tokens(tokenizer, s, w),
        on_skip=log_skip,
    )

    if not embed_records:
        print("No embeddings produced; skipping plots.")
        return

    common.write_visualizations(embed_records, model_name=model_name, slug=slug, docs_dir="docs")
    common.write_tokens_per_word_summary(embed_records, model_slug=slug, docs_dir="docs")
    common.write_skipped_html(skipped_rows, model_slug=slug, docs_dir="docs")
    common.update_docs_indexes("docs")
    print("Updated docs indexes.")

    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} records; see docs/{slug}_skipped.html")


if __name__ == "__main__":
    main()
