# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "scikit-learn>=1.0.0",
#   "protobuf>=4.0.0",
#   "torch>=2.0.0",
#   "tqdm>=4.0.0",
#   "transformers>=4.48.0",
#   "numba>=0.59.0",
#   "llvmlite>=0.42.0",
#   "umap-learn",
# ]
# ///

"""Run the experiment with bert-base-uncased.

Outputs: docs/bert_base_uncased_{PCA|TSNE|UMAP}_{2D|3D}.html.

Also removes the legacy docs/modernbert_*.html files to prevent confusion.
"""

import os

import overabundance_common as common


def main() -> None:
    args = common.parse_runner_args()

    common.setup_environment()

    if args.deps_only:
        common.deps_only_report()
        return

    model_name = "bert-base-uncased"
    # Use the project’s typical underscore naming convention for filenames.
    slug = "bert_base_uncased"
    cache_path, use_cache = common.setup_cache(model_name=model_name, cache_slug=slug)
    records = common.load_records(max_records=args.max_records)

    if use_cache:
        cache = common.load_cache(cache_path)
        missing = [r for r in records if r.get("cache_key") not in cache or cache[r.get("cache_key")].get("delta") is None]
        if not missing:
            embed_records = [cache[r["cache_key"]] for r in records]
            common.remove_legacy_modernbert_docs("docs")
            common.write_visualizations(embed_records, model_name=model_name, slug=slug, docs_dir="docs")
            common.write_tokens_per_word_summary(embed_records, model_slug=slug, docs_dir="docs")
            common.update_docs_indexes("docs")
            print("Updated docs indexes (cache-only run).")
            return

    from transformers import AutoModel, AutoTokenizer

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def get_embedding(sentence: str, word: str):
        return common.hf_get_embedding(tokenizer, model, sentence, word)

    embed_records = common.compute_delta_records(
        records,
        get_embedding,
        cache_path=cache_path,
        use_cache=use_cache,
        desc=f"Extracting embeddings ({model_name})",
        get_tokenization=lambda s, w: common.hf_target_tokens(tokenizer, s, w),
    )

    if not embed_records:
        print("No embeddings produced; skipping plots.")
        return

    common.remove_legacy_modernbert_docs("docs")

    common.write_visualizations(embed_records, model_name=model_name, slug=slug, docs_dir="docs")
    common.write_tokens_per_word_summary(embed_records, model_slug=slug, docs_dir="docs")
    common.update_docs_indexes("docs")
    print("Updated docs indexes.")


if __name__ == "__main__":
    main()
