"""Shared utilities for the overabundance embedding-delta experiment.

This file deliberately avoids importing model-specific libraries (transformers).
The model runner scripts supply a `get_embedding(sentence, word) -> np.ndarray | None`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def setup_environment() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_runner_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only verify dependencies import; do not download models or run the experiment.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="If set, only process the first N records (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--embedding-source",
        choices=["delta", "delta_from_raw", "head", "head_delta", "head_delta_from_raw", "orig", "art", "orig_head", "art_head"],
        default="delta",
        help="Embedding source for downstream visualization and analysis (delta, raw contextual, or head-based).",
    )
    parser.add_argument(
        "--head-indices",
        type=str,
        default="",
        help="Comma-separated head indices for --embedding-source=head (default: all heads).",
    )
    return parser.parse_args()


def parse_head_indices(raw: str) -> Optional[List[int]]:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    out: List[int] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(int(piece))
    return out or None


def _version_of(module) -> str:
    return getattr(module, "__version__", "(unknown)")


def deps_only_report(*, extra_imports: Iterable[str] = ()) -> None:
    """Import common dependencies and print versions.

    This is meant for quick reproducibility checks and debugging on a new machine.
    """

    core = [
        "numpy",
        "pandas",
        "plotly",
        "sklearn",
        "torch",
        "transformers",
        "umap",
    ]
    modules = core + list(extra_imports)

    loaded = {}
    for name in modules:
        loaded[name] = importlib.import_module(name)

    print("Dependency check OK")
    print(f"python: {sys.version.split()[0]}")
    for name, module in loaded.items():
        # print friendly names for a couple of packages
        label = "scikit-learn" if name == "sklearn" else ("umap-learn" if name == "umap" else name)
        print(f"{label}: {_version_of(module)}")


def load_records(*, max_records: int = 0, tsv_path: str = "flexemes.tsv") -> List[Dict[str, Any]]:
    df = load_flexemes_tsv(tsv_path)
    meaning_cols = ["meaning_index", "Meaning index", "meaning", "Meaning"]
    if not any(col in df.columns for col in meaning_cols):
        print("Warning: TSV missing meaning columns; meaning colors/tooltips will be empty.")
    else:
        present = [col for col in meaning_cols if col in df.columns]
        if present and df[present].isna().all().all():
            print("Warning: TSV has meaning columns but all values are empty.")
    records = build_records(df)
    if max_records and max_records > 0:
        return records[:max_records]
    return records


def setup_cache(*, model_name: str, cache_slug: str) -> tuple[str, bool]:
    cache_path = f"head_embed_cache_{cache_slug}.jsonl"
    maybe_move_legacy_cache(f"delta_cache_{cache_slug}.jsonl", cache_path)
    maybe_move_legacy_cache("delta_cache.jsonl", cache_path)
    maybe_move_legacy_cache("head_embed_cache.jsonl", cache_path)
    use_cache = prompt_cache_choice(cache_path, model_name)
    return cache_path, use_cache


def find_sublist(haystack: List[str], needle: List[str]) -> Optional[int]:
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _hf_locate_target_in_encoding(tokenizer, encoding_tokens: List[str], word: str) -> Optional[tuple[int, List[str]]]:
    # Some tokenizers treat different whitespace characters differently (notably
    # TransfoXL with NBSP in this dataset). Try a few common “word-start”
    # variants so we can robustly locate the target span.
    prefixes = ["", " ", "\u00a0", "\n", "\t"]

    candidate_token_seqs: List[List[str]] = []
    for prefix in prefixes:
        try:
            toks = tokenizer.tokenize(prefix + word)
        except Exception:
            continue
        if toks:
            candidate_token_seqs.append(toks)

    seen: set[tuple[str, ...]] = set()
    for word_tokens in candidate_token_seqs:
        key = tuple(word_tokens)
        if key in seen:
            continue
        seen.add(key)

        word_start = find_sublist(encoding_tokens, word_tokens)
        if word_start is not None:
            return word_start, encoding_tokens[word_start : word_start + len(word_tokens)]
    return None


def _hf_locate_target(tokenizer, sentence: str, word: str) -> Optional[tuple[int, List[str]]]:
    """Locate `word` token sequence inside tokenized `sentence`.

    Returns (start_index, tokens_slice) where tokens_slice are the *sentence*
    tokens corresponding to the target word.
    """

    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    encoding_tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)

    return _hf_locate_target_in_encoding(tokenizer, encoding_tokens, word)


def hf_get_embedding(tokenizer, model, sentence: str, word: str):
    """Extract a contextual embedding for `word` within `sentence`.

    Works for typical HF encoder/decoder models that return `last_hidden_state`.
    Returns a NumPy vector or None if the target word tokens can’t be located.
    """

    import torch

    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    encoding_tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)

    located = _hf_locate_target_in_encoding(tokenizer, encoding_tokens, word)
    if located is None:
        return None
    word_start, word_tokens = located

    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        hidden_states = hidden[0]

    word_embeds = hidden_states[word_start : word_start + len(word_tokens)].mean(dim=0)
    return word_embeds.float().cpu().numpy()


def hf_get_embedding_with_heads(tokenizer, model, sentence: str, word: str):
    """Extract both full contextual embedding and per-head slices.

    Returns (embedding, heads) where embedding is a NumPy vector and heads is a
    list of NumPy vectors (or None if head slicing is unavailable).
    """

    import torch

    tokens = tokenizer(sentence, return_tensors="pt")
    input_ids = tokens["input_ids"][0]
    encoding_tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)

    located = _hf_locate_target_in_encoding(tokenizer, encoding_tokens, word)
    if located is None:
        return None
    word_start, word_tokens = located

    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]
        hidden_states = hidden[0]

    word_embeds = hidden_states[word_start : word_start + len(word_tokens)].mean(dim=0)
    embedding = word_embeds.float().cpu().numpy()

    n_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
    if not isinstance(n_heads, int) or n_heads <= 0:
        return embedding, None

    hidden_dim = int(word_embeds.shape[0])
    if hidden_dim % n_heads != 0:
        return embedding, None

    head_dim = hidden_dim // n_heads
    heads = word_embeds.reshape(n_heads, head_dim)
    return embedding, [heads[i].float().cpu().numpy() for i in range(n_heads)]


def hf_get_embedding_heads(tokenizer, model, sentence: str, word: str) -> Optional[List[object]]:
    """Extract per-head target embeddings by slicing hidden dimensions.

    This uses equal-size hidden-dimension slices based on num_attention_heads.
    """

    out = hf_get_embedding_with_heads(tokenizer, model, sentence, word)
    if out is None:
        return None
    _, heads = out
    return heads


def _as_np_vector(x):
    import numpy as np

    if not isinstance(x, list):
        return None
    return np.asarray(x, dtype=float)


def _as_np_heads(x):
    import numpy as np

    if not isinstance(x, list) or not x:
        return None
    heads = []
    for h in x:
        if not isinstance(h, list):
            return None
        heads.append(np.asarray(h, dtype=float))
    return heads


def select_record_embedding(
    rec: Dict[str, Any],
    *,
    embedding_source: str = "delta",
    head_indices: Optional[List[int]] = None,
):
    import numpy as np

    orig_heads = _as_np_heads(rec.get("orig_head_embeddings"))
    art_heads = _as_np_heads(rec.get("art_head_embeddings"))

    def _concat_heads(heads):
        if heads is None or not heads:
            return None
        return np.concatenate(heads, axis=0)

    def _valid_indices(n: int) -> List[int]:
        return list(range(n)) if head_indices is None else [i for i in head_indices if 0 <= i < n]

    if embedding_source == "orig":
        return _concat_heads(orig_heads)
    if embedding_source == "art":
        return _concat_heads(art_heads)

    if embedding_source in {"delta", "delta_from_raw"}:
        orig = _concat_heads(orig_heads)
        art = _concat_heads(art_heads)
        if orig is None or art is None:
            return None
        return orig - art

    if embedding_source in {"orig_head", "art_head"}:
        src_heads = orig_heads if embedding_source == "orig_head" else art_heads
        if src_heads is None:
            return None
        idxs = _valid_indices(len(src_heads))
        if not idxs:
            return None
        return np.mean([src_heads[i] for i in idxs], axis=0)

    if embedding_source in {"head", "head_delta", "head_delta_from_raw"}:
        if orig_heads is None or art_heads is None or len(orig_heads) != len(art_heads):
            return None
        idxs = _valid_indices(len(orig_heads))
        if not idxs:
            return None
        deltas = [orig_heads[i] - art_heads[i] for i in idxs]
        return np.mean(deltas, axis=0)

    return None


def cache_has_contextual_fields(cached: Dict[str, Any]) -> bool:
    """Whether cache entry contains raw per-head contextual embeddings."""

    def _is_heads(x):
        return isinstance(x, list) and len(x) > 0 and all(isinstance(h, list) and len(h) > 0 for h in x)

    return _is_heads(cached.get("orig_head_embeddings")) and _is_heads(cached.get("art_head_embeddings"))


def hf_target_tokens(tokenizer, sentence: str, word: str) -> Optional[List[str]]:
    located = _hf_locate_target(tokenizer, sentence, word)
    if located is None:
        return None
    _, tokens_slice = located
    return tokens_slice


def hf_count_target_tokens(tokenizer, sentence: str, word: str) -> Optional[int]:
    """Count how many model tokens correspond to `word` in `sentence`.

    Returns None if the word token sequence can’t be located.
    """

    tokens_slice = hf_target_tokens(tokenizer, sentence, word)
    if tokens_slice is None:
        return None
    return len(tokens_slice)


def patch_transfoxl_torch_type_as(torch) -> None:
    """Work around a bug in some transformers builds for deprecated TransfoXL.

    The model calls `tensor.type_as(dtype=...)`, but `torch.Tensor.type_as` expects
    a positional tensor `other`.
    """

    if getattr(torch, "_overabundance_type_as_patched", False):
        return

    _orig_type_as = torch.Tensor.type_as

    def _patched_type_as(self, other=None, *, dtype=None):
        if other is None and dtype is not None:
            return self.to(dtype=dtype)
        return _orig_type_as(self, other)

    torch.Tensor.type_as = _patched_type_as
    torch._overabundance_type_as_patched = True


def load_flexemes_tsv(path: str = "flexemes.tsv"):
    import pandas as pd

    return pd.read_csv(path, sep="\t")


def extract_bold(text: str) -> Optional[str]:
    match = re.search(r"<b>(.*?)</b>", text)
    return match.group(1) if match else None


def build_records(df) -> List[Dict[str, Any]]:
    import pandas as pd

    lexeme_mps_forms = df.groupby(["lexeme", "mps"])["form"].unique().to_dict()

    def _optional(row, *keys):
        for key in keys:
            if key in row.index:
                val = row.get(key)
                if pd.notna(val):
                    return val
        return None

    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        lexeme = row["lexeme"]
        mps = row["mps"]
        original_form = row["form"]
        text = row["text"]

        bold_word = extract_bold(text)
        if bold_word != original_form:
            continue

        key = (lexeme, mps)
        partner_forms = [f for f in lexeme_mps_forms.get(key, []) if f != original_form]
        if not partner_forms:
            continue
        partner_form = partner_forms[0]

        orig_sentence = re.sub(r"<b>(.*?)</b>", r"\1", text)
        artificial_sentence = re.sub(r"<b>(.*?)</b>", partner_form, text)
        artificial_sentence = re.sub(r"<b>|</b>", "", artificial_sentence)

        records.append(
            {
                "ID": row["ID"],
                # IMPORTANT: The dataset's ID is NOT unique (it repeats across many
                # rows), so we must not use it as the cache key.
                # Using the TSV row index is stable and unique for caching.
                "cache_key": f"row:{int(idx)}",
                "lexeme": lexeme,
                "mps": mps,
                "meaning_index": _optional(row, "meaning_index", "Meaning index", "meaning index"),
                "meaning": _optional(row, "meaning", "Meaning"),
                "original_form": original_form,
                "partner_form": partner_form,
                "orig_sentence": orig_sentence,
                "artificial_sentence": artificial_sentence,
                "pair_label": f"{original_form}->{partner_form} ({mps})",
                "target_word": original_form,
            }
        )

    return records


def maybe_move_legacy_cache(legacy_path: str, cache_path: str) -> None:
    if os.path.exists(legacy_path) and not os.path.exists(cache_path):
        import shutil

        shutil.move(legacy_path, cache_path)
        print(f"Moved old cache to {cache_path}")


def prompt_cache_choice(cache_path: str, model_name: str) -> bool:
    if not os.path.exists(cache_path):
        return False
    resp = input(f"Cache file '{cache_path}' exists for {model_name}. Use cache? (y/n/d=delete): ").strip().lower()
    if resp == "y":
        return True
    if resp == "d":
        os.remove(cache_path)
        print("Cache deleted.")
        return False
    print("Ignoring cache.")
    return False


def load_cache(cache_path: str) -> Dict[Any, Dict[str, Any]]:
    import json

    cache: Dict[Any, Dict[str, Any]] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path) as f:
        for line in f:
            item = json.loads(line)
            cache_key = item.get("cache_key")
            if cache_key is None:
                # Legacy caches used ID as the key.
                cache_key = item.get("ID")
            cache[cache_key] = item
    return cache


def merge_cached_records(
    records: List[Dict[str, Any]],
    cache: Dict[Any, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return cached records with current metadata merged in."""

    merged: List[Dict[str, Any]] = []
    for rec in records:
        cache_key = rec.get("cache_key", rec.get("ID"))
        cached = cache.get(cache_key)
        if cached is None:
            continue
        for k, v in rec.items():
            if cached.get(k) is None:
                cached[k] = v
        merged.append(cached)
    return merged


def compute_delta_records(
    records: List[Dict[str, Any]],
    get_embedding: Callable[[str, str], "Optional[object]"],
    *,
    cache_path: str,
    use_cache: bool,
    desc: str,
    get_token_count: Optional[Callable[[str, str], Optional[int]]] = None,
    get_tokenization: Optional[Callable[[str, str], Optional[List[str]]]] = None,
    get_head_embeddings: Optional[Callable[[str, str], Optional[List[object]]]] = None,
    on_skip: Optional[Callable[[Dict[str, Any], str], None]] = None,
) -> List[Dict[str, Any]]:
    import json

    from tqdm import tqdm

    cache: Dict[Any, Dict[str, Any]] = load_cache(cache_path) if use_cache else {}
    cache_out = None
    cache_dirty = False

    out: List[Dict[str, Any]] = []
    for rec in tqdm(records, desc=desc):
        cache_key = rec.get("cache_key", rec.get("ID"))
        if use_cache and cache_key in cache:
            cached = cache[cache_key]
            before_cached = dict(cached)
            # Ensure cached records include the current record metadata.
            # This makes plots robust even if the cache schema changes.
            for k, v in rec.items():
                if cached.get(k) is None:
                    cached[k] = v
            if get_tokenization is not None:
                if cached.get("orig_tokens") is None:
                    cached["orig_tokens"] = get_tokenization(rec["orig_sentence"], rec["original_form"])
                if cached.get("art_tokens") is None:
                    cached["art_tokens"] = get_tokenization(rec["artificial_sentence"], rec["partner_form"])
                if cached.get("orig_token_count") is None and cached.get("orig_tokens") is not None:
                    cached["orig_token_count"] = len(cached["orig_tokens"])
                if cached.get("art_token_count") is None and cached.get("art_tokens") is not None:
                    cached["art_token_count"] = len(cached["art_tokens"])

            if get_tokenization is None and get_token_count is not None:
                if cached.get("orig_token_count") is None:
                    cached["orig_token_count"] = get_token_count(rec["orig_sentence"], rec["original_form"])
                if cached.get("art_token_count") is None:
                    cached["art_token_count"] = get_token_count(rec["artificial_sentence"], rec["partner_form"])

            need_heads = (
                get_head_embeddings is not None
                and (
                    cached.get("orig_head_embeddings") is None
                    or cached.get("art_head_embeddings") is None
                )
            )
            if need_heads:
                try:
                    orig_heads = get_head_embeddings(rec["orig_sentence"], rec["original_form"])
                    art_heads = get_head_embeddings(rec["artificial_sentence"], rec["partner_form"])
                except Exception:
                    orig_heads = None
                    art_heads = None
                if isinstance(orig_heads, list) and isinstance(art_heads, list) and len(orig_heads) == len(art_heads):
                    cached["orig_head_embeddings"] = [h.tolist() for h in orig_heads]
                    cached["art_head_embeddings"] = [h.tolist() for h in art_heads]

            # Compact legacy cache entries to the head-only schema.
            for obsolete_key in ["orig_embedding", "art_embedding", "delta", "head_deltas"]:
                if obsolete_key in cached:
                    cached.pop(obsolete_key, None)

            if cached != before_cached:
                cache_dirty = True

            out.append(cached)
            continue

        out_rec = {**rec}
        if get_tokenization is not None:
            out_rec["orig_tokens"] = get_tokenization(rec["orig_sentence"], rec["original_form"])
            out_rec["art_tokens"] = get_tokenization(rec["artificial_sentence"], rec["partner_form"])
            out_rec["orig_token_count"] = None if out_rec["orig_tokens"] is None else len(out_rec["orig_tokens"])
            out_rec["art_token_count"] = None if out_rec["art_tokens"] is None else len(out_rec["art_tokens"])

        if get_tokenization is None and get_token_count is not None:
            out_rec["orig_token_count"] = get_token_count(rec["orig_sentence"], rec["original_form"])
            out_rec["art_token_count"] = get_token_count(rec["artificial_sentence"], rec["partner_form"])

        if get_head_embeddings is None:
            if on_skip is not None:
                on_skip(rec, "head_embeddings_unavailable")
            continue

        try:
            orig_heads = get_head_embeddings(rec["orig_sentence"], rec["original_form"])
            art_heads = get_head_embeddings(rec["artificial_sentence"], rec["partner_form"])
        except Exception as e:
            if on_skip is not None:
                on_skip(rec, f"head_exception:{type(e).__name__}:{e}")
            continue

        if not isinstance(orig_heads, list) or not isinstance(art_heads, list):
            if on_skip is not None:
                on_skip(rec, "head_none")
            continue

        if len(orig_heads) != len(art_heads):
            if on_skip is not None:
                on_skip(rec, "head_length_mismatch")
            continue

        out_rec["orig_head_embeddings"] = [h.tolist() for h in orig_heads]
        out_rec["art_head_embeddings"] = [h.tolist() for h in art_heads]

        out.append(out_rec)

        if cache_out is None:
            cache_out = open(cache_path, "a", encoding="utf-8")
        cache_out.write(json.dumps(out_rec) + "\n")

    if cache_out:
        cache_out.close()

    if use_cache and cache_dirty:
        import json

        with open(cache_path, "w", encoding="utf-8") as f:
            for item in cache.values():
                f.write(json.dumps(item) + "\n")

    return out


def write_tokens_per_word_summary(
    embed_records: List[Dict[str, Any]],
    *,
    model_slug: str,
    docs_dir: str = "docs",
) -> str:
    """Write docs/{model_slug}_tokens-per-word.html.

    Produces two tables:
    1) Summary stats across word types (mean/median/mode tokens-per-word).
    for lexeme, recs in sorted(lexeme_to_records.items(), key=lambda kv: kv[0]):
        if len(recs) < 2:
            continue

    Note: Within a model, a word should generally be tokenized consistently.
    If multiple tokenizations are observed (e.g., whitespace quirks), we report
    the most frequent one and include a small "variants" count.
    """

    import statistics
    from collections import Counter, defaultdict

    import pandas as pd

    os.makedirs(docs_dir, exist_ok=True)

    legacy_tsv = os.path.join(docs_dir, f"{model_slug}_tokens-per-word.tsv")
    if os.path.exists(legacy_tsv):
        os.remove(legacy_tsv)

    counts_by_word: Dict[str, List[int]] = defaultdict(list)
    toks_by_word: Dict[str, List[str]] = defaultdict(list)

    def _tok_str(tokens) -> Optional[str]:
        if isinstance(tokens, list) and all(isinstance(x, str) for x in tokens) and tokens:
            return "-".join(tokens)
        return None

    for rec in embed_records:
        w1 = rec.get("original_form")
        c1 = rec.get("orig_token_count")
        t1 = _tok_str(rec.get("orig_tokens"))
        if isinstance(w1, str) and isinstance(c1, int):
            counts_by_word[w1].append(c1)
        if isinstance(w1, str) and t1:
            toks_by_word[w1].append(t1)

        w2 = rec.get("partner_form")
        c2 = rec.get("art_token_count")
        t2 = _tok_str(rec.get("art_tokens"))
        if isinstance(w2, str) and isinstance(c2, int):
            counts_by_word[w2].append(c2)
        if isinstance(w2, str) and t2:
            toks_by_word[w2].append(t2)

    per_word_rows: List[Dict[str, Any]] = []
    per_word_token_counts: List[int] = []

    for word in sorted(counts_by_word.keys()):
        vals = counts_by_word[word]
        if not vals:
            continue

        freq = Counter(vals)
        max_count = max(freq.values())
        modes = sorted([k for k, v in freq.items() if v == max_count])
        token_count_mode = modes[0]

        tok_vals = toks_by_word.get(word, [])
        tok_mode = ""
        if tok_vals:
            tok_freq = Counter(tok_vals)
            tok_max = max(tok_freq.values())
            tok_modes = sorted([k for k, v in tok_freq.items() if v == tok_max])
            tok_mode = tok_modes[0]

        per_word_rows.append(
            {
                "word": word,
                "tokenization": tok_mode,
                "token_count": token_count_mode,
                "n_occurrences": len(vals),
                "token_count_variants": len(set(vals)),
                "tokenization_variants": len(set(tok_vals)) if tok_vals else 0,
            }
        )
        per_word_token_counts.append(token_count_mode)

    summary_rows: List[Dict[str, Any]] = []
    if per_word_token_counts:
        mean_val = sum(per_word_token_counts) / len(per_word_token_counts)
        median_val = statistics.median(per_word_token_counts)
        freq = Counter(per_word_token_counts)
        max_count = max(freq.values())
        modes = sorted([k for k, v in freq.items() if v == max_count])
        mode_val = modes[0]
        summary_rows.append(
            {
                "n_word_types": len(per_word_token_counts),
                "mean_tokens_per_word": f"{mean_val:.4f}",
                "median_tokens_per_word": f"{float(median_val):.4f}",
                "mode_tokens_per_word": mode_val,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    per_word_df = pd.DataFrame(per_word_rows)
    if not per_word_df.empty:
        per_word_df = per_word_df.sort_values(["token_count", "word"], ascending=[False, True])

    out_path = os.path.join(docs_dir, f"{model_slug}_tokens-per-word.html")

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      table { border-collapse: collapse; margin: 12px 0 24px 0; }
      th, td { border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }
      th { background: #f6f6f6; text-align: left; }
      code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    </style>
    """

    summary_table_html = (
        "<p><em>No token-count data available.</em></p>"
        if summary_df.empty
        else summary_df.to_html(index=False, escape=False)
    )

    # Render tokenization in <code> for readability.
    if not per_word_df.empty and "tokenization" in per_word_df.columns:
        per_word_df = per_word_df.copy()
        per_word_df["tokenization"] = per_word_df["tokenization"].apply(
            lambda s: "" if not isinstance(s, str) or not s else f"<code>{s}</code>"
        )

    per_word_table_html = (
        "<p><em>No per-word rows available.</em></p>" if per_word_df.empty else per_word_df.to_html(index=False, escape=False)
    )

    html = "\n".join(
        [
            "<html>",
            "<head>",
            f"<title>{model_slug} tokens per word</title>",
            css,
            "</head>",
            "<body>",
            f"<h1>{model_slug}: tokens per word</h1>",
            "<h2>Summary</h2>",
            summary_table_html,
            "<h2>Per word</h2>",
            per_word_table_html,
            "</body>",
            "</html>",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved {out_path}")
    return out_path


def write_skipped_html(
    skipped_rows: List[Dict[str, Any]],
    *,
    model_slug: str,
    docs_dir: str = "docs",
) -> str:
    """Write docs/{model_slug}_skipped.html (table of skipped records)."""

    import pandas as pd

    os.makedirs(docs_dir, exist_ok=True)

    legacy_tsv = os.path.join(docs_dir, f"{model_slug}_skipped.tsv")
    if os.path.exists(legacy_tsv):
        os.remove(legacy_tsv)

    out_path = os.path.join(docs_dir, f"{model_slug}_skipped.html")

    df = pd.DataFrame(skipped_rows)
    if not df.empty:
        # Keep a stable, useful column order when present.
        preferred = [
            "cache_key",
            "ID",
            "lexeme",
            "mps",
            "original_form",
            "partner_form",
            "reason",
        ]
        cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df = df[cols]

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      table { border-collapse: collapse; margin: 12px 0 24px 0; }
      th, td { border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }
      th { background: #f6f6f6; text-align: left; }
    </style>
    """

    table_html = "<p><em>No skipped records.</em></p>" if df.empty else df.to_html(index=False, escape=True)

    html = "\n".join(
        [
            "<html>",
            "<head>",
            f"<title>{model_slug} skipped</title>",
            css,
            "</head>",
            "<body>",
            f"<h1>{model_slug}: skipped records</h1>",
            table_html,
            "</body>",
            "</html>",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved {out_path}")
    return out_path


def write_visualizations(
    embed_records: List[Dict[str, Any]],
    *,
    model_name: str,
    slug: str,
    docs_dir: str = "docs",
    filename_prefix: Optional[str] = None,
    reducers_to_run: Optional[List[str]] = None,
    color_by: Optional[str] = None,
    embedding_source: str = "delta",
    head_indices: Optional[List[int]] = None,
) -> List[str]:
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    import textwrap

    import umap

    os.makedirs(docs_dir, exist_ok=True)

    def _embedding_file_tag(source: str, heads: Optional[List[int]]) -> str:
        """Return a stable filename-safe embedding-source tag."""

        if source in {"delta", "orig", "art"}:
            return source
        if heads:
            return f"{source}_{'-'.join(str(i) for i in heads)}"
        return source

    filtered_records: List[Dict[str, Any]] = []
    vectors = []
    for rec in embed_records:
        vec = select_record_embedding(rec, embedding_source=embedding_source, head_indices=head_indices)
        if vec is None:
            continue
        filtered_records.append(rec)
        vectors.append(vec)

    if not filtered_records:
        print(f"No records available for embedding source '{embedding_source}'; skipping plots.")
        return []

    plot_df = pd.DataFrame(filtered_records)

    def _tok_str(tokens):
        if isinstance(tokens, list) and all(isinstance(x, str) for x in tokens) and tokens:
            return "-".join(tokens)
        return None

    if "orig_tokens" in plot_df.columns:
        plot_df["orig_tokenization"] = plot_df["orig_tokens"].apply(_tok_str)
    else:
        plot_df["orig_tokenization"] = None

    if "art_tokens" in plot_df.columns:
        plot_df["art_tokenization"] = plot_df["art_tokens"].apply(_tok_str)
    else:
        plot_df["art_tokenization"] = None

    if "meaning_index" not in plot_df.columns:
        plot_df["meaning_index"] = None
    if "meaning" not in plot_df.columns:
        plot_df["meaning"] = None

    def _meaning_index_label(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, float) and np.isnan(value):
            return None
        return str(value)

    plot_df["meaning_index_label"] = plot_df["meaning_index"].apply(_meaning_index_label)

    def _word_summary(word, tokenization, token_count) -> str:
        w = "" if word is None else str(word)
        tok = "" if tokenization is None else str(tokenization)
        cnt = "" if token_count is None else str(token_count)

        if tok and cnt:
            return f"{w} – {tok} ({cnt})"
        if tok:
            return f"{w} – {tok}"
        if cnt:
            return f"{w} ({cnt})"
        return w

    def _mark_target(sentence: str, target: str) -> str:
        if not isinstance(sentence, str) or not isinstance(target, str) or not target:
            return "" if sentence is None else str(sentence)

        # Prefer a whole-word match; fall back to first substring occurrence.
        pattern = re.compile(r"\\b" + re.escape(target) + r"\\b")
        marked, n = pattern.subn(f"<<<{target}>>>", sentence, count=1)
        if n:
            return marked
        if target in sentence:
            return sentence.replace(target, f"<<<{target}>>>", 1)
        return sentence

    def _wrap_for_hover(text: str, *, width: int = 88) -> str:
        if text is None:
            return ""
        s = str(text).replace("\r\n", "\n").replace("\r", "\n")
        lines: List[str] = []
        for line in s.split("\n"):
            if not line:
                lines.append("")
                continue
            lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
        return "<br>".join(lines)

    plot_df["original_form_hover"] = plot_df.apply(
        lambda r: _word_summary(r.get("original_form"), r.get("orig_tokenization"), r.get("orig_token_count")),
        axis=1,
    )
    plot_df["partner_form_hover"] = plot_df.apply(
        lambda r: _word_summary(r.get("partner_form"), r.get("art_tokenization"), r.get("art_token_count")),
        axis=1,
    )

    plot_df["orig_sentence_hover"] = plot_df.apply(
        lambda r: _wrap_for_hover(_mark_target(r.get("orig_sentence"), r.get("original_form"))),
        axis=1,
    )
    plot_df["artificial_sentence_hover"] = plot_df.apply(
        lambda r: _wrap_for_hover(_mark_target(r.get("artificial_sentence"), r.get("partner_form"))),
        axis=1,
    )
    plot_df["meaning_hover"] = plot_df["meaning"].apply(_wrap_for_hover)

    def _pair_label_tokenized(row):
        left_tok = row.get("orig_tokenization")
        right_tok = row.get("art_tokenization")
        left = left_tok if isinstance(left_tok, str) and left_tok else row.get("original_form")
        right = right_tok if isinstance(right_tok, str) and right_tok else row.get("partner_form")
        mps = row.get("mps")
        if mps is None:
            return f"{left}->{right}"
        return f"{left}->{right} ({mps})"

    form_order_map: Dict[tuple[str, str], Dict[str, int]] = {}
    for _, row in plot_df.iterrows():
        lexeme = row.get("lexeme")
        mps = row.get("mps")
        if not isinstance(lexeme, str) or not lexeme or not isinstance(mps, str) or not mps:
            continue
        forms = {
            form
            for form in (row.get("original_form"), row.get("partner_form"))
            if isinstance(form, str) and form
        }
        if not forms:
            continue
        key = (lexeme, mps)
        merged = set(form_order_map.get(key, {}).keys()) | forms
        ordered = sorted(merged, key=lambda form: (form.casefold(), form))
        form_order_map[key] = {form: idx for idx, form in enumerate(ordered)}

    lexeme_rank_map: Dict[str, Dict[str, int]] = {}
    for mps in sorted({str(v) for v in plot_df["mps"].dropna().astype(str).tolist()}):
        lexemes = sorted(
            {
                str(v)
                for v in plot_df.loc[plot_df["mps"].astype(str) == mps, "lexeme"].dropna().astype(str).tolist()
                if v
            }
        )
        lexeme_rank_map[mps] = {lexeme: idx for idx, lexeme in enumerate(lexemes)}

    def _open_symbol(base_symbol: str, n_dim: int) -> str:
        if n_dim == 3 and base_symbol == "cross":
            return "x"
        return f"{base_symbol}-open"

    def _orig_marker_symbol(row, n_dim: int) -> str:
        lexeme = row.get("lexeme")
        mps = row.get("mps")
        orig = row.get("original_form")
        if (
            not isinstance(lexeme, str)
            or not lexeme
            or not isinstance(mps, str)
            or not mps
            or not isinstance(orig, str)
            or not orig
        ):
            return "circle"

        if n_dim == 2:
            family_pairs = {
                "pst": [("circle", "square"), ("diamond", "triangle-up")],
                "ppt": [("cross", "x"), ("star", "hexagram")],
            }
        else:
            family_pairs = {
                "pst": [("circle", "square")],
                "ppt": [("diamond", "cross")],
            }

        pairs = family_pairs.get(mps, family_pairs["pst"])
        lexeme_rank = lexeme_rank_map.get(mps, {}).get(lexeme, 0)
        pair = pairs[lexeme_rank % len(pairs)]
        form_idx = form_order_map.get((lexeme, mps), {}).get(orig, 0)
        return pair[min(form_idx, len(pair) - 1)]

    def _delta_marker_symbol(row, n_dim: int) -> str:
        lexeme = row.get("lexeme")
        mps = row.get("mps")
        orig = row.get("original_form")
        art = row.get("partner_form")
        if (
            not isinstance(lexeme, str)
            or not lexeme
            or not isinstance(mps, str)
            or not mps
            or not isinstance(orig, str)
            or not orig
            or not isinstance(art, str)
            or not art
        ):
            return "circle"

        if n_dim == 2:
            family = {
                "pst": ["circle", "square", "diamond", "triangle-up"],
                "ppt": ["cross", "x", "star", "hexagram"],
            }
        else:
            family = {
                "pst": ["circle", "square"],
                "ppt": ["diamond", "cross"],
            }

        symbols = family.get(mps, family["pst"])
        lexeme_rank = lexeme_rank_map.get(mps, {}).get(lexeme, 0)
        base_symbol = symbols[lexeme_rank % len(symbols)]

        form_order = form_order_map.get((lexeme, mps), {})
        orig_idx = form_order.get(orig, 0)
        art_idx = form_order.get(art, 1)
        is_reverse = orig_idx > art_idx
        if not is_reverse:
            return base_symbol

        return _open_symbol(base_symbol, n_dim)

    def _marker_symbol_for_row(row, source: str, n_dim: int) -> str:
        if source in {"delta", "delta_from_raw", "head", "head_delta", "head_delta_from_raw"}:
            return _delta_marker_symbol(row, n_dim)
        return _orig_marker_symbol(row, n_dim)

    # Override pair_label in the plotting DF so hover shows tokenization via hyphens.
    plot_df["pair_label"] = plot_df.apply(_pair_label_tokenized, axis=1)
    deltas = np.stack(vectors)
    n_samples = int(deltas.shape[0])

    reducers = {
        "PCA": PCA,
        "TSNE": TSNE,
        "UMAP": lambda n, n_neighbors: umap.UMAP(n_components=n, n_neighbors=n_neighbors, random_state=42),
    }

    if reducers_to_run is not None:
        reducers = {k: v for k, v in reducers.items() if k in set(reducers_to_run)}

    written: List[str] = []

    if embedding_source == "orig":
        custom_data_fields = [
            "lexeme",
            "mps",
            "meaning_index",
            "meaning_hover",
            "original_form_hover",
            "orig_sentence_hover",
        ]

        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "mps: %{customdata[1]}<br>"
            "meaning_index: %{customdata[2]}<br>"
            "meaning: %{customdata[3]}<br>"
            "original: %{customdata[4]}<br>"
            "<br>"
            "orig sentence: %{customdata[5]}"
            "<extra></extra>"
        )
    else:
        custom_data_fields = [
            "lexeme",
            "mps",
            "meaning_index",
            "meaning_hover",
            "original_form_hover",
            "partner_form_hover",
            "orig_sentence_hover",
            "artificial_sentence_hover",
        ]

        hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "mps: %{customdata[1]}<br>"
            "meaning_index: %{customdata[2]}<br>"
            "meaning: %{customdata[3]}<br>"
            "original: %{customdata[4]}<br>"
            "partner: %{customdata[5]}<br>"
            "<br>"
            "orig sentence: %{customdata[6]}<br>"
            "art sentence: %{customdata[7]}"
            "<extra></extra>"
        )

    prefix = filename_prefix or slug
    source_tag = _embedding_file_tag(embedding_source, head_indices)

    color_column = color_by or "lexeme"
    if color_column == "meaning_index":
        color_column = "meaning_index_label"
    if color_column not in plot_df.columns or plot_df[color_column].isna().all():
        color_column = "lexeme"

    def _legend_label_for_row(row) -> Optional[str]:
        meaning_id = row.get("meaning_index_label")
        show_meaning_id = color_column == "meaning_index_label" and isinstance(meaning_id, str) and meaning_id
        prefix = f"({meaning_id}) " if show_meaning_id else ""
        mps = row.get("mps")
        mps_suffix = f" ({mps})" if isinstance(mps, str) and mps else ""

        if embedding_source == "orig":
            form = row.get("original_form")
            lemma = row.get("lexeme")
            if not isinstance(form, str) or not form or not isinstance(lemma, str) or not lemma:
                return None
            return f"{prefix}{form} ({lemma}){mps_suffix}"

        if embedding_source in {"delta", "delta_from_raw", "head", "head_delta", "head_delta_from_raw"}:
            orig = row.get("original_form")
            art = row.get("partner_form")
            if not isinstance(orig, str) or not orig or not isinstance(art, str) or not art:
                return None
            return f"{prefix}{orig} -> {art}{mps_suffix}"

        return None

    def _rename_legend_traces(fig, symbol_column: str) -> None:
        trace_label_map: Dict[tuple[str, str], str] = {}
        for _, row in plot_df.iterrows():
            color_value = row.get(color_column)
            marker_symbol = row.get(symbol_column)
            if color_value is None or not isinstance(marker_symbol, str):
                continue
            label = _legend_label_for_row(row)
            if label is None:
                continue
            trace_label_map.setdefault((str(color_value), marker_symbol), label)

        if not trace_label_map:
            return

        seen_labels: set[str] = set()
        for trace in fig.data:
            name = getattr(trace, "name", None)
            if not isinstance(name, str):
                continue
            parts = [part.strip() for part in name.split(",")]
            if len(parts) < 2:
                continue
            key = (parts[0], parts[-1])
            label = trace_label_map.get(key)
            if label is None:
                continue
            trace.name = label
            trace.legendgroup = label
            if label in seen_labels:
                trace.showlegend = False
            else:
                trace.showlegend = True
                seen_labels.add(label)

        fig.update_layout(legend_title_text="")

    for name, reducer in reducers.items():
        for n_dim in [2, 3]:
            if name == "TSNE":
                if n_samples < 3:
                    print(f"Skipping TSNE ({n_dim}D): need at least 3 samples, have {n_samples}.")
                    continue
                perplexity = min(50.0, float(n_samples - 1))
                reduced = reducer(n_components=n_dim, random_state=42, perplexity=perplexity).fit_transform(deltas)
            elif name == "PCA":
                if n_samples < n_dim:
                    print(f"Skipping PCA ({n_dim}D): need at least {n_dim} samples, have {n_samples}.")
                    continue
                reduced = reducer(n_components=n_dim).fit_transform(deltas)
            else:
                if n_samples < 3:
                    print(f"Skipping UMAP ({n_dim}D): need at least 3 samples, have {n_samples}.")
                    continue
                n_neighbors = min(15, n_samples - 1)
                reduced = reducer(n_dim, n_neighbors).fit_transform(deltas)

            for i in range(n_dim):
                plot_df[f"{name.lower()}{i+1}_{n_dim}d"] = reduced[:, i]

            symbol_column = f"marker_symbol_{n_dim}d"
            plot_df[symbol_column] = plot_df.apply(
                lambda row: _marker_symbol_for_row(row, embedding_source, n_dim),
                axis=1,
            )

            unique_symbols = [str(v) for v in plot_df[symbol_column].dropna().astype(str).unique().tolist()]
            symbol_map = {symbol: symbol for symbol in unique_symbols}

            if n_dim == 2:
                fig = px.scatter(
                    plot_df,
                    x=f"{name.lower()}1_{n_dim}d",
                    y=f"{name.lower()}2_{n_dim}d",
                    color=color_column,
                    symbol=symbol_column,
                    custom_data=custom_data_fields,
                    symbol_map=symbol_map,
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x=f"{name.lower()}1_{n_dim}d",
                    y=f"{name.lower()}2_{n_dim}d",
                    z=f"{name.lower()}3_{n_dim}d",
                    color=color_column,
                    symbol=symbol_column,
                    custom_data=custom_data_fields,
                    symbol_map=symbol_map,
                )

            _rename_legend_traces(fig, symbol_column)

            is_head_source = embedding_source in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}
            title_suffix = embedding_source
            if is_head_source and head_indices:
                title_suffix = f"{embedding_source}:{','.join(str(i) for i in head_indices)}"
            fig.update_layout(title=f"Embeddings ({model_name}; {name}, {n_dim}D; {title_suffix})")
            fig.update_layout(hoverlabel=dict(align="left"))
            fig.update_traces(hovertemplate=hovertemplate)
            html_path = os.path.join(docs_dir, f"{prefix}_{source_tag}_{name}_{n_dim}D.html")
            fig.write_html(html_path)
            written.append(html_path)
            print(f"Saved {html_path}")

    return written


def write_lexeme_visualizations(
    embed_records: List[Dict[str, Any]],
    *,
    model_name: str,
    slug: str,
    docs_dir: str = "docs",
    reducers_to_run: Optional[List[str]] = None,
    embedding_source: str = "delta",
    head_indices: Optional[List[int]] = None,
) -> str:
    """Write per-lexeme visualizations under docs/<slug>_lexemes/.

    These visualizations are computed *within* each lexeme subset (i.e., the
    dimensionality reduction is not shared across the full dataset).
    """

    # Default to PCA-only: fast and deterministic per-lexeme.
    if reducers_to_run is None:
        reducers_to_run = ["PCA"]

    out_root = os.path.join(docs_dir, f"{slug}_lexemes")
    os.makedirs(out_root, exist_ok=True)

    lexeme_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for rec in embed_records:
        lex = rec.get("lexeme")
        if not isinstance(lex, str) or not lex:
            continue
        lexeme_to_records.setdefault(lex, []).append(rec)

    for lexeme, recs in sorted(lexeme_to_records.items(), key=lambda kv: kv[0]):
        lex_stub = model_slug(lexeme)
        lex_dir = os.path.join(out_root, lex_stub)
        os.makedirs(lex_dir, exist_ok=True)

        # Use lexeme-specific prefix within the lexeme directory.
        write_visualizations(
            recs,
            model_name=model_name,
            slug=slug,
            docs_dir=lex_dir,
            filename_prefix=lex_stub,
            reducers_to_run=reducers_to_run,
            color_by="meaning_index",
            embedding_source=embedding_source,
            head_indices=head_indices,
        )
        generate_index_html(lex_dir)

    generate_index_html(out_root)
    refresh_docs_indexes_for_path(out_root, docs_dir=docs_dir)
    return out_root


def write_visualization_sets(
    embed_records: List[Dict[str, Any]],
    *,
    model_name: str,
    slug: str,
    docs_dir: str = "docs",
    requested_embedding_source: str = "delta",
    head_indices: Optional[List[int]] = None,
) -> None:
    """Write the standard visualization sets for a run.

    Always writes both delta and orig visualizations. If the requested source is
    different (for example head-based), it is also rendered.
    """

    sources = ["delta", "orig"]
    if requested_embedding_source not in sources:
        sources.append(requested_embedding_source)

    for source in sources:
        use_heads = source in {"head", "head_delta", "head_delta_from_raw", "orig_head", "art_head"}
        source_head_indices = head_indices if use_heads else None
        write_visualizations(
            embed_records,
            model_name=model_name,
            slug=slug,
            docs_dir=docs_dir,
            embedding_source=source,
            head_indices=source_head_indices,
        )
        write_lexeme_visualizations(
            embed_records,
            model_name=model_name,
            slug=slug,
            docs_dir=docs_dir,
            embedding_source=source,
            head_indices=source_head_indices,
        )


def generate_index_html(folder: str) -> None:
    items = sorted(os.listdir(folder))
    html_lines = [
        "<html>",
        "<head><title>Index of {}</title></head>".format(os.path.basename(folder)),
        "<body>",
        "<h1>Index of {}</h1>".format(os.path.basename(folder)),
        "<ul>",
    ]
    for item in items:
        if item.startswith("."):
            continue
        if item == "index.html":
            continue
        path = os.path.join(folder, item)
        display = item + "/" if os.path.isdir(path) else item
        html_lines.append(f'<li><a href="{item}">{display}</a></li>')
    html_lines += ["</ul>", "</body>", "</html>"]
    with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))


def _find_docs_root(path: str) -> Optional[str]:
    cur = os.path.abspath(path)
    if not os.path.isdir(cur):
        cur = os.path.dirname(cur)
    while True:
        if os.path.basename(cur) == "docs":
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def update_docs_indexes(docs_dir: str = "docs") -> None:
    if not os.path.isdir(docs_dir):
        return
    for root, dirs, files in os.walk(docs_dir):
        generate_index_html(root)


def refresh_docs_indexes_for_path(path: str, *, docs_dir: Optional[str] = None) -> Optional[str]:
    if os.path.isdir(path):
        generate_index_html(path)

    docs_root = _find_docs_root(path)
    if docs_root is None and docs_dir:
        candidate = os.path.abspath(docs_dir)
        if os.path.isdir(candidate):
            docs_root = candidate

    if docs_root is not None:
        update_docs_indexes(docs_root)
    return docs_root


def remove_legacy_modernbert_docs(docs_dir: str = "docs") -> None:
    """Remove the old docs/modernbert_*.html files if present."""

    for reducer_name in ["PCA", "TSNE", "UMAP"]:
        for n_dim in [2, 3]:
            legacy_path = os.path.join(docs_dir, f"modernbert_{reducer_name}_{n_dim}D.html")
            if os.path.exists(legacy_path):
                os.remove(legacy_path)
