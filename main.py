def main():
    import pandas as pd
    import re
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    import plotly.express as px
    import plotly.graph_objects as go
    from tqdm import tqdm
    import os, sys, json

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load data
    df = pd.read_csv("flexemes.tsv", sep="\t")

    # Caching setup
    cache_path = "delta_cache.jsonl"
    use_cache = False
    if os.path.exists(cache_path):
        resp = input(f"Cache file '{cache_path}' exists. Use cache? (y/n/d=delete): ").strip().lower()
        if resp == 'y':
            use_cache = True
        elif resp == 'd':
            os.remove(cache_path)
            print("Cache deleted.")
        else:
            print("Ignoring cache.")

    # Find all forms for each lexeme

    # Find all forms for each lexeme
    lexeme_forms = df.groupby('lexeme')['form'].unique().to_dict()

    # Helper to extract <b>word</b> from text
    def extract_bold(text):
        match = re.search(r'<b>(.*?)</b>', text)
        return match.group(1) if match else None

    # Generate original and artificial sentences
    records = []
    for idx, row in df.iterrows():
        lexeme = row['lexeme']
        original_form = row['form']
        text = row['text']
        bold_word = extract_bold(text)
        # Confirm form matches <b>word</b>
        if bold_word != original_form:
            continue
        # Find partner form
        partner_forms = [f for f in lexeme_forms[lexeme] if f != original_form]
        if not partner_forms:
            continue
        partner_form = partner_forms[0]
        # Remove <b> tags for original
        orig_sentence = re.sub(r'<b>(.*?)</b>', r'\1', text)
        # Replace <b>word</b> with partner for artificial
        artificial_sentence = re.sub(r'<b>(.*?)</b>', partner_form, text)
        artificial_sentence = re.sub(r'<b>|</b>', '', artificial_sentence)
        records.append({
            'ID': row['ID'],
            'lexeme': lexeme,
            'original_form': original_form,
            'partner_form': partner_form,
            'orig_sentence': orig_sentence,
            'artificial_sentence': artificial_sentence,
            'pair_label': f"{original_form}->{partner_form}",
            'target_word': original_form
        })

    # Load ModernBERT (using bert-base-uncased as placeholder)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    def get_word_embedding(sentence, word):
        """
        Extract contextual embedding for a target word in a sentence using BERT.
        Steps:
        1. Tokenize the sentence using the model's tokenizer. This splits the sentence into subword tokens.
        2. Tokenize the target word. Note: The word may be split into multiple subword tokens (e.g., 'eaten' -> ['eat', '##en']).
        3. Find the position(s) in the tokenized sentence that match the tokenized target word. This is nontrivial because:
           - The target word may be split into multiple tokens.
           - The sentence may contain multiple instances of the word; we select the first match.
        4. Pass the sentence through the model to get hidden states for each token.
        5. Average the hidden states for the target word's tokens to get a single embedding.
        If the word cannot be found in the tokenized sentence, return None.
        """
        tokens = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        input_ids = tokens['input_ids'][0]
        word_tokens = tokenizer.tokenize(word)
        tokenized = tokenizer.tokenize(sentence)
        # Find all possible start indices for word_tokens in tokenized
        for i in range(len(tokenized) - len(word_tokens) + 1):
            if tokenized[i:i+len(word_tokens)] == word_tokens:
                word_start = i + 1  # +1 for [CLS]
                break
        else:
            return None
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state[0]  # (seq_len, hidden_dim)
        word_embeds = hidden_states[word_start:word_start+len(word_tokens)].mean(dim=0)
        return word_embeds.numpy()

    # Extract embeddings and compute deltas, with caching
    embed_records = []
    cache = {}
    if use_cache:
        with open(cache_path) as f:
            for line in f:
                item = json.loads(line)
                cache[item['ID']] = item
    cache_out = open(cache_path, 'a') if not use_cache or len(cache) < len(records) else None
    for rec in tqdm(records, desc="Extracting embeddings and computing deltas"):
        cache_key = rec['ID']
        if use_cache and cache_key in cache:
            embed_records.append(cache[cache_key])
            continue
        orig_emb = get_word_embedding(rec['orig_sentence'], rec['original_form'])
        art_emb = get_word_embedding(rec['artificial_sentence'], rec['partner_form'])
        if orig_emb is None or art_emb is None:
            continue
        delta = orig_emb - art_emb
        out_rec = {**rec, 'delta': delta.tolist()}
        embed_records.append(out_rec)
        if cache_out:
            cache_out.write(json.dumps(out_rec) + '\n')
    if cache_out:
        cache_out.close()

    # Prepare dataframe
    plot_df = pd.DataFrame(embed_records)
    deltas = np.stack(plot_df['delta'].values)

    # Dimensionality reduction methods
    reducers = {
        'PCA': PCA,
        'TSNE': TSNE,
        'UMAP': lambda n: umap.UMAP(n_components=n, random_state=42)
    }
    dims = [2, 3]
    for name, reducer in reducers.items():
        for n_dim in dims:
            if name == 'TSNE':
                reduced = reducer(n_components=n_dim, random_state=42).fit_transform(deltas)
            elif name == 'PCA':
                reduced = reducer(n_components=n_dim).fit_transform(deltas)
            else:  # UMAP
                reduced = reducer(n_dim).fit_transform(deltas)
            for i in range(n_dim):
                plot_df[f'{name.lower()}{i+1}_{n_dim}d'] = reduced[:,i]
            # Plot
            if n_dim == 2:
                fig = px.scatter(
                    plot_df,
                    x=f'{name.lower()}1_{n_dim}d', y=f'{name.lower()}2_{n_dim}d',
                    color='lexeme',
                    hover_data=['pair_label', 'orig_sentence', 'artificial_sentence']
                )
            else:
                fig = px.scatter_3d(
                    plot_df,
                    x=f'{name.lower()}1_{n_dim}d', y=f'{name.lower()}2_{n_dim}d', z=f'{name.lower()}3_{n_dim}d',
                    color='lexeme',
                    hover_data=['pair_label', 'orig_sentence', 'artificial_sentence']
                )
            fig.update_layout(title=f"Delta Embeddings ({name}, {n_dim}D)")
            html_path = f"docs/modernbert_{name}_{n_dim}D.html"
            fig.write_html(html_path)
            print(f"Saved {html_path}")


if __name__ == "__main__":
    main()
