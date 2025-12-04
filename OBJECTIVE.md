This experiment is an attempt to determine whether the contextual embeddings of
language models (including ModernBERT and at least one LLM) can systematically
distinguish between English verb inflections that appear to have no difference
in meaning in certain contexts (e.g. ate/eaten).

## Data

The data are given in the following format:

| lexeme | mps | form | ID | source | text |
|--------|-----|------|----|-----------|------------------------------------|
| eat | ppt | ate | 3 | Chicago Med | Well, I've <b>ate</b> the biscuits |

In each text, the target word is labeled using an HTML `<b>` tag.
This target will be referred to as the `original` target.

While processing the data, we should always confirm that the `form` value matches
the string inside the `<b>` tags.

## Method

Every original target word has a partner (this can be deduced by comparing all
`form` values that share the same `lexeme`). By replacing the original target
with its partner, we create an `artificial` target.

So given an original text `Well, I've <b>ate</b> the biscuits`, we derive the
following two sentences to pass into the transformer (separately):

```
Well, I've ate the biscuits
Well, I've eaten the biscuits
```

Then, we get the contextual embeddings of `ate` (original) and `eaten`
(artificial) from the transformer's embeddings of each sentence.  and compute
the delta embedding by subtracting the artificial embedding from the original
embedding. We store the delta embedding in a new column of the dataframe
called delta.

Then, we create interactive 2D and 3D visualizations (using tSNE, PCA, and maybe
other techniques?) to explore the extent to which delta embeddings cluster,
whether delta embeddings are the inverse of one another (are eaten->ate delta
embeddings the inverse of ate->eaten deltas?). If delta embeddings do cluster,
can they be used to classify different types of morphological pairs? To what
extent can the "meaning" of these delta clusters be inferred (regional dialects,
gender, SES, etc.)?

## Plan

Step 1: Perform experiment using ModernBERT
