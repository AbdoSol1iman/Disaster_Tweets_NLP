# Disaster Tweets — Text Preprocessing & Trigram N-Gram Language Model

University NLP course project. Two tasks are implemented end-to-end in a single
Jupyter notebook, run top-to-bottom with no errors:

1. **Task 1 — Dataset selection and text preprocessing** — inspection of the
   Kaggle *Disaster Tweets* dataset (11,370 tweets; `target`: 1 = real disaster,
   0 = not a real disaster) and a reproducible preprocessing pipeline producing
   a `cleaned_text` column **plus sentence-segmented token sequences**. No record
   is dropped and no label is changed.
2. **Task 2 — N-Gram language model** — a trigram model built with the Markov
   assumption and Maximum Likelihood Estimation, with Laplace (add-one)
   smoothing, trained on **individual sentence sequences**, and step-by-step
   probability calculations for **three sentences taken from the actual dataset**
   (every count comes from real data, traceable via tweet id + sentence index).

## Project structure

```
Disaster_Tweets_NLP/
│
├── Disaster_Tweets_NLP.ipynb          # main deliverable (fully executed)
├── data/
│   └── tweets.csv                     # copy extracted from archive.zip
├── outputs/
│   ├── processed_tweets.csv           # id, keyword, location, text, target,
│   │                                  # cleaned_text, sentence_tokens (JSON), n_sentences
│   ├── trigram_probabilities.csv      # top-100 trigrams: counts + MLE + Laplace probabilities
│   └── sentence_probability_results.csv
├── requirements.txt
└── README.md
```

## Dataset

Source: <https://www.kaggle.com/datasets/vstepanenko/disaster-tweets>
(the archive was already downloaded). `archive.zip` in the parent folder is
**not modified**; its only member, `tweets.csv`, was extracted into `data/`.
Key facts found during inspection:

- 11,370 rows × 5 columns (`id, keyword, location, text, target`)
- missing values only in `location` (3,418); `text`/`target` complete
- 0 fully duplicated rows / ids, 147 duplicated tweet *texts*
- class distribution: 9,256 not-disaster (81.4%) vs 2,114 disaster (18.6%)

## How to run

```bash
cd Disaster_Tweets_NLP
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
jupyter notebook Disaster_Tweets_NLP.ipynb           # run all cells in order
```

Required NLTK resources (`stopwords`, `wordnet`, `omw-1.4`,
`averaged_perceptron_tagger_eng`, `punkt`/`punkt_tab`) are downloaded
automatically by the notebook only if missing. Paths are relative, so the folder
can be moved to another machine.

## Preprocessing pipeline (Task 1)

lowercasing → HTML entity/tag handling → URL/mention/`RT` removal →
hashtag **word** preserved (`#Haiti` → `haiti`) → emoji/non-ASCII symbol removal →
**sentence segmentation** (NLTK `punkt`; every sentence becomes its own model
sequence) → **TweetTokenizer** (contractions kept as single tokens, e.g. `don't`)
→ stop-word removal **preserving negations** (`not, no, never, don't, …`) →
**strict, Unicode-aware punctuation removal**: *all* Unicode punctuation
(categories `P*`) is removed at token edges **and** interiors; the apostrophe
survives only between two letters (English contractions/possessives);
interior punctuation acts as a token boundary (`oku.the` → `oku` + `the`,
`school-book` → `school` + `book`); purely numerical, letter-less and
single-character fragment tokens are dropped → **WordNet lemmatization**
with NLTK POS tags.

Result: 11,370 → 11,370 records (labels and original texts unchanged;
7 noise-only tweets get an empty `cleaned_text` and are kept); punkt segments
them into **17,541 sentence sequences**; **4,559 tweets (40.1%) contain more
than one sentence** — whole-tweet training would have created artificial
cross-boundary trigrams for these. Corpus statistics after preprocessing:
vocabulary **18,258** word types, **78,050** distinct bigram contexts,
**85,649** distinct trigrams over **109,277** trigram positions.
Validated in-notebook: **0 residual punctuation tokens**, all negations
survive (`not` × 733, `no` × 452, `never` × 142, `don't` × 281, …), and the
model's start-context mass sums to exactly **1.0**.

## Trigram model (Task 2)

Each **sentence** sequence (not each tweet) gets two `<s>` start tokens, so every
chain-rule factor is a genuine trigram conditional and **no trigram crosses a
sentence boundary, by construction**. Probabilities:

- **Raw MLE:** `P(w3|w1,w2) = Count(w1,w2,w3) / Count(w1,w2)`
- **Laplace (add-one):** `P(w3|w1,w2) = (Count(w1,w2,w3) + 1) / (Count(w1,w2) + V)`
- sentence probabilities reported both as products and as **log-space sums**
  (underflow-safe), with raw and smoothed values always shown side by side.

An unseen-trigram demonstration (out-of-corpus sentence, clearly labelled)
shows explicit zero counts: raw MLE collapses to 0, Laplace rescues it with
`(0+1)/(Count(context)+V)`.

## The three dataset sentences (results)

| # | tweet id | sentence index | target | tokens | P (raw MLE) | P (Laplace) | ln P (Laplace) |
|---|---|---|---|---|---|---|---|
| 1 | 7 | 0 | 1 (real disaster) | 9 | 6.528 × 10⁻⁶ | 1.174 × 10⁻³⁶ | −82.7 |
| 2 | 98 | 0 | 0 (not disaster) | 9 | 9.792 × 10⁻⁶ | 2.695 × 10⁻³⁵ | −79.6 |
| 3 | 47 | 0 | 1 (real disaster) | 9 | 5.875 × 10⁻⁵ | 8.804 × 10⁻³⁶ | −80.7 |

- Sentence 1: *"Asansol: A BJP office in Salanpur village was set ablaze last night."*
- Sentence 2: *"If two quakes have about the same magnitude, you cannot say one is the other's preshock or aftershock."* — the preserved negation `cannot` and the internal possessive apostrophe in `other's` are visible tokens.
- Sentence 3: *"My friend (an army) just lost her father in an accident and her mom right now is still unconscious."*

Verification: the exported `processed_tweets.csv` re-reads and **reproduces every
n-gram count exactly** (round-trip assert in the notebook); the three probability
calculations were also re-derived in a **separate process from the CSV exports
only** and matched to full float precision; sentence provenance was re-established
by re-segmenting the raw tweets independently.

## Limitations / assumptions

- Add-one smoothing is a simple baseline (production systems use Kneser–Ney).
- No end-of-sentence token, per the assignment's chain formula.
- `punkt` segmentation is heuristic on noisy social media; truncated tweets yield
  final fragment sentences, which are treated as ordinary sequences.
- Digits, punctuation and non-English tweets are removed by design of the
  assignment's preprocessing; non-English tweets get an empty `cleaned_text`.
- 147 verbatim duplicated texts are kept (dropping them is out of scope).
