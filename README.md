# Disaster Tweets — Text Preprocessing & Trigram N-Gram Language Model

University NLP course project. Two tasks are implemented end-to-end in a single
Jupyter notebook, run top-to-bottom with no errors:

1. **Task 1 — Dataset selection and text preprocessing** — inspection of the
   Kaggle *Disaster Tweets* dataset (11,370 tweets; `target`: 1 = real disaster,
   0 = not a real disaster) and a reproducible preprocessing pipeline producing
   a `cleaned_text` column. No record is dropped and no label is changed.
2. **Task 2 — N-Gram language model** — a trigram model built with the Markov
   assumption and Maximum Likelihood Estimation, with Laplace (add-one)
   smoothing, and step-by-step probability calculations for **three sentences
   taken from the actual dataset** (every count comes from real data).

## Project structure

```
Disaster_Tweets_NLP/
│
├── Disaster_Tweets_NLP.ipynb          # main deliverable (fully executed)
├── data/
│   └── tweets.csv                     # copy extracted from archive.zip
├── outputs/
│   ├── processed_tweets.csv           # id, keyword, location, text, target, cleaned_text
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
`averaged_perceptron_tagger_eng`) are downloaded automatically by the notebook
only if missing. Paths are relative, so the folder can be moved to another machine.

## Preprocessing pipeline (Task 1)

lowercasing → HTML entity/tag handling → URL/mention/`RT` removal →
hashtag **word** preserved (`#Haiti` → `haiti`) → emoji/non-ASCII symbol removal →
**TweetTokenizer** (contractions kept as single tokens, e.g. `don't`) →
stop-word removal **preserving negations** (`not, no, never, don't, …`) →
punctuation & purely numerical token removal → **WordNet lemmatization**
with NLTK POS tags. Sentence segmentation was decided **against** (tweets are
short, sentence boundaries unreliable) — justification with real numbers inside
the notebook. Result: 11,370 → 11,370 records (7 tweets become empty
`cleaned_text` because they contain only noise; they are kept).

Corpus statistics after preprocessing: vocabulary **18,711** word types,
**109,574** trigram positions (**90,431** distinct trigrams, **82,277** distinct
bigram contexts).

## Trigram model (Task 2)

For each tweet sequence two `<s>` start tokens are prepended so every chain-rule
factor is a genuine trigram conditional. Probabilities:

- **Raw MLE:** `P(w3|w1,w2) = Count(w1,w2,w3) / Count(w1,w2)`
- **Laplace (add-one):** `P(w3|w1,w2) = (Count(w1,w2,w3) + 1) / (Count(w1,w2) + V)`
- sentence probabilities reported both as products and as **log-space sums**
  (underflow-safe), with raw and smoothed values always shown side by side.

An unseen-trigram demonstration (out-of-corpus sentence, clearly labelled)
shows explicit zero counts: raw MLE collapses to 0, Laplace rescues it with
`(0+1)/(Count(context)+V)`.

## The three dataset sentences (results)

| # | tweet id | target | tokens | P (raw MLE) | P (Laplace) | log₁₀ P (Laplace) |
|---|---|---|---|---|---|---|
| 1 | 47 | 1 (real disaster) | 12 | 1.256 × 10⁻⁵ | 6.218 × 10⁻⁴⁸ | −108.7 (ln) |
| 2 | 43 | 0 (not disaster) | 13 | 8.795 × 10⁻⁶ | 5.170 × 10⁻⁵² | −118.1 (ln) |
| 3 | 27 | 1 (real disaster) | 12 | 9.772 × 10⁻⁶ | 2.208 × 10⁻⁴⁷ | −107.4 (ln) |

All counts and probabilities were recomputed independently from the exported
`processed_tweets.csv` and matched exactly.

## Limitations / assumptions

- Add-one smoothing is a simple baseline (production systems use Kneser–Ney).
- No end-of-sentence token, per the assignment's chain formula (~0.06% of mass
  left unexplained, documented in a sanity check inside the notebook).
- Digits, punctuation and non-English tweets are removed by design of the
  assignment's preprocessing; non-English tweets get an empty `cleaned_text`.
- 147 verbatim duplicated texts are kept (dropping them is out of scope).
