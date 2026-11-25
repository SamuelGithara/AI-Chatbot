# Kenyan Farmer Query Routing with Fine-tuned mBERT

Mini project: Local AI Application. The model classifies farmer questions written in English, Swahili, Sheng or code-mixed text into six advisory categories, so that each question can be sent to the right extension officer or knowledge base.

| Label | Example |
|---|---|
| `pests` | "Mahindi yangu ina viwavi jeshi, nifanye nini?" |
| `crop_disease` | "Nyanya zangu zina madoa meusi kwenye majani na zinakauka." |
| `soil_fertilizer` | "Ni lini niweke mbolea ya CAN kwenye mahindi?" |
| `weather_climate` | "Mvua za masika zitaanza lini mwaka huu?" |
| `market_prices` | "Madalali wananunua maziwa yangu kwa bei ya chini sana." |
| `livestock` | "Kuku wangu wanakufa ghafla na kinyesi ni cha kijani." |

## Project structure

```
farmer_query_classifier/
├── config.py              # labels, paths, hyperparameters, zero-shot label descriptions
├── prepare_data.py        # builds JSONL/CSV, anonymises, stratified 70/15/15 split
├── train.py               # trains the base models, the TF-IDF reference and fine-tuned mBERT
├── evaluate.py            # metrics, per-language results, McNemar tests, charts
├── predict.py             # base vs fine-tuned predictions on new queries
├── streamlit_app.py       # interactive web app (see DEPLOY.md)
├── requirements.txt
├── data/
│   ├── raw_samples.py         # batch 1 (300 queries)
│   ├── raw_samples_extra.py   # batch 2 (300 queries)
│   ├── farmer_queries.jsonl   # final dataset: input, output, metadata
│   ├── farmer_queries.csv
│   └── train.csv / val.csv / test.csv
└── results/               # metrics, predictions, confusion matrices, charts
```

## How to run

```bash
pip install -r requirements.txt
python prepare_data.py   # 600 samples -> data/
python train.py          # about 12 min on CPU, about 1-2 min on a Colab T4 GPU
python evaluate.py       # writes everything to results/
python predict.py "Ng'ombe wangu amevimba tumbo, nifanye nini?"
```

On Google Colab, upload the folder, choose Runtime → Change runtime type → T4 GPU, then run the same commands with `!` in front.

## Web app

`streamlit_app.py` is an interactive demo. You type a farmer question and see:

- the fine-tuned category, its confidence, and which advisory desk to route it to
- the zero-shot and frozen base mBERT predictions alongside, so the local adaptation is visible

Run it locally with `streamlit run streamlit_app.py`. To deploy it on Streamlit Community Cloud, see [DEPLOY.md](DEPLOY.md).

## Dataset and data provenance

| Item | Detail |
|---|---|
| Size | 600 queries, 100 per label (balanced) |
| Languages | Swahili 206, English 204, code-mixed 129, Sheng 61 |
| Format | JSONL (`input`, `output`, `metadata`) plus a CSV copy |
| Metadata | `id`, `language`, `has_local_term`, `source`, `split`, `n_words` |
| Split | Stratified by label: 420 train / 90 validation / 90 test, seed 42 |
| Personal data | None. The queries are synthetic, and `prepare_data.py` also masks any phone numbers and e-mail addresses |

How the data was made: the queries were written in the style of Kenyan farmer SMS and WhatsApp questions, covering local crops (mahindi, sukuma wiki, miwa, pareto, makadamia), pests and diseases (viwavi jeshi, kideri, ndigana), Kenyan institutions (NCPB, KTDA, KMD) and county place names. The first drafts were AI-assisted and were then reviewed by the student. Before submitting:

1. Read every row and correct any wrong Swahili, Sheng or labels.
2. Add queries you collected or wrote yourself, for example from friends, family or farmer WhatsApp groups (with permission and anonymised), and set their `source` to `student_collected`.
3. Update this section with the final counts and describe each source honestly.

Labelling rule: each query gets the one label that describes the main help needed. Borderline cases are assigned as follows:

- Livestock disease goes to `livestock`, not `crop_disease`.
- Rain-caused rot goes to `crop_disease`.
- Questions about irrigation water supply go to `weather_climate`.

## Models compared

| Model | What it is | Trained on task? |
|---|---|---|
| Base mBERT (zero-shot) | Pretrained `bert-base-multilingual-cased`; each query goes to the most similar label description by cosine similarity | No |
| Base mBERT (frozen + LR head) | Frozen mBERT embeddings with a logistic-regression head | Head only |
| TF-IDF + LR (reference) | Character n-gram TF-IDF with logistic regression; a classical, non-transformer reference | Yes |
| Fine-tuned mBERT | All mBERT weights fine-tuned (10 epochs, lr 3e-5, batch 8, best epoch chosen by validation macro-F1) | Yes |

All trained models use the same 420 training rows. The validation set is used only to choose the best epoch.

## Results (test set, n = 90)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Base mBERT (zero-shot) | 0.333 | 0.327 |
| Base mBERT (frozen + LR head) | 0.644 | 0.643 |
| Fine-tuned mBERT | 0.722 | 0.722 |
| TF-IDF + LR (reference) | 0.867 | 0.867 |

Accuracy by language:

| Language (n) | Zero-shot base | Frozen base | Fine-tuned |
|---|---|---|---|
| English (33) | 0.515 | 0.727 | 0.818 |
| Swahili (30) | 0.167 | 0.600 | 0.667 |
| Code-mixed (17) | 0.353 | 0.471 | 0.588 |
| Sheng (10) | 0.200 | 0.800 | 0.800 |

On queries that contain local terms (n = 49), accuracy rises from 0.306 for the zero-shot base and 0.592 for the frozen base to 0.694 after fine-tuning.

Exact McNemar tests on the same 90 test items (`results/mcnemar_test.txt`):

- Zero-shot base vs fine-tuned: the fine-tuned model fixes 41 queries and breaks 6, p = 1.8 × 10⁻⁷. This is a significant improvement.
- Frozen base vs fine-tuned: the fine-tuned model fixes 15 and breaks 8, p = 0.21. The difference is in the right direction but not significant at this test size.

Local-adaptation examples (base wrong → fine-tuned right) are listed in `results/local_adaptation_examples.csv`. For example:

- "Napima udongo wapi hapa Kakamega?": `market_prices` → `soil_fertilizer`
- "Jinsi ya kutengeneza mboji kwa kutumia takataka za jikoni.": `pests` → `soil_fertilizer`
- "Ni mara ngapi nipe mbuzi dawa ya minyoo?": `weather_climate` → `livestock`
- "Bro, ngamia yangu ina swelling kwa shingo.": `crop_disease` → `livestock`
- "Madalali wanataka kununua gunia la maharagwe kwa bei ya kutupa.": `pests` → `market_prices` (`predict.py` demo)

## Limitations and honest notes

- The test set is small (90 items), so one query is worth 1.1 percentage points and the results vary between random seeds.
- The simple TF-IDF model beats fine-tuned mBERT here. Short queries built around keywords favour character n-grams, and 420 training rows is little data for a model with 178M parameters. This is worth discussing in the report rather than hiding.
- Most errors are between labels that genuinely overlap: market vs livestock (for example, an egg market), soil vs weather, and pests vs disease.
- The data is mostly synthetic, so real farmer messages may be noisier (spelling, abbreviations, mother-tongue words).

## Ideas to improve

- Add 200–400 real or self-written queries, especially in Sheng and code-mixed text.
- On a GPU, run 3–5 random seeds and report the mean ± standard deviation.
- Try lr 2e-5 with 15 epochs, or freeze the lower 6 layers.
- Consider African-language models (for example AfroXLMR or SwahBERT) as an extension, if your lecturer allows models beyond BERT, mBERT and GPT-2.
