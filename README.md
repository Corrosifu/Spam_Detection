# Spam Detection: from heuristics to fine-tuned transformers

A historical and experimental overview of how spam classification has been handled, from
word-frequency heuristics to a fine-tuned DistilBERT, compared on the same data with the same
metric.

| Family | Best F1 (CEAS-08 sample) |
|---|---|
| Frequency heuristics | 0.72 |
| Multinomial Naive Bayes (TF-IDF) | ~0.93 |
| Classical ML (logistic regression, random forest, XGBoost) | 0.98 – 0.99 |
| DistilBERT fine-tuning | 0.995 – 0.996 |

> Read the [limitations](#limitations-and-caveats) before comparing these numbers: a single
> 80/20 split on a 4,000-email sample, and a DistilBERT checkpoint that was already tuned on spam.

## Table of contents

- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Experimental setup](#experimental-setup)
- [Results](#results)
- [Limitations and caveats](#limitations-and-caveats)
- [Constraints and future directions](#constraints-and-future-directions)
- [Data](#data)
- [Author](#author)
- [Resources and references](#resources-and-references)

## Quick start

```bash
git lfs install                      # the CSV files are stored with Git LFS
git clone https://github.com/Corrosifu/Spam_Detection.git
cd Spam_Detection

python -m venv venv
source venv/bin/activate             # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords wordnet
```

Open the notebooks from their own folder (they read the data with relative paths such as
`../Data/new_data_clear.csv`). A GPU is recommended for the fine-tuning notebooks.

## Repository layout

| Path | Content |
|---|---|
| `CEAS-08/` | Main study. Run in this order: `CEAS_data_clear.ipynb` (cleaning) → `Filter_model_CEAS.ipynb` (heuristic and Bayesian filters) → `ML_model_body.ipynb`, `ML_model_multiple_features.ipynb` (classical ML) → `LLM_finetuning_body.ipynb`, `LLM_finetuning_multiple_features.ipynb` (DistilBERT) |
| `Kaggle_database/` | Same pipeline on a small Kaggle dataset, used for pre-tests |
| `*/Functions.py` | Helper functions shared by the notebooks of each folder |
| `Data/` | Datasets (Git LFS) |

## Experimental setup

- **Datasets:** [CEAS-08](https://github.com/Corrosifu/Spam_Detection/tree/main/Data) (primary) and a
  [Kaggle dataset](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection) (pre-tests).
- **Sample:** 4,000 emails from CEAS-08, to balance result quality and compute.
- **Split:** 80 % training / 20 % test (single split).
- **Metric:** F1-score, which combines precision and recall; accuracy, precision and recall are also reported.
- **Preprocessing:** tokenization, stop-word removal, lemmatization, TF-IDF weighting; exact duplicates
  are dropped during cleaning.
- **Features:** message text only, or text plus complementary features (sender address, number of URLs).

## Results

### Simple and heuristic filtering
- Filter on the 20 most frequent words: close to random (F1 ≈ 0.56).
- Filter on the proportion of frequent words, special characters and message length: F1 = 0.72.

### Bayesian filtering
- Multinomial Naive Bayes with TF-IDF: accuracy > 93 %, F1 ≈ 0.93.

### Classical machine learning

| Model | Accuracy (text only) | F1 (text only) | Accuracy (multiple features) | F1 (multiple features) |
|---|---|---|---|---|
| KNN | 80.37 % | 0.80 | 75.87 % | 0.76 |
| Logistic regression | 98.12 % | 0.98 | 97.37 % | 0.97 |
| Random forest | 98.50 % | 0.98 | 99.12 % | 0.99 |
| XGBoost | 98.12 % | 0.98 | 98.12 % | 0.98 |

Ensemble models (random forest, XGBoost) capture non-linearities better; logistic regression is
strong on small vectorized data.

### Transformer fine-tuning (DistilBERT)

| Setup | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Text only | 99.37 % | 0.9956 | 0.9934 | 0.9945 |
| Multiple features | 99.50 % | 0.9912 | 1.0000 | 0.9956 |

With multiple features the recall is 1.0: **no spam email was missed** (zero false negatives). The
precision of 0.9912 means that a few legitimate emails were still flagged as spam (false positives).

The best accuracy comes at a higher cost: training needs a GPU and inference is slower than the
classical models, which limits its use to contexts where performance matters more than latency.

## Limitations and caveats

- **Single split, no confidence intervals.** Results come from one 80/20 split of a 4,000-email
  sample; no cross-validation was run, so differences of a few tenths of a point between the best
  models are not statistically meaningful.
- **Pre-tuned checkpoint.** The CEAS DistilBERT notebooks start from
  `mariagrandury/distilbert-base-uncased-finetuned-sms-spam-collection`, a checkpoint **already
  fine-tuned on spam data** (SMS Spam Collection). The reported scores therefore do not isolate what
  fine-tuning from a generic DistilBERT would achieve. The Kaggle notebook starts from `bert-base-uncased`.
- **Possible overlap between train and test.** Exact duplicates are removed, but near-duplicates and
  templated emails (same sender or campaign) were not checked, which can inflate the scores of all models.
- **Dataset age.** CEAS-08 dates from 2008; modern spam (AI-written, targeted) may behave differently.
- **Environment.** Library versions used for the results were not recorded (`requirements.txt` is unpinned).
- **Optional helpers.** `Functions.py` contains a disabled data-augmentation helper based on an
  OpenAI model; it needs your own `OPENAI_API_KEY` and is not used for the reported results. One cell of
  `CEAS_data_clear.ipynb` loads a Hugging Face model with an access token (placeholder `"Votre Token"`): use your own.

## Constraints and future directions

- AI is a double-edged sword: it strengthens defence and also the attackers' capabilities.
- Adversarial attacks try to fool classifiers with subtle content changes. Mitigations include data
  augmentation, scoring of AI-generated mail, and generative adversarial training.
- Open-source models expose known weaknesses; proprietary or retrained models can reduce this risk.
- Datasets and models must be refreshed regularly to limit concept drift.
- Hybrid approaches combining automated learning and human review look promising.

## Data

The CSV files are tracked with Git LFS (`git lfs pull` if the files look like small text pointers).
They are provided for reproducibility; refer to the original CEAS-08 and Kaggle sources for their
licences and terms of use.

## Author

Bilal Bounadja — [GitHub](https://github.com/Corrosifu) · [LinkedIn](https://www.linkedin.com/in/bilal-bounadja-data/).
Questions or suggestions: please open an issue.

## Resources and references

- CEAS 2008 Spam-filter Challenge
- [Kaggle email spam detection dataset](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection)
- Documentation on Multinomial Naive Bayes, classical ML models and transformers
- References on GANs and adversarial attacks in AI
