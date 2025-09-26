# Spam Detection

## Description

The goal of this project is to provide a historical overview of the various methods that have been used over time to handle spam classification in emails or messages.

## Table of Contents

- [Description](#description)
- [Test Structuring and Case Studies](#1-test-structuring-and-case-studies)
- [Datasets and Metric Choices](#2-datasets-and-metric-choices)
- [Data Preprocessing](#3-data-preprocessing)
- [Method Evaluation](#4-method-evaluation)
  - [Simple and Heuristic Filtering](#simple-and-heuristic-filtering)
  - [Bayesian Filtering](#bayesian-filtering)
  - [Conventional Machine Learning Models](#conventional-machine-learning-models)
  - [LLM Fine-Tuning](#llm-fine-tuning)
- [Constraints and Future Directions](#5-constraints-and-future-directions)
- [Installation and Prerequisites](#installation-and-prerequisites)
- [Usage](#usage)
- [Authors and Contributions](#authors-and-contributions)
- [Resources and References](#resources-and-references)

---

## Installation

1. Clone the repository:

git clone https://github.com/Corrosifu/Spam_Detection.git

2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate # on Linux/Mac
venv\Scripts\activate # on Windows

3. Install dependencies:

pip install -r requirements.txt


---

## Usage
If you want to work on CEAS data
- Open and Launch the [**CEAS_Data_clear**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/CEAS_data_clear.ipynb)  files to analyse and preprocess data.
- Open and Launch the [**Filter_model**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/Filter_model_CEAS.ipynb) file to evaluate the classic filter performances.
- Open and Launch the [**ML_model**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/ML_model_body.ipynb) and [**ML_model_all_features**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/ML_model_multiple_features.ipynb) files to evaluate standard ML methods performances such as Logistic Regression or Random Forest.
- Open and Launch the [**LLM_finetuning**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/LLM_finetuning_body.ipynb) and [**LLM_finetuning_all_features**](https://github.com/Corrosifu/Spam_Detection/blob/main/CEAS-08/LLM_finetuning_multiple_features.ipynb) files to evaluate how LLM perform on Spam vs Ham classification Task.
- The [**Functions**](https://github.com/Corrosifu/Spam_Detection/blob/main/Functions.py) file holds all functions designed to simplify the notebooks code.
  
### 1. Test Structuring and Case Studies

- Analysis of spam classification methods based on text.
- Experiments on the CEAS-08 dataset.
- Progression from simple word frequency filtering to advanced LLM fine-tuning.
- Exploration of the impact of adding complementary data such as email addresses and the number of URLs.

### 2. Datasets and Metric Choices

- Utilization of two main datasets: [CEAS_08](https://github.com/Corrosifu/Spam_Detection/blob/main/Data/CEAS-08) (primary) and a simple [Kaggle dataset](https://www.kaggle.com/datasets/abdmental01/email-spam-dedection) (for pre-tests).
- A 4000-email sample from CEAS-08 balances result quality and computational constraints.
- Data cleaning and preparation adapted to natural language processing (NLP).
- Selection of F1-score as the key metric, combining precision and recall.
- Dataset split: 80% training, 20% testing, according to machine learning standards.

### 3. Data Preprocessing

- Conversion of textual data into tokens via tokenization.
- Removal of stopwords and use of a lemmatizer.
- Application of TF-IDF weighting for word occurrence-based methods.

### 4. Method Evaluation

#### Simple and Heuristic Filtering

- Filtering based on the 20 most frequent words: performance near random (F1 ~ 0.56).
- Filtering using proportions of frequent words, special characters, and mail length: F1 improved to 0.72.

#### Bayesian Filtering

- Multinomial Naive Bayes model with TF-IDF.
- Superior performance (Accuracy > 93%, F1 ~ 0.93).

#### Conventional Machine Learning Models

| Model               | Accuracy (text only) | F1-Score (text only) | Accuracy (multiple features) | F1-Score (multiple features) |
|---------------------|---------------------|---------------------|------------------------------|------------------------------|
| KNN                 | 80.37%              | 0.80                | 75.87%                       | 0.76                         |
| Logistic Regression  | 98.12%              | 0.98                | 97.37%                       | 0.97                         |
| Random Forest       | 98.50%              | 0.98                | 99.12%                       | 0.99                         |
| XGBoost             | 98.12%              | 0.98                | 98.12%                       | 0.98                         |

- Ensemble models (Random Forest, XGBoost) better model non-linearities and large datasets.
- Logistic regression performs well on small, vectorized data batches.

#### LLM Fine-Tuning

- Use of DistilBERT, an encoder transformer architecture optimized for classification.
- Wordpiece tokenizer for fine-grained handling of rare words.
- Results:

| Setup                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| LLM with 1 feature   | 99.37%   | 0.9956    | 0.9934 | 0.9945   |
| LLM with multiple features | 99.50%   | 0.9912    | 1.0    | 0.9956   |

- Perfect recall indicates zero false positives on the test set.
- Computational resource demand and inference time remain higher, limiting usage to contexts prioritizing performance.

### 5. Constraints and Future Directions

- AI serves as a double-edged sword, enhancing both cybersecurity defense and cybercriminal capabilities.
- Increasing threats from adversarial attacks aim to deceive classification algorithms by subtle content modifications.
- Mitigation strategies include data augmentation, AI-generated mail scoring, and use of Generative Adversarial Networks (GANs) to reinforce model robustness.
- Open-source model usage exposes vulnerabilities; proprietary or retrained models could mitigate risks.
- Regular dataset and model updates are required to prevent concept drift.
- Hybrid approaches combining automated learning and human expertise appear promising for refined spam detection.

---

## Installation and Prerequisites

- Python environment with machine learning libraries (scikit-learn, transformers, etc.)
- Access to CEAS-08 and Kaggle datasets.
- GPU recommended for LLM training.

---

## Usage

1. Data preprocessing: cleaning, tokenization, vectorization (TF-IDF or specific tokenizer).
2. Model training.
3. Evaluation on test set.
4. Comparative analysis of results (accuracy, F1-score, precision, recall).

---

## Authors and Contributions

This study was carried out independently, providing a neutral and impersonal synthesis and analysis of current AI spam detection methods.

---

## Resources and References

- CEAS Spam-filter Challenge 2008
- Kaggle email spam detection datasets
- Documentation on Multinomial Naive Bayes, machine learning, and LLMs
- References on GANs and adversarial attacks in AI

---

For questions or contributions, please use the issue tracker in this repository.

     
  
  


