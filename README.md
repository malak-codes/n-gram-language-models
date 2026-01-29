# N-Gram Language Models: Unigrams, Bigrams, and Interpolation

This repository contains the implementation of fundamental Natural Language Processing (NLP) concepts, specifically focusing on **N-gram language models**. The project explores the training, evaluation, and application of Unigram and Bigram models using the **WikiText-2** dataset.

## Project Overview

The core objective of this project is to build and evaluate statistical language models that can predict the probability of word sequences and the next word in a sentence.

### Key Features

- **Preprocessing Pipeline**: Utilizes **spaCy** for lemmatization and tokenization, ensuring a clean and standardized vocabulary.
- **Unigram & Bigram Training**: Implements the calculation of log-probabilities for single words and word pairs based on their frequencies in the training corpus.
- **Next-Word Prediction**: A functional bigram-based predictor that suggests the most likely next word given a sentence fragment.
- **Perplexity Evaluation**: Measures the performance of the language models by calculating perplexity, a standard metric for evaluating how well a probability model predicts a sample.
- **Linear Interpolation**: Combines Unigram and Bigram models using weighted averages to handle the data sparsity problem (where bigrams might not appear in the training set).

## Repository Structure

| File | Description |
| :--- | :--- |
| `main.py` | The primary script containing the preprocessing logic, model training functions, and evaluation tasks. |
| `NlpEx1Report.pdf` | A detailed report covering the theoretical background, implementation details, and experimental results. |
| `README.md` | Documentation providing an overview of the project and its components. |

## Requirements

The project requires the following Python libraries:
- `spacy` (with the `en_core_web_sm` model)
- `datasets` (Hugging Face)
- `math`

You can install the necessary dependencies using:
```bash
pip install spacy datasets
python -m spacy download en_core_web_sm
```

## Usage

To run the language model training and evaluation, execute the main script:
```bash
python main.py
```
The script will load the WikiText-2 dataset, train the models, and output results for next-word prediction, sentence probabilities, and perplexity calculations.

## Authors
- Malak
- Zenab
