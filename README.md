# Text Classification with Different Text Representations

**Author**: Dena Baghery
**Project Type**: University Term Paper
**Dataset**: AG News

## 📚 Project Overview

This project investigates how different text representations influence the performance of text classification models. Using the AG News dataset, we evaluate traditional and neural classifiers across multiple vectorization strategies.

## 🎯 Objectives

- Compare text classification models (k-Nearest Neighbors, MLP)
- Analyze different text representations:
  - Bag-of-Words (BoW)
  - TF-IDF
  - Word Embeddings (Word2Vec, GloVe)
  - Sentence Embeddings (Sentence-BERT)
- Visualize classification performance and model behavior

## 🧠 Methodology

- **Dataset**: AG News (news headlines categorized into 4 topics)
- **Preprocessing**: Tokenization, stopword removal, vectorization
- **Models**: kNN, Multilayer Perceptron (MLP)
- **Evaluation**: Accuracy, F1-score, Confusion Matrix, optional PCA/t-SNE visualizations

## 🗂️ Repository Structure



## 🔧 Requirements

- Python 3.8+
- numpy
- scikit-learn
- pandas
- matplotlib
- sentence-transformers

Install dependencies:

```bash
pip install -r requirements.txt
A comparative study of text classification methods based on different text representations using the AG News dataset.
