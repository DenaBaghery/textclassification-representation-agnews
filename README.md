# Text Classification with AG News

This project explores how different text representation methods affect the performance of classification models on the AG News dataset. It compares traditional methods like Bag-of-Words and TF-IDF against modern embedding techniques like Word2Vec and Sentence-BERT.

## Features

- **Representation Methods**: Bag-of-Words, TF-IDF, Word2Vec, and Sentence-BERT.
- **Classification Models**: K-Nearest Neighbors (KNN) and a Multi-Layer Perceptron (MLP).
- **Interactive Pipeline**: A command-line interface to run the entire workflow from preprocessing to classification.
- **Result Visualization**: A Jupyter notebook to visualize and compare model performance.

## Workflow Overview

The project follows a clear, sequential pipeline to process the data and evaluate the models.

```text
[AG News CSV]
      |
      v
[1. Preprocessing] -> (Lowercase, Punctuation/Stopword Removal, Tokenization)
      |
      +----------------+----------------+----------------+
      |                |                |                |
      v                v                v                v
[2. Representation] [BoW]         [TF-IDF]      [Word2Vec]   [Sentence-BERT]
      |                |                |                |
      v                v                v                v
[3. Classification] (Train/Test Split)
      |
      +----------------+----------------+
      |                |
      v                v
   [k-NN]            [MLP]
      |                |
      +----------------+
      |
      v
[4. Evaluation] -> (Accuracy, F1-Score)
      |
      v
[Results.json] -> [Visualization Notebook]
```

## Getting Started

### 1. Setup the Environment

First, clone the repository and navigate into the project directory.

```bash
git clone https://github.com/DenaBaghery/textclassification-representation-agnews.git
cd textclassification-representation-agnews
```

Next, create and activate a Python virtual environment.

```bash
# Create the virtual environment
python3 -m venv agnews-env

# Activate it (macOS/Linux)
source agnews-env/bin/activate
```

### 2. Install Dependencies

Install all the necessary packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Run the Classification Pipeline

The main script `src/main.py` runs an interactive pipeline. To start it, run:

```bash
python src/main.py
```

You will see a menu with several options. The easiest way to get started is to **select option 7 to run the full pipeline**. This will:
1.  Load the dataset.
2.  Preprocess the text for all representation methods.
3.  Create the vector representations.
4.  Train and evaluate the KNN and MLP models.
5.  Save the final results to a JSON file in the `results/` directory.
6.  Show Results Summary
7.  Run Full Pipeline
8.  Exit

### 4. Visualize the Results

After running the pipeline, you can visualize the performance metrics in the included Jupyter Notebook.

1.  Open the `notebooks/results_visualization.ipynb` file in VS Code.
2.  Make sure you select the `agnews-env` kernel (your virtual environment).
3.  Run the cells to see the performance comparison charts.

## Project Structure

- **/data/**: Contains the raw and processed AG News dataset.
- **/notebooks/**: Jupyter notebooks for exploration and visualization.
- **/results/**: Output directory for saved models, vectorizers, and performance metrics (JSON).
- **/src/**: All Python source code for the pipeline, including preprocessing, representation, and classification logic.