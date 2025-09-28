import pandas as pd
from scipy import sparse
import numpy as np
import logging
from sklearn.model_selection import train_test_split

def load_features_and_labels(method):
    """
    Lädt Features und Labels für eine bestimmte Methode.
    
    Args:
        method (str): 'bow', 'tfidf', 'word_emb', oder 'sentence_emb'
    
    Returns:
        X (array/matrix): Feature-Matrix
        y (array): Label-Array
    """
    logging.info(f"Loading features and labels for method: {method}")

    # CSV mit labels laden
    csv_path = f"data/processed/news_cleaned_{method}.csv"
    df = pd.read_csv(csv_path, sep=",")

    # Labels extrahieren
    y = df["Class Index"].values

    # Features ladens
    if method in ['bow', 'tfidf']:
        # Sparse Matrix für BoW/TF-IDF
        feature_path = f"results/{method}_features.npz"
        X = sparse.load_npz(feature_path)
    elif method in ['sentence_emb', 'word_emb']:
        # Dense Matrix für Embeddings
        feature_path = f"results/{method}_features.npy"
        X = np.load(feature_path)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logging.info(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y 

# Train/ Test- Split
def get_features(method = 'bow', test_size=0.2, random_state=42):
    """
    Lädt Features und Labels und teilt sie in Training/Test auf.
    
    Args:
        method (str): Welche Features ('bow', 'tfidf', etc.)
        test_size (float): Anteil für Test (0.2 = 20%)
        random_state (int): Für reproduzierbare Ergebnisse
    
    Returns:
        X_train, X_test, y_train, y_test: Aufgeteilte Daten
    """
    # Daten laden
    X, y = load_features_and_labels(method)

    # Daten aufteilen
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logging.info(f"Split data: Train={X_train.shape}, Test={X_test.shape}")
    return X_train, X_test, y_train, y_test