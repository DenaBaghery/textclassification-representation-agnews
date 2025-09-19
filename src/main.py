import pandas as pd
import logging
import os
import time
import numpy as np
from scipy import sparse
import joblib
from utils.nltk_ensure import ensure_nltk_resources

# Logging Konfugurations
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/agnews_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()  # Also output to console
    ]
)

ensure_nltk_resources()

from preprocessing import Preprocessor
from representation import Representation

logging.info("Starting the preprocessing and representation script.")

try:
    #  AG News Dataset einlesen
    logging.info("Loading AG News dataset...")
    raw_data= pd.read_csv('data/raw/ag.news.csv', sep=',', encoding='utf-8')
    logging.info(f"Loaded dataset with {len(raw_data)} rows and {raw_data.shape[1]} columns")

    # Instanz von preprocessing Klasse erstellen
    logging.info("Initializing preprocessor...")
    preprocessor = Preprocessor(raw_data)

    # Variabel für process Methode
    save_path = 'data/processed'
    os.makedirs(save_path, exist_ok=True)
    #methods = ['bow', 'tfidf', 'word_emb', 'sentence_emb']
    methods = [ 'sentence_emb']

    # Prozess mit jeder Methode
    for method in methods:
        logging.info(f"Starting preprocessing with method: {method}")
        start_time = time.time()

        result = preprocessor.preprocess(method=method)

        # Log preprocessing Ergebnisse
        processing_time = time.time() - start_time
        logging.info(f"Preprocessing with {method} completed in {processing_time:.2f} seconds")
        logging.info(f"Processed data shape: {result.shape}")

        # Ergebnisse speichern
        output_file = f'{save_path}/news_cleaned_{method}.csv'
        result.to_csv(output_file, sep= ",")
        logging.info(f"Saved processed data to {output_file}")


    logging.info("All preprocessing methods completed successfully") 

    # Representation Vektorisierung
    logging.info("Starting representation and classification phase...")
    results_path = 'results'
    os.makedirs(results_path, exist_ok=True)

    for method in methods:
        logging.info(f"Creating {method} representation...")
        start_time = time.time()

        # Preprocess Daten laden
        preprocessed_file = f'{save_path}/news_cleaned_{method}.csv'
        preprocessed_df = pd.read_csv(preprocessed_file)
        logging.info(f"Loaded preprocessed data with shape: {preprocessed_df.shape}")

        # Repräsentation erstellen
        representation = Representation(preprocessed_df, method=method)
        X, model_or_vectorizer = representation.create_representation() # Hier

        # Log representation Details
        processing_time = time.time() - start_time
        is_sparse = sparse.issparse(X)
        feature_type = "sparse" if is_sparse else "dense"

        logging.info(f"{method} representation: {X.shape[0]} documents, {X.shape[1]} features ({feature_type})")
        logging.info(f"Representation created in {processing_time:.2f} seconds")

        # Repräsentationen speichen
        if is_sparse:
            sparse.save_npz(f'{results_path}/{method}_features.npz', X)
        else:
            np.save(f'{results_path}/{method}_features.npy', X)
        
        # Vekrorizer Speichern oder das Modell
        if method in ['bow', 'tfidf']:
            joblib.dump(model_or_vectorizer, f'{results_path}/{method}_vectorizer.pkl')
        elif method == 'word_emb':
            model_or_vectorizer.save(f'{results_path}/{method}_model')
        
        logging.info(f"Saved {method} representation to {results_path}")

except Exception as e: 
    logging.error(f"An error occured {str(e)}", exc_info=True)
    raise

logging.info("Pipeline execution completed")

