import pandas as pd
import numpy as np
from scipy import sparse
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec

def explore_results():
    # Ergebnisverzeichnisse
    processed_dir = 'data/processed'
    results_dir = 'results'
    
    # 1. Preprocessed CSV Files anzeigen
    print("\n=== PREPROCESSED DATA SAMPLES ===")
    for method in ['bow', 'tfidf', 'word_emb']:
        file_path = f'{processed_dir}/news_cleaned_{method}.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"\nMethod: {method}, Shape: {df.shape}")
            print(df.head(2))
            print(f"Columns: {df.columns.tolist()}")
    
    # 2. Feature Matrices anzeigen
    print("\n=== FEATURE MATRICES ===")
    for method in ['bow', 'tfidf', 'word_emb']:
        if method in ['bow', 'tfidf']:
            file_path = f'{results_dir}/{method}_features.npz'
            if os.path.exists(file_path):
                X = sparse.load_npz(file_path)
                print(f"\nMethod: {method}, Shape: {X.shape}")
                print(f"Sparsity: {X.nnz/(X.shape[0]*X.shape[1])*100:.2f}%")
                print(f"Sample (first row, first 5 non-zero elements):")
                row = X[0].toarray()[0]
                non_zeros = [(i, val) for i, val in enumerate(row) if val != 0][:5]
                print(non_zeros)
        else:
            file_path = f'{results_dir}/{method}_features.npy'
            if os.path.exists(file_path):
                X = np.load(file_path)
                print(f"\nMethod: {method}, Shape: {X.shape}")
                print(f"Sample (first row, first 5 elements):")
                print(X[0][:5])
                
                # Einfache Visualisierung für Word Embeddings
                if X.shape[1] <= 100:  # Nur für niedrigdimensionale Embeddings
                    plt.figure(figsize=(10, 6))
                    sns.heatmap(X[:10], cmap='viridis')
                    plt.title(f"{method} - First 10 document embeddings")
                    plt.savefig(f'{results_dir}/{method}_heatmap.png')
                    print(f"Heatmap saved to {results_dir}/{method}_heatmap.png")
    
    # 3. Vectorizers and Models anzeigen
    print("\n=== VECTORIZERS AND MODELS ===")
    for method in ['bow', 'tfidf']:
        file_path = f'{results_dir}/{method}_vectorizer.pkl'
        if os.path.exists(file_path):
            vectorizer = joblib.load(file_path)
            print(f"\nMethod: {method}")
            print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
            print(f"Top 10 features: {list(vectorizer.vocabulary_.items())[:10]}")
    
    # Word2Vec Modell untersuchen
    model_path = f'{results_dir}/word_emb_model'
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
        print("\nWord2Vec Model:")
        print(f"Vocabulary size: {len(model.wv.key_to_index)}")
        
        # Ähnliche Wörter für einige Beispiele anzeigen
        sample_words = ['business', 'technology', 'sports', 'world']
        print("\nWord similarities:")
        for word in sample_words:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=5)
                print(f"\nWords similar to '{word}':")
                for w, sim in similar:
                    print(f"  {w}: {sim:.4f}")
            else:
                print(f"Word '{word}' not in vocabulary")

if __name__ == "__main__":
    explore_results()