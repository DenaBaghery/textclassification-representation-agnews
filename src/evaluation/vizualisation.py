import sys
import os
# Füge das Projekthauptverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import streamlit as st
import pandas as pd
import numpy as np
from scipy import sparse
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, TruncatedSVD
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
from src.representation import Representation
from src.preprocessing import Preprocessor

# Seitentitel
st.set_page_config(page_title="AG News Klassifikation", layout="wide")
st.title("AG News Text Klassifikation Dashboard")

# Pfade
processed_dir = 'data/processed'
results_dir = 'results'

# Sidebar für Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Seite auswählen",
    ["Übersicht", "Datensatz Explorer", "Feature Visualisierung", "Modell Testen", "Ergebnisse Vergleichen"]
)

# Methoden zum Laden
@st.cache_data
def load_data(method):
    return pd.read_csv(f'{processed_dir}/news_cleaned_{method}.csv')

@st.cache_data
def load_features(method):
    if method in ['bow', 'tfidf']:
        return sparse.load_npz(f'{results_dir}/{method}_features.npz')
    else:
        return np.load(f'{results_dir}/{method}_features.npy')

@st.cache_resource
def load_vectorizer(method):
    return joblib.load(f'{results_dir}/{method}_vectorizer.pkl')

@st.cache_resource
def load_word2vec():
    return Word2Vec.load(f'{results_dir}/word_emb_model')

# Übersichtsseite
if page == "Übersicht":
    st.header("Projekt Übersicht")
    st.write("""
    Diese App visualisiert die Ergebnisse der AG News Textklassifikation mit verschiedenen Repräsentationsmethoden:
    - Bag of Words (BoW)
    - TF-IDF
    - Word Embeddings (Word2Vec)
    
    Nutze die Seitenleiste links, um zwischen den verschiedenen Ansichten zu navigieren.
    """)
    
    # Methoden und deren Status anzeigen
    methods = ['bow', 'tfidf', 'word_emb', 'sentence_emb']
    
    st.subheader("Verfügbare Daten")
    for method in methods:
        col1, col2, col3 = st.columns(3)
        
        # Preprocessed data
        if os.path.exists(f'{processed_dir}/news_cleaned_{method}.csv'):
            col1.success(f"✅ {method}: Preprocessed Data")
        else:
            col1.error(f"❌ {method}: Preprocessed Data")
            
        # Features
        feature_path = f'{results_dir}/{method}_features.npz' if method in ['bow', 'tfidf'] else f'{results_dir}/{method}_features.npy'
        if os.path.exists(feature_path):
            col2.success(f"✅ {method}: Feature Matrix")
        else:
            col2.error(f"❌ {method}: Feature Matrix")
            
        # Model/Vectorizer
        model_path = f'{results_dir}/{method}_vectorizer.pkl' if method in ['bow', 'tfidf'] else f'{results_dir}/{method}_model'
        if os.path.exists(model_path):
            col3.success(f"✅ {method}: Model/Vectorizer")
        else:
            col3.error(f"❌ {method}: Model/Vectorizer")

# Datensatz Explorer
elif page == "Datensatz Explorer":
    st.header("Datensatz Explorer")
    
    method = st.selectbox("Repräsentationsmethode auswählen", ['bow', 'tfidf', 'word_emb', 'sentence_emb'])
    
    try:
        df = load_data(method)
        
        st.subheader(f"Datensatz: {method}")
        st.write(f"Form: {df.shape}")
        
        # Klassen-Verteilung anzeigen
        if 'Class' in df.columns:
            st.subheader("Klassenverteilung")
            class_counts = df['Class'].value_counts().reset_index()
            class_counts.columns = ['Klasse', 'Anzahl']
            
            fig = px.bar(class_counts, x='Klasse', y='Anzahl', title="Klassenverteilung")
            st.plotly_chart(fig)
        
        # Beispiel-Dokumente anzeigen
        st.subheader("Beispiel-Dokumente")
        sample_size = min(5, len(df))
        st.dataframe(df.sample(sample_size))
        
        # Textlängen-Statistiken
        if 'Description' in df.columns:
            st.subheader("Textlängen")
            df['text_length'] = df['Description'].astype(str).apply(len)
            
            fig = px.histogram(df, x='text_length', nbins=50, 
                            title="Verteilung der Textlängen")
            st.plotly_chart(fig)
            
            st.write(f"Min Länge: {df['text_length'].min()}")
            st.write(f"Max Länge: {df['text_length'].max()}")
            st.write(f"Durchschnittliche Länge: {df['text_length'].mean():.2f}")
            
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {str(e)}")

# Feature Visualisierung
elif page == "Feature Visualisierung":
    st.header("Feature Visualisierung")
    
    method = st.selectbox("Repräsentationsmethode auswählen", ['bow', 'tfidf', 'word_emb', 'sentence_emb'])
    
    try:
        X = load_features(method)
        
        st.subheader(f"{method} Feature Matrix")
        st.write(f"Form: {X.shape}")
        
        if method in ['bow', 'tfidf']:
            # Für BoW und TF-IDF - Vokabular und Top-Features anzeigen
            vectorizer = load_vectorizer(method)
            st.write(f"Vokabulargröße: {len(vectorizer.vocabulary_)}")
            
            # Top words by frequency (for BoW) or importance (for TF-IDF)
            if sparse.issparse(X):
                feature_names = vectorizer.get_feature_names_out()
                
                # Summe über alle Dokumente für jedes Feature
                total_counts = X.sum(axis=0).A1
                
                # Top Features
                top_n = min(20, len(feature_names))
                top_indices = total_counts.argsort()[-top_n:][::-1]
                top_words = [(feature_names[i], total_counts[i]) for i in top_indices]
                
                # DataFrame für die Visualisierung
                top_df = pd.DataFrame(top_words, columns=['Wort', 'Häufigkeit/Gewicht'])
                
                # Visualisierung
                fig = px.bar(top_df, x='Wort', y='Häufigkeit/Gewicht', 
                          title=f"Top {top_n} Features ({method})")
                st.plotly_chart(fig)
        
        elif method == 'word_emb':
            # Für Word Embeddings - t-SNE oder PCA Visualisierung
            st.subheader("Word Embedding Visualisierung")
            
            # Lade Word2Vec Modell
            try:
                model = load_word2vec()
                
                # Liste wichtiger Wörter
                important_words = ['business', 'technology', 'sports', 'world', 'economy', 
                                   'politics', 'market', 'science', 'health', 'computer']
                
                # Filtere auf Wörter, die im Vokabular sind
                words = [w for w in important_words if w in model.wv]
                
                if words:
                    # Hole Vektoren für diese Wörter
                    vectors = [model.wv[word] for word in words]
                    
                    # PCA für Dimensionsreduktion
                    pca = PCA(n_components=2)
                    result = pca.fit_transform(vectors)
                    
                    # DataFrame für Plotly
                    df_plot = pd.DataFrame({
                        'x': result[:, 0],
                        'y': result[:, 1],
                        'word': words
                    })
                    
                    # Interaktives Scatter-Plot
                    fig = px.scatter(df_plot, x='x', y='y', text='word',
                                  title="2D-Projektion ausgewählter Wörter")
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig)
                    
                    # Ähnliche Wörter finden
                    st.subheader("Ähnliche Wörter")
                    selected_word = st.selectbox("Wort auswählen", words)
                    
                    if selected_word in model.wv:
                        similar = model.wv.most_similar(selected_word, topn=10)
                        similar_df = pd.DataFrame(similar, columns=['Wort', 'Ähnlichkeit'])
                        st.table(similar_df)
                else:
                    st.warning("Keine der wichtigen Wörter wurde im Vokabular gefunden.")
            except Exception as e:
                st.error(f"Fehler beim Laden des Word2Vec-Modells: {str(e)}")
        
        elif method == 'sentence_emb':
            st.subheader("Sentence Embedding Visualisierung")
            # Lade zugehörige Labels (falls vorhanden)
            try:
                df = load_data('sentence_emb')
                labels = df['Class'] if 'Class' in df.columns else None
            except Exception:
                labels = None
            # PCA auf 2D
            try:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
                plot_df = pd.DataFrame({
                    'x': X_2d[:, 0],
                    'y': X_2d[:, 1],
                    'label': labels if labels is not None else ['']*X_2d.shape[0]
                })
                fig = px.scatter(plot_df, x='x', y='y', color='label' if labels is not None else None,
                                 title="2D-PCA der Sentence Embeddings",
                                 labels={'label': 'Klasse'})
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Fehler bei der Visualisierung der Sentence Embeddings: {str(e)}")
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Features: {str(e)}")

# Modell Testen
elif page == "Modell Testen":
    st.header("Modell mit eigenen Texten testen")
    
    method = st.selectbox("Repräsentationsmethode auswählen", ['bow', 'tfidf', 'word_emb', 'sentence_emb'])
    
    # Texteingabefeld
    user_text = st.text_area("Text eingeben für die Klassifikation", 
                          "Apple unveils new iPhone with revolutionary AI features.")
    
    if st.button("Klassifizieren"):
        try:
            # Preprocessor und Representation erstellen
            # Diese Implementierung ist vereinfacht - in einer realen Anwendung müsstest du
            # das Preprocessing und die Repräsentation genau so durchführen wie im Training
            df = pd.DataFrame({"Description": [user_text]})
            
            # Representation erstellen
            representation = Representation(df, method=method)
            X, _ = representation.create_representation()
            
            # Hier würdest du das gespeicherte Modell laden und die Vorhersage machen
            # Da wir noch kein Klassifikationsmodell gespeichert haben, zeigen wir einen Platzhalter
            st.info("Klassifikationsmodell noch nicht implementiert. Dies würde normalerweise die Klasse vorhersagen.")
            
            # Beispiel für das Feature-Ergebnis
            st.subheader("Feature-Vektor:")
            if sparse.issparse(X):
                st.write(f"Sparse Vektor mit {X.shape[1]} Features und {X.nnz} Nicht-Null-Werten")
                # Konvertiere zu Array für Anzeige
                dense_array = X.toarray()[0]
                non_zeros = [(i, val) for i, val in enumerate(dense_array) if val != 0]
                st.write(f"Erste 10 Nicht-Null-Features: {non_zeros[:10]}")
            else:
                st.write(f"Dense Vektor mit {X.shape[1]} Dimensionen")
                st.write(f"Erste 10 Werte: {X[0][:10]}")
                
        except Exception as e:
            st.error(f"Fehler bei der Klassifikation: {str(e)}")

# Ergebnisse Vergleichen
elif page == "Ergebnisse Vergleichen":
    st.header("Vergleich der verschiedenen Methoden")
    
    # Hier würdest du die Evaluierungsergebnisse der verschiedenen Methoden laden und vergleichen
    # Da wir noch keine gespeicherten Evaluierungsergebnisse haben, zeigen wir Beispieldaten
    
    st.info("Diese Seite würde normalerweise die Ergebnisse verschiedener Klassifikationsmodelle vergleichen.")
    
    # Beispieldaten für die Visualisierung
    methods = ['bow', 'tfidf', 'word_emb', 'sentence_emb']
    metrics = {
        'accuracy': [0.85, 0.88, 0.82],
        'precision': [0.84, 0.87, 0.80],
        'recall': [0.83, 0.86, 0.81],
        'f1': [0.83, 0.86, 0.80]
    }
    
    # Vergleichsdiagramm
    st.subheader("Leistungsvergleich (Beispieldaten)")
    
    metric = st.selectbox("Metrik auswählen", list(metrics.keys()))
    
    df_metrics = pd.DataFrame({
        'Methode': methods,
        metric: metrics[metric]
    })
    
    fig = px.bar(df_metrics, x='Methode', y=metric, 
              title=f"Vergleich nach {metric}",
              color='Methode')
    st.plotly_chart(fig)