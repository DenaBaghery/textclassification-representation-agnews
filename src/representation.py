from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from typing import Optional
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import logging
import os


# Representation needed
class Representation():
    """Erstellt für die cleaned files jeweils repräsentationen"""
    def __init__(self, df: pd.DataFrame, method: str):
        self.df = df
        self.method = method
    
    def get_list(self, col_name: str = 'Description'):
        if col_name not in self.df.columns:
            raise KeyError(f"Spalte '{col_name}' nicht im DataFrame vorhanden.")
        return self.df[col_name].tolist()
    
    # Methoden-Routing
    def create_representation(self, col_name: str = 'Description'):
        """
        Ruft basierend auf self.method die entsprechende Repräsentationsmethode auf.
        Vereinfacht die Verwendung in der main.py.
        
        Returns:
            tuple: (X, model_or_vectorizer) - X ist die Feature-Matrix,
                  model_or_vectorizer ist der zugehörige Vectorizer oder das Modell
        """
        if self.method == 'bow':
            return self.bow_representation(col_name=col_name)
        elif self.method == 'tfidf':
            return self.tfidf_representation(col_name=col_name)
        
        elif self.method == 'word_emb':
            return self.word2vec_representation(col_name=col_name)
        
        elif self.method == 'sentence_emb':
            return self.sentence_embedding_representation(col_name=col_name)
        
        else:
            raise ValueError(f"Unbekannte Methode: {self.method}. " 
                           f"Erlaubte Werte: 'bow', 'tfidf', 'word_emb', 'sentence_emb'")

# BoW Representation
    def bow_representation(self, col_name: str = 'Description'):
        """
        Erzeugt eine Bag-of-Words-Repräsentation (Unigramme) mit sklearn CountVectorizer.
        - Entfernt NaN und reine Leerstrings
        - Speichert vectorizer und X als Instanzattribute
        - Gibt (X, vectorizer) zurück, wobei X eine sparse Dokument-Term-Matrix ist
        """
        if col_name not in self.df.columns:
            raise KeyError(f"Spalte '{col_name}' nicht im DataFrame vorhanden.")

        texts = self.df[col_name].tolist()
        # NaN und reine Leerstrings entfernen, sichere String-Konvertierung
        cleaned = [str(t).strip() for t in texts if pd.notna(t) and str(t).strip() != ""]
        if not cleaned:
            raise ValueError(f"Keine (nicht-leeren) Texte in Spalte '{col_name}' gefunden.")

        # CountVectorizer default = Unigramme => klassisches BOW
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(cleaned)

        # optional für späteren Zugriff speichern
        self.vectorizer = vectorizer
        self.X = X

        return X, vectorizer


# tfidf representation
    def tfidf_representation(
            self, 
            col_name: str = 'Description',
            max_features: Optional[int] = None, 
            norm: str = 'l2',
            use_idf: bool = True, 
            smooth_idf: bool = True,
            sublinear_tf: bool = False):
        """
        Erzeugt eine TF-IDF-Repräsentation mit sklearn.TfidfVectorizer.
        - Entfernt NaN und reine Leerstrings
        - Parameter: max_features, norm, use_idf, smooth_idf, sublinear_tf
        - Speichert self.tfidf_vectorizer und self.tfidf_X
        - Gibt (X, vectorizer) zurück (X ist scipy.sparse)
        """
        if col_name not in self.df.columns:
            raise KeyError(f"Spalte '{col_name}' nicht im DataFrame vorhanden.")
        
        texts = self.get_list(col_name)
        cleaned = [str(t).strip() for t in texts if pd.notna(t) and str(t).strip() != ""]
        if not cleaned: 
            raise ValueError(f"Keine (nicht-leeren) Texte in Spalte '{col_name}' gefunden.")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf
        )

        X = vectorizer.fit_transform(cleaned)
        self.tfidf_vectorizer = vectorizer
        self.tfidf_X = X

        return X, vectorizer

# word embedding
    def word2vec_representation(
            self,
            col_name: str = 'Description',
            vector_size: int = 100,
            window: int = 5,
            min_count: int = 2,
            workers: int = 4,
            epochs: int = 5):
        """
        Erzeugt Word2Vec-Embeddings mit Gensim.
        - Tokenisiert Texte mit simple_preprocess
        - Trainiert Word2Vec-Modell auf den Tokens
        - Erzeugt Dokument-Vektoren durch Mittelung der Wort-Vektoren
        - Speichert self.w2v_model und self.w2v_X
        - Gibt (X, model) zurück, wobei X eine numpy.ndarray ist

        """
        if col_name not in self.df.columns:
            raise KeyError(f"Spalte '{col_name}' nicht im DataFrame vorhanden.")
        
        # logging für Word2Vec-Training einrichten
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        #Tokenisierung mit simple_preprocess
        tokenized_texts = [simple_preprocess(str(t)) for t in self.df[col_name] if pd.notna(t) and str(t).strip() != ""]
        if not tokenized_texts:
            raise ValueError(f"Keine (nicht-leeren) Texte in Spalte '{col_name}' gefunden.")
        
        # Word2Vec-Modell trainieren
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs
        )

        # Dokument-Vektoren durch Mittelung der Wort-Vektoren
        doc_vectors = []
        for tokens in tokenized_texts:
            # nur Wörter, die im Vokabular sind
            valid_tokens = [token for token in tokens if token in model.wv]
            if valid_tokens:
                # Durchschnitt der Wort-Vektoren berechnen
                vectors = [model.wv[token] for token in valid_tokens]
                doc_vectors.append(np.mean(vectors, axis=0))
            else:
                # Nullvektor, wenn keine gültigen Tokens
                doc_vectors.append(np.zeros(vector_size))

        # Als numpy Array umwandeln
        X = np.array(doc_vectors)

        #Modell und Vektoren speichern
        self.w2v_model = model
        self.w2v_X = X

        return X, model

# sentence embedding
    def sentence_embedding_representation(
            self,
            col_name: str = 'Description',
            model_name: str = 'all-MiniLM-L6-v2'):
        """
        Erzeugt Sentence Embeddings mit sentence-transformers.
         - Nutzt ein vortrainiertes Modell (z.B. all-MiniLM-L6-v2)
         - Wandelt alle Sätze in dichte Vektoren um
         - Speichert self.sentence_model und self.sentence_X
         - Gibt (X, model) zurück (X ist numpy.ndarray)
        """
        if col_name not in self.df.columns:
            raise KeyError(f"Spalte '{col_name}' nicht im Dataframe vorhanden.")
        
        # TEST
        texts = [str(t).strip() for t in self.df[col_name] if pd.notna(t) and str(t).strip() !=""]
        texts = texts[:10]
        
        if not texts:
            raise ValueError(f"Keine (nicht-leeren) Texte in Spalte '{col_name}' gefunden.")

        model = SentenceTransformer(model_name)
        X = model.encode(texts, show_progress_bar=True)

        print("Shape der Embeddings:", X.shape)
        print("Erste Embeddings:", X[:2])

        self.sentence_model = model
        self.sentence_X = X

        return X, model
