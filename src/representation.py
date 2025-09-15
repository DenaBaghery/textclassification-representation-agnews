from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import joblib
from scipy import sparse
from typing import Optional
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
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
# sentence embedding