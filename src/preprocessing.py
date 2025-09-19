import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#nltk.download('stopwords') # einmaligiger Download der Stopwörter
#nltk.download('punkt') # einmaligiger Download des Punkt-Tokenizers
#nltk.download('punkt_tab')

class Preprocessor:
    def __init__(self, raw_data: pd.DataFrame):
         """
        Konstruktor- bekommt rohe Daten (z. B. DataFrame) und speichert sie.
        raw_data: erwartet einen Pandas DataFrame mit einer Spalte 'text'.

        """
         self.raw_data = raw_data

    def lowercase(self, df: pd.DataFrame, col: str = 'Description') -> pd.DataFrame:
        """
        Alles in Kleinbuchstaben umwandeln (nur Textspalte).
        """
        df = df.copy()
        df[col] = df[col].map(lambda x: x.lower() if isinstance(x, str) else x)
        return df
    def remove_punctuation(self, df: pd.DataFrame, col: str = 'Description') -> pd.DataFrame:
        """
        Entfernt Satzzeichen aus der Textspalte.
        """
        df = df.copy()
        df[col] = df[col].map(lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x)
        return df
    def remove_stopwords(self, df: pd.DataFrame, col: str = 'Description') -> pd.DataFrame:
        """
        Entfernt Stopwörter mithilfe der NLTK-Stopwortliste (Englisch).
        """
        stop_words = set(stopwords.words('english'))
        df = df.copy()
        df[col] = df[col].map(
            lambda x: " ".join(word for word in x.split() if word.lower() not in stop_words)
            if isinstance(x, str) else x
        )
        return df
    def tokenize(self, df: pd.DataFrame, col: str = 'Description') -> pd.DataFrame:
        """
        Tokenisiert den Text in der Textspalte mit NLTK (nur bei Strings)
        """
        df = df.copy()
        df[col] = df[col].map(
            lambda x: word_tokenize(x) if isinstance(x, str) else x
        )
        return df
    def join_tokens(self, df: pd.DataFrame, col: str = 'Description') -> pd.DataFrame:
        """
        Tokenliste wieder in einen String umwandeln.
        Wichtig für BoW/TF-IDF, weil deren Vectorizer Strings erwarten.
        """
        df = df.copy()
        df[col] = df[col].map(lambda x: " ".join(x) if isinstance(x, list) else x)
        return df

    def preprocess(self, method=None, col: str = 'Description') -> pd.DataFrame:
        """
        Führt je nach Repräsentationsart die passenden Schritte aus.
        method: 'bow', 'tfidf', 'word_emb', 'sentence_emb'
        col: Name der Textspalte (default: 'Description')
        """
        if method is None:
            raise ValueError("No preprocessing method specified.")
        df = self.raw_data.copy()

        if method == "bow":
            print("Started BOW preprocessing...")
            df = self.lowercase(df, col)
            df = self.remove_punctuation(df, col)
            df = self.remove_stopwords(df, col)
            df = self.tokenize(df, col)
            df = self.join_tokens(df, col)
            return df

        elif method == "tfidf":
            print("Started TFIDF preprocessing...")
            df = self.lowercase(df, col)
            df = self.remove_punctuation(df, col)
            df = self.tokenize(df, col)
            df = self.remove_stopwords(df, col)
            return df

        elif method == "word_emb":
            print("Started Word Embedding preprocessing...")
            df = self.lowercase(df, col)
            df = self.remove_punctuation(df, col)
            df = self.tokenize(df, col)
            return df

        elif method == "sentence_emb":
            print("Started Sentence Embedding preprocessing...")
            df = self.lowercase(df, col)
            return df

        else:
            raise ValueError(f"Unknown preprocessing method: {method}")


