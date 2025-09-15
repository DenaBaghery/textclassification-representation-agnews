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

    def lowercase(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Alles in Kleinbuchstaben umwandeln (nur Textspalte).
       """
       return df.map(lambda x: x.lower() if isinstance(x, str) else x)
    def remove_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Entfernt Satzzeichen aus der Textspalte.
        """
        return df.map(lambda x: re.sub(r'[^\w\s]', '', x) if isinstance(x, str) else x)
    def remove_stopwords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Entfernt Stopwörter mithilfe der NLTK-Stopwortliste (Englisch).
        """
        stop_words = set(stopwords.words('english'))
        return df.map(
          lambda x: " ".join(word for word in x.split() if word.lower() not in stop_words) 
          if isinstance(x, str) else x
        )
    def tokenize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenisiert den Text in der Textspalte mit NLTK (nur bei Strings)
        """
        return df.map(
            lambda x: word_tokenize(x) if isinstance(x, str) else x
        )
    def join_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenliste wieder in einen String umwandeln.
        Wichtig für BoW/TF-IDF, weil deren Vectorizer Strings erwarten.
        """
        return df.map(lambda x: " ".join(x) if isinstance(x,list) else x)

    def preprocess(self, method=None) -> pd.DataFrame:
        """
        Führt je nach Repräsentationsart die passenden Schritte aus.
        method: 'bow', 'tfidf', 'word_emb', 'sentence_emb'
        """
        if method is None:
            raise ValueError("No preprocessing method specified.")
        df = self.raw_data.copy()

    # pipeline für Bag-of-Words (BoW)
        if method == "bow":
            print("Started BOW preprocessing...")
            df = self.raw_data.copy()
            df = self.lowercase(df)
            df = self.remove_punctuation(df)
            df = self.remove_stopwords(df)
            df = self.tokenize(df)
            df = self.join_tokens(df)
            return df

    # pipeline für TF-IDF
        elif method == "tfidf":
            print("Started TFIDF preprocessing...")
            df = self.raw_data.copy()
            df = self.lowercase(df)
            df = self.remove_punctuation(df)
            df = self.tokenize(df)
            df = self.remove_stopwords(df)
            #df = self.join_tokens(df)
            return df
    
    #pipeline für Word Embeddings
        elif method == "word_emb":
            print("Started Word Embedding preprocessing...")
            df = self.raw_data.copy()
            df = self.lowercase(df)
            df = self.remove_punctuation(df)
            df = self.tokenize(df)
            return df
    
    # pipeline für Sentence Embeddings
        elif method == "sentence_emb":
            ("Started Sentence Embedding preprocessing")
            df = self.raw_data.copy()
            df = self.lowercase(df)
            return df


