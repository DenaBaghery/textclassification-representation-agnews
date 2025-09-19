import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.representation import Representation

# Lade ein paar Zeilen deines Datensatzes
df = pd.read_csv("data/processed/news_cleaned_sentence_emb.csv")  # Passe den Pfad ggf. an

rep = Representation(df, method='sentence_emb')
X, model = rep.sentence_embedding_representation()