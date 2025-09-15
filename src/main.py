import pandas as pd
# preprocessing klasse importieren 
from preprocessing import Preprocessor

# pandas dataframe von rohdaten erstellen, Beispielhafte AG News Daten
raw_data= pd.read_csv('data\\raw\\ag.news.csv', sep=',', encoding='utf-8')
print(raw_data)

# Instanz von preprocessing klasse erstellen
preprocessor = Preprocessor(raw_data)

# variable für precessing Methode

save_path = 'data\\processed'
methods = ['bow', 'tfidf', 'word_emb', 'sentence_emb']

for method in methods:
    results=preprocessor.preprocess(method=method)
    results.to_csv(f'{save_path}\\news_cleaned_{method}.csv')
    print(f'{method} gespeichert!')

# pandas dataframe übergeben an preprocessing klasse
# jeweils ein preprocessing datei für Bow,tfidf,word_emb,sentence_emb erstellen
# diese dann abspeichern in dataprocessed folde