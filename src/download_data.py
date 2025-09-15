import kagglehub
import pandas as pd
import os
import shutil

print("Starte Download des AG News Datasets...")

try:
    # Dataset von Kaggle herunterladen
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")
    print(f"Dataset heruntergeladen nach: {path}")
    
    # Pfad zur train.csv finden
    train_file = os.path.join(path, "train.csv")
    
    if os.path.exists(train_file):
        print("train.csv gefunden!")
        
        # Daten laden und prüfen
        df = pd.read_csv(train_file, header=None)
        print(f"Dataset hat {len(df)} Zeilen und {len(df.columns)} Spalten")
        
        # Erste 3 Zeilen anzeigen
        print("Erste 3 Zeilen:")
        for i in range(min(3, len(df))):
            print(f"Zeile {i+1}: {df.iloc[i].tolist()}")
        
        # Zum data Ordner kopieren
        target_file = "../data/ag.news.csv"
        shutil.copy2(train_file, target_file)
        print(f"Datei kopiert nach: {target_file}")
        
        print("✅ Download erfolgreich!")
        
    else:
        print("❌ train.csv nicht gefunden")
        
except Exception as e:
    print(f"❌ Fehler: {e}")
    print("Bitte versuchen Sie manuellen Download von:")
    print("https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset")