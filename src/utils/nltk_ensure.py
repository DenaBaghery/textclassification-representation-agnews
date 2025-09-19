import os
import logging
import nltk
from pathlib import Path

def ensure_nltk_resources():
    """
    Überprüft, ob NLTK-Ressourcen vorhanden sind und lädt sie herunter, falls nicht.
    Diese Funktion sollte einmalig am Anfang des Programms aufgerufen werden.
    """
    nltk_resources = ['stopwords', 'punkt', 'wordnet', 'punkt_tab']
    
    # Prüfe, ob NLTK-Datenverzeichnis existiert
    nltk_data_dir = Path(os.path.expanduser('~/nltk_data'))
    if not nltk_data_dir.exists():
        nltk_data_dir.mkdir(parents=True, exist_ok=True)
    
    resources_to_download = []
    for resource in nltk_resources:
        # Prüfe für jede Ressource, ob sie bereits existiert
        resource_path = nltk_data_dir / ('corpora' if resource != 'punkt' else 'tokenizers') / resource
        if not resource_path.exists():
            resources_to_download.append(resource)
    
    # Download nur wenn nötig
    if resources_to_download:
        logging.info(f"Downloading NLTK resources: {', '.join(resources_to_download)}")
        try:
            # Lade die fehlenden Ressourcen herunter
            for resource in resources_to_download:
                nltk.download(resource)
                logging.info(f"Successfully downloaded {resource}")
        except Exception as e:
            logging.error(f"Failed to download NLTK resources: {str(e)}")
            raise
    else:
        logging.info("All required NLTK resources already exist.")