import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

RAW_DATA_PATH = Path("data/raw/events_raw.parquet")
INDEX_DIR = Path("data/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_vector_store():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Données introuvables : {RAW_DATA_PATH}")

    df = pd.read_parquet(RAW_DATA_PATH)
    print(f"Chargement de {len(df)} événements...")

    documents = []
    for _, row in df.iterrows():
        # Construction d'un contenu textuel riche pour l'embedding
        content = f"Titre : {row['title']}\nDescription : {row['description']}\nVille : {row['location_city']}"
        
        # Métadonnées conservées pour le filtrage et la réponse
        metadata = {
            "uid": str(row["uid"]),
            "title": str(row["title"]),
            "date_start": str(row["date_start"]),
            "date_end": str(row["date_end"]),
            "location_name": str(row["location_name"]),
            "location_address": str(row["location_address"]),
            "location_city": str(row["location_city"]),
        }
        
        documents.append(Document(page_content=content, metadata=metadata))

    # Découpage des textes longs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs_chunked = text_splitter.split_documents(documents)
    print(f"Total de chunks générés : {len(docs_chunked)}")

    # Vectorisation via Mistral Embeddings
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    
    print("Génération des embeddings et création de l'index Faiss...")
    vector_store = FAISS.from_documents(docs_chunked, embeddings)
    
    # Sauvegarde locale de l'index
    vector_store.save_local(str(INDEX_DIR))
    print(f"Index Faiss sauvegardé avec succès dans {INDEX_DIR}")

if __name__ == "__main__":
    build_vector_store()