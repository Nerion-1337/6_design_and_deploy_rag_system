"""
Construction de l'index vectoriel Faiss à partir des événements Open Agenda.
"""
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.embeddings_provider import get_embeddings

load_dotenv()

RAW_DATA_PATH = Path("data/raw/events_raw.parquet")
INDEX_DIR = Path("data/faiss_index")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """Convertit chaque ligne du DataFrame d'événements en Document LangChain."""
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Titre : {row['title']}\n"
            f"Description : {row['description']}\n"
            f"Ville : {row['location_city']}"
        )
        metadata = {
            "uid": str(row.get("uid", "")),
            "title": str(row.get("title", "")),
            "date_start": str(row.get("date_start", "")),
            "date_end": str(row.get("date_end", "")),
            "location_name": str(row.get("location_name", "")),
            "location_address": str(row.get("location_address", "")),
            "location_city": str(row.get("location_city", "")),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Découpe les documents longs en chunks pour une vectorisation optimale."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def build_vector_store(
    raw_data_path: Path = RAW_DATA_PATH, index_dir: Path = INDEX_DIR
) -> FAISS:
    """Pipeline complet : chargement, chunking, embeddings, sauvegarde de l'index."""
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Données introuvables : {raw_data_path}")

    df = pd.read_parquet(raw_data_path)
    print(f"Chargement de {len(df)} événements...")

    if df.empty:
        raise ValueError("Le jeu de données est vide : impossible de construire l'index.")

    documents = dataframe_to_documents(df)
    docs_chunked = chunk_documents(documents)
    print(f"Total de chunks générés : {len(docs_chunked)}")

    embeddings = get_embeddings()
    print("Génération des embeddings et création de l'index Faiss...")
    vector_store = FAISS.from_documents(docs_chunked, embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    print(f"Index Faiss sauvegardé avec succès dans {index_dir}")
    return vector_store


if __name__ == "__main__":
    build_vector_store()
