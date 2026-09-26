"""
Tests unitaires du module d'indexation vectorielle (conversion, chunking,
construction et persistance de l'index Faiss).
"""
import ctypes
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.indexing.build_index import (
    build_vector_store,
    chunk_documents,
    dataframe_to_documents,
)


def _to_short_path(path: Path) -> Path:
    """Convertit un chemin en format court 8.3 sous Windows pour Faiss."""
    if os.name != "nt":
        return path
    buffer = ctypes.create_unicode_buffer(260)
    ret = ctypes.windll.kernel32.GetShortPathNameW(str(path.resolve()), buffer, 260)
    return Path(buffer.value) if ret != 0 else path


class FakeEmbeddings(Embeddings):
    """Embeddings déterministes (sans appel réseau ni modèle réel) pour les tests."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), float(len(t) % 5)] for t in texts]

    def embed_query(self, text):
        return [float(len(text) % 7), float(len(text) % 5)]


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Concert de Jazz",
                "description": "Un concert exceptionnel " * 40,
                "date_start": "2026-10-01",
                "date_end": "2026-10-01",
                "location_name": "Salle A",
                "location_address": "1 rue A",
                "location_city": "Bordeaux",
            }
        ]
    )


def test_dataframe_to_documents_builds_expected_content(sample_df):
    docs = dataframe_to_documents(sample_df)
    assert len(docs) == 1
    assert "Titre : Concert de Jazz" in docs[0].page_content
    assert docs[0].metadata["location_city"] == "Bordeaux"


def test_chunk_documents_splits_long_text():
    long_doc = Document(page_content="Phrase de test. " * 100, metadata={"uid": "1"})
    chunks = chunk_documents([long_doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["uid"] == "1"


def test_chunk_documents_keeps_short_text_as_single_chunk():
    short_doc = Document(page_content="Texte court.", metadata={"uid": "2"})
    chunks = chunk_documents([short_doc])
    assert len(chunks) == 1


def test_build_vector_store_raises_if_data_missing(tmp_path):
    missing_path = tmp_path / "inexistant.parquet"
    with pytest.raises(FileNotFoundError):
        build_vector_store(raw_data_path=missing_path, index_dir=tmp_path / "index")


def test_build_vector_store_raises_on_empty_dataframe(tmp_path):
    raw_path = tmp_path / "events_raw.parquet"
    pd.DataFrame(columns=["uid", "title", "description", "location_city"]).to_parquet(raw_path)
    with pytest.raises(ValueError):
        build_vector_store(raw_data_path=raw_path, index_dir=tmp_path / "index")


@patch("src.indexing.build_index.get_embeddings")
def test_build_vector_store_creates_persisted_index(mock_get_embeddings, sample_df, tmp_path):
    mock_get_embeddings.return_value = FakeEmbeddings()

    raw_path = tmp_path / "events_raw.parquet"
    sample_df.to_parquet(raw_path)

    # Création préalable du dossier puis conversion en chemin court 8.3
    index_dir = tmp_path / "faiss_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_dir = _to_short_path(index_dir)

    vector_store = build_vector_store(raw_data_path=raw_path, index_dir=index_dir)

    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "index.pkl").exists()
    assert vector_store.index.ntotal > 0