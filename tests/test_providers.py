"""
Tests unitaires des fournisseurs LLM et embeddings gratuits, sélectionnables
par variable d'environnement (RAG_LLM_PROVIDER, EMBEDDING_MODEL).
"""
from unittest.mock import patch

import pytest

from src.rag.embeddings_provider import DEFAULT_EMBEDDING_MODEL, get_embeddings
from src.rag.llm_provider import get_llm


# --- LLM provider ---------------------------------------------------------------


def test_get_llm_raises_clear_error_without_nvidia_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("RAG_LLM_PROVIDER", "nvidia")
    with pytest.raises(EnvironmentError):
        get_llm()


def test_get_llm_raises_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "provider_inexistant")
    with pytest.raises(ValueError):
        get_llm()


def test_get_llm_returns_client_when_nvidia_key_present(monkeypatch):
    monkeypatch.setenv("RAG_LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key-for-test")
    llm = get_llm()
    assert llm is not None


# --- Embeddings provider ---------------------------------------------------------


@patch("src.rag.embeddings_provider.HuggingFaceEmbeddings")
def test_get_embeddings_uses_default_model_when_env_unset(mock_hf_embeddings, monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    get_embeddings.cache_clear()

    get_embeddings()

    _, kwargs = mock_hf_embeddings.call_args
    assert kwargs["model_name"] == DEFAULT_EMBEDDING_MODEL
    assert kwargs["model_kwargs"] == {"device": "cpu"}
    get_embeddings.cache_clear()


@patch("src.rag.embeddings_provider.HuggingFaceEmbeddings")
def test_get_embeddings_uses_custom_model_from_env(mock_hf_embeddings, monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    get_embeddings.cache_clear()

    get_embeddings()

    _, kwargs = mock_hf_embeddings.call_args
    assert kwargs["model_name"] == "sentence-transformers/all-mpnet-base-v2"
    get_embeddings.cache_clear()
