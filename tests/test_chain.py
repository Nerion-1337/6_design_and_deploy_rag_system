"""
Tests unitaires de la chaîne RAG : formatage du contexte, chargement de
l'index, interrogation du retriever et génération de la réponse par le LLM.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag.chain import format_docs, get_rag_chain, load_vector_store


def test_format_docs_concatenates_page_content():
    docs = [
        Document(page_content="Concert de jazz à Bordeaux."),
        Document(page_content="Exposition de peinture au musée."),
    ]
    result = format_docs(docs)
    assert "Concert de jazz" in result
    assert "Exposition de peinture" in result
    assert "---" in result


def test_format_docs_handles_empty_list():
    result = format_docs([])
    assert "Aucun événement" in result


def test_load_vector_store_raises_if_index_missing(tmp_path):
    missing_index = tmp_path / "no_index"
    with pytest.raises(FileNotFoundError):
        load_vector_store(index_dir=missing_index)


@patch("src.rag.chain.get_llm")
@patch("src.rag.chain.load_vector_store")
def test_get_rag_chain_invokes_retriever_and_generates_answer(
    mock_load_vector_store, mock_get_llm
):
    fake_docs = [Document(page_content="Concert de jazz le 12 octobre à Bordeaux.")]

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = fake_docs
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_load_vector_store.return_value = mock_vector_store

    mock_get_llm.return_value = FakeListChatModel(
        responses=["Il y a un concert de jazz le 12 octobre à Bordeaux."]
    )

    chain = get_rag_chain()
    response = chain.invoke("Quels concerts de jazz à Bordeaux ?")

    assert "concert de jazz" in response.lower()
    mock_vector_store.as_retriever.assert_called_once()


@patch("src.rag.chain.get_llm")
@patch("src.rag.chain.load_vector_store")
def test_get_rag_chain_handles_no_matching_documents(mock_load_vector_store, mock_get_llm):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_load_vector_store.return_value = mock_vector_store

    mock_get_llm.return_value = FakeListChatModel(
        responses=["Je ne trouve pas d'événement correspondant dans le catalogue actuel."]
    )

    chain = get_rag_chain()
    response = chain.invoke("Une question hors catalogue ?")

    assert "ne trouve pas" in response.lower()
