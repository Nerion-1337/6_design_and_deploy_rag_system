"""
Tests unitaires du script d'évaluation automatisée du système RAG
(similarité sémantique et logique de classification).
"""
from unittest.mock import MagicMock, patch

from tests.evaluate_rag import classify, cosine_similarity, run_similarity_evaluation


def test_cosine_similarity_identical_vectors_returns_one():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_returns_zero():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_classify_thresholds():
    assert classify(0.9) == "Correct"
    assert classify(0.6) == "Partiel"
    assert classify(0.2) == "Incorrect"


@patch("tests.evaluate_rag.get_embeddings")
@patch("tests.evaluate_rag.get_rag_chain")
def test_run_similarity_evaluation_builds_expected_results(mock_get_rag_chain, mock_get_embeddings):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Un concert de jazz à Bordeaux le 12 octobre."
    mock_get_rag_chain.return_value = mock_chain

    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.side_effect = lambda text: [len(text), 0]
    mock_get_embeddings.return_value = mock_embeddings

    test_cases = [
        {"question": "Des concerts de jazz ?", "ground_truth": "Oui, un concert de jazz."}
    ]
    results = run_similarity_evaluation(test_cases)

    assert len(results) == 1
    assert results[0]["question"] == "Des concerts de jazz ?"
    assert "status" in results[0]
    mock_chain.invoke.assert_called_once()


@patch("tests.evaluate_rag.get_embeddings")
@patch("tests.evaluate_rag.get_rag_chain")
def test_run_similarity_evaluation_handles_generation_failure(
    mock_get_rag_chain, mock_get_embeddings
):
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = RuntimeError("panne LLM")
    mock_get_rag_chain.return_value = mock_chain

    # Vecteurs orthogonaux entre une réponse vide et une vérité terrain non vide,
    # pour simuler une similarité nulle en cas d'échec de génération.
    mock_embeddings = MagicMock()
    mock_embeddings.embed_query.side_effect = (
        lambda text: [1.0, 0.0] if text.strip() == "" else [0.0, float(len(text))]
    )
    mock_get_embeddings.return_value = mock_embeddings

    test_cases = [{"question": "Question ?", "ground_truth": "Réponse."}]
    results = run_similarity_evaluation(test_cases)

    assert results[0]["generated_answer"] == ""
    assert results[0]["status"] == "Incorrect"
