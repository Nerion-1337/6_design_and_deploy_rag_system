"""
Tests fonctionnels de l'API REST Puls-Events (endpoints /health, /metadata,
/ask, /rebuild). Le LLM et l'index vectoriel sont mockés : ces tests ne
nécessitent ni clé API, ni index Faiss construit, ni connexion réseau.
"""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_endpoint_when_index_not_ready():
    with patch.object(main, "vector_store", None):
        response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["index_ready"] is False
    assert data["total_vectors"] == 0


def test_metadata_endpoint_when_index_ready():
    fake_store = MagicMock()
    fake_store.index.ntotal = 42
    with patch.object(main, "vector_store", fake_store):
        response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["index_ready"] is True
    assert data["total_vectors"] == 42


def test_ask_empty_query_returns_400():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400


def test_ask_blank_query_returns_400():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_valid_query_returns_generated_answer():
    fake_chain = type(
        "FakeChain",
        (),
        {"invoke": staticmethod(lambda q: "Voici un concert de jazz à Bordeaux.")},
    )()

    with patch.object(main, "rag_chain", fake_chain):
        response = client.post(
            "/ask", json={"question": "Y a-t-il des concerts de musique classique ?"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Voici un concert de jazz à Bordeaux."
    assert data["question"] == "Y a-t-il des concerts de musique classique ?"


def test_ask_returns_503_when_index_unavailable():
    with patch.object(main, "rag_chain", None), patch.object(
        main, "get_rag_chain", side_effect=FileNotFoundError("index manquant")
    ):
        response = client.post("/ask", json={"question": "Une question ?"})
    assert response.status_code == 503


def test_ask_returns_500_on_generation_error():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("erreur du LLM")
    with patch.object(main, "rag_chain", fake_chain):
        response = client.post("/ask", json={"question": "Une question valide ?"})
    assert response.status_code == 500


def test_rebuild_endpoint_success():
    with patch.object(main, "build_vector_store") as mock_build, patch.object(
        main, "load_vector_store", return_value=MagicMock()
    ), patch.object(main, "get_rag_chain", return_value="fake-chain"):
        response = client.post("/rebuild")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_build.assert_called_once()


def test_rebuild_endpoint_failure_returns_500():
    with patch.object(main, "build_vector_store", side_effect=RuntimeError("échec index")):
        response = client.post("/rebuild")
    assert response.status_code == 500
