from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_ask_empty_query():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 400

def test_ask_valid_query():
    response = client.post("/ask", json={"question": "Y a-t-il des concerts de musique classique ?"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert len(data["response"]) > 0