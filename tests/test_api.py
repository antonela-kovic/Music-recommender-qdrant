import pytest

import api


class StubRecommender:
    def recommend(self, query, emotions=(), instruments=(), k=5):
        return {"sentiment": "neutral", "recommendations": [{"genre": "jazz", "query": query, "k": k}]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "recommender", StubRecommender())
    return api.app.test_client()


def test_recommend_requires_a_description(client):
    assert client.post("/recommend", json={"description": "  "}).status_code == 400


def test_recommend_rejects_invalid_lists(client):
    assert client.post("/recommend", json={"description": "jazz", "emotions": "😊"}).status_code == 400


def test_recommend_returns_results(client):
    res = client.post("/recommend", json={"description": "relaxed jazz", "k": 100})
    assert res.status_code == 200
    assert res.get_json()["recommendations"][0]["k"] == 20  # k is capped


def test_index_page_renders(client):
    assert client.get("/").status_code == 200
