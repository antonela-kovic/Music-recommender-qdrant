"""Tests with an in-memory Qdrant and a small word-hashing encoder instead of the
sentence-transformer model, so they run offline and fast."""
import hashlib
import re

import numpy as np
import pandas as pd
import pytest
from qdrant_client import QdrantClient

from recommender import MAX_PER_GENRE, VECTOR_SIZE, MusicRecommender, build_descriptions, load_dataset


class HashingEncoder:
    """Bag-of-words vector: every word adds 1 to a hashed dimension."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            v = np.zeros(VECTOR_SIZE)
            for word in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % VECTOR_SIZE] += 1
            vectors.append(v / (np.linalg.norm(v) or 1))
        return vectors


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("data")


@pytest.fixture(scope="module")
def recommender(dataset):
    rec = MusicRecommender(QdrantClient(":memory:"), encoder=HashingEncoder())
    rec.index(dataset)
    return rec


def test_dataset_has_english_descriptions_for_all_tracks(dataset):
    assert len(dataset) == 1000
    assert dataset["description"].notna().all()
    assert not dataset["description"].str.contains(r"[čćžšđ]").any()
    assert dataset["description"].str.contains("BPM").all()


def test_descriptions_are_deterministic():
    tracks = pd.DataFrame([{"id": 1, "filename": "jazz.00001.wav", "label": "jazz", "filepath": "jazz/jazz.00001.wav"}])
    features = pd.DataFrame([{"filename": "jazz.00001.wav", "tempo": 80.0, "rms_mean": 0.1, "spectral_centroid_mean": 1500.0}])
    first = build_descriptions(tracks, features)["description"][0]
    second = build_descriptions(tracks, features)["description"][0]
    assert first == second
    assert "Slow tempo" in first


def test_all_tracks_are_indexed(recommender):
    assert recommender.client.count(recommender.collection_name).count == 1000


def test_query_returns_matching_genre_first(recommender):
    result = recommender.recommend("heavy metal with distortion and guitar riffs", k=3)
    assert result["recommendations"][0]["genre"] == "metal"


def test_results_are_sorted_and_limited_per_genre(recommender):
    result = recommender.recommend("energetic rock song with electric guitars", k=5)
    recs = result["recommendations"]
    assert len(recs) == 5
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    genres = [r["genre"] for r in recs]
    assert all(genres.count(g) <= MAX_PER_GENRE for g in set(genres))


@pytest.mark.parametrize("query,emotions,expected", [
    ("a sad and lonely evening", [], "negative"),
    ("anything", ["😢"], "negative"),
    ("anything", ["🥳"], "positive"),
    ("a wonderful happy day", [], "positive"),
    ("music for studying", [], "neutral"),
])
def test_sentiment(query, emotions, expected):
    assert MusicRecommender.analyze_sentiment(query, emotions) == expected


def test_negative_sentiment_pushes_down_loud_genres():
    assert MusicRecommender.score_adjustment("metal", "negative") < 0
    assert MusicRecommender.score_adjustment("blues", "negative") > 0


def test_instrument_boost():
    assert MusicRecommender.score_adjustment("jazz", "neutral", instruments=["Piano"]) > 0
    assert MusicRecommender.score_adjustment("metal", "neutral", instruments=["piano"]) == 0
