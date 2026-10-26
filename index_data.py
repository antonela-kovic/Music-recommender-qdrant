"""Build the track descriptions and (re)index them in Qdrant.

Usage: python index_data.py
"""
import os

from dotenv import load_dotenv

from recommender import MusicRecommender, load_dataset, make_client

load_dotenv()

recommender = MusicRecommender(make_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")))
count = recommender.index(load_dataset())
print(f"Indexed {count} tracks in Qdrant.")
