# index_data.py
from recommender import MusicRecommender
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

recommender = MusicRecommender(QDRANT_URL, QDRANT_API_KEY)

# Load CSV, add descriptions, index embeddings
df = recommender.load_data("data/gtzan_data.csv")
df = recommender.add_descriptions(df)
recommender.index_embeddings(df)

print("Embeddingi uspješno generirani i pohranjeni u Qdrant!")
