"""Music recommender: semantic search over GTZAN tracks stored in Qdrant.

Every track gets a short English description built from its genre and its
audio features (tempo, energy, brightness). The descriptions are embedded with
a sentence-transformer model and stored in Qdrant. A free-text query is
embedded the same way, the nearest tracks are retrieved and then re-ranked
with the query's sentiment, the selected emojis and instruments.
"""
import hashlib
import os

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from textblob import TextBlob

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
DEFAULT_COLLECTION = "music-recommender"

GENRE_TEMPLATES = {
    "blues": [
        "Blues song with sad melodies.",
        "Slow blues with emotional guitar solos.",
        "Blues song with melancholic vocals and classic guitar.",
        "Instrumental blues with soulful tones.",
        "Sad blues song with soft tones and a slow pace.",
    ],
    "classical": [
        "Classical instrumental music with rich melodies.",
        "Orchestral music with elegant arrangements.",
        "Calm classical music with violin.",
        "Symphonic piece with piano and strings.",
        "Melancholic classical piece with sad harmonies.",
    ],
    "country": [
        "Country song with acoustic guitars.",
        "Cheerful country song that tells a story.",
        "Country song with banjo and a warm voice.",
        "Acoustic country rhythm with narrative lyrics.",
        "Traditional country song with harmonica.",
    ],
    "disco": [
        "Danceable disco song with a fast beat.",
        "Retro disco hit with a rhythmic bass line.",
        "Cheerful disco song with a catchy chorus.",
        "Disco song with a classic seventies sound.",
        "Fast disco song with funky guitars.",
    ],
    "hiphop": [
        "Hip-hop song with a rhythmic beat.",
        "Modern hip-hop track with an urban style.",
        "Rap song with strong vocals and a heavy beat.",
        "Hip-hop song with an aggressive flow.",
        "Trap-influenced track with modern beats.",
    ],
    "jazz": [
        "Jazz song with a soft saxophone.",
        "Instrumental jazz with a relaxed atmosphere.",
        "Jazz with warm tones and rich harmony.",
        "Improvised jazz with drums and a walking bass line.",
        "Sad jazz ballad with an emotional saxophone.",
    ],
    "metal": [
        "Energetic metal song with heavy distortion.",
        "Intense metal track with fierce guitars.",
        "Metal song with fast drums and powerful vocals.",
        "Heavy metal song with guitar riffs.",
        "Aggressive metal song with a pounding beat.",
    ],
    "pop": [
        "Happy pop song with modern production.",
        "Pop hit with a catchy chorus.",
        "Upbeat pop song with a rhythmic beat.",
        "Modern pop song with a danceable rhythm.",
        "Pop song with a bright mood and a catchy melody.",
    ],
    "reggae": [
        "Reggae song with a Jamaican feel.",
        "Relaxed reggae song with a laid-back beat.",
        "Reggae with warm vocals and an offbeat rhythm.",
        "Danceable reggae song with a deep bass line.",
        "Reggae song with a calm atmosphere.",
    ],
    "rock": [
        "Energetic rock song with loud electric guitars.",
        "Rock song with a fast rhythm and drums.",
        "Rock song with strong vocals and riffs.",
        "Dynamic rock song with distortion.",
        "Rock song with a classic guitar sound.",
    ],
}

# Genre boosts used when re-ranking (added to the cosine similarity)
POSITIVE_GENRES = {"pop", "disco", "country"}
CALM_GENRES = {"blues", "jazz", "classical"}
LOUD_GENRES = {"rock", "metal", "hiphop"}
EMOJI_GENRES = {
    "😊": {"pop", "disco", "rock"}, "😃": {"pop", "disco", "rock"}, "🥳": {"pop", "disco", "rock"},
    "😢": CALM_GENRES, "😭": CALM_GENRES, "😌": CALM_GENRES | {"reggae"},
    "🔥": {"rock", "metal"}, "😎": {"rock", "metal", "hiphop"},
}
INSTRUMENT_GENRES = {
    "guitar": {"rock", "metal", "country", "blues"},
    "piano": {"classical", "jazz"},
    "saxophone": {"jazz"},
    "violin": {"classical"},
    "drums": {"rock", "metal", "hiphop", "disco"},
}
NEGATIVE_WORDS = {"sad", "melancholic", "melancholy", "lonely", "heartbroken", "gloomy", "depressed", "grief"}
POSITIVE_EMOJIS = {"😊", "😃", "🎉", "🥳", "❤️"}
NEGATIVE_EMOJIS = {"😢", "😭", "😞"}
MAX_PER_GENRE = 2


def _pick(options, key):
    """Deterministic choice: the same track always gets the same template."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def _bucket(value, low, high, words):
    if value < low:
        return words[0]
    if value > high:
        return words[2]
    return words[1]


def build_descriptions(tracks: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Add an English `description` per track from its genre template and audio features.

    tracks:   columns id, filename, label, filepath (data/gtzan_data.csv)
    features: GTZAN feature table with filename, tempo, rms_mean, spectral_centroid_mean
    """
    df = tracks.merge(
        features[["filename", "tempo", "rms_mean", "spectral_centroid_mean"]], on="filename", how="left"
    )
    # Energy and brightness are relative to the whole dataset (terciles)
    rms_low, rms_high = df["rms_mean"].quantile([1 / 3, 2 / 3])
    sc_low, sc_high = df["spectral_centroid_mean"].quantile([1 / 3, 2 / 3])

    def describe(row):
        text = _pick(GENRE_TEMPLATES.get(row["label"], ["A piece of music."]), row["filename"])
        if pd.isna(row["tempo"]):
            return text
        tempo = _bucket(row["tempo"], 90, 130, ["Slow", "Moderate", "Fast"])
        energy = _bucket(row["rms_mean"], rms_low, rms_high, ["calm", "steady", "energetic"])
        brightness = _bucket(
            row["spectral_centroid_mean"], sc_low, sc_high, ["warm, mellow sound", "balanced sound", "bright, sharp sound"]
        )
        return f"{text} {tempo} tempo (about {row['tempo']:.0f} BPM), {energy}, with a {brightness}."

    df["description"] = df.apply(describe, axis=1)
    return df


def make_client(url=None, api_key=None, path="qdrant_storage") -> QdrantClient:
    """Qdrant Cloud or a server if a URL is given, otherwise a local on-disk database."""
    if url:
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    return QdrantClient(path=path)


class MusicRecommender:
    def __init__(self, client: QdrantClient, encoder=None, collection_name: str = DEFAULT_COLLECTION):
        """encoder: any object with encode(text or list) -> vector(s); defaults to a
        sentence-transformers model (loaded lazily so tests can pass a fake one)."""
        self.client = client
        self.collection_name = collection_name
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def _embed(self, texts):
        vectors = self.encoder.encode(list(texts))
        return [list(map(float, v)) for v in vectors]

    def is_indexed(self) -> bool:
        return self.client.collection_exists(self.collection_name) and \
            self.client.count(self.collection_name).count > 0

    def index(self, df: pd.DataFrame) -> int:
        """(Re)create the collection and store one point per track."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        vectors = self._embed(df["description"])
        points = [
            PointStruct(
                id=int(row["id"]),
                vector=vector,
                payload={
                    "filename": row["filename"],
                    "genre": row["label"],
                    "description": row["description"],
                    "filepath": row["filepath"],
                },
            )
            for (_, row), vector in zip(df.iterrows(), vectors)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    @staticmethod
    def analyze_sentiment(query: str, emotions=()) -> str:
        """Returns "positive", "negative" or "neutral" from keywords, emojis and TextBlob polarity."""
        words = set(query.lower().replace(",", " ").replace(".", " ").split())
        if words & NEGATIVE_WORDS or set(emotions) & NEGATIVE_EMOJIS:
            return "negative"
        if set(emotions) & POSITIVE_EMOJIS:
            return "positive"
        polarity = TextBlob(query).sentiment.polarity
        if polarity > 0.1:
            return "positive"
        if polarity < -0.1:
            return "negative"
        return "neutral"

    @staticmethod
    def score_adjustment(genre: str, sentiment: str, emotions=(), instruments=()) -> float:
        adjustment = 0.0
        if sentiment == "positive" and genre in POSITIVE_GENRES:
            adjustment += 0.3
        elif sentiment == "negative":
            if genre in CALM_GENRES:
                adjustment += 0.3
            elif genre in LOUD_GENRES:
                adjustment -= 0.5
        for emoji in emotions:
            if genre in EMOJI_GENRES.get(emoji, ()):
                adjustment += 0.15
        for instrument in instruments:
            if genre in INSTRUMENT_GENRES.get(instrument.lower(), ()):
                adjustment += 0.15
        return adjustment

    def recommend(self, query: str, emotions=(), instruments=(), k: int = 5, candidates: int = 50):
        """Retrieve the nearest tracks, re-rank them and keep at most MAX_PER_GENRE per genre."""
        sentiment = self.analyze_sentiment(query, emotions)
        query_vector = self._embed([query])[0]
        hits = self.client.query_points(
            collection_name=self.collection_name, query=query_vector, limit=candidates
        ).points

        ranked = sorted(
            (
                {
                    "filename": hit.payload["filename"],
                    "audio_url": f"/audio/{hit.payload['filepath']}",
                    "genre": hit.payload["genre"],
                    "description": hit.payload["description"],
                    "similarity": round(hit.score, 4),
                    "score": round(hit.score + self.score_adjustment(hit.payload["genre"], sentiment, emotions, instruments), 4),
                }
                for hit in hits
            ),
            key=lambda r: r["score"],
            reverse=True,
        )

        results, per_genre = [], {}
        for rec in ranked:
            if per_genre.get(rec["genre"], 0) < MAX_PER_GENRE:
                results.append(rec)
                per_genre[rec["genre"]] = per_genre.get(rec["genre"], 0) + 1
            if len(results) == k:
                break
        return {"sentiment": sentiment, "recommendations": results}


def load_dataset(data_dir="data") -> pd.DataFrame:
    tracks = pd.read_csv(os.path.join(data_dir, "gtzan_data.csv"))
    features = pd.read_csv(os.path.join(data_dir, "features_30_sec.csv"))
    return build_descriptions(tracks, features)
