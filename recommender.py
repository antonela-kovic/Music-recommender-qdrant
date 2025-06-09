import pandas as pd
import random
import freesound
import os
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from textblob import TextBlob


class AudioDownloader:
    def __init__(self, api_key):
        self.client = freesound.FreesoundClient()
        self.client.set_token(api_key, "token")

    def search_sounds(self, query, max_results=5):
        results = self.client.text_search(query=query, fields="id,name,previews")
        sounds = []
        for sound in results:
            sounds.append({
                "id": sound.id,
                "name": sound.name,
                "preview": sound.previews.preview_hq_mp3
            })
        return sounds

class MusicRecommender:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name="music-recommender-v2"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            prefer_grpc=False,
            timeout=30
        )
        self.collection_name = collection_name

    def load_data(self, csv_path: str):
        df = pd.read_csv(csv_path)
        if 'id' not in df.columns:
            df.insert(0, 'id', range(1, len(df) + 1))
        return df

    def add_descriptions(self, df: pd.DataFrame):
        opis_templates = {
            "blues": [
                "Blues pjesma s tužnim melodijama.",
                "Spor ritam s emotivnim gitarskim solažama.",
                "Blues pjesma s melankoličnim vokalom i klasičnom gitarom.",
                "Instrumentalna blues pjesma s duševnim tonovima.",
                "Klasična blues pjesma sa solo gitarom.",
                "Tužna blues pjesma s nježnim tonovima i sporim tempom."
            ],
            "classical": [
                "Klasična instrumentalna glazba s bogatim melodijama.",
                "Orkestralna glazba s elegantnim aranžmanima.",
                "Mirna klasična glazba s violinom.",
                "Simfonijska pjesma s klavirom i gudačima.",
                "Nježna klasična skladba s toplim tonovima.",
                "Melankolična klasična pjesma s tužnim harmonijama."
            ],
            "jazz": [
                "Jazz pjesma s nježnim saksofonom.",
                "Instrumentalna jazz glazba s opuštenom atmosferom.",
                "Jazz s toplim tonovima i bogatom harmonijom.",
                "Improvozirana jazz pjesma s bubnjevima i bas linijom.",
                "Klasična jazz pjesma s pianom i brass sekcijom.",
                "Tužna jazz balada s emotivnim saksofonom."
            ],
            "country": [
                "Country pjesma s akustičnim gitarama.",
                "Vesela country pjesma s pričom u stihovima.",
                "Country pjesma s bendžom i toplim glasom.",
                "Akustični country ritam s narativnim tekstom.",
                "Tradicionalna country pjesma s harmonikom."
            ],
            "disco": [
                "Plesna disco pjesma s brzim ritmom.",
                "Retro disco hit s ritmičnim basom.",
                "Vesela disco pjesma s pjevnim refrenom.",
                "Disco pjesma s klasičnim 80s zvukom.",
                "Brza disco pjesma s funky gitarama."
            ],
            "hiphop": [
                "Trap pjesma s modernim beatovima.",
                "Hip-hop pjesma s ritmičnim beatom.",
                "Moderna hip-hop pjesma s urbanim stilom.",
                "Rap pjesma s izraženim vokalom i beatom.",
                "Hip-hop pjesma s agresivnim flowom."
            ],
            "metal": [
                "Energična metal pjesma s distorzijom.",
                "Intenzivna metal pjesma s žestokim gitarama.",
                "Metal pjesma s brzim bubnjevima i vokalom.",
                "Teška metal pjesma s gitarskim rifovima.",
                "Metal pjesma s agresivnim beatom."
            ],
            "pop": [
                "Vesela pop pjesma s modernom produkcijom.",
                "Pop hit s lako pamtljivim refrenom.",
                "Optimistična pop pjesma s ritmičnim beatom.",
                "Moderna pop pjesma s plesnim ritmom.",
                "Pop pjesma s vedrim tonom i zaraznom melodijom."
            ],
            "reggae": [
                "Reggae pjesma s jamajčinskim ugođajem.",
                "Opustena reggae pjesma s laganim beatom.",
                "Reggae s toplim vokalom i ritmom.",
                "Plesna reggae pjesma s bas linijom.",
                "Reggae pjesma s mirnom atmosferom."
            ],
            "rock": [
                "Energijska rock pjesma s izraženim električnim gitarama.",
                "Rock pjesma s brzim ritmom i bubnjevima.",
                "Rock pjesma s jakim vokalom i rifovima.",
                "Dinamična rock pjesma s distorzijom.",
                "Rock pjesma s klasičnim zvukom gitare."
            ]
        }

        df['description'] = df['label'].apply(
            lambda žanr: random.choice(opis_templates.get(žanr, ["Glazbena pjesma."]))
        )
        df['description_en'] = df['description']
        return df

    def index_embeddings(self, df: pd.DataFrame):
        if 'description' not in df.columns:
            raise KeyError("DataFrame nema stupac 'description'. Prvo pozovi 'add_descriptions'.")
        df['embedding'] = df['description_en'].apply(lambda x: self.model.encode(x).tolist())

        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self.collection_name in collection_names:
            self.qdrant_client.delete_collection(self.collection_name)

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        points = [
            PointStruct(
                id=int(row['id']),
                vector=row['embedding'],
                payload={
                    "filename": row['filename'],
                    "genre": row['label'],
                    "description": row['description'],
                    "filepath": row['filepath']
                }
            )
            for _, row in df.iterrows()
        ]

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Uspješno pohranjeno {len(points)} pjesama u kolekciju.")

    def analyze_sentiment(self, opis_upita, emotions=None):
        if emotions is None:
            emotions = []

        negative_words = ["tužno", "tužan", "melankolično", "žalostan", "tuga", "melankolija"]
        if any(word in opis_upita.lower() for word in negative_words):
            return "negative"

        negative_emojis = ["😢", "😭", "😞"]
        if any(emoji in emotions for emoji in negative_emojis):
            return "negative"

        positive_emojis = ["😊", "😃", "🎉", "🥳", "❤️"]
        if any(emoji in emotions for emoji in positive_emojis):
            return "positive"

        blob = TextBlob(opis_upita)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.1:
            return "positive"
        elif sentiment < -0.1:
            return "negative"
        else:
            return "neutral"


    def recommend(self, opis_upita: str, emotions=None, instruments=None, k: int = 5):
        if emotions is None:
            emotions = []
        if instruments is None:
            instruments = []

        sentiment = self.analyze_sentiment(opis_upita, emotions)
        query_vector = self.model.encode(opis_upita).tolist()
        all_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=50
        )

        recommendations = []
        genre_counts = {}

        for r in all_results:
            genre = r.payload['genre']
            description = r.payload['description']
            score_adjustment = 0

            # Adjust scores based on sentiment
            if sentiment == "positive" and genre in ["pop", "disco", "country"]:
                score_adjustment += 0.8
            elif sentiment == "negative":
                if genre in ["blues", "jazz", "classical"]:
                    score_adjustment += 1.0
                elif genre in ["rock", "metal", "hiphop"]:
                    score_adjustment -= 2.0

            # Adjust scores based on emojis
            for emoji in emotions:
                if emoji in ["😊", "😃", "🥳"] and genre in ["pop", "disco", "rock"]:
                    score_adjustment += 0.15
                elif emoji in ["😢", "😭"] and genre in ["blues", "jazz", "classical"]:
                    score_adjustment += 0.15
                elif emoji in ["🔥", "😎"] and genre in ["rock", "metal"]:
                    score_adjustment += 0.15

            # Adjust scores based on instruments
            for instr in instruments:
                instr_lower = instr.lower()
                if instr_lower in ["guitar", "gitara", "el.gitara"] and genre in ["rock", "metal", "country"]:
                    score_adjustment += 0.15
                elif instr_lower in ["piano", "klavir"] and genre in ["classical", "jazz"]:
                    score_adjustment += 0.15

            total_score = r.score + score_adjustment
            print(f"[DEBUG] Žanr: {genre}, Base: {r.score:.3f}, Adjusted: {total_score:.3f}")

            # Ograniči broj pjesama po žanru na maksimalno 2
            if genre_counts.get(genre, 0) < 2:
                recommendations.append({
                    "filename": r.payload['filename'],
                    "audio_url": f"/audio/{r.payload['filepath']}",
                    "genre": genre,
                    "description": description,
                    "adjusted_score": total_score
                })
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        # Filtriranje za negative sentiment
        if sentiment == "negative":
            recommendations = [
                rec for rec in recommendations
                if rec["genre"] not in ["pop", "disco", "rock", "metal", "hiphop"]
            ]

        # Ako nakon filtriranja nema dovoljno preporuka, vrati originalnu listu
        if sentiment == "negative" and len(recommendations) < 3:
            recommendations = [
                rec for rec in all_results
                if rec.payload['genre'] in ["blues", "jazz", "classical"]
            ]
            # Dodaj scoring i payload format za fallback
            new_recommendations = []
            for r in recommendations:
                new_recommendations.append({
                    "filename": r.payload['filename'],
                    "audio_url": f"/audio/{r.payload['filepath']}",
                    "genre": r.payload['genre'],
                    "description": r.payload['description'],
                    "adjusted_score": r.score  # fallback bez dodatnih prilagodbi
                })
            recommendations = new_recommendations

        # Sortiranje i shuffle
        random.shuffle(recommendations)
        recommendations.sort(key=lambda x: x['adjusted_score'], reverse=True)

        return recommendations[:k]



if __name__ == "__main__":
    recommender = MusicRecommender(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    df = recommender.load_data("data/gtzan_data.csv")
    df = recommender.add_descriptions(df)
    recommender.index_embeddings(df)
    print("Embeddings su uspješno ažurirani!")
