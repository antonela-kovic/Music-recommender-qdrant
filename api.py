# api.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from recommender import MusicRecommender, AudioDownloader
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
API_KEY = os.getenv("FREESOUND_API_KEY")

# Inicijaliziraj recommender
recommender = MusicRecommender(QDRANT_URL, QDRANT_API_KEY)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    opis = data.get('opis', '').strip()
    emotions = data.get('emotions', [])  # Očekujemo listu emojija ili tekstualnih oznaka
    instruments = data.get('instruments', [])  # Očekujemo listu instrumenata (npr. 'guitar', 'piano', ...)

    if not opis:
        return jsonify({"error": "Opis nije unesen."}), 400

    recommendations = recommender.recommend(opis, emotions=emotions, instruments=instruments, k=5)
    app.logger.info(f"Primljeni opis: {opis}")
    app.logger.info(f"Emocije: {emotions}")
    app.logger.info(f"Instrumenti: {instruments}")
    app.logger.info(f"Preporuke: {recommendations}")
    return jsonify({"recommendations": recommendations})



@app.route('/')
def index():
    return render_template("index.html")

downloader = AudioDownloader(API_KEY)

@app.route('/search', methods=['POST'])
def search_audio():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "Nema upita"}), 400

    sounds = downloader.search_sounds(query)
    return jsonify({"sounds": sounds})

@app.route('/reindex', methods=['POST'])
def reindex():
    df = recommender.load_data("data/gtzan_data.csv")
    df = recommender.add_descriptions(df)
    recommender.index_embeddings(df)
    app.logger.info("Embeddings uspješno ažurirani!")
    return jsonify({"message": "Embeddings su uspješno ažurirani!"}), 200

@app.route('/audio/<path:filepath>')
def serve_audio(filepath):
    base_dir = os.path.join('data', 'genres_original')
    return send_from_directory(base_dir, filepath)

if __name__ == '__main__':
    app.run(debug=False)  # production mode: debug=False
