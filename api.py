"""Flask web app and REST API for the music recommender."""
import os

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from recommender import MusicRecommender, load_dataset, make_client

load_dotenv()

AUDIO_DIR = os.path.join("data", "genres_original")  # optional: GTZAN audio files

app = Flask(__name__)
recommender = MusicRecommender(make_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")))


def ensure_index():
    """Index the dataset on the first start (or when the collection is empty)."""
    if not recommender.is_indexed():
        count = recommender.index(load_dataset())
        app.logger.info("Indexed %d tracks", count)


@app.route("/")
def index():
    return render_template("index.html", audio_available=os.path.isdir(AUDIO_DIR))


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True) or {}
    query = str(data.get("description", "")).strip()
    if not query:
        return jsonify({"error": "Please enter a description."}), 400

    emotions = data.get("emotions") or []
    instruments = data.get("instruments") or []
    if not isinstance(emotions, list) or not isinstance(instruments, list):
        return jsonify({"error": "emotions and instruments must be lists."}), 400

    k = min(max(int(data.get("k", 5)), 1), 20)
    return jsonify(recommender.recommend(query, emotions=emotions, instruments=instruments, k=k))


@app.route("/reindex", methods=["POST"])
def reindex():
    count = recommender.index(load_dataset())
    return jsonify({"message": f"Indexed {count} tracks."})


@app.route("/audio/<path:filepath>")
def serve_audio(filepath):
    if not os.path.isdir(AUDIO_DIR):
        abort(404)
    return send_from_directory(AUDIO_DIR, filepath)  # send_from_directory blocks path traversal


if __name__ == "__main__":
    ensure_index()
    app.run(debug=False)
