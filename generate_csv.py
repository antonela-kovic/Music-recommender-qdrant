"""Create data/gtzan_data.csv from the GTZAN audio folder (data/genres_original/<genre>/*.wav).

Only needed if you download the audio files yourself; the CSV is already in the repository.
"""
import os

import pandas as pd

ROOT_DIR = os.path.join("data", "genres_original")

rows = []
for genre in sorted(os.listdir(ROOT_DIR)):
    genre_path = os.path.join(ROOT_DIR, genre)
    if not os.path.isdir(genre_path):
        continue
    for file in sorted(os.listdir(genre_path)):
        if file.endswith(".wav"):
            rows.append({
                "id": len(rows) + 1,
                "filename": file,
                "label": genre,
                "filepath": f"{genre}/{file}",  # relative to ROOT_DIR
            })

df = pd.DataFrame(rows)
df.to_csv(os.path.join("data", "gtzan_data.csv"), index=False)
print(f"CSV created with {len(df)} tracks.")
