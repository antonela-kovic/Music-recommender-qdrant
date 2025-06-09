import os
import pandas as pd

root_dir = 'data/genres_original'  # Gdje su svi žanrovi

data = []
for genre in os.listdir(root_dir):
    genre_path = os.path.join(root_dir, genre)
    if os.path.isdir(genre_path):
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                data.append({
                    'id': len(data) + 1,
                    'filename': file,
                    'label': genre,
                    'filepath': f"{genre}/{file}"  # Relativna putanja
                })

df = pd.DataFrame(data)
df.to_csv('data/gtzan_data.csv', index=False)
print(f"CSV generiran s {len(df)} pjesama!")
