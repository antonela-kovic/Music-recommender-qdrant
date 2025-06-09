# Content Recommendation System with Qdrant

Ovaj projekt omogućava korisnicima da dobiju preporuke pjesama na temelju opisa, emotikona i instrumenata. Također koristi sentiment analizu kako bi prilagodio preporuke raspoloženju korisnika.

## Značajke
- Preporuka pjesama na temelju korisničkog unosa (tekstualni opis, emotikoni, instrumenti).
- Sentiment analiza korisničkog opisa.
- Vizualizacija distribucije žanrova u datasetu i prikaz sentimenta unosa.
- Podrška za tamni i svijetli način rada.

## Korišteni podaci
Projekt koristi **GTZAN dataset** s **Kaggle-a**, koji sadrži pjesme podijeljene u 10 žanrova. Svaka pjesma traje 30 sekundi. Dataset uključuje CSV datoteke s metapodacima:
- `gtzan_data.csv` — metapodaci pjesama
- `features_30_sec.csv` — značajke pjesama (30 sekundi)
- `features_3_sec.csv` — značajke pjesama (3 sekunde)

## Tehnologije
- Python 3.12
- Flask (za backend)
- Qdrant (vektorska baza podataka za pretraživanje)
- SentenceTransformer (generiranje embeddingsa)
- Matplotlib (vizualizacije)
- Bootstrap (UI)
- JavaScript (frontend)

## Pokretanje projekta

1. Kloniraj repozitorij:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   

2. Instaliraj virtualno okruženje i pokreni ga:

   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
```bash
   source venv/bin/activate  # macOS/Linux
   

3. Instaliraj ovisnosti:

   ```bash
   pip install -r requirements.txt
   

4. Postavi varijable okruženja u `.env` datoteku:
 ```bash
   env
   QDRANT_URL=<tvoj_qdrant_url>
   QDRANT_API_KEY=<tvoj_qdrant_api_key>
   

5. Pokreni aplikaciju:

   ```bash
   python index_data.py
   

6. Posjeti aplikaciju na `http://127.0.0.1:5000`.

## Vizualizacija

Za prikaz vizualizacija (distribucija žanrova, sentiment):

```bash
python visualization.py


## Korištenje

* Otvori aplikaciju u pregledniku.
* Unesi opis pjesme (npr. "Želim tužnu pjesmu koja me smiruje").
* Dodaj emotikone i/ili instrumente ako želiš.
* Klikni "Pretraži" za preporuke.
* Klikni gumb za tamni/svijetli način rada za prilagodbu teme.

## Struktura projekta

```
projekat/
│
├── data/
│   ├── gtzan_data.csv
│   ├── features_30_sec.csv
│   └── features_3_sec.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── recommender.py
├── index_data.py
├── visualization.py
├── requirements.txt
└── README.md
```


