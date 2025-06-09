import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Distribucija pjesama po žanrovima (dataset)
def prikazi_distribuciju(df):
    genre_counts = df['label'].value_counts()

    plt.figure(figsize=(10, 6))
    genre_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Žanrovi')
    plt.ylabel('Broj pjesama')
    plt.title('Broj pjesama po žanrovima u kolekciji')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 2. Korelacija značajki
def prikazi_korelaciju(df):
    plt.figure(figsize=(12, 10))
    # Ovdje filtriramo samo numeričke podatke
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Korelacija između značajki')
    plt.tight_layout()
    plt.show()


# 3. Distribucija sentimenta iz liste
def prikazi_sentiment_distribuciju(sentiments):
    sentiment_counts = pd.Series(sentiments).value_counts()
    plt.figure(figsize=(6, 4))
    sentiment_counts.plot(kind='bar', color=['lightgreen', 'lightcoral', 'lightgray'])
    plt.xlabel('Sentiment')
    plt.ylabel('Broj pojavljivanja')
    plt.title('Distribucija sentimenta')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 4. Prikaz distribucije preporučenih žanrova
def prikazi_zanrove(recommendations):
    genre_counts = {}
    for rec in recommendations:
        genre = rec['genre']
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())

    plt.figure(figsize=(8, 7))
    plt.bar(genres, counts, color='lightblue')
    plt.xlabel('Žanrovi')
    plt.ylabel('Broj pjesama')
    plt.title('Raspodjela preporučenih žanrova')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Glavni dio za testiranje
if __name__ == "__main__":
    # 1. Distribucija pjesama po žanrovima (gtzan_data)
    df = pd.read_csv("data/gtzan_data.csv")
    prikazi_distribuciju(df)

    # 2. Korelacija značajki (features_30_sec)
    features_df = pd.read_csv("data/features_30_sec.csv")
    prikazi_korelaciju(features_df)

    # 3. Distribucija sentimenta za više unosa (test lista)
    test_sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']
    prikazi_sentiment_distribuciju(test_sentiments)

    # 4. Prikaz preporučenih žanrova (test lista)
    test_recommendations = [
        {"genre": "pop"},
        {"genre": "pop"},
        {"genre": "jazz"},
        {"genre": "classical"},
        {"genre": "jazz"},
    ]
    prikazi_zanrove(test_recommendations)
