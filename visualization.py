"""Dataset plots: tracks per genre, audio features per genre and the feature correlation matrix.

Usage: python visualization.py   (saves PNG files to docs/)
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT_DIR = "docs"


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_genre_distribution(tracks):
    fig, ax = plt.subplots(figsize=(9, 4))
    tracks["label"].value_counts().sort_index().plot(kind="bar", color="#4C72B0", ax=ax)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Number of tracks")
    ax.set_title("Tracks per genre (GTZAN)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "genre_distribution.png")


def plot_features_by_genre(features):
    features = features.assign(genre=features["filename"].str.split(".").str[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (column, label) in zip(axes, [
        ("tempo", "Tempo (BPM)"),
        ("rms_mean", "Energy (mean RMS)"),
        ("spectral_centroid_mean", "Brightness (spectral centroid, Hz)"),
    ]):
        order = features.groupby("genre")[column].median().sort_values().index
        sns.boxplot(data=features, x="genre", y=column, order=order, ax=ax, color="#9ecae1", fliersize=2)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Audio features used in the track descriptions", y=1.04)
    save(fig, "features_by_genre.png")


def plot_feature_correlation(features):
    numeric = features.select_dtypes(include="number").drop(columns=["length"], errors="ignore")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation between audio features")
    save(fig, "feature_correlation.png")


if __name__ == "__main__":
    tracks = pd.read_csv(os.path.join("data", "gtzan_data.csv"))
    features = pd.read_csv(os.path.join("data", "features_30_sec.csv"))
    plot_genre_distribution(tracks)
    plot_features_by_genre(features)
    plot_feature_correlation(features)
