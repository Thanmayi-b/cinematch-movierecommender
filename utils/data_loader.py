"""
Data loading utilities for MovieLens dataset.
Downloads the small dataset automatically if not present.
"""

import os
import zipfile
import urllib.request

import pandas as pd

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "ml-latest-small")
REQUIRED_FILES = ("movies.csv", "ratings.csv", "tags.csv")


def _dataset_ready() -> bool:
    return all(os.path.exists(os.path.join(EXTRACT_DIR, f)) for f in REQUIRED_FILES)


def download_movielens():
    """Download MovieLens small dataset if not already (fully) present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if _dataset_ready():
        return
    print("Downloading MovieLens dataset...")
    try:
        urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
    finally:
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)  # also removes a corrupt partial download
    print("Dataset downloaded and extracted.")


def load_data():
    """Load and return (movies, ratings, tags) DataFrames."""
    download_movielens()

    movies = pd.read_csv(os.path.join(EXTRACT_DIR, "movies.csv"))
    ratings = pd.read_csv(os.path.join(EXTRACT_DIR, "ratings.csv"))
    tags = pd.read_csv(os.path.join(EXTRACT_DIR, "tags.csv"))

    movies["title"] = movies["title"].str.strip()
    movies["genres"] = movies["genres"].str.replace("|", ", ", regex=False)
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("float")
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)

    print(f"Loaded: {len(movies)} movies, {len(ratings)} ratings, {len(tags)} tags")
    return movies, ratings, tags


def get_movie_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Merge movies with aggregated rating stats."""
    stats = ratings.groupby("movieId").agg(
        rating_count=("rating", "count"),
        rating_mean=("rating", "mean"),
    ).reset_index()
    return movies.merge(stats, on="movieId", how="left")
