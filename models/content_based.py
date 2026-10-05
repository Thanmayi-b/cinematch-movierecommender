"""
Content-Based Recommender
Uses TF-IDF on genres + tags + title features, plus optional
sentence-transformer embeddings for semantic similarity.
"""

from difflib import get_close_matches

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based movie recommender using TF-IDF on movie metadata.

    Features used:
        - Genres (weighted 3x by repetition)
        - User-provided tags (aggregated per movie)
        - Title keywords (year removed so "same release year" isn't a feature)
    """

    def __init__(self, use_embeddings: bool = False):
        self.use_embeddings = use_embeddings
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=10_000,
            stop_words="english",
        )
        self.tfidf_matrix = None
        self.movie_indices = None  # title -> row index
        self.movies_df = None
        self.embeddings = None

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, movies: pd.DataFrame, tags: pd.DataFrame = None):
        """
        Args:
            movies: DataFrame with columns [movieId, title, genres]
            tags: Optional DataFrame with columns [movieId, tag]
        """
        df = movies.copy().reset_index(drop=True)
        df["genres"] = df["genres"].fillna("")

        # Aggregate tags per movie (simple map; no string-add tricks)
        if tags is not None and len(tags) > 0:
            agg = tags.groupby("movieId")["tag"].apply(
                lambda x: " ".join(x.dropna().astype(str))
            )
            df["tag_text"] = df["movieId"].map(agg).fillna("")
        else:
            df["tag_text"] = ""

        # genres * 3 (upweight) + tags + title keywords
        df["genres_clean"] = (
            df["genres"]
            .str.replace(", ", " ", regex=False)
            .str.replace("|", " ", regex=False)
            .str.replace("-", "", regex=False)
            .str.lower()
        )
        df["title_keywords"] = (
            df["title"]
            .str.replace(r"\(\d{4}\)\s*$", "", regex=True)  # drop the year
            .str.lower()
            .str.replace(r"[^a-z0-9 ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["soup"] = (
            df["genres_clean"] + " " + df["genres_clean"] + " " + df["genres_clean"] + " "
            + df["tag_text"].str.lower() + " "
            + df["title_keywords"]
        )

        self.movies_df = df
        self.tfidf_matrix = self.vectorizer.fit_transform(df["soup"])
        self.movie_indices = pd.Series(df.index, index=df["title"])

        if self.use_embeddings:
            self._fit_embeddings(df)

        print(f"Content model fitted: {self.tfidf_matrix.shape[0]} movies, "
              f"{self.tfidf_matrix.shape[1]} features")

    def _fit_embeddings(self, df: pd.DataFrame):
        """Optional: sentence-transformer embeddings on title + genres."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = df["title_keywords"] + " " + df["genres_clean"]
            self.embeddings = model.encode(texts.tolist(), show_progress_bar=True, batch_size=256)
            print("Sentence embeddings computed.")
        except ImportError:
            print("sentence-transformers not installed. Using TF-IDF only.")
            self.embeddings = None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _resolve_index(self, movie_title: str) -> int:
        """Title -> row index (handles fuzzy match and duplicate titles)."""
        if movie_title not in self.movie_indices.index:
            matches = get_close_matches(
                movie_title, self.movie_indices.index.tolist(), n=1, cutoff=0.5
            )
            if not matches:
                raise ValueError(f"Movie '{movie_title}' not found in dataset.")
            movie_title = matches[0]
        idx = self.movie_indices[movie_title]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        return int(idx)

    # ── Recommendation ──────────────────────────────────────────────────────

    def recommend(self, movie_title: str, n: int = 10) -> pd.DataFrame:
        """
        Return top-n movies most similar to movie_title.

        Returns:
            DataFrame with [movieId, title, genres, score]
        """
        idx = self._resolve_index(movie_title)

        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        if self.embeddings is not None:
            emb_scores = cosine_similarity(
                self.embeddings[idx].reshape(1, -1), self.embeddings
            ).flatten()
            scores = 0.5 * scores + 0.5 * emb_scores

        scores[idx] = -1  # exclude the query movie itself

        top_indices = np.argsort(scores)[::-1][:n]
        result = self.movies_df.iloc[top_indices][["movieId", "title", "genres"]].copy()
        result["score"] = scores[top_indices]
        return result.reset_index(drop=True)

    def get_movie_vector(self, movie_title: str):
        """Return the TF-IDF feature vector for a movie."""
        return self.tfidf_matrix[self._resolve_index(movie_title)]

    def get_similarity_scores(self, movie_title: str) -> pd.Series:
        """Return cosine similarity scores for all movies vs. the query."""
        idx = self._resolve_index(movie_title)
        scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        return pd.Series(scores, index=self.movies_df["movieId"])
