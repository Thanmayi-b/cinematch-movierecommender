"""
Collaborative Filtering Recommender
Uses Singular Value Decomposition (SVD) via the `surprise` library.
Falls back to a simple user/item bias model if surprise is not installed.
"""

import pickle

import numpy as np
import pandas as pd


class CollaborativeRecommender:
    """
    Collaborative filtering via matrix factorization (SVD).
    Trains on explicit user-item ratings; predicts ratings for unseen pairs.
    """

    def __init__(self, n_factors: int = 100, n_epochs: int = 20,
                 lr_all: float = 0.005, reg_all: float = 0.02,
                 random_state: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.algo = None
        self.trainset = None
        self.all_movie_ids = None
        self._rated = {}
        self._use_surprise = True

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, ratings: pd.DataFrame):
        """ratings: DataFrame with columns [userId, movieId, rating]"""
        self.all_movie_ids = ratings["movieId"].unique()
        # Single source of truth for "already rated" (both backends use it)
        self._rated = ratings.groupby("userId")["movieId"].apply(set).to_dict()

        try:
            from surprise import Dataset, Reader, SVD

            reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
            data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
            self.trainset = data.build_full_trainset()

            self.algo = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                random_state=self.random_state,
                verbose=False,
            )
            self.algo.fit(self.trainset)
            self._use_surprise = True
            print("CF model (SVD) fitted via surprise.")

        except ImportError:
            print("WARNING: scikit-surprise not installed. Using a bias-only baseline, "
                  "NOT SVD. Don't report these results as SVD.")
            self._fit_fallback(ratings)

    def _fit_fallback(self, ratings: pd.DataFrame):
        """Global mean + user bias + item bias."""
        self._use_surprise = False
        self._global_mean = ratings["rating"].mean()
        self._user_bias = ratings.groupby("userId")["rating"].mean() - self._global_mean
        self._item_bias = ratings.groupby("movieId")["rating"].mean() - self._global_mean

    @property
    def backend(self) -> str:
        return "SVD (surprise)" if self._use_surprise else "bias baseline (fallback)"

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for a (user, movie) pair."""
        if self._use_surprise:
            return self.algo.predict(user_id, movie_id).est
        u_bias = self._user_bias.get(user_id, 0)
        i_bias = self._item_bias.get(movie_id, 0)
        return float(np.clip(self._global_mean + u_bias + i_bias, 0.5, 5.0))

    def has_user(self, user_id: int) -> bool:
        return user_id in self._rated

    def rated_items(self, user_id: int) -> set:
        return self._rated.get(user_id, set())

    # ── Recommendation ──────────────────────────────────────────────────────

    def recommend(self, user_id: int, movies: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Top-n predicted movies for a user, excluding movies they already rated.

        Returns:
            DataFrame with [movieId, title, genres, score]
        """
        rated = self.rated_items(user_id)
        candidate_ids = [m for m in self.all_movie_ids if m not in rated]

        scores = [(mid, self.predict(user_id, mid)) for mid in candidate_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:n]
        top_scores = {mid: s for mid, s in top}

        result = movies[movies["movieId"].isin(top_scores)][["movieId", "title", "genres"]].copy()
        result["score"] = result["movieId"].map(top_scores)
        return result.sort_values("score", ascending=False).reset_index(drop=True)

    def get_all_predictions(self, user_id: int) -> pd.Series:
        """Predicted ratings for all movies, keyed by movieId."""
        return pd.Series({mid: self.predict(user_id, mid) for mid in self.all_movie_ids})

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"CF model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CollaborativeRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)
