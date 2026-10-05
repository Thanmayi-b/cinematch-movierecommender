"""
Hybrid Recommender
Blends Content-Based and Collaborative Filtering scores.

    final = cb_weight * cb_score + (1 - cb_weight) * cf_score
"""

import numpy as np
import pandas as pd

from models.collaborative import CollaborativeRecommender
from models.content_based import ContentBasedRecommender


def _minmax(values: np.ndarray) -> np.ndarray:
    """Min-max scale to [0,1]; a constant vector maps to 0.5 (no information)."""
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-12:
        return np.full_like(values, 0.5)
    return (values - lo) / (hi - lo)


class HybridRecommender:
    """Hybrid recommender combining content-based and collaborative filtering."""

    def __init__(self, content_recommender: ContentBasedRecommender,
                 cf_recommender: CollaborativeRecommender):
        self.content_recommender = content_recommender
        self.cf_recommender = cf_recommender

    def recommend(
        self,
        user_id: int,
        movie_title: str,
        movies: pd.DataFrame,
        n: int = 10,
        cb_weight: float = 0.4,
        candidate_pool: int = 100,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """
        1. Take the top `candidate_pool` movies by content similarity to the seed.
        2. Drop movies the user already rated (if exclude_rated).
        3. Get the CF predicted rating for each remaining candidate.
        4. Min-max both scores to [0,1] and blend with cb_weight.

        Returns:
            DataFrame with [movieId, title, genres, cb_score, cf_score, hybrid_score, score]
        """
        cb_weight = float(np.clip(cb_weight, 0.0, 1.0))

        try:
            cb_recs = self.content_recommender.recommend(movie_title, n=candidate_pool)
        except ValueError:
            return self.cf_recommender.recommend(user_id, movies, n=n)

        if exclude_rated:
            seen = self.cf_recommender.rated_items(user_id)
            cb_recs = cb_recs[~cb_recs["movieId"].isin(seen)]

        if len(cb_recs) == 0:
            return self.cf_recommender.recommend(user_id, movies, n=n)

        cb_recs = cb_recs.copy()
        cb_recs["cf_score_raw"] = [
            self.cf_recommender.predict(user_id, mid) for mid in cb_recs["movieId"]
        ]
        cb_recs["cb_score"] = _minmax(cb_recs["score"].values)
        cb_recs["cf_score"] = _minmax(cb_recs["cf_score_raw"].values)
        cb_recs["hybrid_score"] = (
            cb_weight * cb_recs["cb_score"] + (1 - cb_weight) * cb_recs["cf_score"]
        )

        result = (
            cb_recs[["movieId", "title", "genres", "cb_score", "cf_score", "hybrid_score"]]
            .sort_values("hybrid_score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        result["score"] = result["hybrid_score"]  # UI compatibility
        return result

    def evaluate_cold_start(self, movies: pd.DataFrame, ratings: pd.DataFrame,
                            n_test: int = 20, max_ratings: int = 0,
                            seed: int = 42) -> dict:
        """
        Cold-start check on movies with <= `max_ratings` ratings (default: zero).
        CF has nothing to learn from these; we check content-based still returns
        recommendations and report the mean genre overlap (Jaccard) with the seed.
        """
        counts = ratings.groupby("movieId").size()
        movie_counts = movies["movieId"].map(counts).fillna(0)
        pool = movies[movie_counts <= max_ratings]
        if len(pool) == 0:
            return {"tested": 0, "success": 0, "coverage": 0.0,
                    "mean_genre_jaccard": 0.0, "pool_size": 0}

        sample = pool.sample(min(n_test, len(pool)), random_state=seed)
        successes, jaccards = 0, []
        for _, row in sample.iterrows():
            try:
                recs = self.content_recommender.recommend(row["title"], n=5)
            except Exception:
                continue
            if len(recs) == 0:
                continue
            successes += 1
            seed_g = {g.strip() for g in str(row["genres"]).split(",") if g.strip()}
            for g in recs["genres"]:
                rec_g = {x.strip() for x in str(g).split(",") if x.strip()}
                union = seed_g | rec_g
                jaccards.append(len(seed_g & rec_g) / len(union) if union else 0.0)

        return {
            "pool_size": int(len(pool)),
            "tested": int(len(sample)),
            "success": successes,
            "coverage": successes / len(sample),
            "mean_genre_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        }
