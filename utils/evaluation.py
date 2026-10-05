"""
Evaluation Metrics for Recommender Systems

Implements:
    - RMSE / MAE     (rating prediction accuracy, on held-out ratings)
    - Precision@K, Recall@K, NDCG@K (ranking quality, on held-out ratings)
    - Catalog coverage, Novelty
    - A popularity baseline to compare the CF model against

IMPORTANT: ranking metrics are computed on a per-user HOLDOUT split. The model
is trained only on the train portion, and relevant items come only from the
test portion. (Evaluating on the training data is meaningless: recommenders
exclude already-rated movies, so every "relevant" item would be unreachable.)
"""

from typing import Dict, List

import numpy as np
import pandas as pd


# ── Rating Accuracy ──────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def evaluate_cf_surprise(algo, testset) -> Dict[str, float]:
    """Evaluate a fitted Surprise algorithm on a testset."""
    from surprise import accuracy
    predictions = algo.test(testset)
    return {
        "rmse": accuracy.rmse(predictions, verbose=False),
        "mae": accuracy.mae(predictions, verbose=False),
        "n_predictions": len(predictions),
    }


# ── Ranking Metrics ──────────────────────────────────────────────────────────

def precision_at_k(recommended: List[int], relevant, k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / k


def recall_at_k(recommended: List[int], relevant, k: int) -> float:
    if not len(relevant):
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant, k: int) -> float:
    relevant_set = set(relevant)
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant_set)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── Train / test split ───────────────────────────────────────────────────────

def train_test_split_by_user(ratings: pd.DataFrame, test_frac: float = 0.2, seed: int = 42):
    """Hold out `test_frac` of each user's ratings."""
    ratings = ratings.reset_index(drop=True)
    test = ratings.groupby("userId", group_keys=False).sample(frac=test_frac, random_state=seed)
    train = ratings.drop(index=test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _popularity_recs(pop_order: List[int], rated: set, k: int) -> List[int]:
    out = []
    for mid in pop_order:
        if mid not in rated:
            out.append(mid)
            if len(out) == k:
                break
    return out


# ── Full evaluation ──────────────────────────────────────────────────────────

def evaluate_ranking(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = 10,
    n_users: int = 100,
    threshold: float = 4.0,
    test_frac: float = 0.2,
    seed: int = 42,
    max_rmse_samples: int = 20_000,
) -> pd.DataFrame:
    """
    Fit a fresh CF model on the train split and evaluate on the held-out split.
    Compares it with a most-popular baseline.

    Returns one row per model with precision@k, recall@k, ndcg@k, coverage,
    and RMSE/MAE (CF only).
    """
    from models.collaborative import CollaborativeRecommender

    train, test = train_test_split_by_user(ratings, test_frac, seed)

    cf = CollaborativeRecommender()
    cf.fit(train)

    pop_order = train.groupby("movieId").size().sort_values(ascending=False).index.tolist()

    relevant_by_user = (
        test[test["rating"] >= threshold].groupby("userId")["movieId"].apply(list).to_dict()
    )
    eligible = sorted(u for u in relevant_by_user if cf.has_user(u))
    rng = np.random.default_rng(seed)
    users = list(rng.choice(eligible, size=min(n_users, len(eligible)), replace=False)) if eligible else []

    metrics = {"CF": ([], [], [], []), "Popularity": ([], [], [], [])}  # p, r, ndcg, rec lists
    for u in users:
        relevant = relevant_by_user[u]
        cf_ids = cf.recommend(u, movies, n=k)["movieId"].tolist()
        pop_ids = _popularity_recs(pop_order, cf.rated_items(u), k)
        for name, ids in (("CF", cf_ids), ("Popularity", pop_ids)):
            p, r, n_, recs = metrics[name]
            p.append(precision_at_k(ids, relevant, k))
            r.append(recall_at_k(ids, relevant, k))
            n_.append(ndcg_at_k(ids, relevant, k))
            recs.append(ids)

    # RMSE / MAE of the CF model on held-out ratings
    sample = test.sample(min(max_rmse_samples, len(test)), random_state=seed)
    preds = [cf.predict(u, m) for u, m in zip(sample["userId"], sample["movieId"])]
    cf_rmse, cf_mae = rmse(sample["rating"], preds), mae(sample["rating"], preds)

    all_items = movies["movieId"].tolist()
    rows = []
    for name, (p, r, n_, recs) in metrics.items():
        rows.append({
            "model": cf.backend if name == "CF" else "Most popular",
            f"precision@{k}": float(np.mean(p)) if p else 0.0,
            f"recall@{k}": float(np.mean(r)) if r else 0.0,
            f"ndcg@{k}": float(np.mean(n_)) if n_ else 0.0,
            "coverage": catalog_coverage(recs, all_items) if recs else 0.0,
            "rmse": cf_rmse if name == "CF" else np.nan,
            "mae": cf_mae if name == "CF" else np.nan,
            "n_users": len(users),
        })
    return pd.DataFrame(rows)


# ── Catalog Metrics ──────────────────────────────────────────────────────────

def catalog_coverage(recommended_sets: List[List[int]], all_items: List[int]) -> float:
    """Fraction of the catalog appearing in at least one recommendation list."""
    recommended_all = {item for recs in recommended_sets for item in recs}
    return len(recommended_all) / len(all_items)


def novelty(recommended_sets: List[List[int]], item_popularity: pd.Series) -> float:
    """Mean self-information of recommended items (higher = less popular)."""
    total = item_popularity.sum()
    scores = []
    for recs in recommended_sets:
        for item in recs:
            pop = item_popularity.get(item, 1)
            scores.append(-np.log2(pop / total + 1e-10))
    return float(np.mean(scores)) if scores else 0.0


def full_evaluation_report(ratings: pd.DataFrame, movies: pd.DataFrame,
                           k: int = 10, n_users: int = 50) -> pd.DataFrame:
    """Run the holdout evaluation and return a summary DataFrame."""
    return evaluate_ranking(ratings, movies, k=k, n_users=n_users)
