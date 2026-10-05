"""
Exploratory Data Analysis + Model Training Script
Run this first to understand the dataset, then use app.py for the UI.

Usage (from the project root):
    python notebooks/eda_and_train.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import matplotlib
if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    matplotlib.use("Agg")  # headless servers/CI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.collaborative import CollaborativeRecommender
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender
from utils.data_loader import load_data
from utils.evaluation import full_evaluation_report

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 5)

# 1. LOAD DATA
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

movies, ratings, tags = load_data()
print(f"\nMovies shape:  {movies.shape}")
print(f"Ratings shape: {ratings.shape}")
print(f"Tags shape:    {tags.shape}")

# 2. EDA
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

print(f"\nRating distribution:\n{ratings['rating'].value_counts().sort_index()}")
print(f"\nMean rating: {ratings['rating'].mean():.2f}")
print(f"Ratings per user (mean): {ratings.groupby('userId').size().mean():.1f}")
print(f"Ratings per movie (mean): {ratings.groupby('movieId').size().mean():.1f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=axes[0], color="#E50914")
axes[0].set_title("Rating Distribution")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")

user_counts = ratings.groupby("userId").size()
axes[1].hist(user_counts, bins=50, color="#0074D9", edgecolor="white")
axes[1].set_yscale("log")
axes[1].set_title("Ratings per User (log scale)")
axes[1].set_xlabel("# Ratings")

genre_counts = (
    movies["genres"].dropna().str.split(", ").explode().str.strip()
    .loc[lambda s: s != ""]  # drop movies with no genres listed
    .value_counts().head(12)
)
genre_counts.plot(kind="barh", ax=axes[2], color="#2ECC40")
axes[2].set_title("Top 12 Genres")
axes[2].invert_yaxis()

plt.tight_layout()
plot_path = os.path.join(ROOT, "eda_overview.png")  # same place regardless of cwd
plt.savefig(plot_path, dpi=120)
plt.close(fig)
print(f"\nEDA plot saved to {plot_path}")

# 3. CONTENT-BASED MODEL
print("\n" + "=" * 60)
print("STEP 3: Training Content-Based Model")
print("=" * 60)

cb = ContentBasedRecommender(use_embeddings=False)
cb.fit(movies, tags)

test_movie = "Toy Story (1995)"
print(f"\nContent-based recommendations for: '{test_movie}'")
print(cb.recommend(test_movie, n=5)[["title", "genres", "score"]].to_string(index=False))

# 4. COLLABORATIVE FILTERING MODEL
print("\n" + "=" * 60)
print("STEP 4: Training Collaborative Filtering Model (SVD)")
print("=" * 60)

cf = CollaborativeRecommender(n_factors=100, n_epochs=20)
cf.fit(ratings)
print(f"CF backend: {cf.backend}")

test_user = ratings["userId"].value_counts().index[0]  # most active user
print(f"\nCF recommendations for user {test_user}:")
print(cf.recommend(test_user, movies, n=5)[["title", "genres", "score"]].to_string(index=False))

# 5. HYBRID
print("\n" + "=" * 60)
print("STEP 5: Hybrid Recommendations")
print("=" * 60)

hybrid = HybridRecommender(cb, cf)
print(f"\nHybrid recommendations for user {test_user}, seeded by '{test_movie}':")
h_recs = hybrid.recommend(test_user, test_movie, movies, n=5, cb_weight=0.4)
print(h_recs[["title", "genres", "cb_score", "cf_score", "hybrid_score"]].to_string(index=False))

# 6. EVALUATION (held-out ratings; SVD vs. most-popular baseline)
print("\n" + "=" * 60)
print("STEP 6: Evaluation (80/20 per-user holdout)")
print("=" * 60)

try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    algo = SVD(n_factors=100, n_epochs=20, random_state=SEED, verbose=False)

    print("\nRunning 3-fold cross-validation (this takes ~1 min)...")
    cv = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=3, verbose=False)
    print(f"  RMSE: {cv['test_rmse'].mean():.4f} ± {cv['test_rmse'].std():.4f}")
    print(f"  MAE:  {cv['test_mae'].mean():.4f} ± {cv['test_mae'].std():.4f}")
except ImportError:
    print("Install `scikit-surprise` for cross-validation: pip install scikit-surprise")

print("\nRanking metrics (Precision/Recall/NDCG@10) on held-out ratings...")
eval_df = full_evaluation_report(ratings, movies, k=10, n_users=100)
print(eval_df.round(4).to_string(index=False))

# 7. COLD START
print("\n" + "=" * 60)
print("STEP 7: Cold Start (movies with zero ratings)")
print("=" * 60)

cs = hybrid.evaluate_cold_start(movies, ratings, n_test=30)
print(f"\nZero-rating movies in dataset: {cs['pool_size']}")
print(f"Content-based returned recommendations for {cs['success']}/{cs['tested']} "
      f"({cs['coverage']:.0%}); mean genre overlap with seed (Jaccard): {cs['mean_genre_jaccard']:.2f}")
print("CF has no signal for these movies, so content-based is the only option for them.")
print("\nAll done! Run `streamlit run app.py` to launch the UI.")
