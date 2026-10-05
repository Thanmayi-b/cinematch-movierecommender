"""
CineMatch - Netflix x Spotify Aesthetic Movie Recommender
"""
import html
import os
import re
from difflib import get_close_matches

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from models.content_based import ContentBasedRecommender
from models.collaborative import CollaborativeRecommender
from models.hybrid import HybridRecommender
from utils.data_loader import load_data
from utils.tmdb import get_movie_details

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0a !important; color: #fff; }
.stApp { background-color: #0a0a0a !important; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.hero { padding: 70px 60px 50px; background: #111; border-bottom: 1px solid #1f1f1f; }
.hero-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #1DB954; margin-bottom: 12px; }
.hero-title { font-size: 64px; font-weight: 900; letter-spacing: -3px; line-height: 1; color: #fff; margin-bottom: 10px; }
.hero-title span { color: #E50914; }
.hero-subtitle { font-size: 16px; color: #666; margin-bottom: 0; }
div[data-testid="stForm"] { border: none !important; padding: 0 60px !important; }
div[data-testid="stTextInput"] input {
    background: #1a1a1a !important; border: 1px solid #2a2a2a !important;
    border-radius: 50px !important; color: #fff !important;
    font-size: 16px !important; font-family: 'Inter', sans-serif !important;
    padding: 14px 24px !important;
}
div[data-testid="stTextInput"] input:focus { border-color: #E50914 !important; box-shadow: none !important; }
div[data-testid="stTextInput"] input::placeholder { color: #555 !important; }
div[data-testid="stTextInput"] > div > div { background: transparent !important; border: none !important; }
div[data-testid="stButton"] button,
div[data-testid="stFormSubmitButton"] button {
    background: #E50914 !important; color: #fff !important; border: none !important;
    border-radius: 50px !important; padding: 14px 32px !important;
    font-weight: 700 !important; font-size: 15px !important;
    font-family: 'Inter', sans-serif !important; width: 100% !important;
}
div[data-testid="stButton"] button:hover,
div[data-testid="stFormSubmitButton"] button:hover { background: #b20710 !important; }
.divider { border: none; border-top: 1px solid #1a1a1a; margin: 0; }
</style>
""", unsafe_allow_html=True)

CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: #0a0a0a; color: #fff; padding: 0 60px 60px; }
.section-label { font-size: 11px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #555; margin-bottom: 10px; padding-top: 40px; }
.section-heading { font-size: 28px; font-weight: 900; color: #fff; letter-spacing: -0.5px; margin-bottom: 28px; }
.section-heading span { color: #1DB954; }
.grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }
.card { background: #111; border-radius: 12px; overflow: hidden; border: 1px solid #1a1a1a; transition: transform 0.2s; cursor: pointer; }
.card:hover { transform: translateY(-4px); border-color: #2a2a2a; }
.poster-wrap { position: relative; width: 100%; aspect-ratio: 2/3; background: #1a1a1a; display: flex; align-items: center; justify-content: center; }
.poster-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.poster-overlay { position: absolute; bottom: 0; left: 0; right: 0; height: 60%; background: linear-gradient(transparent, rgba(0,0,0,0.95)); pointer-events: none; }
.match-badge { position: absolute; top: 10px; right: 10px; background: #1DB954; color: #000; font-size: 11px; font-weight: 800; padding: 4px 8px; border-radius: 6px; }
.no-poster-letter { font-size: 56px; font-weight: 900; color: #2a2a2a; }
.card-body { padding: 12px 14px 14px; }
.card-title { font-size: 13px; font-weight: 700; color: #fff; margin-bottom: 4px; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.card-meta { font-size: 11px; color: #555; margin-bottom: 4px; }
.card-genres { font-size: 11px; color: #444; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.seed-section { padding: 40px 60px 0; }
.seed-label { font-size: 11px; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #555; margin-bottom: 16px; }
.seed-card { display: flex; gap: 24px; background: #111; border: 1px solid #1f1f1f; border-radius: 16px; padding: 24px; max-width: 680px; }
.seed-poster { width: 100px; border-radius: 10px; flex-shrink: 0; object-fit: cover; }
.seed-title { font-size: 26px; font-weight: 900; color: #fff; letter-spacing: -0.5px; margin-bottom: 6px; }
.seed-meta { font-size: 13px; color: #555; margin-bottom: 10px; }
.genre-pill { display: inline-block; background: #1f1f1f; color: #888; font-size: 11px; font-weight: 600; padding: 4px 12px; border-radius: 20px; margin: 0 4px 6px 0; }
.seed-overview { font-size: 13px; color: #777; line-height: 1.6; margin-top: 10px; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.divider { border: none; border-top: 1px solid #1a1a1a; margin: 32px 0 0; }
.not-found { padding: 80px 60px; text-align: center; }
.not-found-title { font-size: 24px; font-weight: 800; color: #fff; margin-bottom: 10px; }
.not-found-sub { font-size: 15px; color: #555; }
</style>
"""

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
YEAR_SUFFIX_RE = re.compile(r"\s*\(\d{4}\)\s*$")
TRAILING_ARTICLE_RE = re.compile(r"^(.*),\s*(The|A|An)$", re.IGNORECASE)


def strip_year(title: str) -> str:
    return YEAR_SUFFIX_RE.sub("", str(title)).strip()


def normalize_title(title: str) -> str:
    """Lowercase, drop the year, and un-flip MovieLens-style titles
    ("Dark Knight, The (2008)" -> "the dark knight")."""
    t = strip_year(title)
    m = TRAILING_ARTICLE_RE.match(t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return t.lower().strip()


def fmt_year(year) -> str:
    try:
        return str(int(year))
    except (TypeError, ValueError):
        return ""


def split_genres(genres) -> list:
    """MovieLens uses '|' as the separator; accept ',' too."""
    return [g.strip() for g in re.split(r"[|,]", str(genres)) if g.strip()]


def safe_details(title: str, year, api_key: str) -> dict:
    """TMDB lookup that never takes the page down if the API fails."""
    if not api_key:
        return {}
    try:
        return get_movie_details(title, year, api_key) or {}
    except Exception:
        return {}


@st.cache_resource(show_spinner="Loading models — first run takes ~1 min...")
def load_models():
    movies, ratings, tags = load_data()

    cb = ContentBasedRecommender()
    cb.fit(movies, tags)

    cf = CollaborativeRecommender()
    cf.fit(ratings)

    hybrid = HybridRecommender(cb, cf)

    # Search index (kept separate so the models' input data is untouched)
    search_movies = movies.copy()
    search_movies["_norm"] = search_movies["title"].map(normalize_title)
    counts = ratings.groupby("movieId").size()
    search_movies["_pop"] = search_movies["movieId"].map(counts).fillna(0)
    return hybrid, search_movies, set(ratings["userId"].unique().tolist())


def find_movie(query: str, movies: pd.DataFrame):
    """Exact -> substring -> fuzzy match, preferring the most-rated title."""
    q = normalize_title(query)
    if not q:
        return None

    def most_popular(df):
        return df.sort_values("_pop", ascending=False).iloc[0]

    exact = movies[movies["_norm"] == q]
    if not exact.empty:
        return most_popular(exact)

    # regex=False: titles/queries contain "(", "?", "." etc.
    contains = movies[movies["_norm"].str.contains(q, regex=False, na=False)]
    if not contains.empty:
        return most_popular(contains)

    match = get_close_matches(q, movies["_norm"].tolist(), n=1, cutoff=0.5)
    if match:
        return most_popular(movies[movies["_norm"] == match[0]])
    return None


def build_seed_html(movie_row, api_key: str) -> str:
    title = str(movie_row["title"])
    year = movie_row.get("year", None)
    details = safe_details(title, year, api_key)

    poster = details.get("poster_url") or ""
    overview = details.get("overview") or "No overview available."
    tmdb_rating = details.get("tmdb_rating") or 0
    release_date = details.get("release_date") or ""
    release_year = release_date[:4] if release_date else fmt_year(year)

    genre_pills = "".join(
        f'<span class="genre-pill">{html.escape(g)}</span>'
        for g in split_genres(movie_row.get("genres", ""))[:5]
    )
    poster_img = (
        f'<img class="seed-poster" src="{html.escape(poster, quote=True)}" />' if poster else ""
    )
    rating_str = f"⭐ {round(tmdb_rating, 1)} &nbsp;·&nbsp; " if tmdb_rating else ""

    return f"""
    <div class="seed-section">
        <div class="seed-label">You searched for</div>
        <div class="seed-card">
            {poster_img}
            <div>
                <div class="seed-title">{html.escape(title)}</div>
                <div class="seed-meta">{rating_str}{release_year}</div>
                <div>{genre_pills}</div>
                <div class="seed-overview">{html.escape(overview)}</div>
            </div>
        </div>
    </div>
    <div class="divider"></div>
    """


def build_grid_html(recs: pd.DataFrame, seed_title: str, api_key: str, user_id=None) -> str:
    cards = ""
    for _, row in recs.iterrows():
        title = str(row.get("title", "Unknown"))
        score = float(row.get("score", row.get("hybrid_score", 0)) or 0)
        genres = ", ".join(split_genres(row.get("genres", "")))
        year = row.get("year", None)

        details = safe_details(title, year, api_key)
        poster = details.get("poster_url") or ""
        tmdb_rating = details.get("tmdb_rating") or 0
        release_year = fmt_year(year)
        match_pct = max(0, min(100, int(score * 100)))
        rating_str = f"⭐ {round(tmdb_rating, 1)} &nbsp;·&nbsp; " if tmdb_rating else ""

        if poster:
            poster_html = f"""
            <div class="poster-wrap">
                <img src="{html.escape(poster, quote=True)}" />
                <div class="poster-overlay"></div>
                <div class="match-badge">{match_pct}%</div>
            </div>"""
        else:
            clean = strip_year(title)
            initial = html.escape(clean[0].upper()) if clean else "?"
            poster_html = f"""
            <div class="poster-wrap">
                <span class="no-poster-letter">{initial}</span>
                <div class="match-badge">{match_pct}%</div>
            </div>"""

        cards += f"""
        <div class="card">
            {poster_html}
            <div class="card-body">
                <div class="card-title">{html.escape(title)}</div>
                <div class="card-meta">{rating_str}{release_year}</div>
                <div class="card-genres">{html.escape(genres)}</div>
            </div>
        </div>"""

    clean_seed = html.escape(strip_year(seed_title))
    label = "Because you watched"
    if user_id:
        label += f" &nbsp;·&nbsp; personalized for MovieLens user {int(user_id)}"
    return f"""
    <div class="section-label">{label}</div>
    <div class="section-heading">More like <span>{clean_seed}</span></div>
    <div class="grid">{cards}</div>
    """


def main():
    hybrid, movies, user_ids = load_models()

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Cine<span>Match</span></div>
        <div class="hero-subtitle">Type a movie you love, we'll find what to watch next.</div>
    </div>
    """, unsafe_allow_html=True)

    # st.form so pressing Enter in the text box submits the search too
    with st.form("search_form", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            query = st.text_input(
                "Movie title",
                placeholder="Search a movie (e.g. Inception, The Dark Knight...)",
                label_visibility="collapsed",
            )
        with col2:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            search_btn = st.form_submit_button("Find Recommendations →")
        with st.expander("Personalize (optional) — blend in collaborative filtering"):
            uid_raw = st.number_input(
                f"MovieLens user ID (1–{max(user_ids)}); 0 = off",
                min_value=0, value=0, step=1,
            )
            cb_weight = st.slider(
                "Content vs. collaborative blend", 0.0, 1.0, 0.4, 0.05,
                help="1.0 = pure content similarity, 0.0 = pure collaborative filtering (SVD)",
            )

    st.markdown('<hr class="divider" />', unsafe_allow_html=True)

    if search_btn and query.strip():
        with st.spinner("Finding movies..."):
            movie_row = find_movie(query, movies)

            if movie_row is None:
                not_found_html = f"""
                {CARD_CSS}
                <div class="not-found">
                    <div class="not-found-title">"{html.escape(query)}" isn't in our database</div>
                    <div class="not-found-sub">Try a different title or check the spelling. We have {len(movies):,} movies.</div>
                </div>
                """
                components.html(not_found_html, height=250)
            else:
                seed_html = build_seed_html(movie_row, TMDB_API_KEY)

                user_id = int(uid_raw)
                use_hybrid = user_id in user_ids
                if user_id and not use_hybrid:
                    st.warning(f"User {user_id} not found — showing content-based results.")
                if use_hybrid:
                    recs = hybrid.recommend(user_id, movie_row["title"], movies,
                                            n=11, cb_weight=cb_weight)
                else:
                    recs = hybrid.content_recommender.recommend(movie_row["title"], n=11)
                recs = recs[recs["movieId"] != movie_row["movieId"]].head(10)
                # avoid year_x / year_y if the recommender already returns a year column
                recs = recs.drop(columns=["year"], errors="ignore")
                recs = recs.merge(movies[["movieId", "year"]], on="movieId", how="left")

                grid_html = build_grid_html(recs, movie_row["title"], TMDB_API_KEY,
                                          user_id if use_hybrid else None)
                components.html(CARD_CSS + seed_html + grid_html, height=1400, scrolling=True)

    elif search_btn:
        st.markdown(
            '<p style="color:#555;padding:20px 60px">Type a movie title to get started.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
