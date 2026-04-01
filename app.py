"""
York County Regional News Monitor
Streamlit Dashboard — Professional Edition
"""

import os
import streamlit as st
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="York County News Monitor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

.block-container { padding-top: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.header-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #58a6ff;
    margin: 0;
    letter-spacing: -0.5px;
}

.header-sub {
    color: #8b949e;
    font-size: 0.85rem;
    margin: 0.2rem 0 0 0;
}

/* Alert box */
.alert-box {
    background: #1a0a0a;
    border-left: 3px solid #f85149;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
}

.positive-box {
    background: #0a1a10;
    border-left: 3px solid #3fb950;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
}

.neutral-box {
    background: #111a24;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
}

.headline-text { font-size: 0.88rem; color: #c9d1d9; line-height: 1.4; }
.meta-text { font-size: 0.75rem; color: #6e7681; margin-top: 0.25rem; }
.score-badge {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
}

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 1.5rem 0 0.75rem 0;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
}

/* Tag pills */
.tag {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: #9ca3af;
    margin-right: 4px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RSS FEEDS — Expanded local coverage
# ─────────────────────────────────────────────
RSS_FEEDS = {
    "Google News – Peninsula": (
        "https://news.google.com/rss/search?q=York+County+VA+OR+Newport+News+OR+Hampton+OR+Williamsburg+OR+Poquoson&hl=en-US&gl=US&ceid=US:en"
    ),
    "Google News – Hampton Roads": (
        "https://news.google.com/rss/search?q=Hampton+Roads+Virginia&hl=en-US&gl=US&ceid=US:en"
    ),
    "Google News – Economy": (
        "https://news.google.com/rss/search?q=Newport+News+economy+OR+jobs+OR+business&hl=en-US&gl=US&ceid=US:en"
    ),
    "Google News – Infrastructure": (
        "https://news.google.com/rss/search?q=Hampton+Roads+infrastructure+OR+transportation+OR+housing&hl=en-US&gl=US&ceid=US:en"
    ),
    "Google News – Military/Defense": (
        "https://news.google.com/rss/search?q=Fort+Eustis+OR+Naval+Weapons+Station+Yorktown+OR+Hampton+Roads+military&hl=en-US&gl=US&ceid=US:en"
    ),
    "WTKR": (
        "https://www.wtkr.com/feed/"
    ),
    "WAVY News": (
        "https://www.wavy.com/feed/"
    ),
    "Virginia Gazette": (
        "https://www.vagazette.com/feed/"
    ),
    "Daily Press": (
        "https://www.dailypress.com/feed/"
    ),
}

# Relevance filter keywords
RELEVANCE_KEYWORDS = [
    "york county", "york", "newport news", "hampton", "williamsburg",
    "poquoson", "hampton roads", "fort eustis", "yorktown", "gloucester",
    "james city", "isle of wight", "suffolk", "chesapeake", "norfolk",
    "virginia beach", "peninsula"
]

# ─────────────────────────────────────────────
# TOPIC LABELS (human-readable overrides)
# These are applied after LDA assigns a topic_id,
# matching the dominant LDA keywords to a label.
# ─────────────────────────────────────────────
TOPIC_LABEL_MAP = {
    # These are best-guess seeds; actual topics emerge from LDA
    "housing": "Housing & Development",
    "rent": "Housing & Development",
    "zoning": "Housing & Development",
    "permit": "Housing & Development",
    "school": "Education",
    "education": "Education",
    "student": "Education",
    "teacher": "Education",
    "college": "Education",
    "job": "Labor & Economy",
    "employment": "Labor & Economy",
    "workforce": "Labor & Economy",
    "business": "Business & Commerce",
    "company": "Business & Commerce",
    "economic": "Business & Commerce",
    "road": "Transportation",
    "traffic": "Transportation",
    "transit": "Transportation",
    "bridge": "Transportation",
    "military": "Defense & Military",
    "navy": "Defense & Military",
    "fort": "Defense & Military",
    "naval": "Defense & Military",
    "crime": "Public Safety",
    "police": "Public Safety",
    "fire": "Public Safety",
    "arrest": "Public Safety",
    "storm": "Environment & Weather",
    "flood": "Environment & Weather",
    "weather": "Environment & Weather",
    "park": "Environment & Weather",
    "health": "Health & Community",
    "hospital": "Health & Community",
    "community": "Health & Community",
    "government": "Government & Policy",
    "council": "Government & Policy",
    "election": "Government & Policy",
    "tax": "Government & Policy",
}

DATA_PATH = "york_news_archive.csv"
NEGATIVE_THRESHOLD = -0.2
N_TOPICS = 8

# ─────────────────────────────────────────────
# NLP SETUP (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_nlp_models():
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return sia, nlp

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def is_relevant(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in RELEVANCE_KEYWORDS)

def extract_entities(text: str, nlp) -> str:
    doc = nlp(text)
    ents = [e.text for e in doc.ents if e.label_ in ("ORG", "GPE", "PERSON", "FAC")]
    return ", ".join(dict.fromkeys(ents))  # deduplicate, preserve order

def best_topic_label(keywords_str: str) -> str:
    """Map LDA keyword string to a human-readable topic label."""
    kws = keywords_str.lower()
    for seed, label in TOPIC_LABEL_MAP.items():
        if seed in kws:
            return label
    return "General News"

# ─────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────
def scrape_news(selected_feeds: list, nlp) -> pd.DataFrame:
    rows = []
    for feed_name in selected_feeds:
        url = RSS_FEEDS[feed_name]
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            date = entry.get("published", entry.get("updated", datetime.now().isoformat()))
            if not title:
                continue
            # For local outlet feeds, include all; for Google News, filter by keyword
            if "google" in url.lower() and not is_relevant(title):
                continue
            rows.append({
                "title": title,
                "link": entry.get("link", ""),
                "published": date,
                "source": feed_name,
                "scraped_at": datetime.now().isoformat(),
                "entities": extract_entities(title, nlp),
            })

    if not rows:
        return pd.DataFrame()

    new_df = pd.DataFrame(rows)

    if os.path.exists(DATA_PATH):
        try:
            old_df = pd.read_csv(DATA_PATH)
            df = pd.concat([old_df, new_df]).drop_duplicates(subset=["title"])
        except Exception:
            df = new_df
    else:
        df = new_df

    df.to_csv(DATA_PATH, index=False)
    return df

# ─────────────────────────────────────────────
# TOPIC MODELING
# ─────────────────────────────────────────────
def assign_topics(df: pd.DataFrame, n_topics: int = N_TOPICS):
    if len(df) < n_topics:
        df["topic_id"] = 0
        df["topic_keywords"] = "general"
        df["topic_label"] = "General News"
        return df, ["General News"]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500, min_df=2)
    try:
        X = vectorizer.fit_transform(df["title"])
    except ValueError:
        df["topic_id"] = 0
        df["topic_keywords"] = "general"
        df["topic_label"] = "General News"
        return df, ["General News"]

    actual_n = min(n_topics, X.shape[0], X.shape[1])
    lda = LatentDirichletAllocation(n_components=actual_n, random_state=42, max_iter=20)
    lda.fit(X)

    topic_assignments = lda.transform(X).argmax(axis=1)
    df["topic_id"] = topic_assignments

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords_list = []
    topic_labels_list = []

    for topic_vec in lda.components_:
        top_words = [feature_names[i] for i in topic_vec.argsort()[-10:][::-1]]
        kw_str = ", ".join(top_words)
        topic_keywords_list.append(kw_str)
        topic_labels_list.append(best_topic_label(kw_str))

    df["topic_keywords"] = df["topic_id"].apply(lambda x: topic_keywords_list[x])
    df["topic_label"] = df["topic_id"].apply(lambda x: topic_labels_list[x])

    return df, topic_labels_list

# ─────────────────────────────────────────────
# SENTIMENT
# ─────────────────────────────────────────────
def analyze_sentiment(df: pd.DataFrame, sia) -> tuple:
    df["compound"] = df["title"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    df["sentiment"] = df["compound"].apply(
        lambda s: "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral")
    )
    df["impact"] = df["compound"].apply(
        lambda s: "Growth Opportunity" if s > 0.3 else (
            "Risk Signal" if s < -0.2 else "Monitor"
        )
    )
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)

    weekly = (
        df.dropna(subset=["published"])
        .set_index("published")
        .groupby("topic_label")["compound"]
        .resample("W")
        .mean()
        .reset_index()
    )

    return df, weekly

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "DM Sans"},
    "xaxis": {"gridcolor": "#21262d", "linecolor": "#30363d"},
    "yaxis": {"gridcolor": "#21262d", "linecolor": "#30363d"},
}

def sentiment_distribution_chart(df):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    color_map = {"Positive": "#3fb950", "Negative": "#f85149", "Neutral": "#58a6ff"}
    fig = px.bar(
        counts, x="Sentiment", y="Count", color="Sentiment",
        color_discrete_map=color_map,
        template="plotly_dark",
    )
    fig.update_layout(**CHART_THEME, showlegend=False, height=280,
                      margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(marker_line_width=0)
    return fig

def weekly_trend_chart(weekly):
    if weekly.empty:
        return go.Figure()
    color_seq = ["#58a6ff", "#3fb950", "#d29922", "#f85149",
                 "#bc8cff", "#79c0ff", "#56d364", "#ff7b72"]
    fig = go.Figure()
    for i, label in enumerate(weekly["topic_label"].unique()):
        d = weekly[weekly["topic_label"] == label]
        fig.add_trace(go.Scatter(
            x=d["published"], y=d["compound"],
            mode="lines+markers", name=label,
            line=dict(color=color_seq[i % len(color_seq)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(**CHART_THEME, height=340,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
                      margin=dict(l=10, r=10, t=20, b=10),
                      hovermode="x unified")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Avg. Sentiment Score", range=[-1, 1])
    return fig

def topic_sentiment_heatmap(df):
    pivot = df.groupby(["topic_label", "sentiment"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    fig = px.imshow(
        pivot,
        color_continuous_scale=["#f85149", "#21262d", "#3fb950"],
        aspect="auto",
        template="plotly_dark",
    )
    fig.update_layout(**CHART_THEME, height=300,
                      margin=dict(l=10, r=10, t=20, b=10))
    return fig

def source_breakdown_chart(df):
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["Source", "Count"]
    fig = px.pie(
        counts, names="Source", values="Count",
        hole=0.55, template="plotly_dark",
        color_discrete_sequence=["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff","#79c0ff","#56d364","#ff7b72"],
    )
    fig.update_layout(**CHART_THEME, height=280,
                      showlegend=True,
                      legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=0, r=0, t=20, b=0))
    fig.update_traces(textinfo="percent", textfont_size=11)
    return fig

# ─────────────────────────────────────────────
# HEADLINE CARD
# ─────────────────────────────────────────────
def headline_card(row):
    s = row.get("sentiment", "Neutral")
    box_class = {"Positive": "positive-box", "Negative": "alert-box"}.get(s, "neutral-box")
    score = row.get("compound", 0)
    score_color = "#3fb950" if score > 0.1 else ("#f85149" if score < -0.1 else "#58a6ff")
    label = row.get("topic_label", "")
    entities = row.get("entities", "")
    pub = row.get("published", "")
    pub_str = pub.strftime("%b %d, %Y") if hasattr(pub, "strftime") else str(pub)[:10]
    link = row.get("link", "")
    title = row.get("title", "")
    title_html = f'<a href="{link}" target="_blank" style="color:#c9d1d9;text-decoration:none;">{title}</a>' if link else title

    tags = ""
    if label:
        tags += f'<span class="tag">{label}</span>'
    if entities:
        for e in entities.split(",")[:3]:
            e = e.strip()
            if e:
                tags += f'<span class="tag">{e}</span>'

    st.markdown(f"""
    <div class="{box_class}">
        <div class="headline-text">{title_html}</div>
        <div class="meta-text">
            {pub_str} · {row.get('source','')}
            &nbsp;&nbsp;
            <span class="score-badge" style="color:{score_color};">
                {'▲' if score > 0.1 else ('▼' if score < -0.1 else '–')} {score:+.2f}
            </span>
        </div>
        <div style="margin-top:0.4rem;">{tags}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    sia, nlp = load_nlp_models()

    # ── Header ──
    st.markdown("""
    <div class="header-banner">
        <div>
            <p class="header-title">📰 York County Regional News Monitor</p>
            <p class="header-sub">Hampton Roads Peninsula · Sentiment Intelligence Dashboard</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        selected_feeds = st.multiselect(
            "News Sources",
            options=list(RSS_FEEDS.keys()),
            default=list(RSS_FEEDS.keys()),
            help="Select which RSS feeds to pull from"
        )

        n_topics = st.slider("Topic clusters (LDA)", min_value=3, max_value=12, value=N_TOPICS,
                             help="Number of topics the model will discover automatically")

        st.markdown("---")
        st.markdown("**Sentiment Filter**")
        sentiment_filter = st.multiselect(
            "Show only:",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"]
        )

        st.markdown("**Topic Filter**")
        topic_filter_placeholder = st.empty()

        st.markdown("---")
        st.markdown("**Date Range**")
        days_back = st.slider("Headlines from last N days", 1, 90, 30)

        st.markdown("---")
        run_btn = st.button("🔄 Fetch & Analyze", use_container_width=True, type="primary")

        st.markdown("---")
        st.caption("Data persists in `york_news_archive.csv`")

    # ── Load or fetch data ──
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
        st.session_state.weekly = pd.DataFrame()

    if run_btn or st.session_state.df.empty:
        if not selected_feeds:
            st.warning("Please select at least one news source.")
            return

        with st.spinner("Fetching headlines..."):
            df = scrape_news(selected_feeds, nlp)

        if df.empty:
            st.error("No headlines retrieved. Check your internet connection or RSS feeds.")
            return

        with st.spinner("Modeling topics..."):
            df, topic_labels = assign_topics(df, n_topics)

        with st.spinner("Running sentiment analysis..."):
            df, weekly = analyze_sentiment(df, sia)

        st.session_state.df = df
        st.session_state.weekly = weekly
        st.success(f"✅ Loaded {len(df):,} headlines from {df['source'].nunique()} sources")

    df = st.session_state.df
    weekly = st.session_state.weekly

    if df.empty:
        st.info("Click **Fetch & Analyze** in the sidebar to load headlines.")
        return

    # ── Date filter ──
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    df_filtered = df[df["published"] >= cutoff].copy() if "published" in df.columns else df.copy()

    # ── Sentiment filter ──
    df_filtered = df_filtered[df_filtered["sentiment"].isin(sentiment_filter)]

    # ── Topic filter (dynamic, after data loads) ──
    all_topics = sorted(df_filtered["topic_label"].dropna().unique().tolist())
    selected_topics = topic_filter_placeholder.multiselect(
        "Topics", options=all_topics, default=all_topics
    )
    df_filtered = df_filtered[df_filtered["topic_label"].isin(selected_topics)]

    # ─────────────────────────────────────────────
    # KPI ROW
    # ─────────────────────────────────────────────
    total = len(df_filtered)
    pct_pos = (df_filtered["sentiment"] == "Positive").sum() / max(total, 1) * 100
    pct_neg = (df_filtered["sentiment"] == "Negative").sum() / max(total, 1) * 100
    avg_score = df_filtered["compound"].mean()
    risk_count = (df_filtered["impact"] == "Risk Signal").sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Headlines", f"{total:,}")
    k2.metric("Positive", f"{pct_pos:.1f}%", delta=None)
    k3.metric("Negative", f"{pct_neg:.1f}%", delta=None)
    k4.metric("Avg Sentiment", f"{avg_score:+.3f}")
    k5.metric("Risk Signals", f"{risk_count}")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # CHARTS ROW 1
    # ─────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        st.markdown('<p class="section-title">Sentiment Mix</p>', unsafe_allow_html=True)
        st.plotly_chart(sentiment_distribution_chart(df_filtered), use_container_width=True)

    with c2:
        st.markdown('<p class="section-title">Weekly Sentiment by Topic</p>', unsafe_allow_html=True)
        weekly_f = weekly[weekly["topic_label"].isin(selected_topics)] if not weekly.empty else weekly
        st.plotly_chart(weekly_trend_chart(weekly_f), use_container_width=True)

    with c3:
        st.markdown('<p class="section-title">Source Breakdown</p>', unsafe_allow_html=True)
        st.plotly_chart(source_breakdown_chart(df_filtered), use_container_width=True)

    # ─────────────────────────────────────────────
    # CHARTS ROW 2
    # ─────────────────────────────────────────────
    st.markdown('<p class="section-title">Topic × Sentiment Heatmap</p>', unsafe_allow_html=True)
    st.plotly_chart(topic_sentiment_heatmap(df_filtered), use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # HEADLINES TABS
    # ─────────────────────────────────────────────
    st.markdown('<p class="section-title">Headlines</p>', unsafe_allow_html=True)
    tab_all, tab_neg, tab_pos, tab_risk = st.tabs(
        ["📋 All", "🔴 Negative Alerts", "🟢 Positive Signals", "⚠️ Risk Signals"]
    )

    df_sorted = df_filtered.sort_values("published", ascending=False)

    with tab_all:
        top_n = st.slider("Show top N headlines", 10, 200, 50, key="all_n")
        for _, row in df_sorted.head(top_n).iterrows():
            headline_card(row)

    with tab_neg:
        neg_df = df_sorted[df_sorted["sentiment"] == "Negative"]
        st.caption(f"{len(neg_df)} negative headlines")
        for _, row in neg_df.head(100).iterrows():
            headline_card(row)

    with tab_pos:
        pos_df = df_sorted[df_sorted["sentiment"] == "Positive"]
        st.caption(f"{len(pos_df)} positive headlines")
        for _, row in pos_df.head(100).iterrows():
            headline_card(row)

    with tab_risk:
        risk_df = df_sorted[df_sorted["impact"] == "Risk Signal"]
        st.caption(f"{len(risk_df)} risk-flagged headlines")
        for _, row in risk_df.head(100).iterrows():
            headline_card(row)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # RAW DATA EXPORT
    # ─────────────────────────────────────────────
    with st.expander("📥 Export Data"):
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered headlines as CSV",
            data=csv,
            file_name=f"york_news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        st.dataframe(
            df_filtered[["title", "published", "source", "sentiment", "compound", "topic_label", "entities", "impact"]],
            use_container_width=True,
            height=300,
        )

if __name__ == "__main__":
    main()
