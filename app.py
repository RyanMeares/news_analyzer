"""
York County Regional News Monitor
Streamlit Dashboard — Team Edition
- Keyword search
- Historical volume + topic trend charts
- Today's Briefing summary panel
- Sentiment scoring UI removed
- Cloud-safe: no spaCy, NLTK VADER + sklearn LDA only
"""

import re
import os
import warnings
import feedparser
import pandas as pd
import nltk
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }

section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* ── Header ── */
.header-wrap {
    display: flex; align-items: flex-end; justify-content: space-between;
    border-bottom: 1px solid #21262d; padding-bottom: 1rem; margin-bottom: 1.5rem;
}
.header-title {
    font-family: 'Syne', sans-serif; font-size: 1.75rem; font-weight: 800;
    color: #e6edf3; margin: 0; letter-spacing: -0.5px;
}
.header-sub { color: #6e7681; font-size: 0.82rem; margin: 0.15rem 0 0 0; }
.header-time { color: #6e7681; font-size: 0.78rem; text-align: right; }

/* ── Briefing panel ── */
.briefing-wrap {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1.1rem 1.4rem; margin-bottom: 1.5rem;
}
.briefing-title {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700;
    color: #58a6ff; text-transform: uppercase; letter-spacing: 2px;
    margin: 0 0 0.8rem 0;
}
.briefing-item {
    display: flex; align-items: baseline; gap: 0.6rem;
    padding: 0.35rem 0; border-bottom: 1px solid #21262d;
}
.briefing-item:last-child { border-bottom: none; }
.briefing-num {
    font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700;
    color: #58a6ff; min-width: 1.2rem;
}
.briefing-text { font-size: 0.85rem; color: #c9d1d9; line-height: 1.4; }
.briefing-meta { font-size: 0.72rem; color: #6e7681; white-space: nowrap; }
.topic-chip {
    display: inline-block; font-size: 0.68rem; font-weight: 500;
    background: #1f2937; border: 1px solid #374151;
    border-radius: 4px; padding: 1px 7px; color: #9ca3af;
    margin-left: 0.4rem; vertical-align: middle;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700;
    color: #6e7681; text-transform: uppercase; letter-spacing: 2px;
    margin: 1.4rem 0 0.6rem 0; border-bottom: 1px solid #21262d;
    padding-bottom: 0.4rem;
}

/* ── Headline cards ── */
.hcard {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 0.8rem 1rem; margin: 0.35rem 0;
    transition: border-color 0.15s;
}
.hcard:hover { border-color: #388bfd44; }
.hcard-title { font-size: 0.87rem; color: #c9d1d9; line-height: 1.45; }
.hcard-title a { color: #c9d1d9; text-decoration: none; }
.hcard-title a:hover { color: #58a6ff; }
.hcard-meta { font-size: 0.73rem; color: #6e7681; margin-top: 0.3rem; }
.tag {
    display: inline-block; background: #1c2128; border: 1px solid #30363d;
    border-radius: 20px; padding: 1px 9px; font-size: 0.7rem;
    color: #8b949e; margin: 0.3rem 0.2rem 0 0;
}

/* ── Search box ── */
.stTextInput input {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 8px !important;
    font-size: 0.9rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RSS_FEEDS = {
    "Google News – Peninsula": "https://news.google.com/rss/search?q=York+County+VA+OR+Newport+News+OR+Hampton+OR+Williamsburg+OR+Poquoson&hl=en-US&gl=US&ceid=US:en",
    "Google News – Hampton Roads": "https://news.google.com/rss/search?q=Hampton+Roads+Virginia&hl=en-US&gl=US&ceid=US:en",
    "Google News – Economy": "https://news.google.com/rss/search?q=Newport+News+economy+OR+jobs+OR+business&hl=en-US&gl=US&ceid=US:en",
    "Google News – Infrastructure": "https://news.google.com/rss/search?q=Hampton+Roads+infrastructure+OR+transportation+OR+housing&hl=en-US&gl=US&ceid=US:en",
    "Google News – Military/Defense": "https://news.google.com/rss/search?q=Fort+Eustis+OR+Naval+Weapons+Station+Yorktown+OR+Hampton+Roads+military&hl=en-US&gl=US&ceid=US:en",
    "WTKR News 3": "https://www.wtkr.com/feed/",
    "WAVY News 10": "https://www.wavy.com/feed/",
    "Virginia Gazette": "https://www.vagazette.com/feed/",
    "Daily Press": "https://www.dailypress.com/feed/",
}

RELEVANCE_KEYWORDS = [
    "york county","york","newport news","hampton","williamsburg","poquoson",
    "hampton roads","fort eustis","yorktown","gloucester","james city",
    "isle of wight","suffolk","chesapeake","norfolk","virginia beach","peninsula","virginia",
]

LOCAL_ENTITIES = [
    "York County","Newport News","Hampton","Williamsburg","Poquoson",
    "Norfolk","Virginia Beach","Chesapeake","Suffolk","Portsmouth",
    "Fort Eustis","Naval Weapons Station","Yorktown","Gloucester",
    "James City County","Isle of Wight","Hampton Roads",
    "Sentara","Riverside","Huntington Ingalls","Newport News Shipbuilding",
    "NASA Langley","Thomas Nelson","William & Mary","Christopher Newport",
    "CNU","VDOT","Hampton Roads Transit","HRT",
    "City Council","Board of Supervisors","General Assembly",
]
_ENTITY_PATTERNS = [
    (e, re.compile(r'\b' + re.escape(e) + r'\b', re.IGNORECASE))
    for e in LOCAL_ENTITIES
]

TOPIC_LABEL_MAP = {
    "housing":"Housing & Development","rent":"Housing & Development",
    "zoning":"Housing & Development","permit":"Housing & Development","apartment":"Housing & Development",
    "school":"Education","education":"Education","student":"Education",
    "teacher":"Education","college":"Education","university":"Education",
    "job":"Labor & Economy","employment":"Labor & Economy",
    "workforce":"Labor & Economy","worker":"Labor & Economy",
    "business":"Business & Commerce","company":"Business & Commerce",
    "economic":"Business & Commerce","development":"Business & Commerce",
    "road":"Transportation","traffic":"Transportation","transit":"Transportation",
    "bridge":"Transportation","highway":"Transportation",
    "military":"Defense & Military","navy":"Defense & Military","fort":"Defense & Military",
    "naval":"Defense & Military","shipbuilding":"Defense & Military","defense":"Defense & Military",
    "crime":"Public Safety","police":"Public Safety","fire":"Public Safety",
    "arrest":"Public Safety","shooting":"Public Safety",
    "storm":"Environment & Weather","flood":"Environment & Weather",
    "weather":"Environment & Weather","hurricane":"Environment & Weather",
    "health":"Health & Community","hospital":"Health & Community","community":"Health & Community",
    "government":"Government & Policy","council":"Government & Policy",
    "election":"Government & Policy","tax":"Government & Policy","budget":"Government & Policy",
}

TOPIC_COLORS = {
    "Housing & Development": "#58a6ff",
    "Education": "#3fb950",
    "Labor & Economy": "#d29922",
    "Business & Commerce": "#f0883e",
    "Transportation": "#bc8cff",
    "Defense & Military": "#79c0ff",
    "Public Safety": "#f85149",
    "Environment & Weather": "#56d364",
    "Health & Community": "#ff7b72",
    "Government & Policy": "#ffa657",
    "General News": "#6e7681",
}

DATA_PATH = "york_news_archive.csv"
N_TOPICS = 8
_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="DM Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def extract_entities(text):
    found = []
    for name, pattern in _ENTITY_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return ", ".join(found[:5])

def is_relevant(title):
    return any(k in title.lower() for k in RELEVANCE_KEYWORDS)

def best_topic_label(kw_str):
    kws = kw_str.lower()
    for seed, label in TOPIC_LABEL_MAP.items():
        if seed in kws:
            return label
    return "General News"

# ─────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────
def scrape_news(selected_feeds):
    rows = []
    for feed_name in selected_feeds:
        url = RSS_FEEDS[feed_name]
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            if not title:
                continue
            date = entry.get("published", entry.get("updated", datetime.now().isoformat()))
            if "google" in url.lower() and not is_relevant(title):
                continue
            rows.append({
                "title": title,
                "link": entry.get("link", ""),
                "published": date,
                "source": feed_name,
                "scraped_at": datetime.now().isoformat(),
                "entities": extract_entities(title),
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
    return df.reset_index(drop=True)

# ─────────────────────────────────────────────
# TOPIC MODELING
# ─────────────────────────────────────────────
def assign_topics(df, n_topics=N_TOPICS):
    if len(df) < max(n_topics, 5):
        df["topic_label"] = "General News"
        return df
    vectorizer = TfidfVectorizer(stop_words="english", max_features=300, min_df=2, max_df=0.95)
    try:
        X = vectorizer.fit_transform(df["title"])
    except ValueError:
        df["topic_label"] = "General News"
        return df
    actual_n = min(n_topics, X.shape[0] - 1, X.shape[1])
    if actual_n < 2:
        df["topic_label"] = "General News"
        return df
    lda = LatentDirichletAllocation(
        n_components=actual_n, random_state=42,
        max_iter=15, learning_method="online"
    )
    lda.fit(X)
    assignments = lda.transform(X).argmax(axis=1)
    df["topic_id"] = assignments
    feature_names = vectorizer.get_feature_names_out()
    labels = []
    for topic_vec in lda.components_:
        top_words = [feature_names[i] for i in topic_vec.argsort()[-8:][::-1]]
        labels.append(best_topic_label(", ".join(top_words)))
    df["topic_label"] = df["topic_id"].apply(lambda x: labels[x])
    return df

# ─────────────────────────────────────────────
# DATE PARSING
# ─────────────────────────────────────────────
def parse_dates(df):
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def chart_volume_over_time(df):
    """Daily headline count stacked by topic."""
    d = df.dropna(subset=["published"]).copy()
    d["day"] = d["published"].dt.floor("D")
    grouped = d.groupby(["day","topic_label"]).size().reset_index(name="count")
    colors = [TOPIC_COLORS.get(t, "#6e7681") for t in grouped["topic_label"].unique()]
    fig = px.bar(
        grouped, x="day", y="count", color="topic_label",
        color_discrete_map=TOPIC_COLORS,
        template="plotly_dark", labels={"day":"","count":"Headlines","topic_label":"Topic"},
    )
    fig.update_layout(**_THEME, height=300, barmode="stack",
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11, orientation="h",
                                  yanchor="bottom", y=1.01, xanchor="left", x=0),
                      margin=dict(l=10,r=10,t=40,b=10), hovermode="x unified")
    fig.update_traces(marker_line_width=0)
    return fig

def chart_topic_trend(df):
    """Weekly headline count per topic — line chart."""
    d = df.dropna(subset=["published"]).copy()
    d = d.set_index("published").groupby("topic_label").resample("W").size().reset_index(name="count")
    fig = go.Figure()
    for topic in d["topic_label"].unique():
        td = d[d["topic_label"] == topic]
        fig.add_trace(go.Scatter(
            x=td["published"], y=td["count"],
            mode="lines+markers", name=topic,
            line=dict(color=TOPIC_COLORS.get(topic,"#6e7681"), width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(**_THEME, height=300,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
                      margin=dict(l=10,r=10,t=20,b=10), hovermode="x unified")
    fig.update_yaxes(title="Headlines / week")
    return fig

def chart_source_bar(df):
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["Source","Count"]
    fig = px.bar(counts, x="Count", y="Source", orientation="h",
                 template="plotly_dark", color_discrete_sequence=["#58a6ff"])
    fig.update_layout(**_THEME, height=280, showlegend=False,
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_traces(marker_line_width=0)
    return fig

def chart_topic_donut(df):
    counts = df["topic_label"].value_counts().reset_index()
    counts.columns = ["Topic","Count"]
    fig = px.pie(counts, names="Topic", values="Count", hole=0.6,
                 template="plotly_dark", color="Topic", color_discrete_map=TOPIC_COLORS)
    fig.update_layout(**_THEME, height=280,
                      legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=0,r=0,t=10,b=0))
    fig.update_traces(textinfo="percent", textfont_size=10)
    return fig

# ─────────────────────────────────────────────
# TODAY'S BRIEFING
# ─────────────────────────────────────────────
def render_briefing(df):
    now = pd.Timestamp.now(tz="UTC")
    today = df[df["published"] >= now - pd.Timedelta(hours=24)].copy()

    if today.empty:
        # Fall back to most recent 10 regardless of date
        today = df.sort_values("published", ascending=False).head(10)

    top_topics = today["topic_label"].value_counts().head(3)
    top_headlines = today.sort_values("published", ascending=False).head(5)
    total_24h = len(today)
    sources_24h = today["source"].nunique()

    st.markdown(f"""
    <div class="briefing-wrap">
        <p class="briefing-title">📋 Today's Briefing &nbsp;·&nbsp;
            <span style="font-weight:400;color:#6e7681;">
                {total_24h} headlines across {sources_24h} sources in the last 24 hrs
            </span>
        </p>
    """, unsafe_allow_html=True)

    # Top topics row
    topic_html = ""
    for topic, cnt in top_topics.items():
        color = TOPIC_COLORS.get(topic, "#6e7681")
        topic_html += f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:0.82rem;color:#c9d1d9;"><span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;"></span>{topic} <span style="color:#6e7681;">({cnt})</span></span>'
    st.markdown(f'<div style="margin-bottom:0.8rem;">{topic_html}</div>', unsafe_allow_html=True)

    # Top headlines
    for i, (_, row) in enumerate(top_headlines.iterrows(), 1):
        link = row.get("link","")
        title = row.get("title","")
        topic = row.get("topic_label","")
        pub = row.get("published","")
        pub_str = pub.strftime("%I:%M %p") if hasattr(pub,"strftime") else ""
        source = row.get("source","")
        title_html = f'<a href="{link}" target="_blank" style="color:#c9d1d9;text-decoration:none;">{title}</a>' if link else title
        topic_chip = f'<span class="topic-chip">{topic}</span>' if topic else ""
        st.markdown(f"""
        <div class="briefing-item">
            <span class="briefing-num">{i}</span>
            <span class="briefing-text">{title_html}{topic_chip}</span>
            <span class="briefing-meta">{pub_str} · {source}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADLINE CARD (no sentiment)
# ─────────────────────────────────────────────
def headline_card(row):
    title = row.get("title","")
    link = row.get("link","")
    topic = row.get("topic_label","")
    entities = row.get("entities","")
    pub = row.get("published","")
    pub_str = pub.strftime("%b %d, %Y") if hasattr(pub,"strftime") else str(pub)[:10]
    source = row.get("source","")

    title_html = (
        f'<a href="{link}" target="_blank">{title}</a>' if link else title
    )

    tags = ""
    if topic:
        color = TOPIC_COLORS.get(topic, "#6e7681")
        tags += f'<span class="tag" style="border-color:{color}44;color:{color};">{topic}</span>'
    if entities:
        for e in entities.split(",")[:3]:
            e = e.strip()
            if e:
                tags += f'<span class="tag">{e}</span>'

    st.markdown(f"""
    <div class="hcard">
        <div class="hcard-title">{title_html}</div>
        <div class="hcard-meta">{pub_str} · {source}</div>
        <div style="margin-top:0.25rem;">{tags}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    now_str = datetime.now(timezone.utc).strftime("Updated %b %d, %Y · %H:%M UTC")
    st.markdown(f"""
    <div class="header-wrap">
        <div>
            <p class="header-title">📰 York County Regional News Monitor</p>
            <p class="header-sub">Hampton Roads Peninsula · Team Intelligence Dashboard</p>
        </div>
        <div class="header-time">{now_str}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        selected_feeds = st.multiselect(
            "News Sources", options=list(RSS_FEEDS.keys()),
            default=list(RSS_FEEDS.keys()),
        )
        n_topics = st.slider("Topic clusters", 3, 12, N_TOPICS,
                             help="Number of topics LDA will discover")
        st.markdown("---")
        st.markdown("**Topic Filter**")
        topic_placeholder = st.empty()
        st.markdown("---")
        days_back = st.slider("Headlines from last N days", 1, 90, 30)
        st.markdown("---")
        run_btn = st.button("🔄 Fetch Latest", use_container_width=True, type="primary")
        st.caption("Headlines cached in `york_news_archive.csv`")

    # ── Session state ──
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()

    if run_btn or st.session_state.df.empty:
        if not selected_feeds:
            st.warning("Select at least one news source.")
            return
        with st.spinner("Fetching headlines…"):
            df = scrape_news(selected_feeds)
        if df.empty:
            st.error("No headlines retrieved. Check your connection or sources.")
            return
        with st.spinner("Modeling topics…"):
            df = assign_topics(df, n_topics)
        df = parse_dates(df)
        st.session_state.df = df
        st.success(f"✅ {len(df):,} headlines from {df['source'].nunique()} sources")

    df = st.session_state.df
    if df.empty:
        st.info("Click **Fetch Latest** in the sidebar to get started.")
        return

    # ── Date + topic filter ──
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    df_f = df[df["published"] >= cutoff].copy() if "published" in df.columns else df.copy()

    all_topics = sorted(df_f["topic_label"].dropna().unique().tolist())
    sel_topics = topic_placeholder.multiselect("Topics", options=all_topics, default=all_topics)
    df_f = df_f[df_f["topic_label"].isin(sel_topics)]

    # ── KPIs ──
    total = len(df_f)
    n_sources = df_f["source"].nunique()
    n_topics_active = df_f["topic_label"].nunique()
    top_topic = df_f["topic_label"].value_counts().idxmax() if total else "—"
    now_utc = pd.Timestamp.now(tz="UTC")
    last_24h = int((df_f["published"] >= now_utc - pd.Timedelta(hours=24)).sum()) if "published" in df_f.columns else 0

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Headlines", f"{total:,}")
    k2.metric("Last 24 Hours", f"{last_24h}")
    k3.metric("Sources Active", f"{n_sources}")
    k4.metric("Topics Detected", f"{n_topics_active}")
    k5.metric("Top Topic", top_topic)

    st.markdown("---")

    # ── Today's Briefing ──
    render_briefing(df_f)

    # ── Keyword Search ──
    st.markdown('<p class="section-title">🔍 Keyword Search</p>', unsafe_allow_html=True)
    search_query = st.text_input("", placeholder="Search headlines… e.g. shipyard, school board, flooding", label_visibility="collapsed")
    if search_query.strip():
        mask = df_f["title"].str.contains(search_query.strip(), case=False, na=False)
        df_search = df_f[mask].sort_values("published", ascending=False)
        st.caption(f"{len(df_search)} result{'s' if len(df_search) != 1 else ''} for "{search_query}"")
        if df_search.empty:
            st.info("No headlines matched that search.")
        else:
            for _, row in df_search.head(100).iterrows():
                headline_card(row)
        st.markdown("---")

    # ── Historical Trends ──
    st.markdown('<p class="section-title">📈 Historical Trends</p>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Daily Volume by Topic", "Weekly Topic Trends"])
    with t1:
        st.plotly_chart(chart_volume_over_time(df_f), use_container_width=True)
    with t2:
        st.plotly_chart(chart_topic_trend(df_f), use_container_width=True)

    # ── Topic + Source breakdown ──
    st.markdown('<p class="section-title">📊 Coverage Breakdown</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Topic distribution")
        st.plotly_chart(chart_topic_donut(df_f), use_container_width=True)
    with c2:
        st.caption("Headlines by source")
        st.plotly_chart(chart_source_bar(df_f), use_container_width=True)

    st.markdown("---")

    # ── All Headlines ──
    st.markdown('<p class="section-title">📋 All Headlines</p>', unsafe_allow_html=True)
    df_sorted = df_f.sort_values("published", ascending=False)

    col_sort, col_n, _ = st.columns([2,2,4])
    with col_sort:
        sort_by = st.selectbox("Sort by", ["Newest first","Oldest first","Topic (A–Z)"], label_visibility="collapsed")
    with col_n:
        top_n = st.slider("Show", 10, 300, 50, key="all_n", label_visibility="collapsed")

    if sort_by == "Newest first":
        df_sorted = df_f.sort_values("published", ascending=False)
    elif sort_by == "Oldest first":
        df_sorted = df_f.sort_values("published", ascending=True)
    else:
        df_sorted = df_f.sort_values("topic_label", ascending=True)

    # Group by topic if sorted that way
    if sort_by == "Topic (A–Z)":
        current_topic = None
        for _, row in df_sorted.head(top_n).iterrows():
            if row.get("topic_label") != current_topic:
                current_topic = row.get("topic_label","")
                color = TOPIC_COLORS.get(current_topic,"#6e7681")
                st.markdown(f'<p style="font-family:Syne,sans-serif;font-size:0.78rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:1.5px;margin:1rem 0 0.3rem 0;">{current_topic}</p>', unsafe_allow_html=True)
            headline_card(row)
    else:
        for _, row in df_sorted.head(top_n).iterrows():
            headline_card(row)

    st.markdown("---")

    # ── Export ──
    with st.expander("📥 Export Data"):
        export_df = df_sorted.head(top_n)
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current view as CSV", data=csv,
            file_name=f"york_news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
        cols = [c for c in ["title","published","source","topic_label","entities"] if c in export_df.columns]
        st.dataframe(export_df[cols], use_container_width=True, height=300)


if __name__ == "__main__":
    main()
