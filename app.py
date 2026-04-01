"""
York County Regional News Monitor
Streamlit Dashboard — Cloud-Safe Edition
- No spaCy (replaced with regex NER)
- VADER sentiment (NLTK only)
- LDA topic modeling (scikit-learn, bounded)
- Fully deployable on Streamlit Cloud free tier
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
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="York County News Monitor",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.block-container { padding-top: 2rem; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="metric-container"] { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1rem; }
.header-banner { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2128 100%); border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; }
.header-title { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 800; color: #58a6ff; margin: 0; letter-spacing: -0.5px; }
.header-sub { color: #8b949e; font-size: 0.85rem; margin: 0.2rem 0 0 0; }
.alert-box { background: #1a0a0a; border-left: 3px solid #f85149; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.4rem 0; }
.positive-box { background: #0a1a10; border-left: 3px solid #3fb950; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.4rem 0; }
.neutral-box { background: #111a24; border-left: 3px solid #58a6ff; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.4rem 0; }
.headline-text { font-size: 0.88rem; color: #c9d1d9; line-height: 1.4; }
.meta-text { font-size: 0.75rem; color: #6e7681; margin-top: 0.25rem; }
.score-badge { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 700; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.75rem 0; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }
.tag { display: inline-block; background: #1f2937; border: 1px solid #374151; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; color: #9ca3af; margin-right: 4px; }
</style>
""", unsafe_allow_html=True)

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
    "york county","york","newport news","hampton","williamsburg",
    "poquoson","hampton roads","fort eustis","yorktown","gloucester",
    "james city","isle of wight","suffolk","chesapeake","norfolk",
    "virginia beach","peninsula","virginia",
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
_ENTITY_PATTERNS = [(e, re.compile(r'\b' + re.escape(e) + r'\b', re.IGNORECASE)) for e in LOCAL_ENTITIES]

TOPIC_LABEL_MAP = {
    "housing":"Housing & Development","rent":"Housing & Development",
    "zoning":"Housing & Development","permit":"Housing & Development","apartment":"Housing & Development",
    "school":"Education","education":"Education","student":"Education",
    "teacher":"Education","college":"Education","university":"Education",
    "job":"Labor & Economy","employment":"Labor & Economy","workforce":"Labor & Economy","worker":"Labor & Economy",
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

DATA_PATH = "york_news_archive.csv"
N_TOPICS = 8
_COLORS = ["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff","#79c0ff","#56d364","#ff7b72"]
_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="DM Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)


@st.cache_resource(show_spinner="Loading sentiment model…")
def load_sia():
    nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def extract_entities(text):
    found = []
    for name, pattern in _ENTITY_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return ", ".join(found[:5])


def is_relevant(title):
    t = title.lower()
    return any(k in t for k in RELEVANCE_KEYWORDS)


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


def best_topic_label(kw_str):
    kws = kw_str.lower()
    for seed, label in TOPIC_LABEL_MAP.items():
        if seed in kws:
            return label
    return "General News"


def assign_topics(df, n_topics=N_TOPICS):
    if len(df) < max(n_topics, 5):
        df["topic_label"] = "General News"
        df["topic_keywords"] = ""
        return df
    vectorizer = TfidfVectorizer(stop_words="english", max_features=300, min_df=2, max_df=0.95)
    try:
        X = vectorizer.fit_transform(df["title"])
    except ValueError:
        df["topic_label"] = "General News"
        df["topic_keywords"] = ""
        return df
    actual_n = min(n_topics, X.shape[0] - 1, X.shape[1])
    if actual_n < 2:
        df["topic_label"] = "General News"
        df["topic_keywords"] = ""
        return df
    lda = LatentDirichletAllocation(n_components=actual_n, random_state=42, max_iter=15, learning_method="online")
    lda.fit(X)
    assignments = lda.transform(X).argmax(axis=1)
    df["topic_id"] = assignments
    feature_names = vectorizer.get_feature_names_out()
    labels, kw_strings = [], []
    for topic_vec in lda.components_:
        top_words = [feature_names[i] for i in topic_vec.argsort()[-8:][::-1]]
        kw_str = ", ".join(top_words)
        kw_strings.append(kw_str)
        labels.append(best_topic_label(kw_str))
    df["topic_keywords"] = df["topic_id"].apply(lambda x: kw_strings[x])
    df["topic_label"] = df["topic_id"].apply(lambda x: labels[x])
    return df


def analyze_sentiment(df, sia):
    df["compound"] = df["title"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    df["sentiment"] = df["compound"].apply(lambda s: "Positive" if s > 0.1 else ("Negative" if s < -0.1 else "Neutral"))
    df["impact"] = df["compound"].apply(lambda s: "Growth Opportunity" if s > 0.3 else ("Risk Signal" if s < -0.2 else "Monitor"))
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    weekly = (
        df.dropna(subset=["published"])
        .set_index("published")
        .groupby("topic_label")["compound"]
        .resample("W").mean()
        .reset_index()
    )
    return df, weekly


def chart_sentiment_bar(df):
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment","Count"]
    fig = px.bar(counts, x="Sentiment", y="Count", color="Sentiment",
                 color_discrete_map={"Positive":"#3fb950","Negative":"#f85149","Neutral":"#58a6ff"},
                 template="plotly_dark")
    fig.update_layout(**_THEME, showlegend=False, height=260, margin=dict(l=10,r=10,t=20,b=10))
    fig.update_traces(marker_line_width=0)
    return fig


def chart_weekly_trend(weekly):
    if weekly.empty:
        return go.Figure()
    fig = go.Figure()
    for i, label in enumerate(weekly["topic_label"].unique()):
        d = weekly[weekly["topic_label"] == label]
        fig.add_trace(go.Scatter(x=d["published"], y=d["compound"], mode="lines+markers", name=label,
                                 line=dict(color=_COLORS[i % len(_COLORS)], width=2), marker=dict(size=5)))
    fig.update_layout(**_THEME, height=320, legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
                      margin=dict(l=10,r=10,t=20,b=10), hovermode="x unified")
    fig.update_yaxes(title="Avg Sentiment Score", range=[-1,1])
    return fig


def chart_heatmap(df):
    pivot = df.groupby(["topic_label","sentiment"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=["Positive","Neutral","Negative"], fill_value=0)
    fig = px.imshow(pivot, color_continuous_scale=["#f85149","#21262d","#3fb950"],
                    aspect="auto", template="plotly_dark")
    fig.update_layout(**_THEME, height=280, margin=dict(l=10,r=10,t=20,b=10))
    return fig


def chart_source_pie(df):
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["Source","Count"]
    fig = px.pie(counts, names="Source", values="Count", hole=0.55,
                 template="plotly_dark", color_discrete_sequence=_COLORS)
    fig.update_layout(**_THEME, height=260, legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=0,r=0,t=20,b=0))
    fig.update_traces(textinfo="percent", textfont_size=11)
    return fig


def headline_card(row):
    s = row.get("sentiment","Neutral")
    box_class = {"Positive":"positive-box","Negative":"alert-box"}.get(s,"neutral-box")
    score = row.get("compound", 0)
    score_color = "#3fb950" if score > 0.1 else ("#f85149" if score < -0.1 else "#58a6ff")
    label = row.get("topic_label","")
    entities = row.get("entities","")
    pub = row.get("published","")
    pub_str = pub.strftime("%b %d, %Y") if hasattr(pub,"strftime") else str(pub)[:10]
    link = row.get("link","")
    title = row.get("title","")
    title_html = (f'<a href="{link}" target="_blank" style="color:#c9d1d9;text-decoration:none;">{title}</a>' if link else title)
    tags = ""
    if label:
        tags += f'<span class="tag">{label}</span>'
    if entities:
        for e in entities.split(",")[:3]:
            e = e.strip()
            if e:
                tags += f'<span class="tag">{e}</span>'
    arrow = "▲" if score > 0.1 else ("▼" if score < -0.1 else "–")
    st.markdown(f"""
    <div class="{box_class}">
        <div class="headline-text">{title_html}</div>
        <div class="meta-text">{pub_str} · {row.get('source','')} &nbsp;&nbsp;
            <span class="score-badge" style="color:{score_color};">{arrow} {score:+.2f}</span>
        </div>
        <div style="margin-top:0.4rem;">{tags}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    sia = load_sia()

    st.markdown("""
    <div class="header-banner">
        <p class="header-title">📰 York County Regional News Monitor</p>
        <p class="header-sub">Hampton Roads Peninsula · Sentiment Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        selected_feeds = st.multiselect("News Sources", options=list(RSS_FEEDS.keys()), default=list(RSS_FEEDS.keys()))
        n_topics = st.slider("Topic clusters", 3, 12, N_TOPICS)
        st.markdown("---")
        st.markdown("**Sentiment Filter**")
        sentiment_filter = st.multiselect("Show only:", ["Positive","Neutral","Negative"], default=["Positive","Neutral","Negative"])
        st.markdown("**Topic Filter**")
        topic_placeholder = st.empty()
        st.markdown("---")
        days_back = st.slider("Headlines from last N days", 1, 90, 30)
        st.markdown("---")
        run_btn = st.button("🔄 Fetch & Analyze", use_container_width=True, type="primary")
        st.caption("Headlines cached in `york_news_archive.csv`")

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
        st.session_state.weekly = pd.DataFrame()

    if run_btn or st.session_state.df.empty:
        if not selected_feeds:
            st.warning("Select at least one news source.")
            return
        with st.spinner("Fetching headlines…"):
            df = scrape_news(selected_feeds)
        if df.empty:
            st.error("No headlines retrieved. Check your connection or try different sources.")
            return
        with st.spinner("Modeling topics…"):
            df = assign_topics(df, n_topics)
        with st.spinner("Running sentiment analysis…"):
            df, weekly = analyze_sentiment(df, sia)
        st.session_state.df = df
        st.session_state.weekly = weekly
        st.success(f"✅ {len(df):,} headlines from {df['source'].nunique()} sources")

    df = st.session_state.df
    weekly = st.session_state.weekly

    if df.empty:
        st.info("Click **Fetch & Analyze** in the sidebar to get started.")
        return

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    df_f = df[df["published"] >= cutoff].copy() if "published" in df.columns else df.copy()
    df_f = df_f[df_f["sentiment"].isin(sentiment_filter)]

    all_topics = sorted(df_f["topic_label"].dropna().unique().tolist())
    sel_topics = topic_placeholder.multiselect("Topics", options=all_topics, default=all_topics)
    df_f = df_f[df_f["topic_label"].isin(sel_topics)]

    total = len(df_f)
    pct_pos = (df_f["sentiment"] == "Positive").sum() / max(total,1) * 100
    pct_neg = (df_f["sentiment"] == "Negative").sum() / max(total,1) * 100
    avg_score = df_f["compound"].mean() if total else 0
    risk_count = (df_f["impact"] == "Risk Signal").sum()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Headlines", f"{total:,}")
    k2.metric("Positive", f"{pct_pos:.1f}%")
    k3.metric("Negative", f"{pct_neg:.1f}%")
    k4.metric("Avg Sentiment", f"{avg_score:+.3f}")
    k5.metric("Risk Signals", f"{risk_count}")
    st.markdown("---")

    c1,c2,c3 = st.columns([1,2,1])
    with c1:
        st.markdown('<p class="section-title">Sentiment Mix</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_sentiment_bar(df_f), use_container_width=True)
    with c2:
        st.markdown('<p class="section-title">Weekly Sentiment by Topic</p>', unsafe_allow_html=True)
        wf = weekly[weekly["topic_label"].isin(sel_topics)] if not weekly.empty else weekly
        st.plotly_chart(chart_weekly_trend(wf), use_container_width=True)
    with c3:
        st.markdown('<p class="section-title">Source Breakdown</p>', unsafe_allow_html=True)
        st.plotly_chart(chart_source_pie(df_f), use_container_width=True)

    st.markdown('<p class="section-title">Topic × Sentiment Heatmap</p>', unsafe_allow_html=True)
    st.plotly_chart(chart_heatmap(df_f), use_container_width=True)
    st.markdown("---")

    st.markdown('<p class="section-title">Headlines</p>', unsafe_allow_html=True)
    tab_all, tab_neg, tab_pos, tab_risk = st.tabs(["📋 All","🔴 Negative Alerts","🟢 Positive Signals","⚠️ Risk Signals"])
    df_sorted = df_f.sort_values("published", ascending=False)

    with tab_all:
        top_n = st.slider("Show top N", 10, 200, 50, key="all_n")
        for _, row in df_sorted.head(top_n).iterrows():
            headline_card(row)
    with tab_neg:
        neg = df_sorted[df_sorted["sentiment"] == "Negative"]
        st.caption(f"{len(neg)} headlines")
        for _, row in neg.head(100).iterrows():
            headline_card(row)
    with tab_pos:
        pos = df_sorted[df_sorted["sentiment"] == "Positive"]
        st.caption(f"{len(pos)} headlines")
        for _, row in pos.head(100).iterrows():
            headline_card(row)
    with tab_risk:
        risk = df_sorted[df_sorted["impact"] == "Risk Signal"]
        st.caption(f"{len(risk)} headlines")
        for _, row in risk.head(100).iterrows():
            headline_card(row)

    st.markdown("---")
    with st.expander("📥 Export Data"):
        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv,
                           file_name=f"york_news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv")
        cols = [c for c in ["title","published","source","sentiment","compound","topic_label","entities","impact"] if c in df_f.columns]
        st.dataframe(df_f[cols], use_container_width=True, height=300)


if __name__ == "__main__":
    main()
