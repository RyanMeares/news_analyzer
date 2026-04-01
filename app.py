"""
York County Regional News Monitor — Economic Development Edition
New in this version:
  - Watchlist: named issues with keyword sets, dedicated feeds + trend charts
  - Auto-refresh: fetches every N hours automatically
  - AI Briefing: Claude API generates a narrative economic intelligence summary
  - Weekly Report: one-click formatted HTML report export
  - Expanded RSS sources: VEDP, Port of Virginia, Virginia Business, Bisnow, Chamber
"""

import re
import os
import json
import time
import warnings
import feedparser
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hampton Roads Economic Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
.block-container { padding-top: 1.5rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
[data-testid="metric-container"] { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1rem 1.25rem; }

.header-wrap { display: flex; align-items: flex-end; justify-content: space-between; border-bottom: 1px solid #21262d; padding-bottom: 1rem; margin-bottom: 1.5rem; }
.header-title { font-family: 'Syne', sans-serif; font-size: 1.75rem; font-weight: 800; color: #e6edf3; margin: 0; letter-spacing: -0.5px; }
.header-sub { color: #6e7681; font-size: 0.82rem; margin: 0.15rem 0 0 0; }
.header-time { color: #6e7681; font-size: 0.78rem; text-align: right; line-height: 1.6; }

/* Briefing */
.briefing-wrap { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem; }
.briefing-label { font-family: 'Syne', sans-serif; font-size: 0.68rem; font-weight: 700; color: #58a6ff; text-transform: uppercase; letter-spacing: 2px; margin: 0 0 0.5rem 0; }
.briefing-summary { font-size: 0.9rem; color: #c9d1d9; line-height: 1.7; margin: 0 0 1rem 0; border-bottom: 1px solid #21262d; padding-bottom: 1rem; }
.briefing-item { display: flex; align-items: baseline; gap: 0.6rem; padding: 0.35rem 0; border-bottom: 1px solid #1c2128; }
.briefing-item:last-child { border-bottom: none; }
.briefing-num { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; color: #58a6ff; min-width: 1.2rem; }
.briefing-text { font-size: 0.85rem; color: #c9d1d9; line-height: 1.4; flex: 1; }
.briefing-meta { font-size: 0.72rem; color: #6e7681; white-space: nowrap; }
.topic-chip { display: inline-block; font-size: 0.67rem; font-weight: 500; background: #1f2937; border: 1px solid #374151; border-radius: 4px; padding: 1px 7px; color: #9ca3af; margin-left: 0.4rem; vertical-align: middle; }

/* Watchlist */
.watchlist-card { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; }
.watchlist-name { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #e6edf3; margin: 0 0 0.25rem 0; }
.watchlist-meta { font-size: 0.75rem; color: #6e7681; margin-bottom: 0.5rem; }
.watchlist-headline { font-size: 0.83rem; color: #c9d1d9; padding: 0.25rem 0; border-bottom: 1px solid #1c2128; line-height: 1.4; }
.watchlist-headline:last-child { border-bottom: none; }
.watchlist-headline a { color: #c9d1d9; text-decoration: none; }
.watchlist-headline a:hover { color: #58a6ff; }

/* Section titles */
.section-title { font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 700; color: #6e7681; text-transform: uppercase; letter-spacing: 2px; margin: 1.4rem 0 0.6rem 0; border-bottom: 1px solid #21262d; padding-bottom: 0.4rem; }

/* Headline cards */
.hcard { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.3rem 0; }
.hcard:hover { border-color: #388bfd44; }
.hcard-title { font-size: 0.87rem; color: #c9d1d9; line-height: 1.45; }
.hcard-title a { color: #c9d1d9; text-decoration: none; }
.hcard-title a:hover { color: #58a6ff; }
.hcard-meta { font-size: 0.73rem; color: #6e7681; margin-top: 0.3rem; }
.tag { display: inline-block; background: #1c2128; border: 1px solid #30363d; border-radius: 20px; padding: 1px 9px; font-size: 0.7rem; color: #8b949e; margin: 0.3rem 0.2rem 0 0; }

/* Auto-refresh badge */
.refresh-badge { display: inline-flex; align-items: center; gap: 6px; background: #0f2a1a; border: 1px solid #1a4a2a; border-radius: 6px; padding: 3px 10px; font-size: 0.75rem; color: #3fb950; }
.refresh-dot { width: 6px; height: 6px; border-radius: 50%; background: #3fb950; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.stTextInput input { background: #161b22 !important; border: 1px solid #30363d !important; color: #e6edf3 !important; border-radius: 8px !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RSS FEEDS — expanded for economic development
# ─────────────────────────────────────────────
RSS_FEEDS = {
    # General local coverage
    "Google News – Peninsula":       "https://news.google.com/rss/search?q=York+County+VA+OR+Newport+News+OR+Hampton+OR+Williamsburg+OR+Poquoson&hl=en-US&gl=US&ceid=US:en",
    "Google News – Hampton Roads":   "https://news.google.com/rss/search?q=Hampton+Roads+Virginia&hl=en-US&gl=US&ceid=US:en",
    "Google News – Economy":         "https://news.google.com/rss/search?q=Newport+News+economy+OR+jobs+OR+business+Virginia&hl=en-US&gl=US&ceid=US:en",
    "Google News – Infrastructure":  "https://news.google.com/rss/search?q=Hampton+Roads+infrastructure+OR+transportation+OR+housing&hl=en-US&gl=US&ceid=US:en",
    "Google News – Military/Defense":"https://news.google.com/rss/search?q=Fort+Eustis+OR+Naval+Weapons+Station+Yorktown+OR+Hampton+Roads+military&hl=en-US&gl=US&ceid=US:en",
    # Local outlets
    "WTKR News 3":       "https://www.wtkr.com/feed/",
    "WAVY News 10":      "https://www.wavy.com/feed/",
    "Virginia Gazette":  "https://www.vagazette.com/feed/",
    "Daily Press":       "https://www.dailypress.com/feed/",
    # Economic development specific
    "Virginia Business":      "https://www.virginiabusiness.com/feed/",
    "Google News – VEDP":     "https://news.google.com/rss/search?q=Virginia+Economic+Development+Partnership+OR+VEDP&hl=en-US&gl=US&ceid=US:en",
    "Google News – Port of VA":"https://news.google.com/rss/search?q=%22Port+of+Virginia%22&hl=en-US&gl=US&ceid=US:en",
    "Google News – HR Chamber":"https://news.google.com/rss/search?q=%22Hampton+Roads+Chamber%22+OR+%22HR+Chamber%22&hl=en-US&gl=US&ceid=US:en",
    "Google News – Bisnow HR": "https://news.google.com/rss/search?q=Bisnow+%22Hampton+Roads%22+OR+commercial+real+estate+%22Hampton+Roads%22&hl=en-US&gl=US&ceid=US:en",
    "Google News – WARN Act VA":"https://news.google.com/rss/search?q=WARN+Act+Virginia+OR+layoffs+Virginia+%22Hampton+Roads%22&hl=en-US&gl=US&ceid=US:en",
}

RELEVANCE_KEYWORDS = [
    "york county","york","newport news","hampton","williamsburg","poquoson",
    "hampton roads","fort eustis","yorktown","gloucester","james city",
    "isle of wight","suffolk","chesapeake","norfolk","virginia beach",
    "peninsula","virginia","vedp","port of virginia","warn act",
]

LOCAL_ENTITIES = [
    "York County","Newport News","Hampton","Williamsburg","Poquoson",
    "Norfolk","Virginia Beach","Chesapeake","Suffolk","Portsmouth",
    "Fort Eustis","Naval Weapons Station","Yorktown","Gloucester",
    "James City County","Isle of Wight","Hampton Roads",
    "Sentara","Riverside","Huntington Ingalls","Newport News Shipbuilding",
    "NASA Langley","Thomas Nelson","William & Mary","Christopher Newport",
    "CNU","VDOT","Hampton Roads Transit","HRT","Port of Virginia",
    "City Council","Board of Supervisors","General Assembly","VEDP",
    "Hampton Roads Chamber","Virginia Business",
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
    "workforce":"Labor & Economy","worker":"Labor & Economy","layoff":"Labor & Economy",
    "business":"Business & Commerce","company":"Business & Commerce",
    "economic":"Business & Commerce","development":"Business & Commerce","investment":"Business & Commerce",
    "road":"Transportation","traffic":"Transportation","transit":"Transportation",
    "bridge":"Transportation","highway":"Transportation","port":"Transportation",
    "military":"Defense & Military","navy":"Defense & Military","fort":"Defense & Military",
    "naval":"Defense & Military","shipbuilding":"Defense & Military","defense":"Defense & Military",
    "crime":"Public Safety","police":"Public Safety","fire":"Public Safety",
    "arrest":"Public Safety","shooting":"Public Safety",
    "storm":"Environment & Weather","flood":"Environment & Weather",
    "weather":"Environment & Weather","hurricane":"Environment & Weather",
    "health":"Health & Community","hospital":"Health & Community","community":"Health & Community",
    "government":"Government & Policy","council":"Government & Policy",
    "election":"Government & Policy","tax":"Government & Policy","budget":"Government & Policy",
    "real estate":"Real Estate","commercial":"Real Estate","industrial":"Real Estate",
}

TOPIC_COLORS = {
    "Housing & Development": "#58a6ff",
    "Education":             "#3fb950",
    "Labor & Economy":       "#d29922",
    "Business & Commerce":   "#f0883e",
    "Transportation":        "#bc8cff",
    "Defense & Military":    "#79c0ff",
    "Public Safety":         "#f85149",
    "Environment & Weather": "#56d364",
    "Health & Community":    "#ff7b72",
    "Government & Policy":   "#ffa657",
    "Real Estate":           "#e3b341",
    "General News":          "#6e7681",
}

# Default watchlist issues — users can edit these in the UI
DEFAULT_WATCHLIST = [
    {"name": "Huntington Ingalls / Shipbuilding", "keywords": ["huntington ingalls", "shipbuilding", "shipyard", "newport news shipbuilding"]},
    {"name": "Workforce & Jobs", "keywords": ["layoff", "warn act", "hiring", "workforce", "unemployment", "jobs"]},
    {"name": "Port of Virginia", "keywords": ["port of virginia", "container", "cargo", "maritime"]},
    {"name": "Housing Development", "keywords": ["housing", "apartment", "zoning", "permit", "development"]},
    {"name": "Military / Fort Eustis", "keywords": ["fort eustis", "military", "army", "naval", "defense contract"]},
]

DATA_PATH = "york_news_archive.csv"
WATCHLIST_PATH = "watchlist.json"
N_TOPICS = 8
AUTO_REFRESH_HOURS = 3

_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", family="DM Sans"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)
_COLORS = ["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff","#79c0ff","#56d364","#ff7b72","#ffa657","#e3b341"]

# ─────────────────────────────────────────────
# WATCHLIST PERSISTENCE
# ─────────────────────────────────────────────
def load_watchlist():
    if os.path.exists(WATCHLIST_PATH):
        try:
            with open(WATCHLIST_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_WATCHLIST

def save_watchlist(wl):
    with open(WATCHLIST_PATH, "w") as f:
        json.dump(wl, f, indent=2)

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

def headlines_for_issue(df, keywords):
    mask = pd.Series([False] * len(df), index=df.index)
    for kw in keywords:
        mask |= df["title"].str.contains(kw, case=False, na=False)
    return df[mask].sort_values("published", ascending=False)

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
    lda = LatentDirichletAllocation(n_components=actual_n, random_state=42, max_iter=15, learning_method="online")
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

def parse_dates(df):
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df

# ─────────────────────────────────────────────
# AI BRIEFING — Claude API
# ─────────────────────────────────────────────
def generate_ai_briefing(df):
    """Call Claude API to generate a narrative economic intelligence summary."""
    now_utc = pd.Timestamp.now(tz="UTC")
    recent = df[df["published"] >= now_utc - pd.Timedelta(hours=48)].copy()
    if recent.empty:
        recent = df.sort_values("published", ascending=False).head(30)

    # Build a compact headline list for the prompt
    headlines_text = ""
    for _, row in recent.sort_values("published", ascending=False).head(40).iterrows():
        pub = row.get("published","")
        pub_str = pub.strftime("%b %d") if hasattr(pub,"strftime") else ""
        topic = row.get("topic_label","")
        headlines_text += f"- [{topic}] {row['title']} ({pub_str})\n"

    topic_counts = recent["topic_label"].value_counts().head(5).to_dict()
    topic_summary = ", ".join([f"{t} ({c})" for t,c in topic_counts.items()])

    prompt = f"""You are an economic intelligence analyst for a regional economic development organization in the Hampton Roads / York County, Virginia area.

Below are the most recent local news headlines from the past 48 hours, grouped by topic:

{headlines_text}

Topic coverage summary: {topic_summary}

Write a concise 3-4 sentence economic intelligence briefing for the team. Focus on:
- Key business, workforce, and development signals
- Any notable risks or opportunities for the region
- What the team should be watching closely

Write in clear, professional prose. No bullet points. No headers. Be specific and actionable."""

    try:
        import urllib.request
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}]
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return None

# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────
def chart_volume_over_time(df):
    d = df.dropna(subset=["published"]).copy()
    d["day"] = d["published"].dt.floor("D")
    grouped = d.groupby(["day","topic_label"]).size().reset_index(name="count")
    fig = px.bar(grouped, x="day", y="count", color="topic_label",
                 color_discrete_map=TOPIC_COLORS, template="plotly_dark",
                 labels={"day":"","count":"Headlines","topic_label":"Topic"})
    fig.update_layout(**_THEME, height=300, barmode="stack",
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11,
                                  orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                      margin=dict(l=10,r=10,t=40,b=10), hovermode="x unified")
    fig.update_traces(marker_line_width=0)
    return fig

def chart_topic_trend(df):
    d = df.dropna(subset=["published"]).copy()
    d = d.set_index("published").groupby("topic_label").resample("W").size().reset_index(name="count")
    fig = go.Figure()
    for i, topic in enumerate(d["topic_label"].unique()):
        td = d[d["topic_label"] == topic]
        fig.add_trace(go.Scatter(x=td["published"], y=td["count"], mode="lines+markers",
                                 name=topic, line=dict(color=TOPIC_COLORS.get(topic,"#6e7681"), width=2),
                                 marker=dict(size=4)))
    fig.update_layout(**_THEME, height=300,
                      legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
                      margin=dict(l=10,r=10,t=20,b=10), hovermode="x unified")
    fig.update_yaxes(title="Headlines / week")
    return fig

def chart_watchlist_trend(df, issue):
    """Headline volume over time for a single watchlist issue."""
    d = headlines_for_issue(df, issue["keywords"]).dropna(subset=["published"]).copy()
    if d.empty:
        return None
    d["day"] = d["published"].dt.floor("D")
    grouped = d.groupby("day").size().reset_index(name="count")
    fig = px.area(grouped, x="day", y="count", template="plotly_dark",
                  labels={"day":"","count":"Headlines"})
    fig.update_traces(line_color="#58a6ff", fillcolor="rgba(88,166,255,0.1)", line_width=2)
    fig.update_layout(**_THEME, height=140, showlegend=False,
                      margin=dict(l=10,r=10,t=10,b=10))
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

def chart_source_bar(df):
    counts = df["source"].value_counts().reset_index()
    counts.columns = ["Source","Count"]
    fig = px.bar(counts, x="Count", y="Source", orientation="h",
                 template="plotly_dark", color_discrete_sequence=["#58a6ff"])
    fig.update_layout(**_THEME, height=300, showlegend=False,
                      margin=dict(l=10,r=10,t=10,b=10))
    fig.update_traces(marker_line_width=0)
    return fig

# ─────────────────────────────────────────────
# WEEKLY REPORT EXPORT
# ─────────────────────────────────────────────
def generate_weekly_report(df, ai_summary, watchlist):
    now = datetime.now(timezone.utc)
    week_str = now.strftime("Week of %B %d, %Y")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    week_df = df[df["published"] >= cutoff].copy()

    topic_counts = week_df["topic_label"].value_counts().to_dict()
    topic_rows = "".join([
        f'<tr><td>{t}</td><td style="text-align:center;">{c}</td></tr>'
        for t,c in topic_counts.items()
    ])

    # Top headlines per topic
    topic_sections = ""
    for topic in week_df["topic_label"].unique():
        color = TOPIC_COLORS.get(topic, "#6e7681")
        t_df = week_df[week_df["topic_label"] == topic].sort_values("published", ascending=False).head(5)
        items = ""
        for _, row in t_df.iterrows():
            pub = row.get("published","")
            pub_str = pub.strftime("%b %d") if hasattr(pub,"strftime") else ""
            link = row.get("link","")
            title = row.get("title","")
            href = f'<a href="{link}" style="color:#1a56db;">{title}</a>' if link else title
            items += f'<li style="margin-bottom:6px;">{href} <span style="color:#6b7280;font-size:12px;">— {pub_str} · {row.get("source","")}</span></li>'
        topic_sections += f"""
        <div style="margin-bottom:24px;">
            <h3 style="font-family:Georgia,serif;font-size:15px;color:{color};margin:0 0 8px 0;
                       text-transform:uppercase;letter-spacing:1px;">{topic}</h3>
            <ul style="margin:0;padding-left:18px;font-size:14px;color:#374151;line-height:1.6;">{items}</ul>
        </div>"""

    # Watchlist highlights
    watchlist_html = ""
    for issue in watchlist:
        i_df = headlines_for_issue(week_df, issue["keywords"]).head(3)
        if i_df.empty:
            continue
        items = ""
        for _, row in i_df.iterrows():
            link = row.get("link","")
            title = row.get("title","")
            href = f'<a href="{link}" style="color:#1a56db;">{title}</a>' if link else title
            items += f'<li style="margin-bottom:5px;">{href}</li>'
        watchlist_html += f"""
        <div style="margin-bottom:16px;">
            <strong style="font-size:14px;">{issue['name']}</strong>
            <ul style="margin:4px 0 0 0;padding-left:18px;font-size:13px;color:#374151;">{items}</ul>
        </div>"""

    summary_block = f'<p style="font-size:15px;color:#1f2937;line-height:1.75;margin:0;">{ai_summary}</p>' if ai_summary else '<p style="color:#6b7280;font-style:italic;">AI summary not available.</p>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hampton Roads Economic Intelligence Brief — {week_str}</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; background: #f9fafb; color: #111827; margin: 0; padding: 0; }}
  .wrapper {{ max-width: 720px; margin: 0 auto; background: white; padding: 48px 56px; }}
  h1 {{ font-size: 26px; font-weight: bold; margin: 0 0 4px 0; color: #111827; }}
  h2 {{ font-size: 17px; font-weight: bold; margin: 32px 0 12px 0; color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
  .label {{ font-family: Arial, sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #6b7280; margin: 0 0 6px 0; }}
  .week {{ font-size: 14px; color: #6b7280; margin: 0 0 32px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: Arial,sans-serif; }}
  th {{ background: #f3f4f6; padding: 8px 12px; text-align: left; color: #374151; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #f3f4f6; color: #374151; }}
  .footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-family: Arial,sans-serif; font-size: 11px; color: #9ca3af; }}
</style>
</head>
<body>
<div class="wrapper">
  <p class="label">Hampton Roads Economic Development</p>
  <h1>Economic Intelligence Brief</h1>
  <p class="week">{week_str} &nbsp;·&nbsp; Generated {now.strftime("%B %d, %Y at %H:%M UTC")}</p>

  <h2>Executive Summary</h2>
  {summary_block}

  <h2>Coverage This Week</h2>
  <table>
    <tr><th>Topic</th><th style="text-align:center;">Headlines</th></tr>
    {topic_rows}
  </table>

  <h2>Watchlist Highlights</h2>
  {watchlist_html if watchlist_html else '<p style="color:#6b7280;font-size:14px;">No watchlist matches this week.</p>'}

  <h2>Headlines by Topic</h2>
  {topic_sections}

  <div class="footer">
    Hampton Roads Economic Intelligence Monitor &nbsp;·&nbsp; Auto-generated report &nbsp;·&nbsp; {now.strftime("%Y")}
  </div>
</div>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────
def render_briefing(df, watchlist):
    now_utc = pd.Timestamp.now(tz="UTC")
    recent = df[df["published"] >= now_utc - pd.Timedelta(hours=24)]
    if recent.empty:
        recent = df.sort_values("published", ascending=False).head(10)

    top_topics = recent["topic_label"].value_counts().head(4)
    top_headlines = recent.sort_values("published", ascending=False).head(6)
    total_24h = len(recent)
    sources_24h = recent["source"].nunique()

    # Topic activity row
    topic_html = ""
    for topic, cnt in top_topics.items():
        color = TOPIC_COLORS.get(topic, "#6e7681")
        topic_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin-right:14px;font-size:0.8rem;color:#c9d1d9;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{color};'
            f'display:inline-block;"></span>{topic} '
            f'<span style="color:#6e7681;">({cnt})</span></span>'
        )

    # AI summary (cached per session to avoid repeated API calls)
    if "ai_briefing" not in st.session_state or st.session_state.get("ai_briefing_stale", True):
        with st.spinner("Generating AI briefing…"):
            summary = generate_ai_briefing(df)
        st.session_state.ai_briefing = summary
        st.session_state.ai_briefing_stale = False
    else:
        summary = st.session_state.ai_briefing

    st.markdown(f"""
    <div class="briefing-wrap">
        <p class="briefing-label">📋 Today's Briefing &nbsp;·&nbsp;
            <span style="font-weight:400;color:#6e7681;font-size:0.75rem;">
                {total_24h} headlines · {sources_24h} sources · last 24 hrs
            </span>
        </p>
    """, unsafe_allow_html=True)

    if summary:
        st.markdown(f'<p class="briefing-summary">{summary}</p>', unsafe_allow_html=True)

    st.markdown(f'<div style="margin-bottom:0.75rem;">{topic_html}</div>', unsafe_allow_html=True)

    for i, (_, row) in enumerate(top_headlines.iterrows(), 1):
        link = row.get("link","")
        title = row.get("title","")
        topic = row.get("topic_label","")
        pub = row.get("published","")
        pub_str = pub.strftime("%I:%M %p") if hasattr(pub,"strftime") else ""
        source = row.get("source","")
        title_html = f'<a href="{link}" target="_blank" style="color:#c9d1d9;text-decoration:none;">{title}</a>' if link else title
        chip = f'<span class="topic-chip">{topic}</span>' if topic else ""
        st.markdown(f"""
        <div class="briefing-item">
            <span class="briefing-num">{i}</span>
            <span class="briefing-text">{title_html}{chip}</span>
            <span class="briefing-meta">{pub_str} · {source}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_watchlist(df, watchlist):
    if not watchlist:
        st.info("No watchlist issues defined. Add some below.")
        return

    cols = st.columns(min(len(watchlist), 2))
    for i, issue in enumerate(watchlist):
        col = cols[i % 2]
        with col:
            i_df = headlines_for_issue(df, issue["keywords"])
            count = len(i_df)
            now_utc = pd.Timestamp.now(tz="UTC")
            count_24h = len(i_df[i_df["published"] >= now_utc - pd.Timedelta(hours=24)]) if "published" in i_df.columns else 0

            st.markdown(f"""
            <div class="watchlist-card">
                <p class="watchlist-name">📌 {issue['name']}</p>
                <p class="watchlist-meta">{count} total headlines &nbsp;·&nbsp; {count_24h} in last 24h</p>
            """, unsafe_allow_html=True)

            trend_fig = chart_watchlist_trend(df, issue)
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True, key=f"wl_trend_{i}")

            for _, row in i_df.head(4).iterrows():
                link = row.get("link","")
                title = row.get("title","")
                pub = row.get("published","")
                pub_str = pub.strftime("%b %d") if hasattr(pub,"strftime") else ""
                href = f'<a href="{link}" target="_blank">{title}</a>' if link else title
                st.markdown(
                    f'<div class="watchlist-headline">{href}'
                    f'<span style="color:#6e7681;font-size:0.7rem;"> — {pub_str}</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)


def headline_card(row):
    title = row.get("title","")
    link = row.get("link","")
    topic = row.get("topic_label","")
    entities = row.get("entities","")
    pub = row.get("published","")
    pub_str = pub.strftime("%b %d, %Y") if hasattr(pub,"strftime") else str(pub)[:10]
    source = row.get("source","")
    title_html = f'<a href="{link}" target="_blank">{title}</a>' if link else title
    tags = ""
    if topic:
        color = TOPIC_COLORS.get(topic,"#6e7681")
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
# AUTO-REFRESH LOGIC
# ─────────────────────────────────────────────
def should_auto_refresh():
    last = st.session_state.get("last_fetch_time", 0)
    return (time.time() - last) > (AUTO_REFRESH_HOURS * 3600)

def do_fetch(selected_feeds, n_topics):
    with st.spinner("Fetching headlines…"):
        df = scrape_news(selected_feeds)
    if df.empty:
        return None
    with st.spinner("Modeling topics…"):
        df = assign_topics(df, n_topics)
    df = parse_dates(df)
    st.session_state.df = df
    st.session_state.last_fetch_time = time.time()
    st.session_state.ai_briefing_stale = True
    return df

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # ── Session state init ──
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = 0
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    watchlist = st.session_state.watchlist

    # ── Header ──
    now_str = datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC")
    last_fetch = st.session_state.last_fetch_time
    last_str = datetime.fromtimestamp(last_fetch, tz=timezone.utc).strftime("%H:%M UTC") if last_fetch else "Never"
    next_refresh_mins = max(0, int((AUTO_REFRESH_HOURS * 3600 - (time.time() - last_fetch)) / 60)) if last_fetch else 0

    st.markdown(f"""
    <div class="header-wrap">
        <div>
            <p class="header-title">📊 Hampton Roads Economic Intelligence</p>
            <p class="header-sub">York County · Newport News · Hampton Roads Regional Monitor</p>
        </div>
        <div class="header-time">
            {now_str}<br>
            <span class="refresh-badge">
                <span class="refresh-dot"></span>
                Auto-refresh every {AUTO_REFRESH_HOURS}h &nbsp;·&nbsp; Last fetch {last_str}
            </span>
        </div>
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
        n_topics = st.slider("Topic clusters", 3, 12, N_TOPICS)
        st.markdown("---")
        st.markdown("**Topic Filter**")
        topic_placeholder = st.empty()
        st.markdown("---")
        days_back = st.slider("Headlines from last N days", 1, 90, 30)
        st.markdown("---")
        run_btn = st.button("🔄 Fetch Now", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("**📌 Manage Watchlist**")
        with st.expander("Add / Remove Issues"):
            # Add new issue
            new_name = st.text_input("Issue name", placeholder="e.g. Amazon Warehouse")
            new_kws = st.text_input("Keywords (comma-separated)", placeholder="amazon, warehouse, logistics")
            if st.button("➕ Add Issue"):
                if new_name and new_kws:
                    kw_list = [k.strip().lower() for k in new_kws.split(",") if k.strip()]
                    watchlist.append({"name": new_name, "keywords": kw_list})
                    save_watchlist(watchlist)
                    st.session_state.watchlist = watchlist
                    st.rerun()
            # Remove existing
            if watchlist:
                remove_name = st.selectbox("Remove issue", ["—"] + [w["name"] for w in watchlist])
                if st.button("🗑 Remove") and remove_name != "—":
                    watchlist = [w for w in watchlist if w["name"] != remove_name]
                    save_watchlist(watchlist)
                    st.session_state.watchlist = watchlist
                    st.rerun()

        st.markdown("---")
        st.caption(f"Headlines cached in `york_news_archive.csv`\nAuto-refreshes every {AUTO_REFRESH_HOURS} hours")

    # ── Fetch logic (manual or auto) ──
    if run_btn or st.session_state.df.empty or should_auto_refresh():
        if not selected_feeds:
            st.warning("Select at least one news source.")
            return
        df = do_fetch(selected_feeds, n_topics)
        if df is None:
            st.error("No headlines retrieved. Check your connection or sources.")
            return
        st.success(f"✅ {len(df):,} headlines from {df['source'].nunique()} sources")

    df = st.session_state.df
    if df.empty:
        st.info("Click **Fetch Now** in the sidebar to get started.")
        return

    # ── Filters ──
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
    df_f = df[df["published"] >= cutoff].copy() if "published" in df.columns else df.copy()
    all_topics = sorted(df_f["topic_label"].dropna().unique().tolist())
    sel_topics = topic_placeholder.multiselect("Topics", options=all_topics, default=all_topics)
    df_f = df_f[df_f["topic_label"].isin(sel_topics)]

    # ── KPIs ──
    total = len(df_f)
    now_utc = pd.Timestamp.now(tz="UTC")
    last_24h = int((df_f["published"] >= now_utc - pd.Timedelta(hours=24)).sum()) if "published" in df_f.columns else 0
    n_sources = df_f["source"].nunique()
    top_topic = df_f["topic_label"].value_counts().idxmax() if total else "—"
    watchlist_hits = sum(len(headlines_for_issue(df_f, w["keywords"])) for w in watchlist)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Headlines", f"{total:,}")
    k2.metric("Last 24 Hours", f"{last_24h}")
    k3.metric("Sources Active", f"{n_sources}")
    k4.metric("Top Topic", top_topic)
    k5.metric("Watchlist Hits", f"{watchlist_hits}")
    st.markdown("---")

    # ── Main tabs ──
    tab_brief, tab_watch, tab_search, tab_trends, tab_all, tab_report = st.tabs([
        "📋 Briefing", "📌 Watchlist", "🔍 Search", "📈 Trends", "📰 All Headlines", "📄 Weekly Report"
    ])

    with tab_brief:
        render_briefing(df_f, watchlist)

    with tab_watch:
        st.markdown('<p class="section-title">📌 Watchlist — Tracked Issues</p>', unsafe_allow_html=True)
        render_watchlist(df_f, watchlist)

    with tab_search:
        st.markdown('<p class="section-title">🔍 Keyword Search</p>', unsafe_allow_html=True)
        search_query = st.text_input("", placeholder="Search headlines… e.g. shipyard, zoning, port, layoffs",
                                     label_visibility="collapsed")
        if search_query.strip():
            mask = df_f["title"].str.contains(search_query.strip(), case=False, na=False)
            df_search = df_f[mask].sort_values("published", ascending=False)
            st.caption(f"{len(df_search)} result{'s' if len(df_search) != 1 else ''} for \"{search_query}\"")
            if df_search.empty:
                st.info("No headlines matched.")
            else:
                for _, row in df_search.head(100).iterrows():
                    headline_card(row)
        else:
            st.caption("Type a keyword above to search across all headlines.")

    with tab_trends:
        st.markdown('<p class="section-title">📈 Historical Trends</p>', unsafe_allow_html=True)
        t1, t2 = st.tabs(["Daily Volume by Topic", "Weekly Topic Trends"])
        with t1:
            st.plotly_chart(chart_volume_over_time(df_f), use_container_width=True)
        with t2:
            st.plotly_chart(chart_topic_trend(df_f), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Topic distribution")
            st.plotly_chart(chart_topic_donut(df_f), use_container_width=True)
        with c2:
            st.caption("Headlines by source")
            st.plotly_chart(chart_source_bar(df_f), use_container_width=True)

    with tab_all:
        st.markdown('<p class="section-title">📰 All Headlines</p>', unsafe_allow_html=True)
        col_sort, col_n, _ = st.columns([2,2,4])
        with col_sort:
            sort_by = st.selectbox("Sort", ["Newest first","Oldest first","Topic (A–Z)"],
                                   label_visibility="collapsed")
        with col_n:
            top_n = st.slider("Show", 10, 300, 60, key="all_n", label_visibility="collapsed")

        if sort_by == "Newest first":
            df_sorted = df_f.sort_values("published", ascending=False)
        elif sort_by == "Oldest first":
            df_sorted = df_f.sort_values("published", ascending=True)
        else:
            df_sorted = df_f.sort_values("topic_label", ascending=True)

        if sort_by == "Topic (A–Z)":
            current_topic = None
            for _, row in df_sorted.head(top_n).iterrows():
                if row.get("topic_label") != current_topic:
                    current_topic = row.get("topic_label","")
                    color = TOPIC_COLORS.get(current_topic,"#6e7681")
                    st.markdown(
                        f'<p style="font-family:Syne,sans-serif;font-size:0.78rem;font-weight:700;'
                        f'color:{color};text-transform:uppercase;letter-spacing:1.5px;'
                        f'margin:1rem 0 0.3rem 0;">{current_topic}</p>',
                        unsafe_allow_html=True
                    )
                headline_card(row)
        else:
            for _, row in df_sorted.head(top_n).iterrows():
                headline_card(row)

        st.markdown("---")
        with st.expander("📥 Export CSV"):
            csv = df_sorted.head(top_n).to_csv(index=False).encode("utf-8")
            st.download_button("Download current view", data=csv,
                               file_name=f"york_news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv")

    with tab_report:
        st.markdown('<p class="section-title">📄 Weekly Intelligence Report</p>', unsafe_allow_html=True)
        st.markdown("Generate a formatted HTML report for the past 7 days — ready to email or print.")
        now_utc = pd.Timestamp.now(tz="UTC")
        week_df = df_f[df_f["published"] >= now_utc - pd.Timedelta(days=7)]
        st.caption(f"{len(week_df)} headlines in the last 7 days across {week_df['topic_label'].nunique()} topics")

        if st.button("⚡ Generate Report", type="primary"):
            with st.spinner("Building report…"):
                ai_sum = st.session_state.get("ai_briefing")
                report_html = generate_weekly_report(df_f, ai_sum, watchlist)
            st.download_button(
                "📥 Download Report (HTML)",
                data=report_html.encode("utf-8"),
                file_name=f"hampton_roads_intel_brief_{datetime.now().strftime('%Y%m%d')}.html",
                mime="text/html",
            )
            st.success("Report ready — click Download above. Open the .html file in any browser to view or print.")
            with st.expander("Preview report"):
                st.components.v1.html(report_html, height=600, scrolling=True)


if __name__ == "__main__":
    main()
