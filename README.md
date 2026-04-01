# York County Regional News Monitor

A professional Streamlit dashboard for tracking local news sentiment across Hampton Roads / the Virginia Peninsula.

---

## Features

- **8 curated RSS sources** — Google News topic feeds, WTKR, WAVY, Virginia Gazette, Daily Press, and more
- **Automatic topic discovery** — LDA (Latent Dirichlet Allocation) clusters headlines into topics automatically; no hand-coding required
- **Human-readable topic labels** — Keyword-to-label mapping converts LDA outputs into names like "Public Safety", "Defense & Military", "Housing & Development"
- **Named entity extraction** — spaCy pulls organizations, locations, and persons from each headline
- **VADER sentiment scoring** — Compound scores with Positive / Neutral / Negative / Risk Signal classification
- **Interactive Plotly charts** — Weekly trend lines, sentiment distribution bar chart, topic × sentiment heatmap, source breakdown donut
- **Sidebar controls** — Filter by source, topic, sentiment, and date range on the fly
- **CSV export** — Download filtered headlines with all metadata
- **Persistent archive** — Headlines accumulate in `york_news_archive.csv` across runs

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3. Run the app

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

---

## Usage

1. In the **sidebar**, select which news sources to pull from (all selected by default).
2. Adjust the **Topic clusters** slider (default 8) — higher values = more granular topics.
3. Use **Sentiment Filter** and **Topic Filter** to focus the view.
4. Set the **date range** with the "Last N days" slider.
5. Click **🔄 Fetch & Analyze** to pull fresh headlines.
6. Browse the **Headlines tabs** (All / Negative Alerts / Positive Signals / Risk Signals).
7. Export filtered data to CSV from the **Export Data** expander at the bottom.

---

## Adding More RSS Feeds

Edit the `RSS_FEEDS` dictionary in `app.py`:

```python
RSS_FEEDS = {
    "My Custom Feed": "https://example.com/feed.xml",
    ...
}
```

Any feed that isn't a Google News URL will pass all headlines through (no keyword filter). Google News URLs are filtered against `RELEVANCE_KEYWORDS`.

---

## Improving Topic Labels

Edit the `TOPIC_LABEL_MAP` dictionary in `app.py`. Each key is a word the LDA model might surface; the value is the human-readable label assigned when that word appears in a topic's top keywords:

```python
TOPIC_LABEL_MAP = {
    "shipyard": "Defense & Military",
    "flood": "Environment & Weather",
    ...
}
```

---

## Deployment (Streamlit Cloud)

1. Push this folder to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set the main file path to `app.py`.
4. Add a `packages.txt` file to the repo root containing:
   ```
   python-dev
   ```
5. The spaCy model will be downloaded automatically on first run via the `load_nlp_models()` cache.

---

## File Structure

```
york_news_monitor/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── york_news_archive.csv   # Auto-generated; grows with each fetch run
```
