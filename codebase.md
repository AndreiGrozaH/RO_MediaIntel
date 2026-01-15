# .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
.venv/
venv/
ENV/
env/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Next.js
.next/
out/

# Environment
.env
.env.local
.env*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Build
dist/
build/

# Cache
.cache/
*.egg-info/

```

# app.py

```py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datetime import datetime, timedelta
import dateparser # pip install dateparser
import re

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="RO-MediaIntel Pro", layout="wide", page_icon="📡")

# Custom CSS for Modern UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .article-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        border-left: 5px solid #FF4B4B;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
        color: white;
    }
    .badge-person { background-color: #3498db; }
    .badge-org { background-color: #f1c40f; color: black; }
    .badge-loc { background-color: #2ecc71; }
    .badge-money { background-color: #e74c3c; }
    a { text-decoration: none; color: #2c3e50; font-weight: bold; font-size: 18px; }
    a:hover { color: #FF4B4B; }
    .meta-text { color: #7f8c8d; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# --- 2. AI ENGINE (Cached) ---
@st.cache_resource
def load_model():
    model_name = "dumitrescustefan/bert-base-romanian-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # Map IDs to Labels
    model.config.id2label = {
        0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-ORG", 4: "I-ORG",
        5: "B-GPE", 6: "I-GPE", 7: "B-LOC", 8: "I-LOC",
        9: "B-NAT_REL_POL", 10: "I-NAT_REL_POL", 11: "B-EVENT", 12: "I-EVENT",
        13: "B-LANGUAGE", 14: "I-LANGUAGE", 15: "B-WORK_OF_ART", 16: "I-WORK_OF_ART",
        17: "B-DATETIME", 18: "I-DATETIME", 19: "B-PERIOD", 20: "I-PERIOD",
        21: "B-MONEY", 22: "I-MONEY", 23: "B-QUANTITY", 24: "I-QUANTITY",
        25: "B-NUMERIC_VALUE", 26: "I-NUMERIC_VALUE", 27: "B-ORDINAL", 28: "I-ORDINAL",
        29: "B-FACILITY", 30: "I-FACILITY"
    }
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

nlp = load_model()

# --- 3. INTELLIGENT SCRAPER ENGINE ---
# --- IMPROVED DATE PARSER (Specific Selectors) ---
import unicodedata

def try_parse_date(card_item, url_link):
    """
    Aggressive Date Hunter: Looks for ANY date pattern in the text or URL.
    """
    try:
        # 1. URL STRATEGY (HotNews/G4Media)
        # Matches /2025/12/13/
        url_match = re.search(r'/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/', url_link)
        if url_match:
            return datetime(int(url_match.group(1)), int(url_match.group(2)), int(url_match.group(3)))

        # 2. RAW TEXT HARVEST
        # Get every piece of text from the card
        raw_text = card_item.get_text(" ", strip=True)
        # Normalize: "13.  12. 2025" -> "13.12.2025"
        clean_text = " ".join(raw_text.split())

        # 3. DIGI24 "BLIND" PATTERN
        # We stop looking for "Data publicării". We just look for "13.12.2025"
        # The regex matches: 2 digits + dot + 2 digits + dot + 4 digits
        simple_date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', clean_text)
        if simple_date_match:
            return datetime.strptime(simple_date_match.group(1), "%d.%m.%Y")

        # 4. HOTNEWS SPECIFIC (Long Format)
        # Matches: "luni, 10 februarie 2025" or "10 februarie 2025"
        # We look for: (Day Name optional) + Number + Month Name + Year
        hn_match = re.search(r'(?:luni|mar[t\u021Bi]|miercuri|joi|vineri|s[a\u0103]mb[a\u0103]t[a\u0103]|duminic[a\u0103])?,?\s*(\d{1,2})\s+(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+(\d{4})', clean_text, re.IGNORECASE)
        if hn_match:
            return dateparser.parse(hn_match.group(0), languages=['ro'])
        
        # return parse_short_ro_date(card_item)
            
        # 5. RELATIVE TIME
        # "Acum 2 ore"
        rel_match = re.search(r'acum\s+(\d+)\s+(min|ore|zile)', clean_text, re.IGNORECASE)
        if rel_match:
            val = int(rel_match.group(1))
            unit = rel_match.group(2).lower()
            now = datetime.now()
            if 'min' in unit: return now - timedelta(minutes=val)
            if 'ore' in unit: return now - timedelta(hours=val)
            if 'zile' in unit: return now - timedelta(days=val)

    except Exception:
        pass

    # If all fails, return None so we can see "(Date Failed)"
    return None

# --- IMPROVED SCRAPER WITH PAGINATION ---
def scrape_generic(base_url, max_pages=3):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    all_articles = []
    page_bar = st.progress(0)
    
    for page_num in range(1, max_pages + 1):
        # 1. Pagination Logic
        if page_num == 1: current_url = base_url
        elif "digi24.ro" in base_url: current_url = f"{base_url}?p={page_num}"
        elif "hotnews.ro" in base_url: current_url = f"{base_url}/page/{page_num}"
        else: current_url = f"{base_url}/page/{page_num}"
        
        try:
            response = requests.get(current_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # --- CONTAINER STRATEGY ---
            # Instead of looking for links, we look for the "Boxes" that hold content.
            cards = []
            
            # DIGI24 / HOTNEWS (Standard HTML5)
            cards.extend(soup.find_all('article'))
            
            # G4MEDIA / WORDPRESS (Div based)
            cards.extend(soup.find_all('div', class_=re.compile(r'(post-review|post-card|entry-content|articol-card)')))
            
            # Safety limit
            cards = cards[:40] 

            for card in cards:
                # A. Find Headline inside the Box
                link_tag = card.find('a')
                # Sometimes the first link is an image, find the text link
                if link_tag and not link_tag.get_text(strip=True):
                    headers_tags = card.find_all(['h2', 'h3', 'h4'])
                    for h in headers_tags:
                        if h.find('a'):
                            link_tag = h.find('a')
                            break
                            
                if link_tag and link_tag.has_attr('href'):
                    title = link_tag.get_text(strip=True)
                    link = link_tag['href']
                    if link.startswith('/'): link = "https://" + base_url.replace("https://", "").split('/')[0] + link
                    
                    if len(title) > 25:
                        # B. Find Date inside the SAME Box (The Context is now perfect)
                        pub_date = try_parse_date(card, link)
                        
                        # DEBUG LABEL: Mark if date was found or not
                        if pub_date is None:
                            # If failed, check URL as last resort
                            url_date = try_parse_date_from_url(link)
                            if url_date:
                                pub_date = url_date
                                source_label = f"{base_url.split('/')[2]} (URL Date)"
                            else:
                                pub_date = datetime.now()
                                source_label = f"{base_url.split('/')[2]} (Date Failed)"
                        else:
                             source_label = base_url.split('/')[2]

                        # C. Add to list
                        if not any(a['Link'] == link for a in all_articles):
                            all_articles.append({
                                "Source": source_label,
                                "Headline": title,
                                "Link": link,
                                "Date": pub_date
                            })

        except Exception:
            pass
        
        page_bar.progress(page_num / max_pages)
        
    page_bar.empty()
    return all_articles

# Helper function just for URLs
def try_parse_date_from_url(url_link):
    match = re.search(r'/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/', url_link)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None

# --- NEW HELPER FOR SHORT DATES ---
def parse_short_ro_date(card_item):
    """
    Handles dates like '23 ian.' by adding the current year.
    """
    try:
        text = card_item.get_text(" ", strip=True)
        # Match "23 ian" or "23 ian."
        short_match = re.search(r'(\d{1,2})\s+(ian|feb|mar|apr|mai|iun|iul|aug|sep|oct|nov|dec)\.?', text, re.IGNORECASE)
        if short_match:
            # We found "23 ian". We must assume the year.
            day = short_match.group(1)
            month_str = short_match.group(2)
            
            # Use dateparser to parse "23 ian 2025" (adding current year)
            current_year = datetime.now().year
            date_str = f"{day} {month_str} {current_year}"
            parsed = dateparser.parse(date_str, languages=['ro'])
            
            # Logic check: If today is Jan 2025, and we see "23 Dec", it was likely Dec 2024.
            if parsed > datetime.now() + timedelta(days=2): # If date is in future
                date_str = f"{day} {month_str} {current_year - 1}"
                parsed = dateparser.parse(date_str, languages=['ro'])
            
            return parsed
    except:
        pass
    return None


def analyze_headline(text):
    results = nlp(text)
    meta = {"PERSON": [], "ORG": [], "GPE": [], "VALUES": [], "CONTEXT": []}
    
    for item in results:
        tag = item['entity_group']
        word = item['word'].replace("##", "")
        if tag == 'PERSON': meta['PERSON'].append(word)
        elif tag == 'ORG': meta['ORG'].append(word)
        elif tag in ['GPE', 'LOC']: meta['GPE'].append(word)
        elif tag in ['MONEY', 'QUANTITY']: meta['VALUES'].append(f"{word} ({tag})")
        elif tag in ['EVENT', 'FACILITY']: meta['CONTEXT'].append(f"{word} ({tag})")
    return meta

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🎛️ Control Panel")
    
    st.subheader("1. Data Sources")
    default_sites = [
        "https://hotnews.ro/c/politica", 
        "https://www.digi24.ro/stiri/actualitate/politica",        
        "https://spotmedia.ro/stiri/politica"
    ]
    
    # DYNAMIC URL INPUT
    custom_url = st.text_input("➕ Add Custom News URL", placeholder="e.g. https://techcrunch.com")
    if custom_url:
        default_sites.append(custom_url)
        st.success(f"Added: {custom_url}")
    
    selected_sites = st.multiselect("Active Sources", default_sites, default=default_sites)
    
    st.divider()
    
    st.subheader("2. Date Filter")
    # Date Range Picker
    today = datetime.now()
    last_month = today - timedelta(days=30)
    date_range = st.date_input(
        "Select Date Range",
        (last_month, today),
        format="YYYY-MM-DD"
    )
    
    st.divider()
    
    run_btn = st.button("🚀 LAUNCH SCRAPER")

# --- 5. MAIN DASHBOARD ---
st.title("📡 RO-MediaIntel Pro")
st.markdown("### Thematic News Aggregation & Intelligence System")

if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()

if run_btn:
    with st.status("🔍 Scanning Intelligence Stream...", expanded=True) as status:
        all_articles = []
        
        for site in selected_sites:
            st.write(f"Connecting to {site}...")
            news = scrape_generic(site, max_pages=10)
            st.write(f"Found {len(news)} articles.")
            all_articles.extend(news)
            
        status.update(label="Analysis Complete!", state="complete", expanded=False)
        
        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Run NLP Analysis
            processed_data = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                meta = analyze_headline(row['Headline'])
                processed_data.append({
                    **row,
                    "Person": meta['PERSON'],
                    "Org": meta['ORG'],
                    "Loc": meta['GPE'],
                    "Values": meta['VALUES'],
                    "Context": meta['CONTEXT']
                })
                progress_bar.progress((i + 1) / len(df))
            
            st.session_state['data'] = pd.DataFrame(processed_data)
            progress_bar.empty()
        else:
            st.error("No articles found. Check your URLs.")

# --- 6. DISPLAY RESULTS (MODERN UI) ---
if not st.session_state['data'].empty:
    df = st.session_state['data']
    
    show_all = st.checkbox("⚠️ Debug: Ignore Date Filter (Show All Articles)", value=True)
    # FILTER LOGIC (DATE)
    if not show_all:
        # Only apply filter if the box is UNCHECKED
        if len(date_range) == 2:
            start_date, end_date = date_range
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            df = df.loc[mask]
    
    st.subheader(f"Found {len(df)} Articles")
    
    # Render Cards
    for index, row in df.iterrows():
        # HTML Injection for Custom Card Look
        
        # Badges Generation
        badges_html = ""
        for p in row['Person']: badges_html += f'<span class="badge badge-person">👤 {p}</span>'
        for o in row['Org']: badges_html += f'<span class="badge badge-org">🏢 {o}</span>'
        for l in row['Loc']: badges_html += f'<span class="badge badge-loc">📍 {l}</span>'
        for v in row['Values']: badges_html += f'<span class="badge badge-money">💰 {v}</span>'
        
        display_date = row['Date'].strftime('%Y-%m-%d %H:%M')
    
        card_html = f"""
        <div class="article-card">
            <div style="display:flex; justify-content:space-between;">
                <span class="meta-text">{row['Source']} • {display_date}</span>
            </div>
            <a href="{row['Link']}" target="_blank">{row['Headline']}</a>
            <div style="margin-top: 10px;">
                {badges_html if badges_html else "<span class='meta-text'>No entities detected</span>"}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

else:
    st.info("👋 Click 'Launch Scraper' in the sidebar to begin.")
```

# backend\__init__.py

```py

```

# backend\.gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.pyc
*.pyo
*.pyd

# Virtual environments
.venv/
venv/
ENV/
env/
.virtualenv/

# Distribution / packaging
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Unit test / coverage
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Logs
*.log
logs/

# Database
*.db
*.sqlite
*.sqlite3

# Model cache
models/
.cache/
transformers_cache/
huggingface/

# OS
Thumbs.db
.DS_Store

```

# backend\app\__init__.py

```py

```

# backend\app\config.py

```py
from functools import lru_cache
from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "RO-MediaIntel API"
    default_sources: List[AnyHttpUrl] = [
        "https://www.digi24.ro/stiri/actualitate/politica",
        "https://spotmedia.ro/stiri/politica",
        "https://www.antena3.ro/politica",
        "https://www.stiripesurse.ro/politica",
        "https://www.dcnews.ro/politica",
        "https://www.digi24.ro/stiri/extern",
        "https://adevarul.ro/politica",
        "https://www.libertatea.ro/politica",
        "https://www.g4media.ro/category/politica",
    ]
    model_name: str = "dumitrescustefan/bert-base-romanian-ner"

    model_config = {
        "env_file": ".env",
        "protected_namespaces": ("settings_",),
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

```

# backend\app\constants.py

```py
LABEL_ID2LABEL = {
    0: "O", 1: "B-PERSON", 2: "I-PERSON", 3: "B-ORG", 4: "I-ORG",
    5: "B-GPE", 6: "I-GPE", 7: "B-LOC", 8: "I-LOC",
    9: "B-NAT_REL_POL", 10: "I-NAT_REL_POL", 11: "B-EVENT", 12: "I-EVENT",
    13: "B-LANGUAGE", 14: "I-LANGUAGE", 15: "B-WORK_OF_ART", 16: "I-WORK_OF_ART",
    17: "B-DATETIME", 18: "I-DATETIME", 19: "B-PERIOD", 20: "I-PERIOD",
    21: "B-MONEY", 22: "I-MONEY", 23: "B-QUANTITY", 24: "I-QUANTITY",
    25: "B-NUMERIC_VALUE", 26: "I-NUMERIC_VALUE", 27: "B-ORDINAL", 28: "I-ORDINAL",
    29: "B-FACILITY", 30: "I-FACILITY",
}

```

# backend\app\main.py

```py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import ScrapeRequest, ScrapeResponse, Article
from .services.ner import get_ner_pipeline
from .services.scraper import run_pipeline

logger = logging.getLogger(__name__)

app = FastAPI(title="RO-MediaIntel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scrape", response_model=ScrapeResponse)
def scrape(payload: ScrapeRequest):
    settings = get_settings()
    sources = [str(u) for u in (payload.sources or settings.default_sources)]
    max_pages = payload.max_pages or 3
    try:
        nlp = get_ner_pipeline()
        articles_raw = run_pipeline(nlp, sources=sources, max_pages=max_pages)
    except Exception as exc:
        logger.exception("scrape_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Date filtering if provided
    filtered = []
    for art in articles_raw:
        if payload.date_from or payload.date_to:
            dt = art["date"].date()
            if payload.date_from and dt < payload.date_from:
                continue
            if payload.date_to and dt > payload.date_to:
                continue
        filtered.append(art)

    articles = [
        Article(
            source=a["source"],
            headline=a["headline"],
            link=a["link"],
            date=a["date"],
            person=a.get("person", []),
            org=a.get("org", []),
            loc=a.get("loc", []),
            values=a.get("values", []),
            context=a.get("context", []),
        )
        for a in filtered
    ]

    return ScrapeResponse(total=len(articles), articles=articles)

```

# backend\app\schemas.py

```py
from datetime import datetime, date
from typing import List, Optional, Dict
from pydantic import AnyUrl, BaseModel, HttpUrl, Field, validator


class ScrapeRequest(BaseModel):
    sources: Optional[List[AnyUrl]] = Field(
        None, description="List of news site URLs to scrape (optional, defaults to server settings)"
    )
    max_pages: Optional[int] = Field(
        None, ge=1, le=15, description="How many pages to fetch per source (optional, defaults to 3)"
    )
    date_from: Optional[date] = Field(None, description="Start date filter (inclusive)")
    date_to: Optional[date] = Field(None, description="End date filter (inclusive)")

    @validator("sources", pre=True)
    def normalize_sources(cls, v):
        if v in ("", [], None):
            return None
        if isinstance(v, str):
            return [v]
        return v

    @validator("date_from", "date_to", pre=True)
    def empty_string_to_none(cls, v):
        if v == "":
            return None
        return v

    @validator("date_to")
    def validate_dates(cls, v, values):
        start = values.get("date_from")
        if start and v and v < start:
            raise ValueError("date_to must be after date_from")
        return v


class Article(BaseModel):
    source: str
    headline: str
    link: HttpUrl
    date: datetime
    person: List[str]
    org: List[str]
    loc: List[str]
    values: List[str]
    context: List[str]
    sentiment : Dict[str, float] = {}


class ScrapeResponse(BaseModel):
    total: int
    articles: List[Article]

```

# backend\app\services\ner.py

```py
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from ..constants import LABEL_ID2LABEL
from ..config import get_settings


@lru_cache(maxsize=1)
def get_ner_pipeline():
    settings = get_settings()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    model = AutoModelForTokenClassification.from_pretrained(settings.model_name)
    model.config.id2label = LABEL_ID2LABEL
    model.config.label2id = {v: k for k, v in LABEL_ID2LABEL.items()}
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

```

# backend\app\services\scraper.py

```py
import re
from datetime import datetime, timedelta, time
from typing import List, Optional, Dict

import dateparser
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

from .sentiment import SentimentService


def try_parse_date(card_item, url_link: str) -> Optional[datetime]:
    """Aggressively searches card and URL for a publish date, including time when present."""

    def attach_time(dt: datetime, t: Optional[time]) -> datetime:
        if t is None:
            return dt
        return dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)

    try:
        # 1) Prefer explicit <time datetime="..."></time> used by hotnews.ro listings
        time_tag = card_item.find("time")
        if time_tag and time_tag.has_attr("datetime"):
            try:
                return datetime.fromisoformat(time_tag["datetime"])
            except Exception:
                try:
                    parsed = dateparser.parse(time_tag["datetime"], languages=["ro"])
                    if parsed:
                        return parsed
                except Exception:
                    pass

        raw_text = card_item.get_text(" ", strip=True)
        clean_text = " ".join(raw_text.split())

        time_match = re.search(r"(?:ora\s*)?(\d{1,2}:\d{2})", clean_text, re.IGNORECASE)
        parsed_time = None
        if time_match:
            try:
                hh, mm = time_match.group(1).split(":")
                parsed_time = time(int(hh), int(mm))
            except Exception:
                parsed_time = None

        url_match = re.search(r"/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/", url_link)
        if url_match:
            return attach_time(
                datetime(int(url_match.group(1)), int(url_match.group(2)), int(url_match.group(3))),
                parsed_time,
            )

        datetime_with_time_match = re.search(
            r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})\s*,?\s*(\d{1,2}:\d{2})",
            clean_text,
        )
        if datetime_with_time_match:
            try:
                return datetime.strptime(" ".join(datetime_with_time_match.groups()), "%d.%m.%Y %H:%M")
            except Exception:
                try:
                    return datetime.strptime(" ".join(datetime_with_time_match.groups()), "%d-%m-%Y %H:%M")
                except Exception:
                    pass

        simple_date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", clean_text)
        if simple_date_match:
            return attach_time(datetime.strptime(simple_date_match.group(1), "%d.%m.%Y"), parsed_time)

        hn_match = re.search(
            r"(?:luni|mar[tț]i|miercuri|joi|vineri|s[aă]mb[aă]t[aă]|duminic[aă])?,?\s*(\d{1,2})\s+"
            r"(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\s+(\d{4})",
            clean_text,
            re.IGNORECASE,
        )
        if hn_match:
            parsed = dateparser.parse(hn_match.group(0), languages=["ro"])
            return attach_time(parsed, parsed_time) if parsed else None

        rel_match = re.search(r"acum\s+(\d+)\s+(min|ore|zile)", clean_text, re.IGNORECASE)
        if rel_match:
            val = int(rel_match.group(1))
            unit = rel_match.group(2).lower()
            now = datetime.now()
            if "min" in unit:
                return attach_time(now - timedelta(minutes=val), parsed_time)
            if "ore" in unit:
                return attach_time(now - timedelta(hours=val), parsed_time)
            if "zile" in unit:
                return attach_time(now - timedelta(days=val), parsed_time)

    except Exception:
        pass

    return None


def try_parse_date_from_url(url_link: str) -> Optional[datetime]:
    match = re.search(r"/(\d{4})[-/](\d{1,2})[-/](\d{1,2})/", url_link)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def parse_short_ro_date(card_item) -> Optional[datetime]:
    try:
        text = card_item.get_text(" ", strip=True)
        short_match = re.search(
            r"(\d{1,2})\s+(ian|feb|mar|apr|mai|iun|iul|aug|sep|oct|nov|dec)\.?",
            text,
            re.IGNORECASE,
        )
        if short_match:
            day = short_match.group(1)
            month_str = short_match.group(2)
            current_year = datetime.now().year
            date_str = f"{day} {month_str} {current_year}"
            parsed = dateparser.parse(date_str, languages=["ro"])
            if parsed and parsed > datetime.now() + timedelta(days=2):
                date_str = f"{day} {month_str} {current_year - 1}"
                parsed = dateparser.parse(date_str, languages=["ro"])
            return parsed
    except Exception:
        pass
    return None


def _normalize_link(base_url: str, link: str) -> Optional[str]:
    """Ensure links have an absolute https scheme for pydantic validation."""
    if link.startswith("//"):
        return "https:" + link
    if link.startswith("/"):
        return "https://" + base_url.replace("https://", "").replace("http://", "").split("/")[0] + link
    if link.startswith("http://") or link.startswith("https://"):
        return link
    # skip non-http schemes
    return None


def _fetch_rss(source_url: str) -> List[Dict]:
    rss_map: Dict[str, List[str]] = {
        "adevarul.ro": ["https://adevarul.ro/rss/politica", "https://adevarul.ro/rss"],
        "libertatea.ro": ["https://www.libertatea.ro/politica/feed", "https://www.libertatea.ro/feed"],
        "g4media.ro": ["https://www.g4media.ro/feed"],
        "digi24.ro": ["https://www.digi24.ro/rss"],
        "spotmedia.ro": ["https://spotmedia.ro/rss"],
        "antena3.ro": ["https://www.antena3.ro/rss", "https://www.antena3.ro/rss.xml"],
        "stiripesurse.ro": ["https://www.stiripesurse.ro/rss", "https://www.stiripesurse.ro/rss/politica.xml"],
        "dcnews.ro": ["https://www.dcnews.ro/rss", "https://www.dcnews.ro/rss/politica"],
        "hotnews.ro": ["https://www.hotnews.ro/rss"],
    }

    host = source_url.replace("https://", "").replace("http://", "").split("/")[0]
    if host.startswith("www."):
        host = host[4:]
    feed_urls = rss_map.get(host, [])
    articles: List[Dict] = []

    for feed_url in feed_urls:
        try:
            resp = requests.get(feed_url, timeout=8)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title = item.findtext("title") or ""
                link = item.findtext("link") or ""
                pub = item.findtext("pubDate") or ""
                dt = dateparser.parse(pub) if pub else None
                if not link or not title or not dt:
                    continue
                if not any(a.get("link") == link for a in articles):
                    articles.append(
                        {
                            "source": host,
                            "headline": title.strip(),
                            "link": link.strip(),
                            "date": dt,
                        }
                    )
            if articles:
                break  # got data from a working feed URL
        except Exception:
            continue

    return articles


def scrape_generic(base_url: str, max_pages: int = 3) -> List[Dict]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    all_articles: List[Dict] = []

    for page_num in range(1, max_pages + 1):
        if page_num == 1:
            current_url = base_url
        elif "digi24.ro" in base_url:
            current_url = f"{base_url}?p={page_num}"
        elif "hotnews.ro" in base_url:
            current_url = f"{base_url}/page/{page_num}"
        else:
            current_url = f"{base_url}/page/{page_num}"

        try:
            response = requests.get(current_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            cards = []
            cards.extend(soup.find_all("article"))
            cards.extend(
                soup.find_all("div", class_=re.compile(r"(post-review|post-card|entry-content|articol-card)"))
            )
            cards = cards[:40]

            for card in cards:
                link_tag = card.find("a")
                if link_tag and not link_tag.get_text(strip=True):
                    headers_tags = card.find_all(["h2", "h3", "h4"])
                    for h in headers_tags:
                        if h.find("a"):
                            link_tag = h.find("a")
                            break

                if link_tag and link_tag.has_attr("href"):
                    title = link_tag.get_text(strip=True)
                    raw_link = link_tag["href"]
                    link = _normalize_link(base_url, raw_link)

                    if link and len(title) > 25:
                        pub_date = try_parse_date(card, link)
                        if pub_date is None:
                            url_date = try_parse_date_from_url(link)
                            if url_date:
                                pub_date = url_date
                                source_label = f"{base_url.split('/')[2]} (URL Date)"
                            else:
                                # Skip if we can't determine a publish date to avoid misleading "now" timestamps
                                continue
                        else:
                            source_label = base_url.split("/")[2]

                        if not any(a["link"] == link for a in all_articles):
                            all_articles.append(
                                {
                                    "source": source_label,
                                    "headline": title,
                                    "link": link,
                                    "date": pub_date,
                                }
                            )

            # Fallback: if nothing collected yet, try link-based extraction with date-in-URL
            if not all_articles:
                anchors = soup.find_all("a", href=True)
                for a in anchors:
                    href = a["href"]
                    link = _normalize_link(base_url, href)
                    if not link:
                        continue
                    # Require date pattern in URL
                    url_date = try_parse_date_from_url(link)
                    if not url_date:
                        continue
                    title = a.get_text(" ", strip=True)
                    if len(title) < 20:
                        # fall back to slug words
                        parts = [p for p in re.split(r"[-/]+", link) if p.isalpha()]
                        title = " ".join(parts[-6:]).title()
                    if len(title) < 20:
                        continue
                    if not any(x["link"] == link for x in all_articles):
                        all_articles.append(
                            {
                                "source": base_url.split("/")[2],
                                "headline": title,
                                "link": link,
                                "date": url_date,
                            }
                        )

        except Exception:
            continue

    return all_articles


def analyze_headline(nlp, text: str) -> Dict[str, List[str]]:
    results = nlp(text)
    meta = {"person": [], "org": [], "loc": [], "values": [], "context": []}

    for item in results:
        tag = item["entity_group"]
        word = item["word"].replace("##", "")
        if tag == "PERSON":
            meta["person"].append(word)
        elif tag == "ORG":
            meta["org"].append(word)
        elif tag in ["GPE", "LOC"]:
            meta["loc"].append(word)
        elif tag in ["MONEY", "QUANTITY"]:
            meta["values"].append(f"{word} ({tag})")
        elif tag in ["EVENT", "FACILITY"]:
            meta["context"].append(f"{word} ({tag})")
    return meta


def run_pipeline(nlp, sources: List[str], max_pages: int = 3) -> List[Dict]:
    sentiment_service = SentimentService()
    all_articles: List[Dict] = []

    for site in sources:
        # Try RSS first (more stable), then supplement with HTML scrape for extras
        rss_news = _fetch_rss(site)
        news = list(rss_news)

        html_news = scrape_generic(site, max_pages=max_pages)
        if html_news:
            for art in html_news:
                if not any(a.get("link") == art.get("link") for a in news):
                    news.append(art)

        for article in news:
            meta = analyze_headline(nlp, article["headline"])
            # Calculăm sentimentul doar pentru Persoane și Organizații găsite
            entity_sentiments = {}
            targets = meta.get("person", []) + meta.get("org", [])
            
            #all_articles.append({**article, **meta})

            for entity in targets:
                # Analizăm titlul (headline) în raport cu entitatea
                score = sentiment_service.get_entity_sentiment(article["headline"], entity)
                
                # Păstrăm doar scorurile non-neutre pentru a nu încărca baza de date
                if score != 0:                    
                    entity_sentiments[entity] = round(score, 2)
                
            
            # Adăugăm dicționarul de sentimente în obiectul articolului
            all_articles.append({
                **article, 
                **meta, 
                "sentiment": entity_sentiments # <--- Câmp nou
            })

    if len(all_articles) > 0:
        print(f"DEBUG FINAL: Primul articol are sentiment? {all_articles[0].get('sentiment')}")
    return all_articles

```

# backend\app\services\sentiment.py

```py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax

class SentimentService:
    _instance = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        # Singleton: Asigură încărcarea modelului o singură dată
        if cls._instance is None:
            cls._instance = super(SentimentService, cls).__new__(cls)
            print(" Loading Sentiment Model (readerbench/ro-sentiment)...")
            try:
                model_name = "readerbench/ro-sentiment"
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print(" Sentiment Model Loaded!")
            except Exception as e:
                print(f" Failed to load Sentiment Model: {e}")
                cls._instance = None
        return cls._instance

    def _get_score(self, text: str) -> float:
        """Helper intern: Returnează un scor între -1.0 (Negativ) și 1.0 (Pozitiv)"""
        if not self._model or not self._tokenizer:
            return 0.0

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        probs = softmax(outputs.logits, dim=1).numpy()[0]
        # Label 0 = Negativ, Label 1 = Pozitiv
        return float(probs[1] - probs[0])

    def get_entity_sentiment(self, text: str, entity_name: str) -> float:
        if not self._model or not entity_name:
            return 0.0
        
        # 1. Normalizăm textul (litere mici)
        text_lower = text.lower()
        entity_lower = entity_name.lower()

        # 2. Verificare Inteligentă:
        # Spargem numele în bucăți (ex: "Ion Marcel Ciolacu" -> ["ion", "marcel", "ciolacu"])
        name_parts = entity_lower.split()
        
        # Considerăm că entitatea este prezentă dacă MĂCAR UN nume (mai lung de 2 litere) apare în text
        # Ex: Dacă găsește "Ciolacu" în text, e suficient.
        is_present = any(part in text_lower for part in name_parts if len(part) > 2)
        
        if not is_present:
            return 0.0

        # 3. Calculăm sentimentul
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        
        probs = softmax(outputs.logits, dim=1).numpy()[0]
        # Label 1 = Pozitiv, Label 0 = Negativ
        score = float(probs[1] - probs[0])
        
        return score
```

# backend\requirements.txt

```txt
fastapi==0.110.0
uvicorn[standard]==0.23.2
requests==2.31.0
beautifulsoup4==4.12.2
dateparser==1.2.0
pydantic==2.6.4
pydantic-settings==2.2.1
transformers==4.35.2
torch>=2.1.0

```

# frontend\.eslintrc.json

```json
{
  "extends": [
    "next/core-web-vitals",
    "next/typescript"
  ]
}

```

# frontend\.gitignore

```
# Dependencies
/node_modules
/.pnp
.pnp.js

# Testing
/coverage

# Next.js
/.next/
/out/

# Production
/build

# Misc
.DS_Store
*.pem

# Debug
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Local env files
.env
.env*.local
.env.development.local
.env.test.local
.env.production.local

# Vercel
.vercel

# TypeScript
*.tsbuildinfo
next-env.d.ts

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
Thumbs.db
.DS_Store

# Logs
*.log
logs/

```

# frontend\app\globals.css

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

* {
  box-sizing: border-box;
}

:root {
  color-scheme: light;
  background: #f8fafc;
}

body {
  margin: 0;
  background: #f8fafc;
  color: #0f172a;
}

```

# frontend\app\insights\page.tsx

```tsx
"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
  ArcElement,
  PointElement,
  LineElement,
} from "chart.js";
import { Bar, Doughnut, Line } from "react-chartjs-2";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend, ArcElement);

ChartJS.register(PointElement, LineElement);

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const ARTICLES_CACHE_KEY = "ro-mediaintel-cache-v2";

// Force graph is client-only; load dynamically to avoid SSR issues
type GraphNode = { id: string; group: "source" | "person" | "org"; val: number; x?: number; y?: number; fx?: number | null; fy?: number | null;};
type GraphLink = { source: string; target: string; value: number; type: "source-entity" | "entity-entity" };
type ForceGraphMethods = {
  zoomToFit: (ms?: number, padding?: number) => void;
  d3Force: (name: string) => unknown;
  d3ReheatSimulation: () => void;
  centerAt?: (x: number, y: number, ms?: number) => void;
};

type ForceGraphProps = {
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
  nodeRelSize?: number;
  nodeCanvasObjectMode?: () => string;
  nodeCanvasObject?: (node: GraphNode, ctx: CanvasRenderingContext2D) => void;
  linkColor?: (link: GraphLink) => string;
  linkWidth?: (link: GraphLink) => number;
  linkDirectionalParticles?: number;
  linkDirectionalParticleWidth?: number;
  linkDirectionalParticleColor?: (link: GraphLink) => string;
  backgroundColor?: string;
  ref?: React.Ref<ForceGraphMethods>;
  cooldownTime?: number;
  d3AlphaDecay?: number;    // Add this (optional but good for physics)
  d3VelocityDecay?: number; // Add this
  onNodeHover?: (node: GraphNode | null, prevNode?: GraphNode | null) => void;
  // ADD THESE TWO LINES:
  onNodeDrag?: (node: GraphNode, translate: { x: number; y: number }) => void;
  onNodeDragEnd?: (node: GraphNode, translate: { x: number; y: number }) => void;
};

const ForceGraph2D = dynamic<ForceGraphProps>(() => import("react-force-graph-2d"), { ssr: false });

interface Article {
  source: string;
  headline: string;
  link: string;
  date: string;
  person: string[];
  org: string[];
  loc: string[];
  values: string[];
  context: string[];
  sentiment: Record<string, number>;
}

export default function InsightsPage() {
  const fgRef = useRef<ForceGraphMethods | null>(null);
  const [data, setData] = useState<Article[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);
  const todayIso = new Date().toISOString().slice(0, 10);
  const weekAgoIso = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const [dateFrom, setDateFrom] = useState<string>(weekAgoIso);
  const [dateTo, setDateTo] = useState<string>(todayIso);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/scrape`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ date_from: dateFrom || undefined, date_to: dateTo || undefined }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json = await res.json();
      const articles = json.articles || [];
      setData(articles);

      if (typeof window !== "undefined") {
        const cached = {
          articles,
          date_from: dateFrom,
          date_to: dateTo,
          cached_at: Date.now(),
        };
        localStorage.setItem(ARTICLES_CACHE_KEY, JSON.stringify(cached));
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;

    const cachedRaw = localStorage.getItem(ARTICLES_CACHE_KEY);
    if (cachedRaw) {
      try {
        const cached = JSON.parse(cachedRaw);
        if (cached.date_from) setDateFrom(cached.date_from);
        if (cached.date_to) setDateTo(cached.date_to);
        if (cached.articles) setData(cached.articles);
        return; // use cached data, avoid re-scraping when arriving from main page
      } catch (e) {
        console.warn("Failed to read cache", e);
      }
    }

    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const perSource = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => {
      counts[a.source] = (counts[a.source] || 0) + 1;
    });
    return counts;
  }, [data]);

  const entityCounts = useMemo(() => {
    const buckets: Record<string, number> = { person: 0, org: 0, loc: 0, values: 0, context: 0 };
    data.forEach((a) => {
      buckets.person += a.person.length;
      buckets.org += a.org.length;
      buckets.loc += a.loc.length;
      buckets.values += a.values.length;
      buckets.context += a.context.length;
    });
    return buckets;
  }, [data]);

  const perDay = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => {
      const day = new Date(a.date).toISOString().slice(0, 10);
      counts[day] = (counts[day] || 0) + 1;
    });
    return counts;
  }, [data]);

  const topPeople = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => a.person.forEach((p) => (counts[p] = (counts[p] || 0) + 1)));
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  }, [data]);

  const sentimentChartData = useMemo(() => {
    // 1. Luăm doar numele celor mai menționați 10 oameni
    // (topPeople este deja sortat descrescător după numărul de apariții)
    const topPersonNames = topPeople.map(([name]) => name).slice(0, 10);
    
    const scores = topPersonNames.map((name) => {
      let total = 0;
      let count = 0;
      
      data.forEach((article) => {
        // --- ADĂUGĂM LOG-UL AICI ---
        if (article.sentiment && article.sentiment[name]) {
             // Dacă găsim sentiment, îl afișăm în consola browserului
             console.log(`Frontend a găsit sentiment: ${name} = ${article.sentiment[name]}`);
        }
        // ---------------------------

        if (article.sentiment && typeof article.sentiment[name] === 'number') {
          total += article.sentiment[name];
          count++;
        }
      });
      
      return count > 0 ? (total / count) : 0;
    });

    return {
      labels: topPersonNames,
      datasets: [
        {
          label: 'Sentiment Mediu',
          data: scores,
          // Culori dinamice: Verde pt Pozitiv, Roșu pt Negativ
          backgroundColor: scores.map(s => s >= 0 ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)'),
          borderColor: scores.map(s => s >= 0 ? 'rgb(22, 163, 74)' : 'rgb(220, 38, 38)'),
          borderWidth: 1,
          borderRadius: 4,
          borderSkipped: false,
        },
      ],
    };
  }, [data, topPeople]);

  const topOrgs = useMemo(() => {
    const counts: Record<string, number> = {};
    data.forEach((a) => a.org.forEach((o) => (counts[o] = (counts[o] || 0) + 1)));
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  }, [data]);

  const sourceChart = {
    labels: Object.keys(perSource),
    datasets: [
      {
        label: "Articles",
        data: Object.values(perSource),
        backgroundColor: "rgba(79, 70, 229, 0.6)",
        borderColor: "rgba(79, 70, 229, 1)",
        borderWidth: 1,
      },
    ],
  };

  const entityChart = {
    labels: ["Person", "Org", "Location", "Values", "Context"],
    datasets: [
      {
        label: "Entities",
        data: [
          entityCounts.person,
          entityCounts.org,
          entityCounts.loc,
          entityCounts.values,
          entityCounts.context,
        ],
        backgroundColor: [
          "#3b82f6",
          "#f59e0b",
          "#10b981",
          "#ef4444",
          "#94a3b8",
        ],
      },
    ],
  };

  const timelineChart = {
    labels: Object.keys(perDay).sort(),
    datasets: [
      {
        label: "Articles per day",
        data: Object.keys(perDay)
          .sort()
          .map((d) => perDay[d]),
        fill: false,
        borderColor: "rgba(79, 70, 229, 1)",
        backgroundColor: "rgba(79, 70, 229, 0.2)",
        tension: 0.25,
      },
    ],
  };

  const topPeopleChart = {
    labels: topPeople.map(([name]) => name),
    datasets: [
      {
        label: "Mentions",
        data: topPeople.map(([, count]) => count),
        backgroundColor: "rgba(99, 102, 241, 0.7)",
        borderColor: "rgba(99, 102, 241, 1)",
        borderWidth: 1,
      },
    ],
  };

  const topOrgsChart = {
    labels: topOrgs.map(([name]) => name),
    datasets: [
      {
        label: "Mentions",
        data: topOrgs.map(([, count]) => count),
        backgroundColor: "rgba(16, 185, 129, 0.7)",
        borderColor: "rgba(16, 185, 129, 1)",
        borderWidth: 1,
      },
    ],
  };

const graphData = useMemo(() => {
    if (!data.length) return { nodes: [], links: [] };

    const sourceCounts = new Map<string, number>();
    const personCounts = new Map<string, number>();
    const orgCounts = new Map<string, number>();

    data.forEach((a) => {
      sourceCounts.set(a.source, (sourceCounts.get(a.source) || 0) + 1);
      a.person.forEach((p) => personCounts.set(p, (personCounts.get(p) || 0) + 1));
      a.org.forEach((o) => orgCounts.set(o, (orgCounts.get(o) || 0) + 1));
    });

    // INCREASED LIMIT: Show top 20 instead of 10 for a richer graph
    const topPersons = Array.from(personCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20) 
      .map(([id]) => id);
    const topOrganizations = Array.from(orgCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([id]) => id);

    const selectedEntities = new Set<string>([...topPersons, ...topOrganizations]);

    const sourceEntityLinks: Record<string, number> = {};
    const cooccurrence: Record<string, number> = {};

    data.forEach((a) => {
      const entities = [...a.person.filter((p) => selectedEntities.has(p)), ...a.org.filter((o) => selectedEntities.has(o))];

      // Link Source -> Entity
      entities.forEach((ent) => {
        const key = `${a.source}::${ent}`;
        sourceEntityLinks[key] = (sourceEntityLinks[key] || 0) + 1;
      });

      // Link Entity <-> Entity (Co-occurrence)
      const uniqueEntities = Array.from(new Set(entities));
      for (let i = 0; i < uniqueEntities.length; i += 1) {
        for (let j = i + 1; j < uniqueEntities.length; j += 1) {
          const [aEnt, bEnt] = [uniqueEntities[i], uniqueEntities[j]].sort();
          const key = `${aEnt}||${bEnt}`;
          cooccurrence[key] = (cooccurrence[key] || 0) + 1;
        }
      }
    });

    const nodes: GraphNode[] = [
      // Use raw count for 'val' so we can scale it visually later
      ...Array.from(sourceCounts.entries()).map(([id, count]) => ({ id, group: "source" as const, val: count })),
      ...topPersons.map((id) => ({ id, group: "person" as const, val: personCounts.get(id) || 1 })),
      ...topOrganizations.map((id) => ({ id, group: "org" as const, val: orgCounts.get(id) || 1 })),
    ];

    const links: GraphLink[] = [
      ...Object.entries(sourceEntityLinks).map(([key, value]) => {
        const [source, target] = key.split("::");
        return { source, target, value, type: "source-entity" } as GraphLink;
      }),
      ...Object.entries(cooccurrence)
        // Filter out weak connections (only show if they appear together at least twice)
        .filter(([, value]) => value >= 2)
        .map(([key, value]) => {
          const [aEnt, bEnt] = key.split("||");
          return { source: aEnt, target: bEnt, value, type: "entity-entity" } as GraphLink;
        }),
    ];

    return { nodes, links };
  }, [data]);

useEffect(() => {
    const fg = fgRef.current;
    if (!fg || !graphData.nodes.length) return;

    // Use 'as any' to avoid TS errors
    const chargeForce = fg.d3Force("charge") as any;
    const linkForce = fg.d3Force("link") as any;
    const collideForce = fg.d3Force("collide") as any;
    const xForce = fg.d3Force("x") as any;
    const yForce = fg.d3Force("y") as any;

    // 1. MASSIVE REPULSION (The Key Fix)
    // Increased to -4000. This acts like powerful magnets pushing nodes apart.
    // This forces the "starburst" shape you saw in the screenshot.
    if (chargeForce) {
      chargeForce.strength(-4000); 
    }

    // 2. LONG LINKS
    // Tells the "ropes" between nodes to be much longer (200px+).
    // This prevents the "clump" in the middle.
    if (linkForce) {
      linkForce.distance((link: GraphLink) => 
        link.type === "entity-entity" ? 120 : 250
      );
    }

    // 3. AGGRESSIVE COLLISION
    // Adds a 15px invisible shield around every node so they never touch.
    if (collideForce) {
      collideForce.radius((node: GraphNode) => {
         const visualRadius = Math.min(40, Math.sqrt(node.val) * 4 + 4);
         return visualRadius + 15; 
      });
      collideForce.iterations(4); // Run collision check 4x more often
    }

    // 4. WEAK GRAVITY
    // Lowered to 0.02. This lets nodes float all the way to the edges of the screen.
    if (xForce) xForce.strength(0.02);
    if (yForce) yForce.strength(0.02);

    // 5. EXTENDED WARM-UP (Crucial!)
    // We reheat the simulation but give it more "energy" (alpha) so it runs longer.
    fg.d3ReheatSimulation();
    
    // TRICK: Manually keep the simulation running for 2 seconds to ensure it unfolds
    setTimeout(() => {
        fg.d3ReheatSimulation(); 
    }, 500);

  }, [graphData]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-white">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8 space-y-8">
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-indigo-500">RO-MediaIntel</p>
            <h1 className="mt-1 text-3xl font-bold text-slate-900">Insights</h1>
            <p className="text-slate-600">Source volume and entity distribution</p>
          </div>
          <div className="flex gap-2">
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-slate-700 shadow-sm transition hover:-translate-y-[1px] hover:bg-slate-50"
            >
              Back to articles
            </Link>
            <button
              onClick={fetchData}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-white shadow-md transition hover:-translate-y-[1px] hover:bg-indigo-700 disabled:opacity-60"
            >
              {loading ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </header>

        <section className="rounded-2xl border border-slate-200 bg-white/70 p-6 shadow-sm backdrop-blur">
          <h2 className="text-lg font-semibold text-slate-900">Date interval</h2>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-slate-700">From</label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700">To</label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
              />
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-500">Charts refresh using this interval.</p>
        </section>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700 text-sm">
            {error}
          </div>
        )}

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-900">Articles per source</h3>
              <span className="text-xs text-slate-500">Total: {data.length}</span>
            </div>
            <div className="mt-4">
              {Object.keys(perSource).length ? <Bar data={sourceChart} /> : <p className="text-sm text-slate-500">No data.</p>}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-900">Entity distribution</h3>
              <span className="text-xs text-slate-500">Across all articles</span>
            </div>
            <div className="mt-4 flex items-center justify-center">
              {data.length ? <Doughnut data={entityChart} /> : <p className="text-sm text-slate-500">No data.</p>}
            </div>
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-900">Timeline</h3>
              <span className="text-xs text-slate-500">Daily volume</span>
            </div>
            <div className="mt-4">
              {timelineChart.labels.length ? <Line data={timelineChart} /> : <p className="text-sm text-slate-500">No data.</p>}
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">Top people</h3>
                <span className="text-xs text-slate-500">Mentions</span>
              </div>
              <div className="mt-4">
                {topPeople.length ? <Bar data={topPeopleChart} options={{ indexAxis: "y" }} /> : <p className="text-sm text-slate-500">No data.</p>}
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900">Top organisations</h3>
                <span className="text-xs text-slate-500">Mentions</span>
              </div>
              <div className="mt-4">
                {topOrgs.length ? <Bar data={topOrgsChart} options={{ indexAxis: "y" }} /> : <p className="text-sm text-slate-500">No data.</p>}
              </div>
            </div>
          </div>
        </section>
        
        {/* --- SECȚIUNE NOUĂ: BAROMETRU SENTIMENT --- */}
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-slate-900">Barometru de Imagine (Sentiment AI)</h3>
            <p className="text-sm text-slate-500">
              Analiza tonului din titluri: <span className="text-green-600 font-bold">Dreapta (Pozitiv)</span> vs <span className="text-red-600 font-bold">Stânga (Negativ)</span>.
            </p>
          </div>

          <div className="h-[400px]">
            {topPeople.length > 0 ? (
              <Bar
                data={sentimentChartData}
                options={{
                  indexAxis: 'y', // Face graficul orizontal
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    x: {
                      min: -1, // Scala fixă de la -1
                      max: 1,  // la +1
                      grid: { color: '#f1f5f9' },
                      ticks: {
                        callback: (value) => Number(value).toFixed(1) // Arată 0.5, 0.2 etc.
                      }
                    },
                    y: {
                      grid: { display: false },
                      ticks: { font: { weight: 'bold' } }
                    }
                  },
                  plugins: {
                    legend: { display: false }, // Nu avem nevoie de legendă aici
                    tooltip: {
                      callbacks: {
                        label: (ctx) => {
                          const val = Number(ctx.raw);
                          const label = val > 0 ? "Pozitiv" : val < 0 ? "Negativ" : "Neutru";
                          return `${label}: ${val.toFixed(2)}`;
                        }
                      }
                    }
                  }
                }}
              />
            ) : (
              <div className="flex h-full items-center justify-center text-slate-400">
                Nu sunt date suficiente pentru analiză. Apasă Refresh.
              </div>
            )}
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Knowledge graph</h3>
              <p className="text-xs text-slate-500">Sources ↔ entities plus co-occurrence between entities</p>
            </div>
            <div className="flex items-center gap-3 text-xs text-slate-600">
              <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-indigo-600" /> Source</span>
              <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-sky-500" /> Person</span>
              <span className="inline-flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-amber-500" /> Org</span>
            </div>
          </div>
          <div className="mt-2 min-h-[820px] rounded-xl border border-slate-100 bg-gradient-to-br from-white via-slate-50 to-slate-100">
            {graphData.nodes.length ? (
              <ForceGraph2D
                ref={fgRef}
                graphData={graphData}
                nodeRelSize={7}
                backgroundColor="rgba(248, 250, 252, 0.95)"
                cooldownTime={4000}
                d3AlphaDecay={0.008}
                linkColor={(link) => (link.type === "entity-entity" ? "rgba(71, 85, 105, 0.35)" : "rgba(79, 70, 229, 0.5)")}
                linkWidth={(link) => Math.max(1, Math.min(5, link.value))}
                linkDirectionalParticles={1}
                linkDirectionalParticleWidth={1.5}
                linkDirectionalParticleColor={(link) => (link.type === "entity-entity" ? "#475569" : "#4f46e5")}
                onNodeHover={(node) => setHoverNode((node as GraphNode) || null)}
                // 1. PIN NODE ON DROP: This tells the physics engine "Lock this node here"
                onNodeDragEnd={(node) => {
                  node.fx = node.x;
                  node.fy = node.y;
                }}
                nodeCanvasObjectMode={() => "after"}
                nodeCanvasObject={(node, ctx) => {
                  const colors: Record<GraphNode["group"], string> = {
                    source: "#4f46e5",
                    person: "#0ea5e9",
                    org: "#f59e0b",
                  };
              // NEW SIZING MATH: Square root allows big numbers to grow without taking over the screen
                  const radius = Math.min(30, Math.sqrt(node.val) * 4 + 2);

                  ctx.beginPath();
                  ctx.fillStyle = colors[node.group];
                  // Add a white border to make them pop against other nodes
                  ctx.strokeStyle = "#ffffff";
                  ctx.lineWidth = 1.5;
                  ctx.arc(node.x || 0, node.y || 0, radius, 0, 2 * Math.PI, false);
                  ctx.fill();
                  ctx.stroke();

                  // Label Logic: Show if it's a Source, a BIG node, or hovered
                  const isBigNode = node.val > 10; 
                  const isSource = node.group === "source";
                  const isHovered = hoverNode?.id === node.id;

                  if (isSource || isBigNode || isHovered) {
                    const label = node.id as string;
                    ctx.font = `${isBigNode ? 'bold' : ''} 12px Inter, sans-serif`;
                    const textWidth = ctx.measureText(label).width;
                    const x = (node.x || 0) + radius + 4;
                    const y = (node.y || 0) + 4;
                    
                    // Label Background
                    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
                    ctx.fillRect(x - 2, y - 10, textWidth + 4, 14);
                    
                    // Label Text
                    ctx.fillStyle = "#0f172a";
                    ctx.fillText(label, x, y);
                  }
                }}
              />
            ) : (
              <p className="p-4 text-sm text-slate-500">No data for graph. Run a scrape first.</p>
            )}
          </div>
        </section>
      </div>
    </main>
    
  );
}

```

# frontend\app\layout.tsx

```tsx
import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'], display: 'swap' })

export const metadata: Metadata = {
  title: 'RO-MediaIntel',
  description: 'Thematic news aggregation & intelligence dashboard',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-slate-50 text-slate-900`}>{children}</body>
    </html>
  )
}

```

# frontend\app\page.tsx

```tsx
"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
const ARTICLES_CACHE_KEY = "ro-mediaintel-cache";

interface Article {
  source: string;
  headline: string;
  link: string;
  date: string;
  person: string[];
  org: string[];
  loc: string[];
  values: string[];
  context: string[];
}

export default function Home() {
  const [sources, setSources] = useState<string[]>([
    "https://www.digi24.ro/stiri/actualitate/politica",
    "https://spotmedia.ro/stiri/politica",
    "https://www.antena3.ro/politica",
    "https://www.stiripesurse.ro/politica",
    "https://www.dcnews.ro/politica",
    "https://www.digi24.ro/stiri/extern",
    "https://adevarul.ro/politica",
    "https://www.libertatea.ro/politica",
    "https://www.g4media.ro/category/politica",
  ]);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<Article[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<string[]>([]);
  const [entityFilter, setEntityFilter] = useState<{
    person: boolean;
    org: boolean;
    loc: boolean;
  }>({ person: false, org: false, loc: false });
  const [entityQuery, setEntityQuery] = useState<{
    person: string;
    org: string;
    loc: string;
    text: string;
  }>({ person: "", org: "", loc: "", text: "" });
  const [exactMatch, setExactMatch] = useState<boolean>(true);

  const todayIso = new Date().toISOString().slice(0, 10);
  const weekAgoIso = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);
  const [dateFrom, setDateFrom] = useState<string>(weekAgoIso);
  const [dateTo, setDateTo] = useState<string>(todayIso);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/scrape`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sources,
          date_from: dateFrom || undefined,
          date_to: dateTo || undefined,
        }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const json = await res.json();
      const articles = json.articles || [];
      setData(articles);

      if (typeof window !== "undefined") {
        const payload = {
          articles,
          date_from: dateFrom,
          date_to: dateTo,
          sources,
          cached_at: Date.now(),
        };
        localStorage.setItem(ARTICLES_CACHE_KEY, JSON.stringify(payload));
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (typeof window === "undefined") return;
    const cached = localStorage.getItem(ARTICLES_CACHE_KEY);
    if (cached) {
      try {
        const parsed = JSON.parse(cached);
        if (parsed.articles) setData(parsed.articles);
        if (parsed.date_from) setDateFrom(parsed.date_from);
        if (parsed.date_to) setDateTo(parsed.date_to);
        if (
          parsed.sources &&
          Array.isArray(parsed.sources) &&
          parsed.sources.length
        ) {
          setSources(parsed.sources);
        }
      } catch (e) {
        console.warn("Failed to read cache", e);
      }
    }
  }, []);

  const filteredData = useMemo(() => {
    const normalize = (s: string) =>
      s
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLowerCase()
        .trim();

    const matchesList = (list: string[], rawNeedle: string) => {
      const needle = normalize(rawNeedle);
      if (!needle) return true;
      return list.some((item) => {
        const norm = normalize(item);
        return exactMatch ? norm === needle : norm.includes(needle);
      });
    };

    const matchesBag = (rawNeedle: string, item: Article) => {
      const needle = normalize(rawNeedle);
      if (!needle) return true;
      const bag = [item.headline, ...item.person, ...item.org, ...item.loc]
        .map((x) => normalize(x))
        .join(" ");
      return bag.includes(needle);
    };

    return data.filter((item) => {
      if (sourceFilter.length && !sourceFilter.includes(item.source))
        return false;
      if (entityFilter.person && item.person.length === 0) return false;
      if (entityFilter.org && item.org.length === 0) return false;
      if (entityFilter.loc && item.loc.length === 0) return false;

      if (!matchesList(item.person, entityQuery.person)) return false;
      if (!matchesList(item.org, entityQuery.org)) return false;
      if (!matchesList(item.loc, entityQuery.loc)) return false;
      if (!matchesBag(entityQuery.text, item)) return false;

      return true;
    });
  }, [data, sourceFilter, entityFilter, entityQuery, exactMatch]);

  const allSources = useMemo(() => {
    const set = new Set<string>();
    data.forEach((d) => set.add(d.source));
    return Array.from(set).sort();
  }, [data]);

  const addSource = (value: string) => {
    if (!value) return;
    try {
      const url = new URL(value.trim());
      if (!sources.includes(url.toString()))
        setSources((prev) => [...prev, url.toString()]);
    } catch {
      setError("Invalid URL");
    }
  };

  const sourceBadges = useMemo(
    () => (
      <div className="flex flex-wrap gap-2">
        {sources.map((s) => (
          <span
            key={s}
            className="group inline-flex items-center gap-2 rounded-full bg-blue-50 px-3 py-1 text-sm text-blue-700 border border-blue-200"
            title={s}
          >
            <span className="truncate max-w-[200px] sm:max-w-xs">{s}</span>
            <button
              onClick={() =>
                setSources((prev) => prev.filter((item) => item !== s))
              }
              className="opacity-0 transition group-hover:opacity-100 text-blue-700 hover:text-blue-900"
              aria-label={`Remove ${s}`}
            >
              ×
            </button>
          </span>
        ))}
      </div>
    ),
    [sources]
  );

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-white">
      <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 lg:px-8">
        <header className="mb-10 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-indigo-500">
              RO-MediaIntel
            </p>
            <h1 className="mt-2 text-3xl font-bold text-slate-900 sm:text-4xl">
              Thematic News Aggregation & Intelligence
            </h1>
          </div>
          <div className="flex gap-2">
            <Link
              href="/insights"
              className="inline-flex items-center gap-2 self-start rounded-lg border border-indigo-200 px-4 py-2 text-indigo-700 bg-indigo-50 shadow-sm transition hover:-translate-y-[1px] hover:bg-indigo-100"
            >
              Insights
            </Link>
            <button
              onClick={fetchData}
              disabled={loading}
              className="inline-flex items-center gap-2 self-start rounded-lg bg-indigo-600 px-4 py-2 text-white shadow-md transition hover:-translate-y-[1px] hover:bg-indigo-700 disabled:translate-y-0 disabled:opacity-60"
            >
              {loading ? "Scanning..." : "Launch Scraper"}
            </button>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-3">
          <section className="lg:col-span-2 rounded-2xl border border-slate-200 bg-white/70 p-6 shadow-sm backdrop-blur">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-xl font-semibold text-slate-900">Sources</h2>
              <span className="text-xs font-medium text-slate-500">
                {sources.length} active
              </span>
            </div>
            <div className="mt-4 space-y-2">{sourceBadges}</div>
            <div className="mt-4 flex flex-col gap-3 sm:flex-row">
              <input
                type="url"
                placeholder="https://example.com"
                className="flex-1 rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    addSource((e.target as HTMLInputElement).value);
                    (e.target as HTMLInputElement).value = "";
                  }
                }}
              />
              <button
                onClick={() => {
                  const input =
                    document.querySelector<HTMLInputElement>("input[type=url]");
                  if (input) {
                    addSource(input.value);
                    input.value = "";
                  }
                }}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
              >
                Add source
              </button>
            </div>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white/70 p-6 shadow-sm backdrop-blur space-y-4">
            <div>
              <h2 className="text-xl font-semibold text-slate-900">
                Date interval
              </h2>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-slate-700">
                    From
                  </label>
                  <input
                    type="date"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700">
                    To
                  </label>
                  <input
                    type="date"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    className="mt-2 w-full rounded-lg border border-slate-200 px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                  />
                </div>
              </div>
            </div>

            <div className="border-t border-slate-200 pt-4">
              <h3 className="text-sm font-semibold text-slate-900">Filters</h3>
              <p className="text-xs text-slate-500">
                Narrow results by source and entity types/text.
              </p>
              {(sourceFilter.length > 0 ||
                Object.values(entityFilter).some(Boolean) ||
                entityQuery.person ||
                entityQuery.org ||
                entityQuery.loc ||
                entityQuery.text) && (
                <div className="mt-2">
                  <button
                    onClick={() => {
                      setSourceFilter([]);
                      setEntityFilter({
                        person: false,
                        org: false,
                        loc: false,
                      });
                      setEntityQuery({
                        person: "",
                        org: "",
                        loc: "",
                        text: "",
                      });
                    }}
                    className="text-xs font-semibold text-indigo-700 hover:text-indigo-900"
                  >
                    Clear all filters
                  </button>
                </div>
              )}
              <div className="mt-3 space-y-3">
                <div>
                  <span className="text-xs font-medium text-slate-600">
                    Sources
                  </span>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {allSources.length === 0 && (
                      <span className="text-xs text-slate-400">(none yet)</span>
                    )}
                    {allSources.map((s) => {
                      const active = sourceFilter.includes(s);
                      return (
                        <button
                          key={s}
                          onClick={() =>
                            setSourceFilter((prev) =>
                              prev.includes(s)
                                ? prev.filter((x) => x !== s)
                                : [...prev, s]
                            )
                          }
                          className={`rounded-full border px-3 py-1 text-xs transition ${
                            active
                              ? "border-indigo-300 bg-indigo-50 text-indigo-700"
                              : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                          }`}
                          title={s}
                        >
                          {s.replace(/^https?:\/\//, "").replace(/\/$/, "")}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="flex flex-wrap gap-3">
                  {["person", "org", "loc"].map((key) => {
                    const k = key as keyof typeof entityFilter;
                    const active = entityFilter[k];
                    const labels: Record<typeof k, string> = {
                      person: "Has people",
                      org: "Has orgs",
                      loc: "Has locations",
                    } as const;
                    return (
                      <button
                        key={key}
                        onClick={() =>
                          setEntityFilter((prev) => ({
                            ...prev,
                            [k]: !prev[k],
                          }))
                        }
                        className={`rounded-full border px-3 py-1 text-xs transition ${
                          active
                            ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                            : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                        }`}
                      >
                        {labels[k]}
                      </button>
                    );
                  })}
                </div>

                {(sourceFilter.length > 0 ||
                  Object.values(entityFilter).some(Boolean)) && (
                  <div className="flex flex-wrap gap-2 text-xs text-slate-500">
                    <span>
                      Active: {sourceFilter.length} source filter(s),{" "}
                      {Object.values(entityFilter).filter(Boolean).length}{" "}
                      entity toggle(s)
                    </span>
                  </div>
                )}

                <div className="grid gap-2 sm:grid-cols-2">
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Person contains
                    </label>
                    <input
                      type="text"
                      value={entityQuery.person}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          person: e.target.value,
                        }))
                      }
                      placeholder="e.g. Iohannis"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Org contains
                    </label>
                    <input
                      type="text"
                      value={entityQuery.org}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          org: e.target.value,
                        }))
                      }
                      placeholder="e.g. PSD"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Location contains
                    </label>
                    <input
                      type="text"
                      value={entityQuery.loc}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          loc: e.target.value,
                        }))
                      }
                      placeholder="e.g. Bucuresti"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div>
                    <label className="text-xs font-medium text-slate-600">
                      Headline/entities text search
                    </label>
                    <input
                      type="text"
                      value={entityQuery.text}
                      onChange={(e) =>
                        setEntityQuery((prev) => ({
                          ...prev,
                          text: e.target.value,
                        }))
                      }
                      placeholder="keyword"
                      className="mt-1 w-full rounded-lg border border-slate-200 px-2 py-1 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-200"
                    />
                  </div>
                  <div className="flex items-center gap-2 text-xs text-slate-700">
                    <input
                      type="checkbox"
                      checked={exactMatch}
                      onChange={(e) => setExactMatch(e.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-200"
                    />
                    <span>Exact match for person/org/location</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-red-700 text-sm">
            {error}
          </div>
        )}

        <section className="mt-8 space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <h2 className="text-xl font-semibold text-slate-900">
              Articles ({filteredData.length})
            </h2>
          </div>
          {loading && <p className="text-slate-500">Loading…</p>}
          {!loading && filteredData.length === 0 && (
            <p className="text-slate-500">No data yet.</p>
          )}
          <div className="grid gap-4">
            {filteredData.map((item) => (
              <article
                key={item.link}
                className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm transition hover:-translate-y-[1px] hover:shadow-md"
              >
                <div className="flex flex-col gap-1 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between">
                  <span className="font-semibold text-slate-700">
                    {item.source}
                  </span>
                  <span>{new Date(item.date).toLocaleString()}</span>
                </div>
                <a
                  href={item.link}
                  target="_blank"
                  rel="noreferrer"
                  className="mt-1 block text-lg font-semibold text-indigo-700 hover:text-indigo-900"
                >
                  {item.headline}
                </a>
                <div className="mt-3 flex flex-wrap gap-2 text-sm">
                  {item.person.map((p) => (
                    <span
                      key={p}
                      className="rounded-full bg-blue-50 px-2 py-1 text-blue-700"
                    >
                      👤 {p}
                    </span>
                  ))}
                  {item.org.map((o) => (
                    <span
                      key={o}
                      className="rounded-full bg-amber-50 px-2 py-1 text-amber-700"
                    >
                      🏢 {o}
                    </span>
                  ))}
                  {item.loc.map((l) => (
                    <span
                      key={l}
                      className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700"
                    >
                      📍 {l}
                    </span>
                  ))}
                  {item.values.map((v) => (
                    <span
                      key={v}
                      className="rounded-full bg-rose-50 px-2 py-1 text-rose-700"
                    >
                      💰 {v}
                    </span>
                  ))}
                  {item.context.map((c) => (
                    <span
                      key={c}
                      className="rounded-full bg-slate-100 px-2 py-1 text-slate-700"
                    >
                      🧭 {c}
                    </span>
                  ))}
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}

```

# frontend\next-env.d.ts

```ts
/// <reference types="next" />
/// <reference types="next/image-types/global" />

// NOTE: This file should not be edited
// see https://nextjs.org/docs/app/building-your-application/configuring/typescript for more information.

```

# frontend\next.config.js

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone'
};

module.exports = nextConfig;

```

# frontend\package.json

```json
{
  "name": "ro-mediainel-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.2.13",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "@headlessui/react": "1.7.17",
    "@heroicons/react": "2.0.18",
    "chart.js": "4.4.4",
    "react-chartjs-2": "5.2.0",
    "react-force-graph-2d": "^1.0.0"
  },
  "devDependencies": {
    "@types/node": "20.11.10",
    "@types/react": "18.2.47",
    "@types/react-dom": "18.2.18",
    "eslint": "8.57.0",
    "eslint-config-next": "14.2.13",
    "typescript": "5.3.3",
    "autoprefixer": "10.4.16",
    "postcss": "8.4.31",
    "tailwindcss": "3.4.1"
  }
}

```

# frontend\postcss.config.js

```js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};

```

# frontend\tailwind.config.js

```js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};

```

# frontend\tsconfig.json

```json
{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "esnext"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": false,
    "forceConsistentCasingInFileNames": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "paths": {
      "@/*": [
        "./*"
      ]
    },
    "plugins": [
      {
        "name": "next"
      }
    ]
  },
  "include": [
    "next-env.d.ts",
    "**/*.ts",
    "**/*.tsx",
    ".next/types/**/*.ts"
  ],
  "exclude": [
    "node_modules"
  ]
}

```

# frontend\types\react-force-graph-2d.d.ts

```ts
declare module 'react-force-graph-2d';

```

# README.md

```md
# RO-MediaIntel (FastAPI + Next.js)

A more professional split of the original Streamlit prototype into a backend API (FastAPI) and a modern frontend (Next.js). The backend scrapes Romanian news sources, extracts entities with a Romanian NER model, and returns structured articles. The frontend consumes the API and renders a clean, card-based UI.

## Project layout
\`\`\`
backend/            # FastAPI service
  app/
    main.py         # API entry
    schemas.py      # Pydantic models
    services/       # Scraper + NER pipeline
  requirements.txt
frontend/           # Next.js 14 app router UI
  app/page.tsx
  package.json
app.py              # Legacy Streamlit prototype (kept for reference)
\`\`\`

## Quick start
### Backend
\`\`\`bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir . --host 0.0.0.0 --port 8000
\`\`\`

### Frontend
\`\`\`bash
cd frontend
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
\`\`\`

Set `NEXT_PUBLIC_API_BASE` in a `.env.local` under `frontend/` if your API isn’t on `http://localhost:8000`.

## API
- `GET /health` — simple health check
- `POST /scrape` — body:
\`\`\`json
{
  "sources": ["https://hotnews.ro/c/politica"],
  "max_pages": 3,
  "date_from": "2025-01-01",  # optional
  "date_to": "2025-01-31"     # optional
}
\`\`\`
Response:
\`\`\`json
{
  "total": 12,
  "articles": [
    {
      "source": "hotnews.ro",
      "headline": "...",
      "link": "https://...",
      "date": "2025-01-02T10:00:00",
      "person": ["Nume"],
      "org": ["Organizatie"],
      "loc": ["Bucuresti"],
      "values": [],
      "context": []
    }
  ]
}
\`\`\`

## Notes
- The NER model (`dumitrescustefan/bert-base-romanian-ner`) is loaded once and cached.
- Scraper logic mirrors the Streamlit prototype with pagination and date parsing heuristics.
- Keep `torch`/`transformers` installed in the backend environment for inference.

## Next steps
- Add persistence (DB) and background jobs for scheduled crawls.
- Harden scraping with per-source selectors and retries.
- Add filtering, search, and bookmark features in the frontend.

```

