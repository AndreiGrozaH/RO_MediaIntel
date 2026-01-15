import re
from datetime import datetime, timedelta, time, date
from typing import List, Optional, Dict

import dateparser
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

from .sentiment import SentimentService
from .relationships import RelationshipExtractor
from .geo import resolve_location

# --- DATE PARSING HELPERS ---

def try_parse_date(card_item, url_link: str) -> Optional[datetime]:
    """Aggressively searches card and URL for a publish date, including time when present."""

    def attach_time(dt: datetime, t: Optional[time]) -> datetime:
        if t is None:
            return dt
        return dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)

    try:
        # 1) Prefer explicit <time datetime="..."></time>
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


def scrape_generic(base_url: str, max_pages: int = 3, stop_date: date = None) -> List[Dict]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    all_articles: List[Dict] = []

    # INTELLIGENT DIGGING:
    # If a stop_date is provided, we dig deeper (e.g. 35 pages) to find old news.
    # If no date is provided, we default to the quick scan (max_pages).
    effective_max_pages = 35 if stop_date else max_pages

    for page_num in range(1, effective_max_pages + 1):
        if page_num == 1:
            current_url = base_url
        elif "digi24.ro" in base_url:
            current_url = f"{base_url}?p={page_num}"
        elif "hotnews.ro" in base_url:
            current_url = f"{base_url}/page/{page_num}"
        else:
            current_url = f"{base_url}/page/{page_num}"

        try:
            print(f"Scraping {current_url} ...")
            response = requests.get(current_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            cards = []
            cards.extend(soup.find_all("article"))
            cards.extend(
                soup.find_all("div", class_=re.compile(r"(post-review|post-card|entry-content|articol-card)"))
            )
            # Safe limit per page
            cards = cards[:40]

            page_articles = []

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

                    if link and len(title) > 20:
                        pub_date = try_parse_date(card, link)
                        
                        # Fallback for date
                        if pub_date is None:
                            url_date = try_parse_date_from_url(link)
                            if url_date:
                                pub_date = url_date
                                source_label = f"{base_url.split('/')[2]} (URL Date)"
                            else:
                                continue # Skip if no date found, to ensure data quality
                        else:
                            source_label = base_url.split("/")[2]

                        if not any(a["link"] == link for a in all_articles):
                            art_obj = {
                                "source": source_label,
                                "headline": title,
                                "link": link,
                                "date": pub_date,
                            }
                            all_articles.append(art_obj)
                            page_articles.append(art_obj)

            # --- INTELLIGENT STOP LOGIC ---
            # If we are looking for history (stop_date set)
            if page_articles and stop_date:
                # Find the oldest article on this page
                min_date_on_page = min(a["date"] for a in page_articles).date()
                
                # If we have reached a date OLDER than the user's "From" date, we can stop scraping this site.
                if min_date_on_page < stop_date:
                    print(f"Reached limit for {base_url} ({min_date_on_page} < {stop_date}). Stopping.")
                    break
            
            # If no articles found on this page (captcha, layout change, or end of feed), stop.
            if not page_articles and page_num > 1:
                break

        except Exception as e:
            print(f"Error on {current_url}: {e}")
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


def run_pipeline(nlp, sources: List[str], max_pages: int = 3, stop_date: date = None) -> List[Dict]:
    sentiment_service = SentimentService()
    rel_extractor = RelationshipExtractor()
    
    all_articles: List[Dict] = []

    for site in sources:
        # 1. Fetch RSS (Fast, usually recent items)
        rss_news = _fetch_rss(site)
        news = list(rss_news)

        # 2. Fetch HTML (Deep history, now supports date-aware digging)
        html_news = scrape_generic(site, max_pages=max_pages, stop_date=stop_date)
        
        if html_news:
            for art in html_news:
                if not any(a.get("link") == art.get("link") for a in news):
                    news.append(art)

        # 3. Process Intelligence Layers
        for article in news:
            # A. NER
            meta = analyze_headline(nlp, article["headline"])
            
            # B. Sentiment
            entity_sentiments = {}
            targets = meta.get("person", []) + meta.get("org", [])
            for entity in targets:
                score = sentiment_service.get_entity_sentiment(article["headline"], entity)
                if score != 0:                    
                    entity_sentiments[entity] = round(score, 2)

            # C. Relationship Extraction (Alliance/Conflict)
            relationships = rel_extractor.analyze_interactions(article["headline"], meta)
            
            # D. Geo-Location Resolution
            county_code = resolve_location(meta.get("loc", []))

            all_articles.append({
                **article, 
                **meta, 
                "sentiment": entity_sentiments,
                "relationships": relationships,
                "county": county_code
            })

    return all_articles