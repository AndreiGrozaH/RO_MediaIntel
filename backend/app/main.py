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
    
    # max_pages is used as a baseline, but payload.date_from will trigger deep scraping
    max_pages = payload.max_pages or 3
    
    try:
        nlp = get_ner_pipeline()
        
        # Pass the date_from filter as the 'stop_date' to enable deep digging
        articles_raw = run_pipeline(
            nlp, 
            sources=sources, 
            max_pages=max_pages, 
            stop_date=payload.date_from
        )
        
    except Exception as exc:
        logger.exception("scrape_failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Final filtering pass (redundant safety check)
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
            sentiment=a.get("sentiment", {}),
            relationships=a.get("relationships", []),
            county=a.get("county"),
        )
        for a in filtered
    ]

    return ScrapeResponse(total=len(articles), articles=articles)