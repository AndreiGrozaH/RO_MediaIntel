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

class Relationship(BaseModel):
    source: str
    target: str
    type: str  # "conflict" or "alliance"
    verb: str

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
    relationships: List[Relationship] = []
    county: Optional[str] = None


class ScrapeResponse(BaseModel):
    total: int
    articles: List[Article]
