# RO-MediaIntel (FastAPI + Next.js)

A more professional split of the original Streamlit prototype into a backend API (FastAPI) and a modern frontend (Next.js). The backend scrapes Romanian news sources, extracts entities with a Romanian NER model, and returns structured articles. The frontend consumes the API and renders a clean, card-based UI.

## Project layout
```
backend/            # FastAPI service
  app/
    main.py         # API entry
    schemas.py      # Pydantic models
    services/       # Scraper + NER pipeline
  requirements.txt
frontend/           # Next.js 14 app router UI
  app/page.tsx
  package.json

```

## Quick start
### Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --app-dir . --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Set `NEXT_PUBLIC_API_BASE` in a `.env.local` under `frontend/` if your API isn’t on `http://localhost:8000`.

## API
- `GET /health` — simple health check
- `POST /scrape` — body:
```json
{
  "sources": ["https://hotnews.ro/c/politica"],
  "max_pages": 3,
  "date_from": "2025-01-01",  # optional
  "date_to": "2025-01-31"     # optional
}
```
Response:
```json
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
```

## Notes
- The NER model (`dumitrescustefan/bert-base-romanian-ner`) is loaded once and cached.
- Scraper logic mirrors the Streamlit prototype with pagination and date parsing heuristics.
- Keep `torch`/`transformers` installed in the backend environment for inference.

## Next steps
- Add persistence (DB) and background jobs for scheduled crawls.
- Harden scraping with per-source selectors and retries.
- Add filtering, search, and bookmark features in the frontend.
