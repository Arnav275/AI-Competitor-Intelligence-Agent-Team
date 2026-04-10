"""
FastAPI Backend — serves the frontend and exposes SSE + REST endpoints.
Windows-compatible: all blocking agent calls run in thread pool via run_in_executor.
"""

import os
import json
import asyncio
from datetime import datetime
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

try:
    from competitor_agents import CompetitorIntelligenceTeam
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    AGENTS_AVAILABLE = False

app = FastAPI(title="Competitor Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-init team ────────────────────────────────────────────────────────────
_team = None

def get_team() -> "CompetitorIntelligenceTeam":
    global _team
    if _team is None:
        if not AGENTS_AVAILABLE:
            raise HTTPException(status_code=500, detail="competitor_agents module not available.")
        fc_key = os.getenv("FIRECRAWL_API_KEY", "")
        oa_key = os.getenv("OPENAI_API_KEY", "")
        if not fc_key:
            raise HTTPException(status_code=400, detail="FIRECRAWL_API_KEY not set.")
        if not oa_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set.")
        _team = CompetitorIntelligenceTeam(
            firecrawl_api_key=fc_key,
            openai_api_key=oa_key,
        )
    return _team


# ── Pydantic models ───────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str

class CompareRequest(BaseModel):
    urls: list[str]


# ── SSE helper ────────────────────────────────────────────────────────────────
def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agents_available": AGENTS_AVAILABLE,
        "firecrawl_key_set": bool(os.getenv("FIRECRAWL_API_KEY")),
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.get("/analyze/stream")
async def analyze_stream(url: str):
    """Single competitor analysis with SSE live progress."""

    async def generate():
        try:
            team = get_team()
            loop = asyncio.get_event_loop()

            yield sse("progress", {"stage": "scraping", "message": f"🕷️ Scraping {url}…", "pct": 15})

            # Agent 1 — run sync function in thread pool (Windows-safe)
            profile = await loop.run_in_executor(
                None, team.scrape_competitor_sync, url
            )

            yield sse("progress", {"stage": "analyzing", "message": f"🧠 Analyzing {profile.company_name or url}…", "pct": 55})

            # Agent 2
            insight = await loop.run_in_executor(
                None, team.analyze_competitor_sync, profile
            )

            yield sse("progress", {"stage": "done", "message": "✅ Analysis complete!", "pct": 100})
            yield sse("result", {"profile": asdict(profile), "insight": asdict(insight)})

        except HTTPException as e:
            yield sse("error", {"message": e.detail})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/compare/stream")
async def compare_stream(urls: str):
    """Multi-competitor comparison with SSE live progress."""
    url_list = [u.strip() for u in urls.split(",") if u.strip()]

    async def generate():
        if len(url_list) < 2:
            yield sse("error", {"message": "Please provide at least 2 URLs."})
            return

        try:
            team = get_team()
            loop = asyncio.get_event_loop()
            total = len(url_list)
            profiles, insights = [], []

            for idx, url in enumerate(url_list):
                base_pct = int(10 + (idx / total) * 65)

                yield sse("progress", {
                    "stage": "scraping",
                    "message": f"🕷️ Scraping {url} ({idx+1}/{total})…",
                    "pct": base_pct,
                    "current_url": url,
                })

                # Agent 1
                p = await loop.run_in_executor(None, team.scrape_competitor_sync, url)
                profiles.append(p)

                yield sse("progress", {
                    "stage": "analyzing",
                    "message": f"🧠 Analyzing {p.company_name or url}…",
                    "pct": base_pct + 10,
                    "current_url": url,
                })

                # Agent 2
                i = await loop.run_in_executor(None, team.analyze_competitor_sync, p)
                insights.append(i)

                # Stream partial result so UI updates immediately
                yield sse("competitor", {
                    "index": idx,
                    "profile": asdict(p),
                    "insight": asdict(i),
                })

            yield sse("progress", {"stage": "synthesizing", "message": "📈 Synthesizing strategy…", "pct": 90})

            # Agent 3
            strategy = await loop.run_in_executor(
                None, team.synthesize_strategy_sync, profiles, insights
            )

            yield sse("progress", {"stage": "done", "message": "✅ Report ready!", "pct": 100})
            yield sse("strategy", strategy)

        except HTTPException as e:
            yield sse("error", {"message": e.detail})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Serve frontend HTML ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>frontend.html not found</h1><p>Make sure frontend.html is in the same folder as server.py</p>", status_code=404)
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()