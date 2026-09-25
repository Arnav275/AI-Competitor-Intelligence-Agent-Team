"""
evaluation_server.py — Standalone Evaluation Dashboard API
─────────────────────────────────────────────────────────────────────────────
A SEPARATE FastAPI app (own port) that adds the evaluation suite on top of
the existing project WITHOUT modifying competitor_agents.py, main.py,
server.py, frontend.html, or requirements.txt.

It imports CompetitorIntelligenceTeam exactly the way server.py does, and
adds the EvaluationHarness from evaluation.py.

Run alongside your existing server:

    uvicorn server:app --reload --port 8000            # existing app, untouched
    uvicorn evaluation_server:app --reload --port 8001  # this new eval dashboard

Then open http://localhost:8001/
"""

import os
import json
import asyncio
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from competitor_agents import CompetitorIntelligenceTeam
from evaluation import EvaluationHarness, APPROACH_COMPARISON

app = FastAPI(title="Competitor Intelligence — Evaluation Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_team = None
_harness = None


def get_harness() -> EvaluationHarness:
    global _team, _harness
    if _harness is None:
        fc_key = os.getenv("FIRECRAWL_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not fc_key:
            raise HTTPException(status_code=400, detail="FIRECRAWL_API_KEY not set.")
        if not groq_key:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY not set.")
        _team = CompetitorIntelligenceTeam(firecrawl_api_key=fc_key, groq_api_key=groq_key)
        _harness = EvaluationHarness(_team, firecrawl_api_key=fc_key, groq_api_key=groq_key)
    return _harness


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/eval-health")
async def eval_health():
    return {
        "status": "ok",
        "firecrawl_key_set": bool(os.getenv("FIRECRAWL_API_KEY")),
        "groq_key_set": bool(os.getenv("GROQ_API_KEY")),
    }


@app.get("/comparison")
async def comparison():
    """Static architectural comparison vs generic LLM prompting / manual research / commercial SaaS."""
    return {"comparison": APPROACH_COMPARISON}


@app.get("/evaluate/stream")
async def evaluate_stream(url: str):
    """Runs the multi-agent pipeline AND a naive baseline, scores both, streams progress."""

    async def generate():
        try:
            harness = get_harness()
            loop = asyncio.get_event_loop()

            yield sse("progress", {"stage": "grounding", "message": f"🔎 Fetching ground-truth source for {url}…", "pct": 10})
            yield sse("progress", {"stage": "scraping", "message": "🕷️ Running multi-agent scraper…", "pct": 25})
            yield sse("progress", {"stage": "analyzing", "message": "🧠 Running SWOT analyst agent…", "pct": 45})
            yield sse("progress", {"stage": "baseline", "message": "🤖 Running naive single-prompt baseline…", "pct": 60})
            yield sse("progress", {"stage": "judging", "message": "⚖️ Scoring both runs against the rubric…", "pct": 80})

            report = await loop.run_in_executor(None, harness.evaluate_sync, url)

            yield sse("progress", {"stage": "done", "message": "✅ Evaluation complete!", "pct": 100})
            yield sse("report", {
                "url": report.url,
                "evaluated_at": report.evaluated_at,
                "multi_agent": asdict(report.multi_agent),
                "naive_baseline": asdict(report.naive_baseline),
                "verdict": report.verdict,
                "profile": report.profile,
                "insight": report.insight,
                "comparison": APPROACH_COMPARISON,
            })

        except HTTPException as e:
            yield sse("error", {"message": e.detail})
        except Exception as e:
            yield sse("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_dashboard.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>eval_dashboard.html not found</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()