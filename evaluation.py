"""
evaluation.py — Evaluation Harness for the AI Competitor Intelligence Agent Team
─────────────────────────────────────────────────────────────────────────────
This is a NEW, standalone add-on. It does not import-modify or edit
competitor_agents.py, main.py, server.py, frontend.html, or requirements.txt —
it only *imports* CompetitorIntelligenceTeam from competitor_agents.py and
wraps it with an evaluation layer.

It scores every analysis run against 5 metrics:

    1. Accuracy            – are the extracted facts actually true, checked
                              against the raw Firecrawl source text (LLM-judged)
    2. Completeness         – % of the CompetitorProfile schema that got filled,
                              weighted by field importance
    3. Insight Quality      – LLM-judged relevance / specificity / actionability
                              of the SWOT + recommended actions
    4. Hallucination Rate   – % of extracted claims that are NOT grounded in the
                              scraped source (the inverse of accuracy, but scored
                              independently so partial grounding is visible)
    5. Response Time        – wall-clock latency per pipeline stage + end-to-end

It ALSO runs a "naive baseline" — a single un-grounded LLM prompt, i.e. the way
most people try this with plain ChatGPT/ungrounded prompting — and scores it
with the identical rubric, so the comparison is apples-to-apples rather than
a marketing claim.

Requires only packages already in requirements.txt (firecrawl-py, groq).
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

from firecrawl import FirecrawlApp
from groq import Groq


# ─────────────────────────────────────────────
# Weighted schema for Completeness scoring
# (higher weight = a field a real analyst actually needs)
# ─────────────────────────────────────────────
FIELD_WEIGHTS = {
    "company_name": 10,
    "tagline": 5,
    "description": 10,
    "products_services": 15,
    "pricing_model": 10,
    "pricing_tiers": 10,
    "target_audience": 10,
    "key_features": 15,
    "tech_stack": 5,
    "social_links": 3,
    "contact_info": 3,
    "recent_news": 4,
}
assert sum(FIELD_WEIGHTS.values()) == 100

JUDGE_MODEL = "openai/gpt-oss-120b"  # same model family already used by the agent team


# ─────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────

@dataclass
class StageTimings:
    scrape_seconds: float = 0.0
    analyze_seconds: float = 0.0
    grounding_fetch_seconds: float = 0.0
    judging_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class MetricScores:
    accuracy: Optional[float] = None            # 0-100
    completeness: Optional[float] = None         # 0-100
    insight_quality: Optional[float] = None      # 0-100
    hallucination_rate: Optional[float] = None   # 0-100 (lower is better)
    response_time_seconds: Optional[float] = None
    details: dict = field(default_factory=dict)  # judge reasoning, ungrounded claims, etc.


@dataclass
class EvaluationReport:
    url: str
    evaluated_at: str
    multi_agent: MetricScores
    naive_baseline: MetricScores
    verdict: dict = field(default_factory=dict)   # per-metric winner + delta
    profile: dict = field(default_factory=dict)
    insight: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Static "why this beats the alternatives" comparison
# (used by the dashboard alongside the LIVE numbers above)
# ─────────────────────────────────────────────
APPROACH_COMPARISON = [
    {
        "aspect": "Data freshness",
        "this_system": "Live scrape on every run (Firecrawl)",
        "generic_llm_prompt": "Relies on model training data — can be stale or wrong",
        "manual_research": "Fresh, but hours of analyst time per competitor",
        "commercial_saas": "Klue/Crayon/Kompyte poll on a fixed crawl schedule — fresh, but not on-demand",
    },
    {
        "aspect": "Grounding / hallucination control",
        "this_system": "Extraction agent is schema-constrained + separately scored for hallucination on every run",
        "generic_llm_prompt": "No grounding — freely fabricates pricing, features, stats",
        "manual_research": "Fully grounded (human-verified) but slow and inconsistent across analysts",
        "commercial_saas": "Grounded, but tools like Klue/Crayon give no per-run accuracy or hallucination score",
    },
    {
        "aspect": "Depth of analysis",
        "this_system": "3 specialized agents: scrape → SWOT analyst → cross-competitor strategist",
        "generic_llm_prompt": "One pass, one perspective, no structured SWOT discipline",
        "manual_research": "Can be very deep, but doesn't scale past a handful of competitors",
        "commercial_saas": "Kompyte/Similarweb Digital surface change-alerts and dashboards, not a synthesized SWOT",
    },
    {
        "aspect": "Multi-competitor synthesis",
        "this_system": "Dedicated Strategist agent finds white space + clusters across all URLs",
        "generic_llm_prompt": "Would need a hand-written follow-up prompt per comparison",
        "manual_research": "Possible, but synthesis quality depends entirely on the analyst",
        "commercial_saas": "Klue/Contify show side-by-side battlecards, not a written strategic narrative",
    },
    {
        "aspect": "Speed to first insight",
        "this_system": "Minutes per competitor, streamed live (SSE) so you see progress",
        "generic_llm_prompt": "Fast (seconds) but shallow and ungrounded",
        "manual_research": "Hours to days",
        "commercial_saas": "Fast once configured, but Klue/Crayon typically need days-to-weeks of setup and tagging",
    },
    {
        "aspect": "Cost",
        "this_system": "Pay-per-run API cost only (Firecrawl + Groq), no seat licensing",
        "generic_llm_prompt": "Cheapest, but you get what you pay for (low trust output)",
        "manual_research": "Analyst salary cost — highest per-competitor cost",
        "commercial_saas": "Crayon/Klue/Kompyte are recurring per-seat subscriptions, often $1k+/mo",
    },
    {
        "aspect": "Auditability",
        "this_system": "Every run scored against Accuracy / Completeness / Hallucination metrics, visible in this dashboard",
        "generic_llm_prompt": "No scoring — you have to manually fact-check everything",
        "manual_research": "Auditable by peer review, but no standardized metric",
        "commercial_saas": "Black box — vendors don't expose how confident a given data point is",
    },
]


class EvaluationHarness:
    """Wraps a CompetitorIntelligenceTeam instance with scoring + a naive baseline."""

    def __init__(self, team, firecrawl_api_key: str, groq_api_key: str, judge_model: str = JUDGE_MODEL):
        self.team = team
        self.firecrawl = FirecrawlApp(api_key=firecrawl_api_key)
        self.groq_client = Groq(api_key=groq_api_key)
        self.judge_model = judge_model

    # ── low-level helpers ────────────────────────────────────────────────

    def _timed(self, fn, *args):
        start = time.perf_counter()
        result = fn(*args)
        return result, round(time.perf_counter() - start, 3)

    def _parse_json(self, text: str) -> dict:
        text = str(text).strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def _judge(self, prompt: str) -> dict:
        try:
            resp = self.groq_client.chat.completions.create(
                model=self.judge_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a strict, neutral evaluation judge. Reply with ONLY valid JSON, no markdown, no commentary."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = resp.choices[0].message.content
            return self._parse_json(text)
        except Exception as e:
            return {"error": str(e)}

    def _call_scrape(self, url: str, **kwargs):
        """Call whichever scrape method the installed firecrawl-py exposes.

        firecrawl-py v2.0 (Aug 2025) renamed FirecrawlApp.scrape_url() -> .scrape().
        This supports both so the harness doesn't break across SDK upgrades.
        """
        if hasattr(self.firecrawl, "scrape"):
            return self.firecrawl.scrape(url, **kwargs)
        if hasattr(self.firecrawl, "scrape_url"):
            return self.firecrawl.scrape_url(url, **kwargs)
        raise AttributeError(
            "Installed firecrawl-py client exposes neither '.scrape()' nor '.scrape_url()' "
            "— check the firecrawl-py version on the server."
        )

    def _fetch_raw_content(self, url: str) -> tuple:
        """Independent ground-truth fetch, used only for scoring (not for extraction).

        Returns (content, error_message). error_message is "" on success, and is
        propagated up to the dashboard so 'N/A' scores come with an actual reason
        instead of a silent blank.

        Some sites (e.g. leetcode.com) are JS-heavy or bot-gated and return thin/
        empty content on a plain markdown scrape, so this tries a second pass with
        a longer render wait and full-page content before giving up.
        """
        MIN_USABLE_CHARS = 200
        attempts = [
            {"formats": ["markdown"], "only_main_content": True},
            {"formats": ["markdown"], "only_main_content": False, "wait_for": 5000, "timeout": 30000},
        ]
        last_error = ""
        for kwargs in attempts:
            try:
                result = self._call_scrape(url, **kwargs)
                if isinstance(result, dict):
                    raw = result.get("markdown") or result.get("data", {}).get("markdown", "")
                    meta = result.get("metadata") or {}
                    meta_error = meta.get("error") if isinstance(meta, dict) else None
                else:
                    raw = getattr(result, "markdown", "") or ""
                    meta_error = getattr(getattr(result, "metadata", None), "error", None)
                raw = (raw or "").strip()
                if len(raw) >= MIN_USABLE_CHARS:
                    return raw[:8000], ""
                last_error = (
                    f"Scrape succeeded but returned only {len(raw)} chars"
                    + (f" (site reported: {meta_error})" if meta_error else "")
                    + " — page is likely JS-rendered or blocking scrapers."
                )
            except TypeError:
                # Older SDK's scrape_url signature may not accept these kwarg names directly —
                # retry via the legacy `params={...}` calling convention.
                try:
                    result = self._call_scrape(url, params=kwargs)
                    if isinstance(result, dict):
                        raw = (result.get("markdown") or result.get("data", {}).get("markdown", "") or "").strip()
                    else:
                        raw = (getattr(result, "markdown", "") or "").strip()
                    if len(raw) >= MIN_USABLE_CHARS:
                        return raw[:8000], ""
                    last_error = f"Scrape (legacy params call) returned only {len(raw)} chars."
                except Exception as e2:
                    last_error = f"{type(e2).__name__}: {e2}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
        return "", last_error or "Unknown scrape failure."

    # ── metric 2: completeness ──────────────────────────────────────────

    def score_completeness(self, profile: dict) -> float:
        score = 0
        for field_name, weight in FIELD_WEIGHTS.items():
            value = profile.get(field_name)
            filled = bool(value) and (
                (isinstance(value, str) and value.strip()) or
                (isinstance(value, (list, dict)) and len(value) > 0) or
                (not isinstance(value, (str, list, dict)))
            )
            if filled:
                score += weight
        return round(score, 1)

    # ── metrics 1 + 4: accuracy & hallucination (grounded against raw source) ──

    def score_accuracy_and_hallucination(self, profile: dict, raw_content: str, fetch_error: str = "") -> dict:
        if not raw_content:
            return {
                "accuracy_score": None,
                "hallucination_rate": None,
                "ungrounded_claims": [],
                "note": fetch_error or "No source content available to ground against.",
            }
        prompt = f"""Compare the EXTRACTED_DATA against the SOURCE_TEXT it was supposedly extracted from.

SOURCE_TEXT (raw scraped website content, truncated):
\"\"\"{raw_content}\"\"\"

EXTRACTED_DATA (JSON claimed to be derived from the source above):
{json.dumps(profile, indent=2)[:4000]}

Score two things:
1. "accuracy_score": 0-100, how factually correct are the extracted fields relative to what's actually in SOURCE_TEXT (100 = every checkable claim matches the source).
2. "hallucination_rate": 0-100, the percentage of extracted claims (features, pricing, audience, news, etc.) that are NOT supported anywhere in SOURCE_TEXT (0 = fully grounded, 100 = entirely fabricated).
3. "ungrounded_claims": a short list (max 5) of specific extracted claims that could not be verified in SOURCE_TEXT.

Reply with ONLY this JSON shape:
{{"accuracy_score": <number>, "hallucination_rate": <number>, "ungrounded_claims": ["..."]}}"""
        result = self._judge(prompt)
        return {
            "accuracy_score": result.get("accuracy_score"),
            "hallucination_rate": result.get("hallucination_rate"),
            "ungrounded_claims": result.get("ungrounded_claims", []),
        }

    # ── metric 3: insight quality ───────────────────────────────────────

    def score_insight_quality(self, insight: dict) -> dict:
        prompt = f"""Judge the quality of this competitive-intelligence SWOT + recommendations.
Score each dimension 0-10:

- "relevance": are the points specific to THIS company, not generic boilerplate?
- "specificity": concrete detail vs vague statements?
- "actionability": could someone actually act on the recommended_actions tomorrow?
- "non_generic": would this differ from what you'd write about a random competitor with no info?

INSIGHT_DATA:
{json.dumps(insight, indent=2)[:3000]}

Reply with ONLY this JSON shape:
{{"relevance": <0-10>, "specificity": <0-10>, "actionability": <0-10>, "non_generic": <0-10>, "reasoning": "<one sentence>"}}"""
        result = self._judge(prompt)
        dims = ["relevance", "specificity", "actionability", "non_generic"]
        values = [result.get(d) for d in dims if isinstance(result.get(d), (int, float))]
        overall = round((sum(values) / len(values)) * 10, 1) if values else None
        return {
            "overall_score": overall,
            "dimensions": {d: result.get(d) for d in dims},
            "reasoning": result.get("reasoning", ""),
        }

    # ── naive baseline (what "just ask an LLM" looks like) ─────────────

    def run_naive_baseline(self, url: str) -> tuple:
        """One ungrounded prompt, no scraping, no multi-agent pipeline — the comparison point."""
        prompt = f"""You are analyzing the competitor at this URL: {url}
Without browsing or scraping, produce your best-guess competitor profile and SWOT analysis
based on what you already know. Reply with ONLY this JSON shape:
{{
  "profile": {{
    "company_name": "", "tagline": "", "description": "", "products_services": [],
    "pricing_model": "", "pricing_tiers": [], "target_audience": [],
    "key_features": [], "tech_stack": [], "social_links": {{}}, "contact_info": {{}}, "recent_news": []
  }},
  "insight": {{
    "strengths": [], "weaknesses": [], "opportunities": [], "threats": [],
    "unique_selling_points": [], "market_positioning": "", "recommended_actions": []
  }}
}}"""
        start = time.perf_counter()
        resp = self.groq_client.chat.completions.create(
            model=self.judge_model,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = round(time.perf_counter() - start, 3)
        data = self._parse_json(resp.choices[0].message.content)
        return data, elapsed

    # ── orchestration ────────────────────────────────────────────────────

    def evaluate_sync(self, url: str) -> EvaluationReport:
        """Runs BOTH the real multi-agent pipeline and the naive baseline, scores both."""

        # 1) Ground truth fetch (independent of the extraction agent, used only for judging)
        (raw_content, ground_error), t_ground = self._timed(self._fetch_raw_content, url)

        # 2) Multi-agent pipeline (the actual product)
        profile_obj, t_scrape = self._timed(self.team.scrape_competitor_sync, url)
        insight_obj, t_analyze = self._timed(self.team.analyze_competitor_sync, profile_obj)
        profile, insight = asdict(profile_obj), asdict(insight_obj)

        t_judge_start = time.perf_counter()
        ma_completeness = self.score_completeness(profile)
        ma_acc = self.score_accuracy_and_hallucination(profile, raw_content, ground_error)
        ma_quality = self.score_insight_quality(insight)
        t_judge = round(time.perf_counter() - t_judge_start, 3)

        multi_agent = MetricScores(
            accuracy=ma_acc.get("accuracy_score"),
            completeness=ma_completeness,
            insight_quality=ma_quality.get("overall_score"),
            hallucination_rate=ma_acc.get("hallucination_rate"),
            response_time_seconds=round(t_scrape + t_analyze, 2),
            details={
                "stage_timings": asdict(StageTimings(
                    scrape_seconds=t_scrape,
                    analyze_seconds=t_analyze,
                    grounding_fetch_seconds=t_ground,
                    judging_seconds=t_judge,
                    total_seconds=round(t_scrape + t_analyze, 2),
                )),
                "ungrounded_claims": ma_acc.get("ungrounded_claims", []),
                "insight_dimensions": ma_quality.get("dimensions", {}),
                "grounding_error": ground_error,
            },
        )

        # 3) Naive baseline (comparison point)
        baseline_data, t_baseline = self.run_naive_baseline(url)
        b_profile = baseline_data.get("profile", {})
        b_insight = baseline_data.get("insight", {})
        b_completeness = self.score_completeness(b_profile)
        b_acc = self.score_accuracy_and_hallucination(b_profile, raw_content, ground_error)
        b_quality = self.score_insight_quality(b_insight)

        naive_baseline = MetricScores(
            accuracy=b_acc.get("accuracy_score"),
            completeness=b_completeness,
            insight_quality=b_quality.get("overall_score"),
            hallucination_rate=b_acc.get("hallucination_rate"),
            response_time_seconds=t_baseline,
            details={
                "ungrounded_claims": b_acc.get("ungrounded_claims", []),
                "insight_dimensions": b_quality.get("dimensions", {}),
                "grounding_error": ground_error,
            },
        )

        # 4) Verdict summary
        verdict = {}
        for metric in ["accuracy", "completeness", "insight_quality"]:
            m_val, b_val = getattr(multi_agent, metric), getattr(naive_baseline, metric)
            if m_val is not None and b_val is not None:
                verdict[metric] = {
                    "winner": "multi_agent" if m_val >= b_val else "naive_baseline",
                    "delta": round(m_val - b_val, 1),
                }
        if multi_agent.hallucination_rate is not None and naive_baseline.hallucination_rate is not None:
            verdict["hallucination_rate"] = {
                "winner": "multi_agent" if multi_agent.hallucination_rate <= naive_baseline.hallucination_rate else "naive_baseline",
                "delta": round(naive_baseline.hallucination_rate - multi_agent.hallucination_rate, 1),
            }
        verdict["response_time_seconds"] = {
            "multi_agent": multi_agent.response_time_seconds,
            "naive_baseline": naive_baseline.response_time_seconds,
            "note": "Baseline is faster but ungrounded — see accuracy/hallucination for why that speed is misleading.",
        }

        return EvaluationReport(
            url=url,
            evaluated_at=datetime.now().isoformat(),
            multi_agent=multi_agent,
            naive_baseline=naive_baseline,
            verdict=verdict,
            profile=profile,
            insight=insight,
        )