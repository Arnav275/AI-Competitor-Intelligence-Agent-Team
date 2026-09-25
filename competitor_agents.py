"""
Competitor Intelligence Agents
Uses Firecrawl for web scraping and Agno's AI Agent framework for analysis.
All agent calls are SYNCHRONOUS — async wrappers live in server.py only.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime

from firecrawl import FirecrawlApp
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.firecrawl import FirecrawlTools
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich import box

console = Console()


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class CompetitorProfile:
    url: str
    company_name: str = ""
    tagline: str = ""
    description: str = ""
    products_services: list = field(default_factory=list)
    pricing_model: str = ""
    pricing_tiers: list = field(default_factory=list)
    target_audience: str = ""
    key_features: list = field(default_factory=list)
    tech_stack: list = field(default_factory=list)
    social_links: dict = field(default_factory=dict)
    contact_info: dict = field(default_factory=dict)
    recent_news: list = field(default_factory=list)
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CompetitiveInsight:
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    opportunities: list = field(default_factory=list)
    threats: list = field(default_factory=list)
    unique_selling_points: list = field(default_factory=list)
    market_positioning: str = ""
    recommended_actions: list = field(default_factory=list)


# ─────────────────────────────────────────────
# Agent Prompts
# ─────────────────────────────────────────────

SCRAPER_INSTRUCTIONS = """
You are a web intelligence specialist. Your job is to extract structured competitor data from websites.

When given a URL, use Firecrawl to scrape the website and extract:
1. Company name, tagline, and description
2. Products/services offered
3. Pricing information and tiers
4. Target audience and market segment
5. Key features and differentiators
6. Technology stack hints (from footer, meta tags, integrations mentioned)
7. Social media links
8. Contact information
9. Any recent news, blog posts, or announcements

Return ONLY a valid JSON object — no markdown, no explanation, no backticks:
{
  "company_name": "",
  "tagline": "",
  "description": "",
  "products_services": [],
  "pricing_model": "",
  "pricing_tiers": [],
  "target_audience": "",
  "key_features": [],
  "tech_stack": [],
  "social_links": {},
  "contact_info": {},
  "recent_news": []
}
"""

ANALYST_INSTRUCTIONS = """
You are a senior competitive intelligence analyst with expertise in market strategy.

Given structured competitor data, perform a deep SWOT analysis and generate actionable insights.

Return ONLY a valid JSON object — no markdown, no explanation, no backticks:
{
  "strengths": [],
  "weaknesses": [],
  "opportunities": [],
  "threats": [],
  "unique_selling_points": [],
  "market_positioning": "",
  "recommended_actions": []
}
"""

STRATEGIST_INSTRUCTIONS = """
You are a chief strategy officer specializing in competitive positioning.

Given multiple competitor profiles and analyses, synthesize a comprehensive strategic report.

Return ONLY a valid JSON object — no markdown, no explanation, no backticks:
{
  "market_landscape": "",
  "competitive_clusters": [],
  "industry_trends": [],
  "white_space_opportunities": [],
  "differentiation_strategy": "",
  "priority_actions": [],
  "executive_summary": ""
}
"""


# ─────────────────────────────────────────────
# Agent Team
# ─────────────────────────────────────────────

class CompetitorIntelligenceTeam:

    def __init__(self, firecrawl_api_key: str, groq_api_key: str):
        self.firecrawl_api_key = firecrawl_api_key
        self.groq_api_key = groq_api_key
        firecrawl_tools = FirecrawlTools(api_key=firecrawl_api_key, enable_scrape=True, enable_crawl=False)
        model = Groq(id="openai/gpt-oss-120b", api_key=groq_api_key)

        self.scraper_agent = Agent(
            name="Web Scraper Agent",
            role="Extract structured data from competitor websites",
            model=model,
            tools=[firecrawl_tools],
            instructions=SCRAPER_INSTRUCTIONS,
            debug_mode=False,
            markdown=False,
        )

        self.analyst_agent = Agent(
            name="Intelligence Analyst Agent",
            role="Perform SWOT analysis and generate competitive insights",
            model=model,
            instructions=ANALYST_INSTRUCTIONS,
            debug_mode=False,
            markdown=False,
        )

        self.strategist_agent = Agent(
            name="Strategy Synthesizer Agent",
            role="Synthesize multi-competitor analysis into strategic recommendations",
            model=model,
            instructions=STRATEGIST_INSTRUCTIONS,
            debug_mode=False,
            markdown=False,
        )

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _parse_json_response(self, response_text: str) -> dict:
        """Safely extract JSON from agent response."""
        text = str(response_text).strip()
        # Strip markdown code fences if present
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

    def _get_agent_text(self, response) -> str:
        """Extract plain text from Agno agent response object."""
        # RunResponse object
        if hasattr(response, 'content') and response.content:
            return str(response.content)
        # Message list
        if hasattr(response, 'messages') and response.messages:
            last = response.messages[-1]
            if hasattr(last, 'content'):
                return str(last.content)
        return str(response)

    # ─── Sync agent calls (safe to run in thread pool) ──────────────────────

    def scrape_competitor_sync(self, url: str) -> CompetitorProfile:
        """Agent 1 — fully synchronous. Call from thread pool only."""
        profile = CompetitorProfile(url=url)
        try:
            response = self.scraper_agent.run(
                f"Scrape this competitor website and extract all available information: {url}"
            )
            text = self._get_agent_text(response)
            data = self._parse_json_response(text)
            for key, value in data.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
        except Exception as e:
            console.print(f"[red]Scraper error: {e}[/red]")
        return profile

    def analyze_competitor_sync(self, profile: CompetitorProfile) -> CompetitiveInsight:
        """Agent 2 — fully synchronous. Call from thread pool only."""
        insight = CompetitiveInsight()
        try:
            profile_json = json.dumps(asdict(profile), indent=2)
            response = self.analyst_agent.run(
                f"Analyze this competitor data and generate a comprehensive SWOT analysis "
                f"with strategic insights:\n\n{profile_json}"
            )
            text = self._get_agent_text(response)
            data = self._parse_json_response(text)
            for key, value in data.items():
                if hasattr(insight, key):
                    setattr(insight, key, value)
        except Exception as e:
            console.print(f"[red]Analyst error: {e}[/red]")
        return insight

    def synthesize_strategy_sync(
        self,
        profiles: list,
        insights: list,
    ) -> dict:
        """Agent 3 — fully synchronous. Call from thread pool only."""
        try:
            combined = [
                {"competitor": asdict(p), "analysis": asdict(i)}
                for p, i in zip(profiles, insights)
            ]
            response = self.strategist_agent.run(
                f"Synthesize this multi-competitor analysis into a comprehensive "
                f"strategic intelligence report:\n\n{json.dumps(combined, indent=2)}"
            )
            text = self._get_agent_text(response)
            return self._parse_json_response(text)
        except Exception as e:
            console.print(f"[red]Strategist error: {e}[/red]")
            return {}

    # ─── Async public API (used by main.py CLI) ──────────────────────────────

    async def analyze_competitor(self, url: str) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()
        profile = await loop.run_in_executor(None, self.scrape_competitor_sync, url)
        insight = await loop.run_in_executor(None, self.analyze_competitor_sync, profile)
        return {"profile": asdict(profile), "insight": asdict(insight)}

    async def compare_competitors(self, urls: list) -> dict:
        import asyncio
        loop = asyncio.get_event_loop()
        profiles, insights = [], []
        for url in urls:
            p = await loop.run_in_executor(None, self.scrape_competitor_sync, url)
            i = await loop.run_in_executor(None, self.analyze_competitor_sync, p)
            profiles.append(p)
            insights.append(i)
        strategy = await loop.run_in_executor(
            None, self.synthesize_strategy_sync, profiles, insights
        )
        return {
            "competitors": [
                {"profile": asdict(p), "insight": asdict(i)}
                for p, i in zip(profiles, insights)
            ],
            "strategy": strategy,
        }

    async def generate_full_report(self, urls: list, output_file: str) -> dict:
        result = await self.compare_competitors(urls)
        result["generated_at"] = datetime.now().isoformat()
        result["total_competitors"] = len(urls)
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # ─── Rich display (CLI) ──────────────────────────────────────────────────

    def display_analysis(self, result: dict):
        profile = result.get("profile", {})
        insight = result.get("insight", {})
        company = profile.get("company_name", "Unknown Company")
        tagline = profile.get("tagline", "")
        console.print(Panel(
            f"[bold white]{company}[/bold white]\n[dim]{tagline}[/dim]",
            title="🏢 Competitor Profile", border_style="cyan",
        ))
        t = Table(box=box.ROUNDED, show_header=False, border_style="dim")
        t.add_column("Field", style="cyan", width=22)
        t.add_column("Value", style="white")
        t.add_row("URL", profile.get("url", ""))
        desc = profile.get("description", "")
        t.add_row("Description", (desc[:120] + "...") if len(desc) > 120 else desc)
        t.add_row("Target Audience", profile.get("target_audience", ""))
        t.add_row("Pricing Model", profile.get("pricing_model", ""))
        t.add_row("Key Features", "\n".join(f"• {f}" for f in profile.get("key_features", [])[:5]))
        t.add_row("Products/Services", "\n".join(f"• {p}" for p in profile.get("products_services", [])[:5]))
        console.print(t)
        console.print("\n[bold yellow]📊 SWOT Analysis[/bold yellow]")
        def sp(title, items, color):
            content = "\n".join(f"• {i}" for i in items[:5]) if items else "None identified"
            return Panel(content, title=title, border_style=color, padding=(0, 1))
        console.print(Columns([
            sp("💪 Strengths", insight.get("strengths", []), "green"),
            sp("⚠️  Weaknesses", insight.get("weaknesses", []), "red"),
        ], equal=True))
        console.print(Columns([
            sp("🚀 Opportunities", insight.get("opportunities", []), "blue"),
            sp("🔥 Threats", insight.get("threats", []), "magenta"),
        ], equal=True))
        actions = insight.get("recommended_actions", [])
        if actions:
            console.print("\n[bold green]✅ Recommended Actions[/bold green]")
            for i, a in enumerate(actions, 1):
                console.print(f"  [cyan]{i}.[/cyan] {a}")

    def display_comparison(self, result: dict):
        competitors = result.get("competitors", [])
        strategy = result.get("strategy", {})
        console.print(Panel(
            f"Analyzed [bold]{len(competitors)}[/bold] competitors",
            title="🔄 Competitive Comparison", border_style="cyan",
        ))
        t = Table(box=box.DOUBLE_EDGE, border_style="cyan", show_lines=True)
        t.add_column("Competitor", style="bold white", width=20)
        t.add_column("Positioning", style="yellow", width=30)
        t.add_column("Top Strength", style="green", width=30)
        t.add_column("Top Weakness", style="red", width=30)
        for comp in competitors:
            p, i = comp.get("profile", {}), comp.get("insight", {})
            t.add_row(
                p.get("company_name", p.get("url", ""))[:20],
                i.get("market_positioning", "")[:60],
                (i.get("strengths") or [""])[0][:40],
                (i.get("weaknesses") or [""])[0][:40],
            )
        console.print(t)
        if strategy:
            console.print(Panel(
                strategy.get("executive_summary", "No summary available"),
                title="🧭 Strategic Summary", border_style="magenta",
            ))

    def display_report_summary(self, result: dict):
        strategy = result.get("strategy", {})
        console.print(Panel(
            f"[bold]Competitors Analyzed:[/bold] {result.get('total_competitors', 0)}\n"
            f"[bold]Generated:[/bold] {result.get('generated_at', '')}\n\n"
            f"[bold]Executive Summary:[/bold]\n{strategy.get('executive_summary', 'N/A')}",
            title="📋 Report Summary", border_style="green",
        ))