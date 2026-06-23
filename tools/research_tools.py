# tools/research_tools.py
"""
ADK tool definitions used by the Researcher agent.

In Google ADK, tools are plain Python functions decorated with
type annotations — the framework auto-generates the JSON schema.
"""

import json


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Web search simulator
# In production: replace body with a real search API call
# (e.g. Tavily, SerpAPI, or Google Custom Search)
# ─────────────────────────────────────────────────────────────────────────────
def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web for information on a given query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 3).

    Returns:
        A JSON string containing a list of search results with
        title, url, and snippet fields.
    """
    # --- Simulated results (replace with real API call) ---
    simulated_results = [
        {
            "title": f"[Result 1] Overview of: {query}",
            "url": f"https://example.com/article-1?q={query.replace(' ', '+')}",
            "snippet": (
                f"Comprehensive overview of {query}. "
                "This source covers key concepts, recent developments, "
                "and practical applications in the field."
            ),
            "published_date": "2024-11-01",
        },
        {
            "title": f"[Result 2] Deep dive into {query}",
            "url": f"https://research.example.org/deep-dive?topic={query.replace(' ', '+')}",
            "snippet": (
                f"In-depth analysis of {query} from leading researchers. "
                "Includes statistical data, case studies, and expert opinions."
            ),
            "published_date": "2025-01-15",
        },
        {
            "title": f"[Result 3] {query} — latest trends 2025",
            "url": f"https://techblog.example.io/trends/{query.replace(' ', '-')}",
            "snippet": (
                f"Emerging trends and future outlook for {query}. "
                "Industry experts weigh in on what to expect in the coming years."
            ),
            "published_date": "2025-03-20",
        },
    ]

    results = simulated_results[:max_results]
    return json.dumps({"query": query, "results": results, "total_found": len(results)})


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Fetch page content simulator
# In production: replace with httpx/requests + HTML parser
# ─────────────────────────────────────────────────────────────────────────────
def fetch_page_content(url: str) -> str:
    """
    Fetch and extract the main text content from a web page URL.

    Args:
        url: The full URL of the page to fetch.

    Returns:
        A JSON string with the page title and extracted text content.
    """
    # --- Simulated page content ---
    simulated_content = {
        "url": url,
        "title": f"Content from {url}",
        "content": (
            "This article explores the topic in depth. "
            "Key findings include: (1) significant growth in adoption rates "
            "over the past two years, (2) emerging use cases in enterprise settings, "
            "and (3) ongoing challenges around scalability and cost-efficiency. "
            "Experts predict continued innovation driven by open-source communities "
            "and major cloud providers. The article concludes with recommendations "
            "for practitioners looking to adopt these technologies."
        ),
        "word_count": 1200,
        "fetched_at": "2025-01-01T00:00:00",
    }
    return json.dumps(simulated_content)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — Get current date/time (useful for research context)
# ─────────────────────────────────────────────────────────────────────────────
def get_current_datetime() -> str:
    """
    Get the current UTC date and time.

    Returns:
        A JSON string with the current date, time, and timezone.
    """
    return json.dumps({
        "date": "2025-01-01",
        "time": "00:00:00",
        "timezone": "UTC",
        "iso": "2025-01-01T00:00:00",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Export all tools as a list for easy registration in ADK agents
# ─────────────────────────────────────────────────────────────────────────────
RESEARCHER_TOOLS = [search_web, fetch_page_content, get_current_datetime]
