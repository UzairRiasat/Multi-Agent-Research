import os
import httpx
from typing import List, Dict

# FIX #4: Use httpx with both sync and async interfaces.
# The sync version is used by searcher_node (run inside asyncio.to_thread).
# An async version is provided if you ever want to call it directly from async code.

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

def _get_api_key() -> str:
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not set in environment")
    return api_key

def _parse_results(data: dict) -> List[Dict]:
    results = data.get("web", {}).get("results", [])
    return [
        {
            "title": r.get("title", ""),
            "url":   r.get("url", ""),
            "snippet": r.get("description", ""),
        }
        for r in results
    ]

# FIX #8: Timeout + error handling so a bad search never crashes the graph.
# searcher_node already wraps this in try/except, but defence in depth is good.
def search_web(query: str, limit: int = 5) -> List[Dict]:
    """Synchronous search — safe to call from a thread pool."""
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": _get_api_key(),
    }
    params = {"q": query, "count": limit}

    with httpx.Client(timeout=10.0) as client:
        resp = client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        return _parse_results(resp.json())

async def search_web_async(query: str, limit: int = 5) -> List[Dict]:
    """Async version — use if calling directly from an async context."""
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": _get_api_key(),
    }
    params = {"q": query, "count": limit}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        resp.raise_for_status()
        return _parse_results(resp.json())