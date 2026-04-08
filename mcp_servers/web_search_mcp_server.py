# web_search_mcp_server.py - MCP Server for Web Search using Tavily
import os
import asyncio
import logging
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

mcp = FastMCP("web_search")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for information using Tavily.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return (default 5)
    """
    if not TAVILY_API_KEY:
        return "ERROR: Tavily API key not configured. Set TAVILY_API_KEY in .env"
    
    if not query:
        return "ERROR: Please provide a search query"
    
    try:
        from tavily import TavilyClient
        
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = await asyncio.to_thread(
            client.search,
            query,
            max_results=max_results
        )
        
        results = response.get("results", [])
        
        if not results:
            return f"No search results found for '{query}'"
        
        formatted = []
        for i, r in enumerate(results[:max_results], 1):
            title = r.get("title", "No title")
            snippet = r.get("content", "")[:300]
            url = r.get("url", "")
            
            formatted.append(f"{i}. {title}\n   {snippet}...\n   Source: {url}")
        
        return f"Search results for '{query}':\n\n" + "\n\n".join(formatted)
    
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return "ERROR: Invalid Tavily API key"
        return f"ERROR: Search failed: {error_msg}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
