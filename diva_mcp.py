# diva_mcp.py - MCP Tools for Diva with LLM-native Tool Calling
import asyncio
import json
from typing import Optional, List, Dict, Any
from mcp_servers import get_weather, get_forecast, web_search


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temperature, conditions, humidity, and wind.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (e.g., 'London', 'New York', 'Tokyo')"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_forecast",
            "description": "Get weather forecast for multiple days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Number of days (1-5)", "default": 3}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information using DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]


TOOL_MAP = {
    "get_weather": get_weather,
    "get_forecast": get_forecast,
    "web_search": web_search,
}


SYSTEM_PROMPT = """You are Diva, a smart voice assistant. Always show your step-by-step thinking before answering.

When answering questions, follow this format:
1. First, think out loud: "Let me think about this..."
2. Break down the question: "The user is asking about..."
3. If tools are needed: "I should use [tool_name] to get accurate data"
4. Then provide your answer

You have access to these tools:
- get_weather(location): Get current weather for a city
- get_forecast(location, days): Get weather forecast for multiple days
- web_search(query): Search the web for information

IMPORTANT RULES:
- ALWAYS show your reasoning first, then use tools if needed
- Use ONLY the data from tool results - do NOT make up any information
- If tool returns ERROR:, explain the error to the user
- Be conversational but show your thinking process"""



class DivaMCP:
    """
    MCP tools wrapper with LLM-native tool calling.
    Uses Ollama's tool calling feature for intelligent tool selection.
    """
    
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.tools = TOOL_SCHEMAS
        self.tool_map = TOOL_MAP
    
    def get_tools_for_llm(self) -> List[Dict]:
        """Return tool schemas for LLM"""
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool by name with given arguments.
        """
        if tool_name not in self.tool_map:
            return f"ERROR: Unknown tool '{tool_name}'"
        
        try:
            tool_func = self.tool_map[tool_name]
            result = await tool_func(**arguments)
            return result
        
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def parse_tool_calls(self, response: Dict) -> Optional[List[Dict]]:
        """
        Parse tool calls from LLM response.
        Handles different response formats from various LLM providers.
        """
        # Ollama with tool support
        if "tool_calls" in response:
            return response["tool_calls"]
        
        # Alternative format
        if "message" in response and "tool_calls" in response["message"]:
            return response["message"]["tool_calls"]
        
        return None


# Global instance
diva_mcp = DivaMCP(debug=True)
