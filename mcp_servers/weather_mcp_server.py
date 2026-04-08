# weather_mcp_server.py - MCP Server for Weather (FastMCP)
import os
import sys
import logging
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load .env from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

mcp = FastMCP("weather")

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")


@mcp.tool()
async def get_weather(location: str, units: str = "metric") -> str:
    """
    Get current weather for a location.
    
    Args:
        location: City name (e.g., 'London', 'New York', 'Tokyo')
        units: Temperature units - 'metric' (Celsius) or 'imperial' (Fahrenheit)
    """
    if not OPENWEATHERMAP_API_KEY:
        return "ERROR: OpenWeatherMap API key not configured. Set OPENWEATHERMAP_API_KEY in .env"
    
    if not location:
        return "ERROR: Please specify a location"
    
    temp_unit = "°C" if units == "metric" else "°F"
    speed_unit = "m/s" if units == "metric" else "mph"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": location, "appid": OPENWEATHERMAP_API_KEY, "units": units},
                timeout=10.0
            )
            
            if response.status_code == 404:
                return f"ERROR: Location '{location}' not found"
            elif response.status_code == 401:
                return "ERROR: Invalid OpenWeatherMap API key"
            elif response.status_code != 200:
                return f"ERROR: Weather API error: {response.status_code}"
            
            data = response.json()
            weather = data["weather"][0]
            
            result = f"Weather in {data['name']}, {data['sys']['country']}: Temperature: {data['main']['temp']}{temp_unit}, Feels like: {data['main']['feels_like']}{temp_unit}, Conditions: {weather['description'].capitalize()}, Humidity: {data['main']['humidity']}%, Wind: {data['wind']['speed']} {speed_unit}"
            return result
        
        except httpx.TimeoutException:
            return "ERROR: Weather API request timed out"
        except Exception as e:
            return f"ERROR: {str(e)}"


@mcp.tool()
async def get_forecast(location: str, days: int = 3, units: str = "metric") -> str:
    """
    Get weather forecast for multiple days.
    
    Args:
        location: City name
        days: Number of days (1-5)
        units: Temperature units - 'metric' or 'imperial'
    """
    if not OPENWEATHERMAP_API_KEY:
        return "ERROR: OpenWeatherMap API key not configured"
    
    if not location:
        return "ERROR: Please specify a location"
    
    # Convert days to int if it's a string
    try:
        days = int(days)
    except (ValueError, TypeError):
        days = 3
    
    temp_unit = "°C" if units == "metric" else "°F"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"q": location, "appid": OPENWEATHERMAP_API_KEY, "units": units, "cnt": days * 8},
                timeout=10.0
            )
            
            if response.status_code != 200:
                return f"ERROR: Forecast API error: {response.status_code}"
            
            data = response.json()
            
            daily_data = {}
            for item in data["list"]:
                date = item["dt_txt"].split(" ")[0]
                if date not in daily_data:
                    daily_data[date] = []
                daily_data[date].append(item)
            
            forecasts = []
            for date, items in list(daily_data.items())[:days]:
                temps = [item["main"]["temp"] for item in items]
                desc = items[len(items)//2]["weather"][0]["description"]
                forecasts.append(f"{date}: High {max(temps)}{temp_unit}, Low {min(temps)}{temp_unit}, {desc.capitalize()}")
            
            return f"Forecast for {data['city']['name']}:\n" + "\n".join(forecasts)
        
        except Exception as e:
            return f"ERROR: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
