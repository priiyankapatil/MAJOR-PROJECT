# step8_weather_rag.py
# =============================================
# Detects weather-related queries
# Injects real-time weather into context
# before generating answer
# =============================================

import re
from weather_fetcher import get_farm_weather, weather_to_context

# Weather-related keywords
WEATHER_KEYWORDS = [
    "weather", "rain", "rainfall", "irrigation today",
    "spray today", "fertilize today", "harvest today",
    "temperature", "humidity", "monsoon", "drought",
    "flood", "should i water", "should i spray",
    "tomorrow", "this week", "forecast", "climate today",
]

# Location extraction pattern
LOCATION_PATTERNS = [
    r"in ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
    r"at ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
    r"near ([A-Z][a-z]+(?: [A-Z][a-z]+)*)",
    r"([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*) farm",
    r"([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*) district",
]

DEFAULT_LOCATION = "Coimbatore, Tamil Nadu"


def is_weather_query(query):
    """Check if query needs real-time weather data."""
    q_lower = query.lower()
    return any(kw in q_lower for kw in WEATHER_KEYWORDS)


def extract_location(query):
    """Extract location from query text."""
    for pattern in LOCATION_PATTERNS:
        match = re.search(pattern, query)
        if match:
            return match.group(1)
    return DEFAULT_LOCATION


def enrich_with_weather(query, base_context):
    """
    If query is weather-related:
    1. Extract location from query
    2. Fetch real-time weather
    3. Prepend weather context to base context

    Returns enriched context string.
    """
    if not is_weather_query(query):
        return base_context, None

    location = extract_location(query)
    print(f"\n🌦️  Weather query detected")
    print(f"   Location: {location}")

    weather = get_farm_weather(location)

    if weather:
        weather_ctx = weather_to_context(weather)
        enriched    = weather_ctx + "\n\n" + base_context
        return enriched, weather
    else:
        return base_context, None