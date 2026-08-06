# weather_fetcher.py
# =============================================
# Real-time weather data for Indian farms
# Uses Open-Meteo API — completely FREE
# No API key required
# =============================================

import requests
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim

# ── Geocoder to convert city name to coordinates ──
geocoder = Nominatim(user_agent="agri_rag_india")


# ─────────────────────────────────────────────
# CORE: GET COORDINATES FROM PLACE NAME
# ─────────────────────────────────────────────

def get_coordinates(location_name):
    """
    Convert a place name to lat/lon coordinates.

    Examples:
    "Coimbatore, Tamil Nadu" → (11.0168, 76.9558)
    "Thrissur, Kerala"       → (10.5276, 76.2144)
    "Bellary, Karnataka"     → (15.1394, 76.9214)
    """
    try:
        # Add India to improve accuracy
        query = f"{location_name}, India"
        loc   = geocoder.geocode(query, timeout=5)

        if loc:
            return {
                "lat"     : round(loc.latitude,  4),
                "lon"     : round(loc.longitude, 4),
                "address" : loc.address,
                "found"   : True
            }
        else:
            # Default to Chennai if not found
            print(f"   ⚠️  Location '{location_name}' "
                  f"not found — using Chennai")
            return {
                "lat"    : 13.0827,
                "lon"    : 80.2707,
                "address": "Chennai, Tamil Nadu",
                "found"  : False
            }
    except Exception as e:
        print(f"   ⚠️  Geocoding error: {e}")
        return {"lat":13.0827, "lon":80.2707,
                "address":"Chennai", "found":False}


# ─────────────────────────────────────────────
# CORE: FETCH WEATHER FROM OPEN-METEO
# ─────────────────────────────────────────────

def fetch_weather(lat, lon, days=7):
    """
    Fetch real-time + forecast weather data.
    Uses Open-Meteo — no API key needed.

    Returns agricultural weather variables:
    - Temperature (min, max, current)
    - Rainfall (today + next 7 days)
    - Humidity
    - Wind speed
    - Soil temperature
    - ET0 (evapotranspiration — crop water need)
    - UV index
    """
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude"               : lat,
        "longitude"              : lon,
        "current"                : [
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "wind_speed_10m",
            "weather_code",
        ],
        "daily"                  : [
            "temperature_2m_max",
            "temperature_2m_min",
            "rain_sum",
            "precipitation_probability_max",
            "et0_fao_evapotranspiration",
            "uv_index_max",
            "wind_speed_10m_max",
        ],
        "forecast_days"          : days,
        "timezone"               : "Asia/Kolkata",
    }

    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"   ⚠️  Weather API error: {e}")
        return None


# ─────────────────────────────────────────────
# PROCESS: CONVERT RAW DATA TO READABLE FORMAT
# ─────────────────────────────────────────────

WEATHER_CODES = {
    0:"Clear sky", 1:"Mainly clear", 2:"Partly cloudy",
    3:"Overcast", 45:"Foggy", 48:"Icy fog",
    51:"Light drizzle", 53:"Drizzle", 55:"Heavy drizzle",
    61:"Slight rain", 63:"Moderate rain", 65:"Heavy rain",
    71:"Slight snow", 73:"Moderate snow", 75:"Heavy snow",
    80:"Slight showers", 81:"Moderate showers",
    82:"Heavy showers", 95:"Thunderstorm",
    96:"Thunderstorm with hail",
}


def process_weather(raw, location_name):
    """
    Convert raw API response into clean dict
    ready to inject into agricultural prompts.
    """
    if not raw:
        return None

    current = raw.get("current", {})
    daily   = raw.get("daily",   {})

    # Current conditions
    temp_now  = current.get("temperature_2m", "N/A")
    humidity  = current.get("relative_humidity_2m", "N/A")
    rain_now  = current.get("rain", 0)
    wind_now  = current.get("wind_speed_10m", "N/A")
    wcode     = current.get("weather_code", 0)
    condition = WEATHER_CODES.get(wcode, "Unknown")

    # 7-day forecast summary
    dates     = daily.get("time", [])
    rain_list = daily.get("rain_sum", [])
    tmax_list = daily.get("temperature_2m_max", [])
    tmin_list = daily.get("temperature_2m_min", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    et0_list  = daily.get("et0_fao_evapotranspiration", [])
    uv_list   = daily.get("uv_index_max", [])

    # Build 7-day forecast
    forecast = []
    for i, date in enumerate(dates[:7]):
        forecast.append({
            "date"      : date,
            "tmax"      : tmax_list[i] if i < len(tmax_list) else None,
            "tmin"      : tmin_list[i] if i < len(tmin_list) else None,
            "rain_mm"   : rain_list[i] if i < len(rain_list) else 0,
            "rain_prob" : rain_prob[i] if i < len(rain_prob) else 0,
            "et0"       : et0_list[i]  if i < len(et0_list)  else None,
            "uv"        : uv_list[i]   if i < len(uv_list)   else None,
        })

    # Agricultural alerts
    alerts = []
    total_rain_7d = sum(r for r in rain_list[:7] if r)

    if total_rain_7d > 100:
        alerts.append("Heavy rainfall expected — "
                      "avoid pesticide spraying")
    if total_rain_7d < 5:
        alerts.append("Dry week ahead — "
                      "irrigation required")
    if temp_now and temp_now > 38:
        alerts.append("High heat stress risk — "
                      "water crops in early morning")
    if temp_now and temp_now < 15:
        alerts.append("Cold temperature — "
                      "protect sensitive seedlings")
    if humidity and humidity > 85:
        alerts.append("High humidity — "
                      "fungal disease risk is elevated")
    if any(p > 70 for p in rain_prob[:3] if p):
        alerts.append("Rain likely in next 3 days — "
                      "delay fertilizer application")

    return {
        "location"      : location_name,
        "timestamp"     : datetime.now().strftime(
                            "%Y-%m-%d %H:%M IST"),
        "current"       : {
            "temperature_c": temp_now,
            "humidity_pct" : humidity,
            "rain_mm"      : rain_now,
            "wind_kmh"     : wind_now,
            "condition"    : condition,
        },
        "forecast_7d"   : forecast,
        "total_rain_7d" : round(total_rain_7d, 1),
        "alerts"        : alerts,
    }


# ─────────────────────────────────────────────
# MAIN: GET WEATHER FOR ANY INDIAN LOCATION
# ─────────────────────────────────────────────

def get_farm_weather(location_name):
    """
    Single function to call for weather data.
    Returns clean weather dict for any Indian location.

    Usage:
        weather = get_farm_weather("Coimbatore")
        weather = get_farm_weather("Thrissur, Kerala")
        weather = get_farm_weather("Bellary, Karnataka")
    """
    print(f"\n🌦️  Fetching weather for: {location_name}")

    # Step 1: Get coordinates
    coords = get_coordinates(location_name)
    print(f"   📍 Coordinates: "
          f"{coords['lat']}, {coords['lon']}")
    print(f"   📍 Address: {coords['address']}")

    # Step 2: Fetch weather
    raw = fetch_weather(coords["lat"], coords["lon"])

    if not raw:
        return None

    # Step 3: Process
    weather = process_weather(raw, location_name)

    # Display summary
    if weather:
        c = weather["current"]
        print(f"   🌡️  Temperature : {c['temperature_c']}°C")
        print(f"   💧 Humidity    : {c['humidity_pct']}%")
        print(f"   🌧️  Condition   : {c['condition']}")
        print(f"   🌧️  Rain 7 days : "
              f"{weather['total_rain_7d']} mm")
        if weather["alerts"]:
            print(f"   ⚠️  Alerts:")
            for a in weather["alerts"]:
                print(f"      • {a}")

    return weather


def weather_to_context(weather):
    """
    Convert weather data to text for LLM prompt injection.
    This text gets added to the agricultural context.
    """
    if not weather:
        return "Weather data unavailable."

    c = weather["current"]
    f = weather["forecast_7d"]

    lines = [
        f"=== REAL-TIME WEATHER: {weather['location']} ===",
        f"As of {weather['timestamp']}:",
        f"Current: {c['condition']}, "
        f"{c['temperature_c']}°C, "
        f"Humidity {c['humidity_pct']}%, "
        f"Wind {c['wind_kmh']} km/h",
        f"Rainfall today: {c['rain_mm']} mm",
        f"",
        f"7-Day Forecast:"
    ]

    for d in f[:7]:
        rain_info = (f"{d['rain_mm']}mm rain "
                    f"({d['rain_prob']}% chance)"
                    if d['rain_mm'] else
                    f"No rain ({d['rain_prob']}% chance)")
        et0_info  = (f", ET0: {d['et0']}mm"
                    if d['et0'] else "")
        lines.append(
            f"  {d['date']}: "
            f"{d['tmin']}–{d['tmax']}°C, "
            f"{rain_info}{et0_info}"
        )

    if weather["alerts"]:
        lines.append(f"\n⚠️  Agricultural Alerts:")
        for a in weather["alerts"]:
            lines.append(f"  • {a}")

    lines.append("=" * 40)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_locations = [
        "Coimbatore, Tamil Nadu",
        "Thrissur, Kerala",
        "Bellary, Karnataka",
    ]

    for loc in test_locations:
        weather = get_farm_weather(loc)
        if weather:
            print("\n📋 PROMPT CONTEXT:")
            print(weather_to_context(weather))
        print("\n" + "─"*50)