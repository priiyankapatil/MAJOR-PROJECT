import requests
from datetime import datetime, timedelta


def get_soil_moisture(lat: float, lon: float) -> dict:
    URL = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,et0_fao_evapotranspiration",
        "hourly": "soil_moisture_0_to_1cm",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1
    }

    try:
        resp = requests.get(URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        daily = data.get("daily", {})

        soil_list = hourly.get("soil_moisture_0_to_1cm", [])
        soil_moisture = soil_list[0] if soil_list else None

        precipitation = daily.get("precipitation_sum", [None])[0]
        evapotranspiration = daily.get("et0_fao_evapotranspiration", [None])[0]

        if soil_moisture is None:
            return {"soil_moisture_m3": None, "moisture_label": "UNAVAILABLE", "advisory": "Soil moisture data unavailable"}

        if soil_moisture > 0.3:
            moisture_label = "WET — delay fertilizer application"
        elif soil_moisture > 0.15:
            moisture_label = "OPTIMAL — good time for fertilizer"
        else:
            moisture_label = "DRY — irrigate before fertilizing"

        print(f"🌍 Soil Moisture : {soil_moisture} m³/m³ → {moisture_label}")

        return {
            "soil_moisture_m3": soil_moisture,
            "moisture_label": moisture_label,
            "precipitation_mm": precipitation,
            "evapotranspiration_mm": evapotranspiration,
            "advisory": moisture_label
        }

    except Exception as e:
        print(f"   ⚠️  Soil moisture API error: {e}")
        return {"soil_moisture_m3": None, "moisture_label": "UNAVAILABLE", "advisory": "Soil moisture data unavailable"}
