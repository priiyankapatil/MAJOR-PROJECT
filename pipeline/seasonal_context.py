from crop_calendar import get_crop_stage
from soil_moisture import get_soil_moisture
from pest_alerts import get_pest_alerts
from datetime import datetime


def get_full_seasonal_context(query: str, lat: float, lon: float, month: int = None) -> dict:
    if month is None:
        month = datetime.now().month

    print("\n🌿 Step 4D: Building Seasonal Context Layer...")

    # 1. Crop stage
    crop_stage = get_crop_stage(query, month)

    # 2. Soil moisture
    soil = get_soil_moisture(lat, lon)

    # 3. Pest alerts
    pests = get_pest_alerts(query, month)

    # Build context string to inject into prompt
    context_string = f"""
=== SEASONAL AGRICULTURAL CONTEXT ===
Crop          : {crop_stage['crop']}
Growth Stage  : {crop_stage['stage']} (Month {month})
Soil Moisture : {soil.get('moisture_label')}
Soil M³/M³    : {soil.get('soil_moisture_m3')}
Season        : {pests['season']}
Pest Alerts   :
"""
    for alert in pests['alerts']:
        context_string += f"  ⚠️  {alert}\n"

    context_string += "======================================"

    print(context_string)

    return {
        "crop_stage": crop_stage,
        "soil_moisture": soil,
        "pest_alerts": pests,
        "context_string": context_string
    }
