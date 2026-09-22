from datetime import datetime

CROP_CALENDAR = {
    "rice": {
        1:  "harvesting",
        2:  "land preparation",
        3:  "land preparation",
        4:  "nursery sowing",
        5:  "nursery sowing",
        6:  "transplanting",
        7:  "vegetative growth",
        8:  "vegetative growth",
        9:  "flowering",
        10: "grain filling",
        11: "maturity",
        12: "harvesting"
    },
    "tomato": {
        1:  "harvesting",
        2:  "harvesting",
        3:  "land preparation",
        4:  "nursery sowing",
        5:  "transplanting",
        6:  "vegetative growth",
        7:  "flowering",
        8:  "fruit development",
        9:  "harvesting",
        10: "land preparation",
        11: "nursery sowing",
        12: "transplanting"
    },
    "coconut": {
        1:  "dry season management",
        2:  "dry season management",
        3:  "pre-monsoon fertilization",
        4:  "pre-monsoon fertilization",
        5:  "pre-monsoon irrigation",
        6:  "monsoon planting",
        7:  "monsoon fertilization",
        8:  "monsoon fertilization",
        9:  "post-monsoon care",
        10: "post-monsoon care",
        11: "harvesting",
        12: "harvesting"
    },
    "wheat": {
        1:  "flowering",
        2:  "grain filling",
        3:  "maturity",
        4:  "harvesting",
        5:  "land preparation",
        6:  "fallow",
        7:  "fallow",
        8:  "fallow",
        9:  "fallow",
        10: "land preparation",
        11: "sowing",
        12: "vegetative growth"
    },
    "cotton": {
        1:  "harvesting",
        2:  "harvesting",
        3:  "land preparation",
        4:  "sowing",
        5:  "vegetative growth",
        6:  "squaring",
        7:  "flowering",
        8:  "boll development",
        9:  "boll opening",
        10: "harvesting",
        11: "harvesting",
        12: "land preparation"
    },
    "default": {
        1: "winter dormancy or harvesting",
        2: "land preparation",
        3: "sowing season",
        4: "vegetative growth",
        5: "vegetative growth",
        6: "monsoon onset",
        7: "active growth",
        8: "active growth",
        9: "flowering or fruiting",
        10: "maturity",
        11: "harvesting",
        12: "post-harvest"
    }
}


def detect_crop_from_query(query: str) -> str:
    q = (query or "").lower()
    for crop in CROP_CALENDAR.keys():
        if crop == "default":
            continue
        if crop in q:
            return crop
    return "default"


def get_crop_stage(query: str, month: int = None) -> dict:
    if month is None:
        month = datetime.now().month

    crop = detect_crop_from_query(query)
    stages = CROP_CALENDAR.get(crop, CROP_CALENDAR["default"])
    stage = stages.get(month, CROP_CALENDAR["default"].get(month))

    print(f"🌱 Crop Stage: {crop} is in [{stage}] stage (Month {month})")

    return {
        "crop": crop,
        "month": month,
        "stage": stage,
        "advisory_note": f"Crop is currently in {stage} stage. Recommendations should align with this growth phase."
    }
