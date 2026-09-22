from datetime import datetime

PEST_ALERT_RULES = {
    "rice": {
        "monsoon": ["Brown Plant Hopper (BPH) risk HIGH — apply Buprofezin 25SC @ 1ml/L",
                    "Blast disease risk — apply Tricyclazole 75WP @ 0.6g/L at tillering"],
        "summer":  ["Stem borer active — use pheromone traps",
                    "Leaf folder risk — apply Chlorpyriphos 2.5ml/L"],
        "winter":  ["Sheath blight risk during cool humid nights — monitor closely"]
    },
    "tomato": {
        "monsoon": ["Late blight (Phytophthora) HIGH risk — apply Mancozeb 2.5g/L",
                    "Fruit borer active — use Bacillus thuringiensis spray"],
        "summer":  ["Whitefly and TYLCV risk — use yellow sticky traps + Imidacloprid 0.3ml/L",
                    "Thrips infestation likely — apply Spinosad 0.5ml/L"],
        "winter":  ["Damping off risk in nursery — drench with Captan 2g/L"]
    },
    "coconut": {
        "monsoon": ["Rhinoceros beetle activity HIGH — use pheromone traps in trunk",
                    "Bud rot risk — apply Bordeaux mixture 1% at crown"],
        "summer":  ["Red palm weevil alert — inspect trunk regularly",
                    "Leaf caterpillar active — apply Carbaryl 0.1%"],
        "winter":  ["Eriophyid mite risk in dry cool weather — apply Wettable Sulphur 3g/L"]
    },
    "wheat": {
        "monsoon": ["Karnal bunt risk — use certified disease-free seed"],
        "summer":  ["Aphid infestation — apply Dimethoate 1.5ml/L"],
        "winter":  ["Yellow rust HIGH risk in cool humid conditions — apply Propiconazole 1ml/L",
                    "Termite damage — apply Chlorpyriphos to soil before sowing"]
    },
    "default": {
        "monsoon": ["High humidity — general fungal disease risk elevated. Monitor crops daily."],
        "summer":  ["Heat stress — ensure adequate irrigation. Watch for sucking pests."],
        "winter":  ["Cool dry conditions — watch for mites and powdery mildew."]
    }
}


def get_season_from_month(month: int) -> str:
    if month in [6, 7, 8, 9]:
        return "monsoon"
    if month in [3, 4, 5]:
        return "summer"
    return "winter"


def get_pest_alerts(query: str, month: int = None) -> dict:
    from crop_calendar import detect_crop_from_query

    if month is None:
        month = datetime.now().month

    crop = detect_crop_from_query(query)
    season = get_season_from_month(month)
    rules = PEST_ALERT_RULES.get(crop, PEST_ALERT_RULES["default"]) or PEST_ALERT_RULES["default"]
    alerts = rules.get(season, [])

    print(f"🐛 Pest Alerts [{season}] for {crop}:")
    for a in alerts:
        print(f"   ⚠️  {a}")

    return {
        "crop": crop,
        "season": season,
        "month": month,
        "alerts": alerts
    }
