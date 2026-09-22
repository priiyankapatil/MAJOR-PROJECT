STAGE_COMPATIBILITY = {
    "land preparation": {
        "allowed": ["soil preparation", "tillage", "ploughing", "levelling",
                    "soil testing", "drainage", "irrigation setup",
                    "organic matter", "green manure", "lime application"],
        "blocked": ["harvesting", "post-harvest", "grain filling", "flowering",
                    "fruit development", "maturity", "pest spray at harvest",
                    "yield estimation", "storage"]
    },
    "nursery sowing": {
        "allowed": ["seed treatment", "nursery management", "germination",
                    "seedbed preparation", "damping off", "nursery fertilizer",
                    "seed rate", "seed selection"],
        "blocked": ["harvesting", "post-harvest", "grain filling", "transplanting advice",
                    "maturity", "fruit development", "storage", "yield estimation"]
    },
    "transplanting": {
        "allowed": ["transplanting", "spacing", "seedling", "root establishment",
                    "irrigation after transplant", "weed control early",
                    "gap filling", "starter fertilizer"],
        "blocked": ["harvesting", "post-harvest", "grain filling",
                    "maturity", "fruit development", "storage", "yield estimation"]
    },
    "vegetative growth": {
        "allowed": ["nitrogen fertilizer", "weed management", "irrigation",
                    "leaf area", "canopy management", "intercropping",
                    "growth regulators", "top dressing", "pest monitoring"],
        "blocked": ["harvesting", "post-harvest", "grain filling",
                    "maturity", "storage", "yield estimation", "fruit thinning"]
    },
    "flowering": {
        "allowed": ["pollination", "micronutrients", "boron application",
                    "flower drop prevention", "irrigation at flowering",
                    "pest control at flowering", "phosphorus", "potassium"],
        "blocked": ["harvesting", "post-harvest", "storage",
                    "nursery sowing", "land preparation", "transplanting",
                    "yield estimation", "grain drying"]
    },
    "fruit development": {
        "allowed": ["potassium fertilizer", "calcium application", "fruit set",
                    "irrigation for fruit fill", "fruit borer control",
                    "thinning", "size improvement", "color development"],
        "blocked": ["harvesting", "post-harvest", "storage",
                    "nursery sowing", "land preparation", "grain drying"]
    },
    "grain filling": {
        "allowed": ["potassium", "irrigation at grain fill", "foliar spray",
                    "pest control", "disease monitoring", "lodging prevention"],
        "blocked": ["harvesting", "post-harvest", "storage",
                    "nursery sowing", "land preparation", "transplanting"]
    },
    "maturity": {
        "allowed": ["maturity indicators", "moisture content", "harvest readiness",
                    "pre-harvest interval", "desiccation", "stopping irrigation"],
        "blocked": ["nursery sowing", "land preparation", "transplanting",
                    "vegetative fertilizer", "nitrogen top dressing",
                    "flowering advice", "fruit set"]
    },
    "harvesting": {
        "allowed": ["harvesting", "cutting", "threshing", "combine harvester",
                    "harvest loss", "timing of harvest", "moisture at harvest",
                    "yield estimation", "post-harvest handling"],
        "blocked": ["nursery sowing", "land preparation", "transplanting",
                    "nitrogen fertilizer", "growth regulators", "flowering advice",
                    "fruit set", "pollination"]
    },
    "post-harvest": {
        "allowed": ["storage", "drying", "milling", "grading", "packaging",
                    "cold storage", "fumigation", "market price",
                    "soil health restoration", "residue management"],
        "blocked": ["fertilizer application", "pest spray", "irrigation scheduling",
                    "flowering advice", "fruit development", "nursery sowing"]
    },
    "monsoon planting": {
        "allowed": ["seed treatment", "planting", "spacing", "basal fertilizer",
                    "drainage management", "waterlogging prevention",
                    "monsoon pest alert", "green manure"],
        "blocked": ["harvesting", "post-harvest", "storage",
                    "maturity", "grain drying", "yield estimation"]
    },
    "monsoon fertilization": {
        "allowed": ["organic fertilizer", "PGPR", "potassium", "micronutrients",
                    "split application", "fertigation", "soil moisture based application",
                    "monsoon irrigation management", "pest alert monsoon"],
        "blocked": ["harvesting", "post-harvest", "storage",
                    "nursery sowing", "grain drying", "yield estimation"]
    },
    "pre-monsoon fertilization": {
        "allowed": ["pre-monsoon soil preparation", "basal dose", "organic manure",
                    "lime application", "soil testing", "irrigation before monsoon"],
        "blocked": ["harvesting", "post-harvest", "storage", "grain drying"]
    },
    "dry season management": {
        "allowed": ["irrigation", "mulching", "drought management",
                    "water conservation", "soil moisture retention",
                    "shade management", "drip irrigation"],
        "blocked": ["harvesting", "monsoon pest", "waterlogging",
                    "nursery sowing in rain", "grain drying"]
    },
    "default": {
        "allowed": [],
        "blocked": []
    }
}


def get_compatible_stages(current_stage: str) -> dict:
    """Lookup a stage in STAGE_COMPATIBILITY.

    Matching is case-insensitive and allows partial matches.
    If no match is found, return the 'default' entry.
    """
    if not current_stage:
        return STAGE_COMPATIBILITY["default"]

    cs = current_stage.lower().strip()
    # Exact match
    if cs in STAGE_COMPATIBILITY:
        return STAGE_COMPATIBILITY[cs]

    # Partial match: look for key that contains cs or cs contains key
    for key in STAGE_COMPATIBILITY.keys():
        lk = key.lower()
        if cs in lk or lk in cs:
            return STAGE_COMPATIBILITY[key]

    return STAGE_COMPATIBILITY["default"]


def _contains_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()


def check_chunk_compatibility(chunk_text: str, current_stage: str) -> dict:
    """Check whether a chunk's text is compatible with the given stage.

    Decision logic:
    - If allowed list is empty -> compatible=True
    - If any blocked keyword found -> compatible=False
    - Elif any allowed keyword found -> compatible=True
    - Else -> compatible=True (neutral)
    """
    rules = get_compatible_stages(current_stage)
    allowed = rules.get("allowed", [])
    blocked = rules.get("blocked", [])

    # If allowed empty, treat as permissive
    if not allowed:
        return {
            "compatible": True,
            "reason": "Default stage — permissive",
            "matched_blocked": None,
            "matched_allowed": None,
            "current_stage": current_stage
        }

    text = chunk_text or ""

    # Check blocked keywords first
    for kw in blocked:
        if _contains_keyword(text, kw):
            return {
                "compatible": False,
                "reason": f"Contains blocked topic: {kw}",
                "matched_blocked": kw,
                "matched_allowed": None,
                "current_stage": current_stage
            }

    # Then check allowed keywords
    for kw in allowed:
        if _contains_keyword(text, kw):
            return {
                "compatible": True,
                "reason": f"Matches allowed topic: {kw}",
                "matched_blocked": None,
                "matched_allowed": kw,
                "current_stage": current_stage
            }

    # Neutral fallback
    return {
        "compatible": True,
        "reason": "Neutral content — allowed by default",
        "matched_blocked": None,
        "matched_allowed": None,
        "current_stage": current_stage
    }
