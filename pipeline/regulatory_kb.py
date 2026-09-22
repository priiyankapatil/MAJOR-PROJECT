# ============================================================
# REGULATORY KNOWLEDGE BASE
# Sources:
#   - CIB&RC (Central Insecticides Board & Registration Committee)
#   - WHO Pesticide Hazard Classification
#   - EU Regulation 396/2005 (MRL database)
#   - Codex Alimentarius MRL Standards
#   - NPOP (National Programme for Organic Production) India
#   - PGS-India (Participatory Guarantee System)
# ============================================================

# ─── WHO HAZARD CLASSES ───
WHO_HAZARD = {
    "Ia":  {"label": "Extremely Hazardous",   "score_penalty": 0.90},
    "Ib":  {"label": "Highly Hazardous",       "score_penalty": 0.75},
    "II":  {"label": "Moderately Hazardous",   "score_penalty": 0.45},
    "III": {"label": "Slightly Hazardous",     "score_penalty": 0.20},
    "U":   {"label": "Unlikely Hazardous",     "score_penalty": 0.05},
    "FM":  {"label": "Fumigant",               "score_penalty": 0.60},
}

# ─── CIB&RC INDIA STATUS ───
INDIA_STATUS = {
    "BANNED":      {"label": "Banned in India",           "score_penalty": 1.00},
    "RESTRICTED":  {"label": "Restricted Use in India",   "score_penalty": 0.55},
    "APPROVED":    {"label": "Approved in India",         "score_penalty": 0.00},
    "UNDER_REVIEW":{"label": "Under Review by CIB&RC",   "score_penalty": 0.30},
}

# ─── EU EXPORT RISK ───
EU_EXPORT_RISK = {
    "BANNED_EU":    {"label": "Banned in EU",              "score_penalty": 0.95},
    "MRL_VERY_LOW": {"label": "EU MRL ≤ 0.01 mg/kg",      "score_penalty": 0.70},
    "MRL_LOW":      {"label": "EU MRL 0.01–0.1 mg/kg",    "score_penalty": 0.40},
    "MRL_MODERATE": {"label": "EU MRL 0.1–1.0 mg/kg",     "score_penalty": 0.15},
    "MRL_HIGH":     {"label": "EU MRL > 1.0 mg/kg",       "score_penalty": 0.05},
    "NOT_ASSESSED": {"label": "Not assessed by EU",        "score_penalty": 0.25},
}

# ─── ORGANIC CERTIFICATION ───
ORGANIC_STATUS = {
    "NPOP_APPROVED":   {"label": "Approved under NPOP",         "score_penalty": 0.00},
    "PGS_APPROVED":    {"label": "Approved under PGS-India",    "score_penalty": 0.00},
    "NOT_ORGANIC":     {"label": "Not permitted in organic",    "score_penalty": 0.50},
    "CONDITIONALLY":   {"label": "Conditionally permitted",     "score_penalty": 0.20},
}

# ─── MAIN PESTICIDE DATABASE ───
PESTICIDE_DB = {
    "dimethoate": {
        "common_names": ["dimethoate", "rogor", "tafgor", "dimecron"],
        "who_class": "II",
        "india_status": "RESTRICTED",
        "eu_export_risk": "MRL_VERY_LOW",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 0.01,
        "pre_harvest_interval_days": 14,
        "notes": "Restricted for use on vegetables in India. EU MRL extremely low — high export rejection risk for produce like grapes and mangoes."
    },
    "monocrotophos": {
        "common_names": ["monocrotophos", "nuvacron", "azodrin"],
        "who_class": "Ib",
        "india_status": "BANNED",
        "eu_export_risk": "BANNED_EU",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": None,
        "pre_harvest_interval_days": None,
        "notes": "Completely banned in India since 2005 by CIB&RC. Also banned in EU. Highly toxic to birds and humans."
    },
    "endosulfan": {
        "common_names": ["endosulfan", "thiodan", "beosit"],
        "who_class": "II",
        "india_status": "BANNED",
        "eu_export_risk": "BANNED_EU",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": None,
        "pre_harvest_interval_days": None,
        "notes": "Banned in India by Supreme Court order 2011. Listed under Stockholm Convention as Persistent Organic Pollutant."
    },
    "chlorpyrifos": {
        "common_names": ["chlorpyrifos", "dursban", "lorsban", "radar"],
        "who_class": "II",
        "india_status": "UNDER_REVIEW",
        "eu_export_risk": "BANNED_EU",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 0.01,
        "pre_harvest_interval_days": 15,
        "notes": "Under review by CIB&RC for potential ban. Banned in EU since 2020. High export risk for all produce."
    },
    "imidacloprid": {
        "common_names": ["imidacloprid", "confidor", "admire", "gaucho"],
        "who_class": "II",
        "india_status": "RESTRICTED",
        "eu_export_risk": "MRL_LOW",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 0.05,
        "pre_harvest_interval_days": 7,
        "notes": "Restricted in India — not permitted for use on flowering crops due to bee toxicity. EU MRL low — moderate export risk."
    },
    "mancozeb": {
        "common_names": ["mancozeb", "dithane m-45", "indofil m-45", "mancocide"],
        "who_class": "III",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_LOW",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 2.0,
        "pre_harvest_interval_days": 7,
        "notes": "Approved in India. EU MRL moderate. Codex MRL relatively high. Low export risk for most crops."
    },
    "streptocycline": {
        "common_names": ["streptocycline", "streptomycin", "kcycline"],
        "who_class": "III",
        "india_status": "APPROVED",
        "eu_export_risk": "BANNED_EU",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": None,
        "pre_harvest_interval_days": 10,
        "notes": "Approved in India for bacterial diseases. Banned in EU due to antibiotic resistance concerns. High export risk."
    },
    "copper oxychloride": {
        "common_names": ["copper oxychloride", "blitox", "fytolan", "copper"],
        "who_class": "III",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_HIGH",
        "organic_status": "NPOP_APPROVED",
        "codex_mrl_mg_kg": 50.0,
        "pre_harvest_interval_days": 3,
        "notes": "Approved in India and EU. Permitted under NPOP organic certification. Low export risk. Preferred in organic farming."
    },
    "tricyclazole": {
        "common_names": ["tricyclazole", "beam", "blascide"],
        "who_class": "III",
        "india_status": "APPROVED",
        "eu_export_risk": "BANNED_EU",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": None,
        "pre_harvest_interval_days": 14,
        "notes": "Approved in India for rice blast. Banned in EU. High export risk for basmati rice — major trade issue since 2016."
    },
    "carbaryl": {
        "common_names": ["carbaryl", "sevin", "ravyon"],
        "who_class": "II",
        "india_status": "RESTRICTED",
        "eu_export_risk": "BANNED_EU",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": None,
        "pre_harvest_interval_days": 7,
        "notes": "Restricted in India. Banned in EU. High export risk. Not recommended for export-oriented farming."
    },
    "propiconazole": {
        "common_names": ["propiconazole", "tilt", "bumper"],
        "who_class": "II",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_MODERATE",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 0.08,
        "pre_harvest_interval_days": 21,
        "notes": "Approved in India and EU for wheat rust. Moderate export risk. Long pre-harvest interval — must be timed carefully."
    },
    "spinosad": {
        "common_names": ["spinosad", "tracer", "success", "delegate"],
        "who_class": "U",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_HIGH",
        "organic_status": "NPOP_APPROVED",
        "codex_mrl_mg_kg": 1.0,
        "pre_harvest_interval_days": 3,
        "notes": "Approved in India and EU. Permitted under NPOP organic certification. Derived from soil bacteria — safe for beneficial insects."
    },
    "neem oil": {
        "common_names": ["neem oil", "neem", "azadirachtin", "neemicide", "econeem"],
        "who_class": "U",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_HIGH",
        "organic_status": "NPOP_APPROVED",
        "codex_mrl_mg_kg": None,
        "pre_harvest_interval_days": 0,
        "notes": "Fully approved in India and EU. Certified under NPOP and PGS-India. Zero pre-harvest interval. Preferred for organic and export farming."
    },
    "buprofezin": {
        "common_names": ["buprofezin", "applaud", "courier"],
        "who_class": "III",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_LOW",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 0.05,
        "pre_harvest_interval_days": 14,
        "notes": "Approved in India for BPH in rice. Low EU MRL — moderate export risk for rice."
    },
    "bordeaux mixture": {
        "common_names": ["bordeaux mixture", "bordeaux", "copper sulfate", "lime"],
        "who_class": "U",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_HIGH",
        "organic_status": "NPOP_APPROVED",
        "codex_mrl_mg_kg": 50.0,
        "pre_harvest_interval_days": 0,
        "notes": "Traditional fungicide. Approved globally. Permitted under organic certification. Safe for export crops."
    },
    "glyphosate": {
        "common_names": ["glyphosate", "roundup", "glycel", "spark"],
        "who_class": "II",
        "india_status": "APPROVED",
        "eu_export_risk": "MRL_VERY_LOW",
        "organic_status": "NOT_ORGANIC",
        "codex_mrl_mg_kg": 0.1,
        "pre_harvest_interval_days": 30,
        "notes": "Approved in India. Under heavy EU scrutiny — MRL very low. High export residue risk. Banned by several EU member states individually."
    },
}

# ─── ORGANIC-APPROVED ALTERNATIVES ───
# Maps non-organic pesticide to its best organic alternative
ORGANIC_ALTERNATIVES = {
    "dimethoate":       "neem oil (Azadirachtin 0.03%) or Spinosad",
    "monocrotophos":    "neem oil or PGPR-based biocontrol",
    "endosulfan":       "neem oil or Bacillus thuringiensis",
    "chlorpyrifos":     "spinosad or neem oil",
    "imidacloprid":     "neem oil or buprofezin (conventional alt.)",
    "carbaryl":         "spinosad or neem oil",
    "streptocycline":   "copper oxychloride or Bordeaux mixture",
    "tricyclazole":     "Trichoderma-based biocontrol",
    "glyphosate":       "manual weeding or flame weeding",
}