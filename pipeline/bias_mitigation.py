import json
import os
import re
from datetime import datetime
from rapidfuzz import fuzz

BIAS_LOG_PATH = "bias_mitigation_log.json"

# ─── ALPHA for combined score weighting ───
COMMERCIAL_WEIGHT   = 0.40
TEMPORAL_WEIGHT     = 0.35
DEMOGRAPHIC_WEIGHT  = 0.25

# ─── PENALTY VALUES ───
PENALTY_HIGH   = 0.80
PENALTY_MEDIUM = 0.45
PENALTY_LOW    = 0.20
PENALTY_NONE   = 0.00

# ─── TRIGGER 1: Commercial Source Bias ───
COMMERCIAL_TRIGGERS = {
    # High penalty — direct product promotion
    "high": [
        "buy now", "purchase", "order today", "limited offer",
        "discount", "sale", "price", "cost per bottle",
        "our product", "our brand", "contact us to buy",
        "available at", "shop now", "add to cart",
        "sponsored", "advertisement", "promoted",
        "best seller", "top rated product", "customer reviews",
        "only rs", "only $", "per kg price", "market price"
    ],
    # Medium penalty — brand-specific but not outright selling
    "medium": [
        "brand name", "registered trademark", "patent pending",
        "proprietary formula", "exclusive", "our research shows",
        "according to our trials", "company tested",
        "manufactured by", "produced by", "distributed by"
    ],
    # Low penalty — general commercial context
    "low": [
        "commercial fertilizer", "commercial pesticide",
        "available in market", "sold in stores"
    ]
}

# ─── TRIGGER 2: Temporal/Outdated Bias ───
TEMPORAL_TRIGGERS = {
    "high": [
        "in the 1980s", "in the 1990s", "in the 1970s",
        "traditional method", "old practice",
        "previously recommended", "formerly used",
        "outdated", "deprecated", "no longer recommended",
        "historical data", "past studies show"
    ],
    "medium": [
        "in 2000", "in 2001", "in 2002", "in 2003",
        "in 2004", "in 2005", "in 2006", "in 2007",
        "in 2008", "in 2009", "in 2010",
        "early studies", "initial research",
        "first generation", "early trials"
    ],
    "low": [
        "in 2011", "in 2012", "in 2013", "in 2014",
        "in 2015", "older variety", "conventional method"
    ]
}

# ─── TRIGGER 3: Demographic/Farm-Scale Bias ───
DEMOGRAPHIC_TRIGGERS = {
    "high": [
        "large farm only", "commercial farm", "industrial farm",
        "hectares of land", "1000 acres", "500 acres",
        "mechanized farming", "only for large scale",
        "not suitable for small farmers",
        "requires heavy machinery",
        "bulk purchase only", "minimum order",
        "plantation scale", "estate farming"
    ],
    "medium": [
        "large scale", "commercial scale", "bulk application",
        "tractor mounted", "combine harvester",
        "precision agriculture equipment",
        "gps guided", "drone spraying at scale",
        "automated irrigation system",
        "requires investment of lakhs"
    ],
    "low": [
        "recommended for medium farms",
        "suitable for larger holdings",
        "best for organized farming"
    ]
}

# ─── POSITIVE SIGNALS (boost score) ───
POSITIVE_SIGNALS = [
    "small farmer", "marginal farmer", "smallholder",
    "subsistence farming", "family farm",
    "low cost", "affordable", "cost effective",
    "manual application", "hand application",
    "suitable for all farm sizes",
    "government recommended", "icar", "tnau", "kau",
    "university research", "peer reviewed",
    "organic", "traditional knowledge", "farmer friendly",
    "backyard", "kitchen garden", "home garden"
]


# ─────────────────────────────────────────────────────────────
# BIAS CHECK FUNCTIONS
# ─────────────────────────────────────────────────────────────

def check_commercial_bias(text: str) -> dict:
    """Check text for commercial source bias triggers."""
    text_lower = text.lower()

    hits_high   = [t for t in COMMERCIAL_TRIGGERS["high"]
                   if t in text_lower]
    hits_medium = [t for t in COMMERCIAL_TRIGGERS["medium"]
                   if t in text_lower]
    hits_low    = [t for t in COMMERCIAL_TRIGGERS["low"]
                   if t in text_lower]

    # Calculate penalty
    if hits_high:
        penalty  = PENALTY_HIGH
        severity = "HIGH"
    elif hits_medium:
        penalty  = PENALTY_MEDIUM
        severity = "MEDIUM"
    elif hits_low:
        penalty  = PENALTY_LOW
        severity = "LOW"
    else:
        penalty  = PENALTY_NONE
        severity = "NONE"

    return {
        "bias_type":     "COMMERCIAL",
        "severity":      severity,
        "penalty":       penalty,
        "triggers_hit":  hits_high + hits_medium + hits_low,
        "trigger_count": len(hits_high + hits_medium + hits_low)
    }


def check_temporal_bias(text: str) -> dict:
    """Check text for outdated/temporal bias triggers."""
    text_lower = text.lower()

    hits_high   = [t for t in TEMPORAL_TRIGGERS["high"]
                   if t in text_lower]
    hits_medium = [t for t in TEMPORAL_TRIGGERS["medium"]
                   if t in text_lower]
    hits_low    = [t for t in TEMPORAL_TRIGGERS["low"]
                   if t in text_lower]

    if hits_high:
        penalty  = PENALTY_HIGH
        severity = "HIGH"
    elif hits_medium:
        penalty  = PENALTY_MEDIUM
        severity = "MEDIUM"
    elif hits_low:
        penalty  = PENALTY_LOW
        severity = "LOW"
    else:
        penalty  = PENALTY_NONE
        severity = "NONE"

    return {
        "bias_type":     "TEMPORAL",
        "severity":      severity,
        "penalty":       penalty,
        "triggers_hit":  hits_high + hits_medium + hits_low,
        "trigger_count": len(hits_high + hits_medium + hits_low)
    }


def check_demographic_bias(text: str) -> dict:
    """Check text for large-farm demographic bias triggers."""
    text_lower = text.lower()

    hits_high   = [t for t in DEMOGRAPHIC_TRIGGERS["high"]
                   if t in text_lower]
    hits_medium = [t for t in DEMOGRAPHIC_TRIGGERS["medium"]
                   if t in text_lower]
    hits_low    = [t for t in DEMOGRAPHIC_TRIGGERS["low"]
                   if t in text_lower]

    if hits_high:
        penalty  = PENALTY_HIGH
        severity = "HIGH"
    elif hits_medium:
        penalty  = PENALTY_MEDIUM
        severity = "MEDIUM"
    elif hits_low:
        penalty  = PENALTY_LOW
        severity = "LOW"
    else:
        penalty  = PENALTY_NONE
        severity = "NONE"

    return {
        "bias_type":     "DEMOGRAPHIC",
        "severity":      severity,
        "penalty":       penalty,
        "triggers_hit":  hits_high + hits_medium + hits_low,
        "trigger_count": len(hits_high + hits_medium + hits_low)
    }


def check_positive_signals(text: str) -> tuple:
    """
    Checks for positive signals that BOOST the chunk score.
    Returns a boost value between 0.0 and 0.15.
    """
    text_lower = text.lower()
    hits = [s for s in POSITIVE_SIGNALS if s in text_lower]

    # Each positive signal gives a small boost, max 0.15
    boost = min(len(hits) * 0.03, 0.15)
    return boost, hits


# ─────────────────────────────────────────────────────────────
# CHUNK-LEVEL SCORING
# ─────────────────────────────────────────────────────────────

def score_chunk_bias(chunk: dict, chunk_index: int) -> dict:
    """
    Master scoring function for a single chunk.

    Final bias score:
    bias_penalty = (commercial_weight × commercial_penalty)
                 + (temporal_weight   × temporal_penalty)
                 + (demographic_weight × demographic_penalty)

    fairness_score = 1.0 - bias_penalty + positive_boost
    fairness_score = clamp(fairness_score, 0.0, 1.0)
    """
    text   = chunk.get("text",   str(chunk)) \
             if isinstance(chunk, dict) else str(chunk)
    source = chunk.get("source", f"Chunk {chunk_index}") \
             if isinstance(chunk, dict) else f"Chunk {chunk_index}"

    # Run all three bias checks
    commercial  = check_commercial_bias(text)
    temporal    = check_temporal_bias(text)
    demographic = check_demographic_bias(text)

    # Positive signal boost
    boost, positive_hits = check_positive_signals(text)

    # Weighted bias penalty
    bias_penalty = (
        COMMERCIAL_WEIGHT  * commercial["penalty"]  +
        TEMPORAL_WEIGHT    * temporal["penalty"]    +
        DEMOGRAPHIC_WEIGHT * demographic["penalty"]
    )

    # Fairness score
    fairness_score = max(0.0, min(1.0,
        1.0 - bias_penalty + boost
    ))
    fairness_score = round(fairness_score, 4)

    # Verdict
    if fairness_score >= 0.85:
        verdict = "UNBIASED [OK]"
        action  = "KEEP"
    elif fairness_score >= 0.70:
        verdict = "MILD BIAS [!]"
        action  = "KEEP WITH FLAG"
    elif fairness_score >= 0.55:
        verdict = "MODERATE BIAS [!!]"
        action  = "DE-RANK"
    else:
        verdict = "HIGH BIAS [XX]"
        action  = "FILTER OUT"

    return {
        "chunk_index":    chunk_index,
        "source":         source,
        "text_snippet":   text[:120] + "..."
                          if len(text) > 120 else text,
        "commercial":     commercial,
        "temporal":       temporal,
        "demographic":    demographic,
        "positive_hits":  positive_hits,
        "positive_boost": round(boost, 4),
        "bias_penalty":   round(bias_penalty, 4),
        "fairness_score": fairness_score,
        "verdict":        verdict,
        "action":         action
    }


# ─────────────────────────────────────────────────────────────
# DEBIASRAG FILTERING + RE-RANKING
# ─────────────────────────────────────────────────────────────

def debias_and_rerank(chunks: list) -> dict:
    """
    DebiasRAG-style filtering + fairness-aware re-ranking.

    1. Score all chunks for bias
    2. Filter out HIGH BIAS chunks
    3. Re-rank remaining by fairness score
    4. Return clean, re-ranked chunk list
    """
    if not chunks:
        return {
            "original":     [],
            "scored":       [],
            "kept":         [],
            "filtered":     [],
            "reranked":     [],
            "filter_count": 0,
            "kept_count":   0,
            "total":        0
        }

    # Score all chunks
    scored = [
        score_chunk_bias(chunk, i + 1)
        for i, chunk in enumerate(chunks)
    ]

    # Separate kept vs filtered
    kept     = [s for s in scored if s["action"] != "FILTER OUT"]
    filtered = [s for s in scored if s["action"] == "FILTER OUT"]

    # Re-rank kept chunks by fairness score descending
    reranked = sorted(kept,
                      key=lambda x: x["fairness_score"],
                      reverse=True)

    return {
        "original":     chunks,
        "scored":       scored,
        "kept":         kept,
        "filtered":     filtered,
        "reranked":     reranked,
        "filter_count": len(filtered),
        "kept_count":   len(kept),
        "total":        len(chunks)
    }


# ─────────────────────────────────────────────────────────────
# REPORTING + LOGGING
# ─────────────────────────────────────────────────────────────

def print_bias_report(result: dict):
    print("\n" + "="*65)
    print("||           DEBIASRAG FAIRNESS REPORT                    ||")
    print("="*65)
    print(f"  Total chunks input : {result['total']}")
    print(f"  [OK] Kept          : {result['kept_count']}")
    print(f"  [XX] Filtered out  : {result['filter_count']}")
    print("="*65)

    print("\n  [CHART] Per-chunk bias analysis:\n")
    for s in result["scored"]:
        print(f"  [{s['chunk_index']}] {s['verdict']}")
        print(f"      Source         : {s['source']}")
        print(f"      Fairness score : {s['fairness_score']:.4f}  "
              f"| Action: {s['action']}")
        print(f"      Commercial     : {s['commercial']['severity']}  "
              f"penalty={s['commercial']['penalty']:.2f}"
              + (f"  triggers={s['commercial']['triggers_hit']}"
                 if s['commercial']['triggers_hit'] else ""))
        print(f"      Temporal       : {s['temporal']['severity']}  "
              f"penalty={s['temporal']['penalty']:.2f}"
              + (f"  triggers={s['temporal']['triggers_hit']}"
                 if s['temporal']['triggers_hit'] else ""))
        print(f"      Demographic    : {s['demographic']['severity']}  "
              f"penalty={s['demographic']['penalty']:.2f}"
              + (f"  triggers={s['demographic']['triggers_hit']}"
                 if s['demographic']['triggers_hit'] else ""))
        if s["positive_hits"]:
            print(f"      [+] Positive   : boost=+{s['positive_boost']:.2f}  "
                  f"signals={s['positive_hits']}")
        print()

    if result["filtered"]:
        print("  [XX] FILTERED OUT (bias too high):\n")
        for s in result["filtered"]:
            print(f"  -> {s['source']} "
                  f"(fairness={s['fairness_score']:.4f})")
        print()

    print("  [OK] RE-RANKED ORDER (cleanest first):\n")
    for i, s in enumerate(result["reranked"]):
        print(f"  {i+1}. [{s['verdict']}] "
              f"{s['source']}  "
              f"score={s['fairness_score']:.4f}")

    print("\n" + "="*65 + "\n")


def save_bias_log(result: dict, query: str):
    log = []
    if os.path.exists(BIAS_LOG_PATH):
        try:
            with open(BIAS_LOG_PATH, "r") as f:
                log = json.load(f)
        except:
            log = []

    log.append({
        "timestamp":      datetime.now().isoformat(),
        "query":          query,
        "total":          result["total"],
        "kept":           result["kept_count"],
        "filtered":       result["filter_count"],
        "reranked_order": [
            s["source"] for s in result["reranked"]
        ]
    })

    with open(BIAS_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

    print(f"  [SAVED] Bias log saved -> {BIAS_LOG_PATH}")


# ─────────────────────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────────────────────

def run_bias_mitigation(query: str, chunks: list) -> dict:
    """
    Standalone bias mitigation component.

    Call with:
      - query  : farmer's question
      - chunks : list of retrieved chunk dicts
                 (each with 'text' and 'source' keys)

    Returns debiased, re-ranked chunk list.
    """
    print("\n[BIAS] DEBIASRAG BIAS MITIGATION")
    print("-" * 45)
    print(f"  Query  : {query[:70]}{'...' if len(query) > 70 else ''}")
    print(f"  Chunks : {len(chunks)} input")

    # Run DebiasRAG
    result = debias_and_rerank(chunks)

    # Print report
    print_bias_report(result)

    # Save log
    save_bias_log(result, query)

    return result


# ─────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── TEST 1: Mix of clean + biased chunks ──
    test_query = (
        "What fertilizer should I apply for tomato "
        "during flowering stage?"
    )

    test_chunks = [
        {
            "source": "TNAU Agriculture PDF.pdf",
            "text":   "During flowering, apply potassium at 50 kg/ha "
                      "to improve fruit set in tomato. ICAR recommends "
                      "phosphorus for root development. Suitable for "
                      "small farmer and marginal farmer use."
        },
        {
            "source": "AgroChemCorp Advertisement",
            "text":   "Buy now our SuperGrow Pro fertilizer! "
                      "Only Rs 299 per bottle. Best seller in market. "
                      "Order today and get 20% discount. "
                      "Our research shows 3x yield improvement. "
                      "Available at all agro stores. Contact us to buy."
        },
        {
            "source": "1985 Farming Manual",
            "text":   "In the 1980s, traditional method was to apply "
                      "farmyard manure at 10 tonnes per hectare. "
                      "Historical data shows good results. "
                      "Previously recommended DDT was used for pests."
        },
        {
            "source": "Large Farm Equipment Guide",
            "text":   "For plantation scale tomato, use GPS guided "
                      "tractor mounted sprayers. Requires investment of "
                      "lakhs for automated irrigation system. "
                      "Only suitable for large scale commercial farm "
                      "with minimum 500 acres."
        },
        {
            "source": "UAS Bangalore.pdf",
            "text":   "Boron deficiency in tomato causes flower drop. "
                      "Apply 0.2% borax spray at flower initiation. "
                      "University research confirms effectiveness for "
                      "smallholder and family farm contexts."
        },
    ]

    print("="*65)
    print("TEST 1: Mixed chunks — expect commercial/old/large-farm filtered")
    print("="*65)

    result = run_bias_mitigation(test_query, test_chunks)

    # Summary
    print("="*65)
    print("BIAS MITIGATION SUMMARY")
    print("="*65)
    print(f"  Input chunks    : {result['total']}")
    print(f"  After filtering : {result['kept_count']}")
    print(f"  Filtered out    : {result['filter_count']}")
    print(f"\n  Re-ranked order (cleanest -> most biased):")
    for i, s in enumerate(result["reranked"]):
        print(f"  {i+1}. {s['source']:35s} "
              f"fairness={s['fairness_score']:.4f}  "
              f"{s['verdict']}")
    print("="*65)
