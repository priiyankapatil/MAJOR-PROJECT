import config
from dialectal_alignment import align_query
from datetime import datetime
import json
import os

BRIDGE_LOG_PATH = "semantic_bridge_log.json"

def apply_semantic_bridge(query: str) -> dict:
    """
    Master controller for the Rural-to-Scientific Semantic Bridge.
    Call this as Step 0 before all other pipeline steps.
    """
    print("\n🌐 Step 0: Rural-to-Scientific Semantic Bridge...")

    if getattr(config, "ENABLE_SB2_GUARDRAILS", False):
        from components.context.guardrails import apply_semantic_bridge_sb2
        result = apply_semantic_bridge_sb2(query)
    else:
        result = align_query(query)

    if not result.get("bridged"):
        print("   ℹ️  No folk/dialect terms detected — query passed through unchanged")
        return result

    # Print detailed mapping report
    print(f"   Original query  : {result['original']}")
    print(f"   Folk terms found: {result.get('term_count', len(result.get('terms_found', [])))}")
    print()
    print("   Mappings applied:")
    for term in result["terms_found"]:
        match_label = f"[{term['match_type'].upper()} {term.get('match_score', 100):.0f}%]"
        print(f"   {match_label} '{term['original_term']}'")
        print(f"      → {term['english']} ({term['scientific']})")
        print(f"      → Category: {term['category']}")

    print()
    print(f"   Enriched query  : {result['enriched']}")
    print(f"   Alignment conf  : {result['confidence']:.4f}")

    # Log to file
    _log_bridge_result(result)

    return result


def _log_bridge_result(result: dict):
    """Append bridge result to log file."""
    log = []
    if os.path.exists(BRIDGE_LOG_PATH):
        try:
            with open(BRIDGE_LOG_PATH, "r", encoding="utf-8") as f:
                log = json.load(f)
        except Exception:
            log = []

    log.append({
        "timestamp": datetime.now().isoformat(),
        "original_query": result["original"],
        "enriched_query": result["enriched"],
        "terms_found": len(result["terms_found"]),
        "confidence": result.get("confidence", 0.0),
        "bridged": result.get("bridged", False)
    })

    try:
        with open(BRIDGE_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
    except Exception:
        pass


def get_bridge_statistics() -> dict:
    """Print and return bridge usage statistics."""
    if not os.path.exists(BRIDGE_LOG_PATH):
        print("No bridge log found.")
        return {}

    with open(BRIDGE_LOG_PATH, "r", encoding="utf-8") as f:
        log = json.load(f)

    total = len(log)
    bridged = sum(1 for e in log if e.get("bridged"))
    avg_conf = sum(e.get("confidence", 0.0) for e in log) / total if total > 0 else 0

    print(f"""
╔══════════════════════════════════════╗
║   SEMANTIC BRIDGE STATISTICS         ║
╠══════════════════════════════════════╣
║ Total queries processed : {total}
║ Folk terms detected     : {bridged}
║ Pass-through (English)  : {total - bridged}
║ Avg alignment confidence: {avg_conf:.4f}
╚══════════════════════════════════════╝
    """)
    return {"total": total, "bridged": bridged, "avg_confidence": avg_conf}
