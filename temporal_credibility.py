import math
from datetime import datetime
import re

SOURCE_REGISTRY = {
    "TNAU": {"base_score": 0.95, "pub_year": 2022},
    "KAU": {"base_score": 0.90, "pub_year": 2019},
    "UAS Bangalore": {"base_score": 0.88, "pub_year": 2021},
    "ANGRAU": {"base_score": 0.87, "pub_year": 2020},
    "Crop Protection": {"base_score": 0.85, "pub_year": 2018},
    "ICAR": {"base_score": 0.92, "pub_year": 2023},
}

DECAY_CONSTANTS = {
    "FACTUAL": 0.03,
    "RECOMMENDATION": 0.10,
    "DIAGNOSTIC": 0.07,
    "PROCEDURAL": 0.05,
}


def _match_source(filename):
    if not filename:
        return None
    fn = filename.lower()
    for key in SOURCE_REGISTRY:
        if key.lower() in fn:
            return key
    return None


def compute_temporal_score(source_name, query_type, current_year=None):
    if current_year is None:
        current_year = datetime.utcnow().year

    src = SOURCE_REGISTRY.get(source_name)
    if not src:
        # Unknown source: return a conservative low score
        return {
            "source": source_name,
            "base_score": 0.5,
            "pub_year": current_year,
            "age_years": 0,
            "decay_lambda": DECAY_CONSTANTS.get(query_type, 0.05),
            "final_score": 0.5,
            "freshness_label": "OUTDATED",
        }

    base = float(src.get("base_score", 0.5))
    pub = int(src.get("pub_year", current_year))
    delta_T = max(0, int(current_year) - pub)
    lam = float(DECAY_CONSTANTS.get(query_type, 0.05))
    final = base * math.exp(-lam * delta_T)

    # Freshness label
    if final >= 0.85:
        label = "FRESH"
    elif final >= 0.70:
        label = "ACCEPTABLE"
    elif final >= 0.55:
        label = "STALE"
    else:
        label = "OUTDATED"

    return {
        "source": source_name,
        "base_score": round(base, 4),
        "pub_year": pub,
        "age_years": delta_T,
        "decay_lambda": lam,
        "final_score": round(final, 4),
        "freshness_label": label,
    }


def score_all_chunks(retrieved_chunks, query_type, current_year=None):
    """Score and annotate a list of chunk dicts.

    Each chunk is expected to have either `source` or `source_file`.
    Returns chunks sorted by `final_score` descending and with
    `temporal_score`, `freshness_label`, `age_years` attached.
    """
    if current_year is None:
        current_year = datetime.utcnow().year

    scored = []
    for c in retrieved_chunks:
        fname = c.get("source") or c.get("source_file") or ""
        matched = _match_source(fname)
        if matched:
            sc = compute_temporal_score(matched, query_type, current_year)
        else:
            # No match: treat as unknown/low trust but recent
            sc = compute_temporal_score(None, query_type, current_year)

        # Attach
        c = dict(c)  # copy
        c["temporal_score"] = sc["final_score"]
        c["freshness_label"] = sc["freshness_label"]
        c["age_years"] = sc["age_years"]
        c["temporal_source_match"] = sc["source"]
        scored.append((sc["final_score"], c))

    # Sort by final_score descending
    scored_sorted = [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]
    return scored_sorted


def filter_stale_chunks(scored_chunks, min_score=0.55):
    """Remove chunks below min_score and print warnings for each."""
    kept = []
    for c in scored_chunks:
        score = c.get("temporal_score") if c.get("temporal_score") is not None else c.get("final_score", 0)
        src = c.get("source") or c.get("source_file") or c.get("temporal_source_match")
        if score < min_score:
            print(f"   ⚠️  Filtering out stale chunk: {src} → score: {score:.4f}")
        else:
            kept.append(c)
    return kept


if __name__ == "__main__":
    # Quick manual test
    fake = [
        {"source_file": "The TNAU Agriculture PDF.pdf", "text": "..."},
        {"source_file": "KAU (Kerala Agricultural University).pdf", "text": "..."},
        {"source_file": "Crop Protection.pdf", "text": "..."},
    ]
    scored = score_all_chunks(fake, query_type="RECOMMENDATION", current_year=2025)
    print(f"{'Source':<40} {'Score':<8} {'Label':<10} {'Age':<4}")
    print(f"{'-'*70}")
    for c in scored:
        src = c.get('source_file') or c.get('source')
        print(f"{src:<40} {c['temporal_score']:<8.4f} {c['freshness_label']:<10} {c['age_years']:<4}")
