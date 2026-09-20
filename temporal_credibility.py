import math
from datetime import datetime
import re

# ── Dynamic sync from config.SOURCE_METADATA ──────────────────────────────────
# Pulls publication years and authority scores for all 16+ crop PDFs registered
# in config.py, so they don't silently fall back to the 0.50 OUTDATED score.
try:
    from config import SOURCE_METADATA
except ImportError:
    SOURCE_METADATA = {}

# ── Institutional base registry ────────────────────────────────────────────────
# Keys are lowercase for consistent lookup in _match_source().
# base_score reflects institutional authority; pub_year drives temporal decay.
SOURCE_REGISTRY = {
    "tnau":           {"base_score": 0.95, "pub_year": 2022},
    "kau":            {"base_score": 0.90, "pub_year": 2019},
    "uas bangalore":  {"base_score": 0.88, "pub_year": 2021},
    "angrau":         {"base_score": 0.87, "pub_year": 2020},
    "crop protection":{"base_score": 0.85, "pub_year": 2018},
    "icar":           {"base_score": 0.92, "pub_year": 2023},
}

# ── Synchronize with config.SOURCE_METADATA ───────────────────────────────────
# For every PDF registered in config.py, derive a normalized key (strip ".pdf",
# lowercase) and add it to SOURCE_REGISTRY if it isn't already covered.
# authority_score from config maps to base_score; year maps to pub_year.
for _filename, _meta in SOURCE_METADATA.items():
    _key = _filename.lower().replace(".pdf", "").strip()
    if _key not in SOURCE_REGISTRY:
        SOURCE_REGISTRY[_key] = {
            "base_score": float(_meta.get("authority_score", _meta.get("source_weight", 0.95))),
            "pub_year":   int(_meta.get("year", _meta.get("pub_year", 2021))),
        }


DECAY_CONSTANTS = {
    "FACTUAL":        0.03,
    "RECOMMENDATION": 0.10,
    "DIAGNOSTIC":     0.07,
    "PROCEDURAL":     0.05,
}


def _match_source(source_str: str) -> str | None:
    """Return the SOURCE_REGISTRY key that best matches *source_str*, or None.

    Lookup order:
    1. Exact match after stripping ".pdf" and lowercasing.
    2. Substring match — registry key contained in source string, or vice-versa.
    """
    if not source_str:
        return None

    # Normalise: strip extension and whitespace, lowercase
    src_lower = str(source_str).lower().replace(".pdf", "").strip()

    # 1. Exact normalised match
    if src_lower in SOURCE_REGISTRY:
        return src_lower

    # 2. Substring match (handles long filenames like "The TNAU Agriculture PDF")
    for key in SOURCE_REGISTRY:
        if key in src_lower or src_lower in key:
            return key

    return None


def compute_temporal_score(source_name, query_type, current_year=None):
    if current_year is None:
        current_year = datetime.utcnow().year

    src = SOURCE_REGISTRY.get(source_name)
    if not src:
        # Unknown source: return a conservative low score
        return {
            "source":          source_name,
            "base_score":      0.5,
            "pub_year":        current_year,
            "age_years":       0,
            "decay_lambda":    DECAY_CONSTANTS.get(query_type, 0.05),
            "final_score":     0.5,
            "freshness_label": "OUTDATED",
        }

    base    = float(src.get("base_score", 0.5))
    pub     = int(src.get("pub_year", current_year))
    delta_T = max(0, int(current_year) - pub)
    lam     = float(DECAY_CONSTANTS.get(query_type, 0.05))
    final   = base * math.exp(-lam * delta_T)

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
        "source":          source_name,
        "base_score":      round(base, 4),
        "pub_year":        pub,
        "age_years":       delta_T,
        "decay_lambda":    lam,
        "final_score":     round(final, 4),
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
        fname   = c.get("source") or c.get("source_file") or ""
        matched = _match_source(fname)
        if matched:
            sc = compute_temporal_score(matched, query_type, current_year)
        elif c.get("temporal_score") is not None and c.get("freshness_label") is not None:
            # Preserve already-evaluated or mock credibility metadata
            sc = {
                "source":          fname,
                "base_score":      float(c.get("temporal_score")),
                "pub_year":        current_year,
                "age_years":       int(c.get("age_years", 0)),
                "decay_lambda":    DECAY_CONSTANTS.get(query_type, 0.05),
                "final_score":     float(c.get("temporal_score")),
                "freshness_label": c.get("freshness_label"),
            }
        else:
            # No match: treat as unknown/low trust but recent
            sc = compute_temporal_score(None, query_type, current_year)

        # Attach scores to chunk copy
        c = dict(c)
        c["temporal_score"]       = sc["final_score"]
        c["freshness_label"]      = sc["freshness_label"]
        c["age_years"]            = sc["age_years"]
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
        src   = c.get("source") or c.get("source_file") or c.get("temporal_source_match")
        if score < min_score:
            print(f"   ⚠️  Filtering out stale chunk: {src} → score: {score:.4f}")
        else:
            kept.append(c)
    return kept


def score_chunk_temporal(chunk: dict, query_type: str = "RECOMMENDATION", current_year: int = None) -> dict:
    """Convenience wrapper: score a single chunk dict and return it annotated.

    Equivalent to ``score_all_chunks([chunk], ...)[0]`` but returns the dict
    directly instead of a one-element list.

    Args:
        chunk:        A chunk dict with at least a ``source`` or ``source_file`` key.
        query_type:   One of FACTUAL / RECOMMENDATION / DIAGNOSTIC / PROCEDURAL.
        current_year: Override the reference year (defaults to UTC today).

    Returns:
        The same chunk dict with ``temporal_score``, ``freshness_label``,
        ``age_years``, and ``temporal_source_match`` attached.
    """
    results = score_all_chunks([chunk], query_type=query_type, current_year=current_year)
    return results[0]


if __name__ == "__main__":
    # Quick manual test — covers institutional + crop-specific PDF filenames
    fake = [
        {"source_file": "The TNAU Agriculture PDF.pdf",       "text": "..."},
        {"source_file": "KAU (Kerala Agricultural University).pdf", "text": "..."},
        {"source_file": "Crop Protection.pdf",                "text": "..."},
        {"source_file": "Banana.pdf",                         "text": "..."},
        {"source_file": "Cardamom.pdf",                       "text": "..."},
        {"source_file": "Blackgram.pdf",                      "text": "..."},
        {"source_file": "Apple.pdf",                          "text": "..."},
        {"source_file": "mango.pdf",                          "text": "..."},
        {"source_file": "unknown_document.pdf",               "text": "..."},
    ]
    scored = score_all_chunks(fake, query_type="RECOMMENDATION", current_year=2026)
    print(f"{'Source':<45} {'Score':<8} {'Label':<10} {'Age':<4} {'Matched Key'}")
    print("-" * 90)
    for c in scored:
        src = c.get("source_file") or c.get("source")
        print(
            f"{src:<45} {c['temporal_score']:<8.4f} "
            f"{c['freshness_label']:<10} {c['age_years']:<4} "
            f"{c.get('temporal_source_match', '—')}"
        )
