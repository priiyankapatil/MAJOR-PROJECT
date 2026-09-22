from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from folk_taxonomy import FOLK_TAXONOMY, ALIAS_INDEX, get_scientific, fuzzy_match, STOPWORDS
import re


# Folk embedder — handles informal, mixed Hindi/English text
FOLK_EMBEDDER = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Scientific embedder — your existing model
SCIENTIFIC_EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def tokenize_query(query: str) -> list:
    q = (query or "").lower()
    # split on whitespace and punctuation
    tokens = [t for t in re.split(r"[\s,;:.!?()\[\]\\/]+", q) if t]

    # build bigrams
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)] if len(tokens) > 1 else []
    return tokens + bigrams


def detect_folk_terms(query: str) -> list:
    tokens = tokenize_query(query)
    found = []
    seen_scientific = set()

    for tok in tokens:
        token_clean = tok.lower().strip()

        # Skip stopwords and short tokens
        if token_clean in STOPWORDS or len(token_clean) <= 3:
            continue

        # exact lookup
        exact = get_scientific(token_clean)
        if exact:
            sci = exact.get('scientific')
            if sci not in seen_scientific:
                seen_scientific.add(sci)
                found.append({
                    "original_term": tok,
                    "scientific": exact.get('scientific'),
                    "english": exact.get('english'),
                    "category": exact.get('category'),
                    "match_type": "exact",
                    "match_score": 100
                })
            continue

        # fuzzy lookup (no threshold param here; folk_taxonomy's default applies)
        fuzzed = fuzzy_match(token_clean)
        if fuzzed:
            sci = fuzzed.get('scientific')
            if sci not in seen_scientific:
                seen_scientific.add(sci)
                found.append({
                    "original_term": tok,
                    "scientific": fuzzed.get('scientific'),
                    "english": fuzzed.get('english'),
                    "category": fuzzed.get('category'),
                    "match_type": "fuzzy",
                    "match_score": fuzzed.get('match_score', 0)
                })

    return found


def enrich_query(query: str, detected_terms: list) -> str:
    enriched = query
    # sort by length to replace longer terms first
    detected_terms_sorted = sorted(detected_terms, key=lambda x: len(x['original_term']), reverse=True)
    for t in detected_terms_sorted:
        folk = t['original_term']
        eng = t.get('english', '')
        sci = t.get('scientific', '')
        replacement = f"{folk} ({eng}, {sci})"
        # replace all case-insensitive occurrences using word boundaries to avoid substring corruption
        enriched = re.sub(rf"\b{re.escape(folk)}\b", replacement, enriched, flags=re.IGNORECASE)
    return enriched


def compute_alignment_confidence(query: str, enriched_query: str, detected_terms: list) -> float:
    if not detected_terms:
        return 0.0

    try:
        # Embed original with folk embedder
        v1 = FOLK_EMBEDDER.encode([query])[0]
        # Embed enriched with scientific embedder
        v2 = SCIENTIFIC_EMBEDDER.encode([enriched_query])[0]

        # Ensure vectors are 2D
        v1 = np.array(v1).reshape(1, -1)
        v2 = np.array(v2).reshape(1, -1)

        # If dims differ, fallback to using folk embedder for both
        if v1.shape[1] != v2.shape[1]:
            v2 = FOLK_EMBEDDER.encode([enriched_query])[0].reshape(1, -1)

        sim = float(cosine_similarity(v1, v2)[0, 0])
        sim = max(0.0, min(1.0, sim))
    except Exception:
        sim = 0.0

    # average match score normalized
    avg_score = sum(t.get('match_score', 0) for t in detected_terms) / (len(detected_terms) * 100.0)
    avg_score = max(0.0, min(1.0, avg_score))

    confidence = sim * avg_score
    return float(max(0.0, min(1.0, confidence)))


def align_query(query: str) -> dict:
    detected = detect_folk_terms(query)
    if not detected:
        return {"original": query, "enriched": query, "terms_found": [], "confidence": 1.0, "bridged": False}

    enriched = enrich_query(query, detected)
    confidence = compute_alignment_confidence(query, enriched, detected)

    return {
        "original": query,
        "enriched": enriched,
        "terms_found": detected,
        "confidence": confidence,
        "bridged": True,
        "term_count": len(detected)
    }
