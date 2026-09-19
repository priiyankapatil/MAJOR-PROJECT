"""
components/context/guardrails.py

Authoritative Production Rural-to-Scientific Semantic Bridge Guardrails (SB2).
DO NOT IMPORT FROM evaluation/.

Implements five precision guardrails:
1. Synonym De-duplication & Tautology Elimination (SB1):
   Prevents repetitive injections like "coconut (Coconut, Cocos nucifera)" -> "coconut (Cocos nucifera)"
   and eliminates no-op expansions like "green manure (Green Manure, Green Manure)".
2. Specificity & Entity Anchor Guardrail (SB1):
   Suppresses broad collective category expansion (e.g. "organic manures" -> FYM)
   when the query is already anchored by specific crop entities and physical practices.
3. Vernacular Confidence Floor for English Terms (SB1):
   Restricts aggressive fuzzy-matching on standard English collective nouns.
4. Multi-Crop Comparative Balance Guardrail (SB2):
   When a query contains multiple co-occurring crop entities in an English list
   (e.g., "tomato, chilli, and brinjal"), suppresses asymmetric botanical Latin binomial
   injection to preserve balanced term representation and multi-crop tabular retrieval.
5. Plural & Compound Pest Hardening (SB2 Hardened):
   Recognizes irregular and standard English crop plurals (tomatoes, chillies, paddies).
   Masks compound pest phrases (tea mosquito bug, mango fruit fly, rice blast) so crop modifiers
   are not falsely counted as crop entities, while strictly protecting agronomic nouns.
"""

import re
from folk_taxonomy import FOLK_TAXONOMY, ALIAS_INDEX, get_scientific, fuzzy_match, STOPWORDS
from dialectal_alignment import tokenize_query, compute_alignment_confidence

# Generic collective agronomic categories that should not be narrowed
# to a single specific input if the query is already entity-anchored.
GENERIC_COLLECTIVES = {
    "organic manure", "organic manures",
    "fertilizer", "fertilizers", "chemical fertilizer", "chemical fertilizers",
    "biofertilizer", "biofertilizers",
    "compost", "composts",
    "manure", "manures",
    "pesticide", "pesticides",
    "insecticide", "insecticides",
    "weedicide", "weedicides",
    "fungicide", "fungicides",
    "practices", "practice",
    "method", "methods"
}

# Recognized crop anchors and comprehensive English singular/plural inflections
CROP_INFLECTIONS: dict[str, list[str]] = {
    # Irregular / spelling-changing plurals
    "tomato": ["tomato", "tomatoes", "tomatos"],
    "potato": ["potato", "potatoes", "potatos"],
    "mango": ["mango", "mangoes", "mangos"],
    "chilli": ["chilli", "chillies", "chillis", "chili", "chilies", "chilis"],
    "paddy": ["paddy", "paddies"],
    "citrus": ["citrus", "citruses"],

    # Standard -s / -es plurals
    "coconut": ["coconut", "coconuts"],
    "brinjal": ["brinjal", "brinjals", "eggplant", "eggplants"],
    "cowpea": ["cowpea", "cowpeas"],
    "chickpea": ["chickpea", "chickpeas"],
    "pigeon pea": ["pigeon pea", "pigeon peas"],
    "pearl millet": ["pearl millet", "pearl millets"],
    "finger millet": ["finger millet", "finger millets"],
    "lentil": ["lentil", "lentils"],
    "brassica": ["brassica", "brassicas"],
    "cashew": ["cashew", "cashews"],
    "black gram": ["black gram", "black grams"],
    "green gram": ["green gram", "green grams"],
    "groundnut": ["groundnut", "groundnuts", "peanut", "peanuts"],
    "soybean": ["soybean", "soybeans"],
    "cardamom": ["cardamom", "cardamoms"],
    "banana": ["banana", "bananas"],
    "onion": ["onion", "onions"],

    # Primarily uncountable / collective agronomic crops (with common plural variants)
    "wheat": ["wheat", "wheats"],
    "rice": ["rice", "rices"],
    "cotton": ["cotton", "cottons"],
    "sugarcane": ["sugarcane", "sugarcanes"],
    "coffee": ["coffee", "coffees"],
    "maize": ["maize", "corn", "corns"],
    "sorghum": ["sorghum", "sorghums"],
    "sesame": ["sesame", "sesames"],
    "mustard": ["mustard", "mustards"],
    "rapeseed": ["rapeseed", "rapeseeds"],
    "linseed": ["linseed", "linseeds"],
    "flax": ["flax", "flaxes"],
    "tea": ["tea", "teas"],
    "garlic": ["garlic", "garlics"]
}

# Inverted surface lookup: surface form -> canonical crop name
SURFACE_TO_CANONICAL: dict[str, str] = {}
for canonical, forms in CROP_INFLECTIONS.items():
    for form in forms:
        SURFACE_TO_CANONICAL[form.lower()] = canonical

# Complete set of all recognized crop surface forms
ALL_CROP_SURFACE_FORMS = set(SURFACE_TO_CANONICAL.keys())

# Preserved for backward compatibility with existing tests and imports
CROP_ANCHORS = set(ALL_CROP_SURFACE_FORMS)

# Sort multi-word forms first (by length descending) to avoid sub-phrase collisions
MULTI_WORD_CROP_FORMS = sorted(
    [f for f in ALL_CROP_SURFACE_FORMS if " " in f],
    key=lambda x: len(x),
    reverse=True
)
SINGLE_WORD_CROP_FORMS = sorted(
    [f for f in ALL_CROP_SURFACE_FORMS if " " not in f],
    key=lambda x: len(x),
    reverse=True
)

# Recognized multi-word pest/disease phrases that contain a crop name as an adjective/modifier
COMPOUND_PEST_EXCLUSIONS = {
    # Insects & Pests
    "tea mosquito bug",
    "coffee berry borer",
    "cotton pink bollworm", "pink bollworm",
    "cotton bollworm",
    "cotton aphid",
    "mango fruit fly",
    "sugarcane top borer", "sugarcane shoot borer", "sugarcane stem borer", "sugarcane pyrilla",
    "rice stem borer", "rice yellow stem borer", "rice gall midge", "rice leaf folder", "rice hispa",
    "citrus canker", "citrus greening", "citrus psylla",
    "banana bunchy top", "banana pseudostem borer",
    "coconut rhinoceros beetle", "coconut black headed caterpillar", "coconut eriophyid mite",
    "coconut red palm weevil",
    "tomato leaf curl", "tomato leaf miner", "tomato fruit borer",
    "potato tuber moth", "potato late blight",
    "cardamom thrips",
    "mustard aphid",
    # Pathogens with specific named compound terms
    "rice blast"
}

# Specific entomological and pathological head nouns that form compound pest names.
# Broad agronomic nouns ("disease", "management", "fertilizer", "pest", "spray", "practice")
# are strictly excluded to prevent suppressing genuine crops in multi-crop queries.
PEST_HEAD_NOUNS = [
    "fly", "borer", "bollworm", "aphid", "weevil", "beetle", "bug",
    "hopper", "caterpillar", "moth", "mite", "thrips", "grub",
    "whitefly", "sawfly", "midge", "leafminer", "pyrilla", "hispa",
    "blast", "canker"
]

# Anatomical plant parts and descriptors that intervene in compound pest names
PEST_MODIFIERS = [
    "fruit", "shoot", "stem", "top", "tuber", "berry", "pod", "leaf",
    "root", "bud", "flower", "seed", "head", "gall", "bark", "cane",
    "pink", "yellow", "black", "red", "green", "brown", "spotted",
    "white", "mosquito"
]

_crop_pattern_str = "|".join(re.escape(f) for f in SINGLE_WORD_CROP_FORMS)
_modifier_pattern_str = "|".join(re.escape(m) for m in PEST_MODIFIERS)
_head_pattern_str = "|".join(re.escape(h) for h in PEST_HEAD_NOUNS)

STRUCTURAL_COMPOUND_PEST_REGEX = re.compile(
    rf"\b({_crop_pattern_str})\s+(?:(?:{_modifier_pattern_str})\s+)?({_head_pattern_str})\b",
    re.IGNORECASE
)

COORDINATED_PEST_REGEX = re.compile(
    rf"\b({_crop_pattern_str})\s+shoot\s+and\s+fruit\s+borer\b",
    re.IGNORECASE
)


def get_detected_crops(query: str) -> list[str]:
    """
    Extract distinct recognized canonical crop entities mentioned in the query.
    - Supports singular and plural inflections (e.g., tomato/tomatoes, chilli/chillies, paddy/paddies).
    - Prevents false positives from crop words functioning as modifiers in compound
      pest/pathogen names (e.g., 'tea' in 'tea mosquito bug', 'mango' in 'mango fruit fly',
      'sugarcane' in 'sugarcane top borer', 'rice' in 'rice blast').
    - If a crop is mentioned both inside a compound pest name and independently (e.g.,
      'tea mosquito bug in tea and cashew'), the independent mention is preserved.
    - Protects genuine crop mentions followed by broad agronomic terms (e.g., 'rice disease
      management and tomato fertilizer').
    - Deduplicates repeated terms and canonicalizes synonyms to prevent false multi-crop triggers.
    """
    if not query or not query.strip():
        return []

    q_lower = query.lower()

    # Step 1: Mask compound pest occurrences in a working copy of the text.
    # Masking with spaces of identical length preserves character offsets and word boundaries.
    masked_chars = list(q_lower)

    # 1a. Catalog-based compound exclusions
    for compound in COMPOUND_PEST_EXCLUSIONS:
        pattern = r"\b" + re.escape(compound) + r"\b"
        for match in re.finditer(pattern, q_lower):
            start, end = match.span()
            for idx in range(start, end):
                masked_chars[idx] = " "

    # 1b. Structural compound pest matches (unlisted phrases like 'mango fruit fly', 'rice blast')
    q_working = "".join(masked_chars)
    for match in STRUCTURAL_COMPOUND_PEST_REGEX.finditer(q_working):
        start, end = match.span()
        for idx in range(start, end):
            masked_chars[idx] = " "

    for match in COORDINATED_PEST_REGEX.finditer(q_working):
        start, end = match.span()
        for idx in range(start, end):
            masked_chars[idx] = " "

    q_clean = "".join(masked_chars)

    # Step 2: Extract distinct canonical crops from unmasked text.
    detected_positions: list[tuple[int, str]] = []

    # 2a. Multi-word crop surface forms first (e.g., 'pigeon pea', 'pearl millet', 'black gram')
    q_scan_chars = list(q_clean)
    for form in MULTI_WORD_CROP_FORMS:
        pattern = r"\b" + re.escape(form) + r"\b"
        q_scan = "".join(q_scan_chars)
        for match in re.finditer(pattern, q_scan):
            start, end = match.span()
            canonical = SURFACE_TO_CANONICAL[form]
            detected_positions.append((start, canonical))
            # Mask out the multi-word span so constituent single words don't match
            for idx in range(start, end):
                q_scan_chars[idx] = " "

    # 2b. Single-word crop surface forms
    q_scan = "".join(q_scan_chars)
    for form in SINGLE_WORD_CROP_FORMS:
        pattern = r"\b" + re.escape(form) + r"\b"
        for match in re.finditer(pattern, q_scan):
            start, end = match.span()
            canonical = SURFACE_TO_CANONICAL[form]
            detected_positions.append((start, canonical))

    # Step 3: Sort by first appearance in query and deduplicate canonical crops
    detected_positions.sort(key=lambda x: x[0])
    seen: set[str] = set()
    ordered_crops: list[str] = []
    for _, canonical in detected_positions:
        if canonical not in seen:
            seen.add(canonical)
            ordered_crops.append(canonical)

    return ordered_crops


# Recognized specific physical/agronomic practice anchors
PRACTICE_ANCHORS = {
    "coir pith", "seed treatment", "happy seeder", "softwood graft",
    "softwood grafting", "drip irrigation", "foliar spray", "tine spacing",
    "pan evaporation", "spacing", "pruning", "nursery transplanting",
    "nursery", "transplanting", "seed rate", "earthing up", "solarization",
    "basin", "basin application", "mulching", "straw mulching"
}


def is_entity_anchored(query: str) -> tuple[bool, list[str]]:
    """
    Checks whether a query already contains explicit domain anchors
    (specific crop or specific physical agronomic practice).
    """
    q_lower = query.lower()
    anchors = []

    # Check crop anchors
    crops = get_detected_crops(query)
    for c in crops:
        anchors.append(f"crop:{c}")

    # Check practice anchors
    for practice in PRACTICE_ANCHORS:
        pattern = r"\b" + re.escape(practice) + r"\b"
        if re.search(pattern, q_lower):
            anchors.append(f"practice:{practice}")

    return len(anchors) > 0, anchors


def normalize_term(s: str) -> str:
    """Normalize a term for clean comparison."""
    if not s:
        return ""
    s_clean = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(s_clean.split())


def build_deduplicated_replacement(folk: str, eng: str, sci: str) -> str | None:
    """
    Constructs a clean, non-redundant replacement string.
    Returns None if the expansion adds zero new semantic information.
    """
    folk_norm = normalize_term(folk)
    eng_norm = normalize_term(eng)
    sci_norm = normalize_term(sci)

    # 1. Total tautology check: all identical -> no-op
    if folk_norm == eng_norm and (not sci_norm or eng_norm == sci_norm):
        return None

    # 2. Extract distinct parenthetical additions
    additions = []

    # Include English if distinct from folk
    if eng and folk_norm != eng_norm:
        if eng_norm != folk_norm and eng_norm not in folk_norm:
            additions.append(eng)

    # Include Scientific if distinct from both folk and English
    if sci:
        eng_collapsed = eng_norm.replace(" ", "")
        sci_collapsed = sci_norm.replace(" ", "")
        folk_collapsed = folk_norm.replace(" ", "")

        # If sci is just a repeat of eng or folk (e.g. Green Manure)
        if sci_collapsed != folk_collapsed and sci_collapsed != eng_collapsed:
            # Check if eng is a substring of sci (e.g., "Farmyard Manure" vs "Farm Yard Manure (FYM)")
            if additions and (eng_collapsed in sci_collapsed or sci_collapsed in eng_collapsed):
                additions = [sci]
            else:
                additions.append(sci)

    if not additions:
        return None

    parenthetical = ", ".join(additions)
    return f"{folk} ({parenthetical})"


def detect_folk_terms_guarded(query: str, enable_multi_crop_balance: bool = False) -> list[dict]:
    """
    Guarded term detection:
    - Identifies if query is already entity-anchored.
    - Suppresses generic collective expansion when anchored.
    - Applies strict vernacular confidence floor for English terms.
    - (SB2) If enable_multi_crop_balance is True: suppresses asymmetric botanical Latin
      binomial injection when multiple crops co-occur in an English list.
    """
    tokens = tokenize_query(query)
    anchored, anchor_list = is_entity_anchored(query)
    detected_crops = get_detected_crops(query)
    is_multi_crop = len(detected_crops) >= 2

    found = []
    seen_scientific = set()

    for tok in tokens:
        token_clean = tok.lower().strip()

        # Skip stopwords and short tokens
        if token_clean in STOPWORDS or len(token_clean) <= 3:
            continue

        # ── Specificity Guardrail (SB1) ──
        # If query is entity-anchored, do not expand broad generic collective terms
        if anchored and token_clean in GENERIC_COLLECTIVES:
            continue

        # ── Multi-Crop Balance Guardrail (SB2) ──
        # If the query lists multiple co-occurring crops in English, do not inject
        # botanical Latin binomials onto an individual crop (prevents asymmetric dilution)
        if enable_multi_crop_balance and is_multi_crop:
            if token_clean in CROP_ANCHORS:
                continue

        # 1. Exact lookup
        exact = get_scientific(token_clean)
        if exact:
            sci = exact.get("scientific")
            # If anchored and the matched canonical or alias is generic collective, suppress
            if anchored and (token_clean in GENERIC_COLLECTIVES or exact.get("folk_term", "").lower() in GENERIC_COLLECTIVES):
                continue

            # Multi-crop check on exact match category
            if enable_multi_crop_balance and is_multi_crop and exact.get("category") == "crop":
                # If the matched token is already an English common crop name, suppress
                if token_clean in CROP_ANCHORS:
                    continue

            if sci not in seen_scientific:
                seen_scientific.add(sci)
                found.append({
                    "original_term": tok,
                    "scientific": exact.get("scientific"),
                    "english": exact.get("english"),
                    "category": exact.get("category"),
                    "match_type": "exact",
                    "match_score": 100,
                    "matched_alias": token_clean
                })
            continue

        # 2. Fuzzy lookup with Guardrails
        # If anchored, never fuzzy match a generic collective term
        if anchored and any(gen in token_clean for gen in ["manure", "fertilizer", "pesticide"]):
            continue

        fuzzed = fuzzy_match(token_clean)
        if fuzzed:
            matched_alias = fuzzed.get("matched_alias", "").lower()
            canonical_folk = fuzzed.get("folk_term", "").lower()

            # Guardrail 3: Vernacular Confidence Floor
            # If the matched alias or canonical is a generic collective, suppress
            if anchored and (matched_alias in GENERIC_COLLECTIVES or canonical_folk in GENERIC_COLLECTIVES):
                continue

            # If the token is standard English and category is fertilizer/soil, require exact match
            if fuzzed.get("category") in {"fertilizer", "soil"} and fuzzed.get("match_score", 0) < 100:
                continue

            # Multi-crop check on fuzzy crop match
            if enable_multi_crop_balance and is_multi_crop and fuzzed.get("category") == "crop":
                if token_clean in CROP_ANCHORS or matched_alias in CROP_ANCHORS:
                    continue

            sci = fuzzed.get("scientific")
            if sci not in seen_scientific:
                seen_scientific.add(sci)
                found.append({
                    "original_term": tok,
                    "scientific": fuzzed.get("scientific"),
                    "english": fuzzed.get("english"),
                    "category": fuzzed.get("category"),
                    "match_type": "fuzzy",
                    "match_score": fuzzed.get("match_score", 0),
                    "matched_alias": matched_alias
                })

    return found


def enrich_query_guarded(query: str, detected_terms: list[dict]) -> str:
    """
    Applies Guardrail 1: Synonym De-duplication and clean replacement.
    """
    enriched = query
    detected_terms_sorted = sorted(detected_terms, key=lambda x: len(x["original_term"]), reverse=True)

    for t in detected_terms_sorted:
        folk = t["original_term"]
        eng = t.get("english", "")
        sci = t.get("scientific", "")

        replacement = build_deduplicated_replacement(folk, eng, sci)
        if replacement:
            pattern = rf"\b{re.escape(folk)}\b"
            enriched = re.sub(pattern, replacement, enriched, flags=re.IGNORECASE)

    return enriched


def apply_semantic_bridge_sb1(query: str) -> dict:
    """
    Evaluation-only entrypoint for Experiment SB1:
    - De-duplication & Specificity Guardrails
    - enable_multi_crop_balance = False
    """
    detected = detect_folk_terms_guarded(query, enable_multi_crop_balance=False)
    if not detected:
        return {
            "original": query,
            "enriched": query,
            "terms_found": [],
            "confidence": 1.0,
            "bridged": False,
            "term_count": 0
        }

    enriched = enrich_query_guarded(query, detected)

    if enriched == query:
        return {
            "original": query,
            "enriched": query,
            "terms_found": detected,
            "confidence": 1.0,
            "bridged": False,
            "term_count": len(detected)
        }

    confidence = compute_alignment_confidence(query, enriched, detected)

    return {
        "original": query,
        "enriched": enriched,
        "terms_found": detected,
        "confidence": confidence,
        "bridged": True,
        "term_count": len(detected)
    }


def apply_semantic_bridge_sb2(query: str) -> dict:
    """
    Production and Evaluation entrypoint for SB2 Guardrails:
    - De-duplication & Specificity Guardrails
    - Multi-Crop Comparative Balance Guardrail (enable_multi_crop_balance = True)
    """
    detected = detect_folk_terms_guarded(query, enable_multi_crop_balance=True)
    if not detected:
        return {
            "original": query,
            "enriched": query,
            "terms_found": [],
            "confidence": 1.0,
            "bridged": False,
            "term_count": 0
        }

    enriched = enrich_query_guarded(query, detected)

    if enriched == query:
        return {
            "original": query,
            "enriched": query,
            "terms_found": detected,
            "confidence": 1.0,
            "bridged": False,
            "term_count": len(detected)
        }

    confidence = compute_alignment_confidence(query, enriched, detected)

    return {
        "original": query,
        "enriched": enriched,
        "terms_found": detected,
        "confidence": confidence,
        "bridged": True,
        "term_count": len(detected)
    }


__all__ = [
    "GENERIC_COLLECTIVES",
    "CROP_INFLECTIONS",
    "SURFACE_TO_CANONICAL",
    "ALL_CROP_SURFACE_FORMS",
    "CROP_ANCHORS",
    "MULTI_WORD_CROP_FORMS",
    "SINGLE_WORD_CROP_FORMS",
    "COMPOUND_PEST_EXCLUSIONS",
    "PEST_HEAD_NOUNS",
    "PEST_MODIFIERS",
    "STRUCTURAL_COMPOUND_PEST_REGEX",
    "COORDINATED_PEST_REGEX",
    "PRACTICE_ANCHORS",
    "get_detected_crops",
    "is_entity_anchored",
    "normalize_term",
    "build_deduplicated_replacement",
    "detect_folk_terms_guarded",
    "enrich_query_guarded",
    "apply_semantic_bridge_sb1",
    "apply_semantic_bridge_sb2",
]
