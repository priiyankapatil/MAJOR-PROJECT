"""
Production Context Compression / Prompt Distillation (Experiment C1).
Prunes irrelevant narrative sentences from retrieved chunks prior to prompt assembly,
while strictly preserving technical numbers, units, dosages, chemical names, formulas,
markdown tables, and actionable agronomic instructions.
"""

import re
from typing import Tuple, List, Dict, Set, Any

# Stopwords for query keyword extraction
STOPWORDS = {
    "what", "which", "how", "when", "where", "why", "who", "whom", "this", "that", "these", "those",
    "from", "with", "about", "into", "through", "during", "before", "after", "above", "below", "to",
    "and", "but", "or", "for", "nor", "on", "at", "by", "is", "are", "was", "were", "be", "been",
    "the", "a", "an", "in", "of", "it", "its", "does", "did", "can", "could", "should", "would",
    "will", "shall", "may", "might", "must"
}

# Regex for agronomic and technical units / dosages
DOSAGE_PATTERN = re.compile(
    r"\b\d+(\.\d+)?\s*(%|kg|g|mg|l|ml|litres?|liters?|ha|hectare|acre|cm|m|mm|inches?|feet|foot|'|’|hours?|days?|tablets?|cum|ppm|ppb|ec|sl|wp|sc|gr|sp|fs|wdg|das)\b",
    re.IGNORECASE
)

# Regex for mathematical/agronomic formulas and ratios
FORMULA_PATTERN = re.compile(
    r"(=|\+|-|×|\/|iw/cpe|ratio|pe\s*x|\b\d+\s*:\s*\d+\b)",
    re.IGNORECASE
)

# Regex for actionable agronomic verbs
ACTION_PATTERN = re.compile(
    r"\b(spray|apply|application|dissolve|dilute|mix|construct|transplant|sow|sowing|prune|pruning|seal|fumigate|irrigate|irrigation|schedule|dose|dosage|spacing|depth|drench|drenching|broadcast)\b",
    re.IGNORECASE
)

# Regex for chemical formulations (e.g. Chlorpyrifos 20 EC, Cartap 4 G)
FORMULATION_PATTERN = re.compile(
    r"\b[A-Za-z\-]{3,}\s+\d+(\.\d+)?\s*(EC|SL|WP|SC|SP|WDG|FS|GR|DP)\b",
    re.IGNORECASE
)


def _get_known_chemicals() -> Set[str]:
    """Retrieve recognized active ingredients and common pesticide names from regulatory KB."""
    chemicals = set()
    try:
        from regulatory_kb import PESTICIDE_DB
        for p_key, p_data in PESTICIDE_DB.items():
            chemicals.add(p_key.lower())
            for c_name in p_data.get("common_names", []):
                chemicals.add(c_name.lower())
    except Exception:
        pass
    return chemicals


KNOWN_CHEMICALS = _get_known_chemicals()


def extract_query_keywords(query: str) -> Set[str]:
    """Extract non-trivial keyword tokens from the query."""
    if not query:
        return set()
    tokens = re.findall(r"\b[a-zA-Z0-9\-\'\’]{3,}\b", query.lower())
    return {t for t in tokens if t not in STOPWORDS}


def is_sentence_relevant(query_kws: Set[str], sentence: str) -> Tuple[bool, str]:
    """
    Evaluates sentence relevance against query keywords, technical formulas,
    dosages, recognized chemicals, and actionable agronomic instructions.
    """
    s_lower = sentence.lower()
    s_tokens = set(re.findall(r"\b[a-zA-Z0-9\-\'\’]{3,}\b", s_lower))

    # 1. Direct keyword overlap
    overlap = query_kws.intersection(s_tokens)
    if len(overlap) >= 1:
        return True, f"overlap({','.join(overlap)})"

    # 2. Markdown table row protection (always preserve table rows)
    if sentence.strip().startswith("|") and sentence.strip().endswith("|"):
        return True, "table_row"

    # 3. Chemical name protection (from regulatory DB)
    for chem in KNOWN_CHEMICALS:
        if len(chem) >= 4 and chem in s_lower:
            return True, f"chemical({chem})"

    # 4. Formulations (e.g., "Chlorpyrifos 20 EC")
    if FORMULATION_PATTERN.search(sentence):
        return True, "chemical_formulation"

    # 5. Formulas and ratios (e.g. "IW/CPE = 0.8", "1:2")
    if FORMULA_PATTERN.search(s_lower):
        return True, "formula"

    # 6. Technical dosage co-occurring with actionable verb or technical context
    has_dosage = bool(DOSAGE_PATTERN.search(s_lower))
    has_action = bool(ACTION_PATTERN.search(s_lower))
    if has_dosage and has_action:
        return True, "dosage+action"
    if has_dosage:
        return True, "dosage"

    return False, "irrelevant"


def distill_chunk_text(query: str, text: str) -> Tuple[str, List[str], List[str]]:
    """
    Prunes irrelevant narrative sentences from a chunk while strictly preserving
    technical facts, numbers, units, dosages, chemical names, and markdown tables.

    Returns:
        (distilled_text, kept_sentences, dropped_sentences)
    """
    if not text or not text.strip():
        return text or "", [], []

    orig_text = text.strip()
    orig_char_len = len(orig_text)
    query_kws = extract_query_keywords(query)

    # Separate markdown table lines from narrative text to avoid splitting table cells
    lines = orig_text.splitlines()
    table_lines = []
    narrative_segments = []

    current_narrative = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if current_narrative:
                narrative_segments.append("\n".join(current_narrative))
                current_narrative = []
            table_lines.append(line)
        else:
            if table_lines:
                narrative_segments.append("\n".join(table_lines))
                table_lines = []
            current_narrative.append(line)
    if current_narrative:
        narrative_segments.append("\n".join(current_narrative))
    if table_lines:
        narrative_segments.append("\n".join(table_lines))

    # Parse and filter segments
    kept = []
    dropped = []

    for seg in narrative_segments:
        # If segment is a markdown table, preserve entirely
        if any(l.strip().startswith("|") and l.strip().endswith("|") for l in seg.splitlines()):
            for tl in seg.splitlines():
                if tl.strip():
                    kept.append(tl.strip())
            continue

        # Otherwise split narrative into sentences
        cleaned = re.sub(r"[\r\n]+", " ", seg)
        cleaned = re.sub(r"[•\t]+", ". ", cleaned)
        raw_sents = re.split(r"(?<=[.?!;])\s+", cleaned)
        sentences = [s.strip() for s in raw_sents if len(s.strip()) > 5]

        for s in sentences:
            rel, _ = is_sentence_relevant(query_kws, s)
            if rel:
                kept.append(s)
            else:
                dropped.append(s)

    # Safeguard 1: If all sentences dropped, preserve original text
    if not kept:
        return orig_text, [orig_text], []

    distilled_text = " ".join(kept).strip()

    # Safeguard 2: If compression would prune > 80% of original text, retain original chunk
    if len(distilled_text) < 0.20 * orig_char_len:
        return orig_text, [orig_text], []

    return distilled_text, kept, dropped


def compress_context_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Applies selective context compression to a list of chunk dictionaries.
    Creates shallow copies of each chunk dict with updated 'text', preserving
    all original chunk metadata and IDs without in-place mutation.
    """
    if not chunks:
        return []

    compressed_chunks = []
    for c in chunks:
        c_copy = dict(c)  # Preserve original chunk object without mutation
        orig_text = c.get("text", "")
        try:
            dist_text, _, _ = distill_chunk_text(query, orig_text)
            c_copy["text"] = dist_text
        except Exception:
            # Fail open to original text on any unexpected error
            c_copy["text"] = orig_text
        compressed_chunks.append(c_copy)

    return compressed_chunks
