"""
evaluation/retrieval/semantic_bridge_guardrails.py

Backward compatibility re-export stub.
Authoritative implementation relocated to production: components/context/guardrails.py
"""
from components.context.guardrails import (
    GENERIC_COLLECTIVES,
    CROP_INFLECTIONS,
    SURFACE_TO_CANONICAL,
    ALL_CROP_SURFACE_FORMS,
    CROP_ANCHORS,
    MULTI_WORD_CROP_FORMS,
    SINGLE_WORD_CROP_FORMS,
    COMPOUND_PEST_EXCLUSIONS,
    PEST_HEAD_NOUNS,
    PEST_MODIFIERS,
    STRUCTURAL_COMPOUND_PEST_REGEX,
    COORDINATED_PEST_REGEX,
    PRACTICE_ANCHORS,
    get_detected_crops,
    is_entity_anchored,
    normalize_term,
    build_deduplicated_replacement,
    detect_folk_terms_guarded,
    enrich_query_guarded,
    apply_semantic_bridge_sb1,
    apply_semantic_bridge_sb2,
    __all__,
)
