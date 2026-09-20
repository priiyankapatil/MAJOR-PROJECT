"""
evaluation/generation/inspect_generation_pipeline.py
=====================================================
Diagnostic inspection script for Agricultural RAG Generation Pipeline.

Inspects:
1. Generation Models (Fast vs Slow path models in config.py)
2. Context Construction & Chunk Slicing (chunks[:3] vs chunks[:5])
3. Prompts & Instruction Templates (SLOW_PATH_PROMPTS, System Prompts)
4. Metadata reaching Generator (Source file, text vs scores, chunk IDs)
5. Provenance & Compliance Scanning
6. Determinism & Generation Parameters (temperature, seed, max_tokens)
7. Latency Measurement Gaps
8. Ground-Truth Answer Rubric Availability in retrieval_benchmark_dataset.json
"""

import os
import sys
import json
import inspect
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import config
import step6_query_gate as s6


def inspect_models_and_config():
    """Inspect model configuration from config.py and step6_query_gate.py."""
    gate_model = getattr(config, "GROQ_GATE_MODEL", "Not Set")
    answer_model = getattr(config, "GROQ_ANSWER_MODEL", "Not Set")
    has_api_key = bool(getattr(config, "GROQ_API_KEY", None))

    return {
        "fast_path_model": gate_model,
        "slow_path_model": answer_model,
        "api_key_configured": has_api_key,
        "entropy_threshold": getattr(config, "QT_ENTROPY_THRESHOLD", 1.5),
        "min_confidence": getattr(config, "QT_MIN_CONFIDENCE", 0.6),
        "entropy_tokens": getattr(config, "QT_LOGPROB_TOKENS", 10),
    }


def inspect_prompt_templates():
    """Inspect system prompts and query-type templates in step6_query_gate."""
    slow_prompts = getattr(s6, "SLOW_PATH_PROMPTS", {})
    categories = list(slow_prompts.keys())
    
    # Inspect fast_path_answer code
    fast_source = inspect.getsource(s6.fast_path_answer)
    slow_source = inspect.getsource(s6.slow_path_answer)
    
    return {
        "slow_path_categories": categories,
        "slow_path_prompts_count": len(slow_prompts),
        "fast_path_temperature": 0.1,
        "fast_path_max_tokens": 250,
        "fast_path_chunk_limit": 3,
        "slow_path_temperature": 0.1,
        "slow_path_max_tokens": 700,
        "slow_path_chunk_limit": 5,
        "has_seed_configured": ("seed" in fast_source and "seed" in slow_source),
    }


def inspect_context_and_metadata():
    """Inspect how retrieved chunks and metadata are transformed for generator input."""
    return {
        "fast_path_context": {
            "chunk_count": 3,
            "format": "Plain text joined by double newlines",
            "metadata_in_context": ["text only"],
            "metadata_excluded_from_context": [
                "chunk_id", "source_file", "trust_weight",
                "dense_score", "sparse_score", "final_score",
                "cross_encoder_score"
            ],
            "metadata_in_return_payload": ["source_file (list of unique sources)"]
        },
        "slow_path_context": {
            "chunk_count": 5,
            "format": "[Source: {source_file}]\\n{text} joined by '\\n\\n---\\n\\n'",
            "metadata_in_context": ["source_file", "text"],
            "metadata_excluded_from_context": [
                "chunk_id", "trust_weight", "dense_score",
                "sparse_score", "final_score", "cross_encoder_score",
                "authority_score", "year"
            ],
            "metadata_in_return_payload": ["source_file (list of unique sources)"]
        },
        "upstream_filters_before_slow_generator": [
            "Step 4C: Temporal Credibility Decay (score_all_chunks, filter_stale_chunks)",
            "Step 4D: Phenological Gate (apply_phenological_gate)",
            "Step 4B: Weather Enrichment & Seasonal Context (enrich_with_weather)"
        ],
        "post_generation_processors": [
            "Step 5B: Sentence-Level Provenance Mapping (build_provenance_map)",
            "Step 5C: Regulatory Compliance Scan (scan_answer_for_compliance & warning box)"
        ]
    }


def inspect_benchmark_dataset():
    """Check retrieval_benchmark_dataset.json for reference answers or rubrics."""
    dataset_path = WORKSPACE / "evaluation" / "retrieval" / "retrieval_benchmark_dataset.json"
    if not dataset_path.exists():
        return {"error": "retrieval_benchmark_dataset.json not found"}

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_queries = len(data)
    has_reference_answer = any("reference_answer" in q for q in data)
    has_ground_truth_answer = any("ground_truth_answer" in q for q in data)
    has_rubric = any("rubric" in q or "evaluation_rubric" in q for q in data)
    has_key_facts = any("key_facts" in q for q in data)
    has_relevance_notes = any("relevance_note" in q for q in data)

    sample_fields = list(data[0].keys()) if data else []

    return {
        "dataset_path": str(dataset_path),
        "total_queries": total_queries,
        "sample_fields": sample_fields,
        "has_reference_answer": has_reference_answer,
        "has_ground_truth_answer": has_ground_truth_answer,
        "has_rubric": has_rubric,
        "has_key_facts": has_key_facts,
        "has_relevance_notes": has_relevance_notes,
    }


def inspect_latency_instrumentation():
    """Inspect latency capture across the pipeline."""
    qg_source = inspect.getsource(s6.query_gate)
    measures_total_latency = "time.perf_counter" in qg_source or "time.time" in qg_source
    returns_latency = "latency" in qg_source or "latency_ms" in qg_source

    return {
        "query_gate_records_latency_internally": returns_latency,
        "query_gate_has_timer": measures_total_latency,
        "required_evaluation_timers": [
            "retrieval_latency_ms (Dense + BM25 + CrossEncoder)",
            "routing_latency_ms (Classification + Entropy)",
            "weather_pheno_latency_ms (Weather/Pheno gate if active)",
            "generation_latency_ms (Groq LLM call)",
            "provenance_latency_ms (Sentence provenance mapping)",
            "compliance_latency_ms (Regulatory compliance scan)",
            "total_e2e_latency_ms (Full wall-clock latency)"
        ]
    }


def run_inspection():
    print("=" * 80)
    print(" 🌾 AGRICULTURAL RAG GENERATION PIPELINE INSPECTION")
    print("=" * 80)

    # 1. Models & Config
    print("\n1. GENERATION MODELS & CONFIGURATION:")
    cfg = inspect_models_and_config()
    print(f"   • Fast Path Model   : {cfg['fast_path_model']}")
    print(f"   • Slow Path Model   : {cfg['slow_path_model']}")
    print(f"   • Groq API Key Set  : {cfg['api_key_configured']}")
    print(f"   • Entropy Threshold : {cfg['entropy_threshold']}")
    print(f"   • Min Confidence    : {cfg['min_confidence']}")

    # 2. Context & Slicing
    print("\n2. RETRIEVED CHUNKS & CONTEXT BUDGET:")
    ctx = inspect_context_and_metadata()
    print(f"   • Fast Path Chunks  : Top {ctx['fast_path_context']['chunk_count']} chunks")
    print(f"     Format            : {ctx['fast_path_context']['format']}")
    print(f"     Context Metadata  : {ctx['fast_path_context']['metadata_in_context']}")
    print(f"   • Slow Path Chunks  : Top {ctx['slow_path_context']['chunk_count']} chunks")
    print(f"     Format            : {ctx['slow_path_context']['format']}")
    print(f"     Context Metadata  : {ctx['slow_path_context']['metadata_in_context']}")

    # 3. Prompts & Templates
    print("\n3. PROMPTS & INSTRUCTION TEMPLATES:")
    pr = inspect_prompt_templates()
    print(f"   • Slow Path Categories : {', '.join(pr['slow_path_categories'])}")
    print(f"   • Fast Path Temp/Tokens: temp={pr['fast_path_temperature']}, max_tokens={pr['fast_path_max_tokens']}")
    print(f"   • Slow Path Temp/Tokens: temp={pr['slow_path_temperature']}, max_tokens={pr['slow_path_max_tokens']}")
    print(f"   • Explicit Seed Fixed  : {pr['has_seed_configured']} (Default uses sampling temp=0.1)")

    # 4. Provenance & Compliance
    print("\n4. PROVENANCE & COMPLIANCE POST-PROCESSING:")
    print("   • Fast Path : Provenance SKIPPED, Compliance SKIPPED")
    print("   • Slow Path :")
    print("     - Step 5B: Sentence-level provenance map (sentence_provenance.py)")
    print("     - Step 5C: Regulatory compliance scan (compliance_scanner.py)")
    print("     - Appends formatted compliance warning box if regulated pesticides detected")

    # 5. Latency Capture
    print("\n5. LATENCY INSTRUMENTATION GAPS:")
    lat = inspect_latency_instrumentation()
    print(f"   • query_gate returns latency : {lat['query_gate_records_latency_internally']}")
    print(f"   • query_gate internal timer  : {lat['query_gate_has_timer']}")
    print("   • Evaluation harness must supply granular step-by-step timers.")

    # 6. Benchmark Dataset Status
    print("\n6. BENCHMARK DATASET & ANSWER RUBRIC STATUS:")
    ds = inspect_benchmark_dataset()
    print(f"   • Queries in Benchmark       : {ds.get('total_queries', 0)}")
    print(f"   • Sample Keys                : {ds.get('sample_fields', [])}")
    print(f"   • Has 'reference_answer'     : {ds.get('has_reference_answer', False)}")
    print(f"   • Has 'ground_truth_answer'  : {ds.get('has_ground_truth_answer', False)}")
    print(f"   • Has 'rubric'               : {ds.get('has_rubric', False)}")
    print(f"   • Has 'key_facts'            : {ds.get('has_key_facts', False)}")
    print(f"   • Has 'relevance_notes'      : {ds.get('has_relevance_notes', False)}")

    if not ds.get("has_reference_answer") and not ds.get("has_rubric"):
        print("\n   ⚠️  CRITICAL FINDING: NO REFERENCE ANSWERS OR RUBRICS EXIST YET!")
        print("   The retrieval benchmark dataset provides ground truth chunk IDs and relevance notes,")
        print("   but lacks reference answers, expected numerical facts, or scoring rubrics for generation.")

    print("\n" + "=" * 80)
    print(" ✅ INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_inspection()
