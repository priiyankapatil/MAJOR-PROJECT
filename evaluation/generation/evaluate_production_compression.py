"""
evaluate_production_compression.py

Phase C1 Context Compression / Prompt Distillation Evaluation Harness:
- Evaluates selective narrative sentence pruning on retrieved chunks.
- Validates key technical entity/fact retention (>= 90% target).
- Measures character & token compression ratio (% reduction).
- Benchmarks compression runtime overhead in milliseconds.
- Writes structured results to evaluation/generation/production_c1_evaluation_results.json.
"""

import os
import sys
import json
import time
import re
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe non-interactive execution
os.environ["NO_INTERACTIVE_FEEDBACK"] = "1"
os.environ["ENABLE_PROVENANCE_LOGGING"] = "0"

import config
import step6_query_gate as s6
from components.context.compression import compress_context_chunks

OUTPUT_PATH = REPO_ROOT / "evaluation" / "generation" / "production_c1_evaluation_results.json"

# Representative evaluation cases covering diverse agronomic scenarios
EVALUATION_CASES = [
    {
        "id": "C1-CASE-01",
        "topic": "Wheat seed rate & irrigation",
        "query": "Wheat seed rate 100 kg/ha and critical CRI stage irrigation",
        "expected_facts": ["100", "kg/ha", "CRI"]
    },
    {
        "id": "C1-CASE-02",
        "topic": "Tomato late blight",
        "query": "Tomato late blight mancozeb fungicide spray in high relative humidity",
        "expected_facts": ["Mancozeb", "fungicide", "humidity"]
    },
    {
        "id": "C1-CASE-03",
        "topic": "Rice nitrogen schedule",
        "query": "Rice nitrogen fertilizer schedule split application tillering kg/ha",
        "expected_facts": ["kg/ha", "split", "tillering"]
    },
    {
        "id": "C1-CASE-04",
        "topic": "Cashew tea mosquito bug",
        "query": "Cashew tea mosquito bug management and spray schedule",
        "expected_facts": ["tea mosquito bug", "spray"]
    },
    {
        "id": "C1-CASE-05",
        "topic": "Cotton pink bollworm ETL",
        "query": "Cotton pink bollworm ETL threshold and monitoring",
        "expected_facts": ["ETL", "bollworm"]
    }
]


def fact_present(fact: str, text: str) -> bool:
    """Case-insensitive fact / entity presence check supporting slash variants (e.g. kg/ha)."""
    if not text or not fact:
        return False
    if fact.lower() in text.lower():
        return True
    if "/" in fact:
        parts = [re.escape(p.strip()) for p in fact.split("/")]
        pattern = r"\s*/\s*".join(parts)
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def estimate_tokens(text: str) -> int:
    """Token count estimation (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def run_evaluation():
    print("=" * 75)
    print("🌿 PHASE C1 EVALUATION: PRODUCTION CONTEXT COMPRESSION & FACT RETENTION")
    print("=" * 75)

    # 1. Warm-up retrieval components
    print("\n⏳ Loading search components (ChromaDB, BM25, Cross-Encoder)...")
    t0 = time.time()
    embedder, collection, bm25, corpus = s6.get_or_load_components()
    print(f"✓ Components ready in {time.time() - t0:.2f}s\n")

    results = []
    total_baseline_chars = 0
    total_distilled_chars = 0
    total_baseline_tokens = 0
    total_distilled_tokens = 0
    all_facts_expected_count = 0
    all_facts_retained_count = 0
    total_overhead_ms = 0.0

    # Ensure config.ENABLE_CONTEXT_COMPRESSION resets cleanly to False
    original_compression_setting = getattr(config, "ENABLE_CONTEXT_COMPRESSION", False)
    config.ENABLE_CONTEXT_COMPRESSION = True

    try:
        for case in EVALUATION_CASES:
            case_id = case["id"]
            topic = case["topic"]
            query = case["query"]
            expected_facts = case["expected_facts"]

            print(f"▶ [{case_id}] {topic}")
            print(f"  Query: \"{query}\"")

            # a. Retrieve top-5 candidate chunks
            chunks = s6.retrieve_chunks(query, embedder, collection, bm25, corpus, top_k=5)
            raw_context = "\n\n".join(c.get("text", "") for c in chunks)
            raw_chars = len(raw_context)
            raw_tokens = estimate_tokens(raw_context)

            # b. Measure compression overhead
            start_t = time.perf_counter()
            compressed_chunks = compress_context_chunks(query, chunks)
            end_t = time.perf_counter()
            overhead_ms = (end_t - start_t) * 1000.0

            distilled_context = "\n\n".join(c.get("text", "") for c in compressed_chunks)
            distilled_chars = len(distilled_context)
            distilled_tokens = estimate_tokens(distilled_context)

            # c. Compression ratio
            reduction_chars_pct = ((raw_chars - distilled_chars) / raw_chars * 100.0) if raw_chars > 0 else 0.0
            reduction_tokens_pct = ((raw_tokens - distilled_tokens) / raw_tokens * 100.0) if raw_tokens > 0 else 0.0

            # d. Fact retention check
            fact_status = {}
            retained_facts_for_case = 0
            for fact in expected_facts:
                in_raw = fact_present(fact, raw_context)
                in_distilled = fact_present(fact, distilled_context)
                fact_status[fact] = {
                    "in_raw": in_raw,
                    "in_distilled": in_distilled,
                    "retained": in_distilled
                }
                if in_distilled:
                    retained_facts_for_case += 1

            retention_rate = (retained_facts_for_case / len(expected_facts) * 100.0) if expected_facts else 100.0

            # Accumulate totals
            total_baseline_chars += raw_chars
            total_distilled_chars += distilled_chars
            total_baseline_tokens += raw_tokens
            total_distilled_tokens += distilled_tokens
            all_facts_expected_count += len(expected_facts)
            all_facts_retained_count += retained_facts_for_case
            total_overhead_ms += overhead_ms

            case_result = {
                "id": case_id,
                "topic": topic,
                "query": query,
                "expected_facts": expected_facts,
                "fact_retention": fact_status,
                "fact_retention_rate_pct": round(retention_rate, 2),
                "baseline_characters": raw_chars,
                "distilled_characters": distilled_chars,
                "character_reduction_pct": round(reduction_chars_pct, 2),
                "baseline_tokens": raw_tokens,
                "distilled_tokens": distilled_tokens,
                "token_reduction_pct": round(reduction_tokens_pct, 2),
                "compression_overhead_ms": round(overhead_ms, 3)
            }
            results.append(case_result)

            print(f"  ├─ Baseline Chars : {raw_chars} (~{raw_tokens} tokens)")
            print(f"  ├─ Distilled Chars: {distilled_chars} (~{distilled_tokens} tokens)")
            print(f"  ├─ Reduction      : {reduction_chars_pct:.2f}% chars ({reduction_tokens_pct:.2f}% tokens)")
            print(f"  ├─ Fact Retention : {retention_rate:.1f}% ({retained_facts_for_case}/{len(expected_facts)} preserved)")
            print(f"  └─ Overhead       : {overhead_ms:.2f} ms\n")

    finally:
        # Guarantee config.ENABLE_CONTEXT_COMPRESSION is reset cleanly to False
        config.ENABLE_CONTEXT_COMPRESSION = False

    # Summary calculations
    overall_char_reduction_pct = (
        (total_baseline_chars - total_distilled_chars) / total_baseline_chars * 100.0
    ) if total_baseline_chars > 0 else 0.0

    overall_token_reduction_pct = (
        (total_baseline_tokens - total_distilled_tokens) / total_baseline_tokens * 100.0
    ) if total_baseline_tokens > 0 else 0.0

    overall_fact_retention_rate = (
        all_facts_retained_count / all_facts_expected_count * 100.0
    ) if all_facts_expected_count > 0 else 100.0

    mean_overhead_ms = total_overhead_ms / len(results) if results else 0.0

    print("=" * 75)
    print("📊 PHASE C1 COMPRESSION EVALUATION SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<35} | {'Value':<18}")
    print("-" * 75)
    print(f"{'Evaluation Cases':<35} | {len(results):<18}")
    print(f"{'Avg Character Reduction':<35} | {overall_char_reduction_pct:<16.2f}%")
    print(f"{'Avg Token Reduction':<35} | {overall_token_reduction_pct:<16.2f}%")
    print(f"{'Key Fact Retention Rate':<35} | {overall_fact_retention_rate:<16.2f}% (Target: >= 90%)")
    print(f"{'Mean Compression Overhead':<35} | {mean_overhead_ms:<16.2f} ms")
    print(f"{'config.ENABLE_CONTEXT_COMPRESSION':<35} | {config.ENABLE_CONTEXT_COMPRESSION}")
    print("=" * 75)

    # Save output JSON artifact
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_reset_verified": config.ENABLE_CONTEXT_COMPRESSION is False,
        "summary": {
            "total_cases": len(results),
            "avg_character_reduction_pct": round(overall_char_reduction_pct, 2),
            "avg_token_reduction_pct": round(overall_token_reduction_pct, 2),
            "overall_fact_retention_rate_pct": round(overall_fact_retention_rate, 2),
            "target_retention_met": overall_fact_retention_rate >= 90.0,
            "mean_overhead_ms": round(mean_overhead_ms, 3)
        },
        "cases": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n📁 Report saved cleanly to: {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run_evaluation()
