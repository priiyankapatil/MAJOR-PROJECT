"""
benchmark_fast_vs_slow.py

Benchmarking End-to-End Latency and Token Usage:
- Fast Path (gpt-oss-20b) vs Slow Path (gpt-oss-120b)
- Measures total wall-clock time, retrieval time, and generation time.
- Approximates prompt/completion token usage and calculates throughput.
- Writes structured artifacts to evaluation/benchmark_results.json.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Safe non-interactive execution
os.environ["NO_INTERACTIVE_FEEDBACK"] = "1"
os.environ["ENABLE_PROVENANCE_LOGGING"] = "0"

from step6_query_gate import query_gate, get_or_load_components

OUTPUT_PATH = REPO_ROOT / "evaluation" / "benchmark_results.json"

BENCHMARK_QUERIES = [
    # --- FAST PATH TARGETS (Factual, Definitions, Low-Entropy) ---
    {
        "id": "FAST-01",
        "query": "What is the seed rate for wheat?",
        "expected_path": "FAST"
    },
    {
        "id": "FAST-02",
        "query": "What does NPK stand for?",
        "expected_path": "FAST"
    },
    {
        "id": "FAST-03",
        "query": "What is the botanical name of rice?",
        "expected_path": "FAST"
    },

    # --- SLOW PATH TARGETS (Procedural, Diagnostic, Complex Recommendations) ---
    {
        "id": "SLOW-01",
        "query": "Should I spray pesticide on my tomato crop in Coimbatore tomorrow?",
        "expected_path": "SLOW"
    },
    {
        "id": "SLOW-02",
        "query": "How to manage tea mosquito bug and stem borer in cashew plantation?",
        "expected_path": "SLOW"
    },
    {
        "id": "SLOW-03",
        "query": "What is the complete fertilizer schedule and split application for sugarcane?",
        "expected_path": "SLOW"
    }
]


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def run_benchmark():
    print("=" * 70)
    print("🌾 AGRI-TRUST PIPELINE: FAST-PATH VS SLOW-PATH LATENCY & TOKEN BENCHMARK")
    print("=" * 70)

    # 1. Warm-up component caches
    print("\n⏳ Pre-loading ChromaDB, BM25, and Cross-Encoder components...")
    t_start = time.time()
    get_or_load_components()
    print(f"✓ Components ready in {time.time() - t_start:.2f}s\n")

    benchmark_runs = []

    for item in BENCHMARK_QUERIES:
        q_id = item["id"]
        query = item["query"]
        expected = item["expected_path"]

        print(f"▶ [{q_id}] ({expected}) Query: \"{query}\"")

        # Execute end-to-end query gate
        start_time = time.perf_counter()
        response = query_gate(query)
        end_time = time.perf_counter()

        total_latency = end_time - start_time
        path_taken = response.get("path", "UNKNOWN").upper()
        model_used = response.get("model_used", "UNKNOWN")
        answer_text = response.get("answer", "")
        sources = response.get("sources", [])

        # Token metrics
        query_tokens = estimate_tokens(query)
        answer_tokens = estimate_tokens(answer_text)
        total_tokens = query_tokens + answer_tokens
        tokens_per_sec = answer_tokens / total_latency if total_latency > 0 else 0.0

        run_result = {
            "id": q_id,
            "query": query,
            "expected_path": expected,
            "actual_path": path_taken,
            "model_used": model_used,
            "latency_seconds": round(total_latency, 3),
            "sources_count": len(sources),
            "approx_query_tokens": query_tokens,
            "approx_answer_tokens": answer_tokens,
            "approx_total_tokens": total_tokens,
            "tokens_per_second": round(tokens_per_sec, 2),
            "routing_match": path_taken == expected
        }

        benchmark_runs.append(run_result)

        print(f"    ├─ Path Selected : {path_taken} (Model: {model_used})")
        print(f"    ├─ Total Latency : {total_latency:.2f}s")
        print(f"    ├─ Output Length : ~{answer_tokens} tokens ({len(answer_text)} chars)")
        print(f"    └─ Throughput    : {tokens_per_sec:.1f} tokens/s\n")

    # Aggregate summaries
    fast_runs = [r for r in benchmark_runs if r["actual_path"] == "FAST"]
    slow_runs = [r for r in benchmark_runs if r["actual_path"] == "SLOW"]

    avg_fast_lat = sum(r["latency_seconds"] for r in fast_runs) / len(fast_runs) if fast_runs else 0
    avg_slow_lat = sum(r["latency_seconds"] for r in slow_runs) / len(slow_runs) if slow_runs else 0

    avg_fast_tok = sum(r["approx_answer_tokens"] for r in fast_runs) / len(fast_runs) if fast_runs else 0
    avg_slow_tok = sum(r["approx_answer_tokens"] for r in slow_runs) / len(slow_runs) if slow_runs else 0

    print("=" * 70)
    print("📊 BENCHMARK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Fast Path':<18} | {'Slow Path':<18} | {'Delta / Ratio'}")
    print("-" * 70)
    print(f"{'Sample Count':<25} | {len(fast_runs):<18} | {len(slow_runs):<18} | -")
    print(f"{'Avg Latency':<25} | {avg_fast_lat:<16.2f}s | {avg_slow_lat:<16.2f}s | {avg_slow_lat/avg_fast_lat if avg_fast_lat else 0:.1f}x slower")
    print(f"{'Avg Output Tokens':<25} | {avg_fast_tok:<18.0f} | {avg_slow_tok:<18.0f} | {avg_slow_tok/avg_fast_tok if avg_fast_tok else 0:.1f}x larger")
    print("=" * 70)

    # Save artifact
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "fast_path": {
                "count": len(fast_runs),
                "avg_latency_s": round(avg_fast_lat, 3),
                "avg_output_tokens": round(avg_fast_tok, 1)
            },
            "slow_path": {
                "count": len(slow_runs),
                "avg_latency_s": round(avg_slow_lat, 3),
                "avg_output_tokens": round(avg_slow_tok, 1)
            },
            "latency_multiplier": round(avg_slow_lat / avg_fast_lat, 2) if avg_fast_lat else 0,
            "token_multiplier": round(avg_slow_tok / avg_fast_tok, 2) if avg_fast_tok else 0
        },
        "runs": benchmark_runs
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n📁 Report saved cleanly to: {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    run_benchmark()
