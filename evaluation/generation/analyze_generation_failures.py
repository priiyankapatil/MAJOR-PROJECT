"""
analyze_generation_failures.py
Detailed attribution analysis of all 30 queries from the generation benchmark.
Inspects retrieval, temporal filtering, reranking, context delivery, LLM output,
grader accuracy, and latency.
"""

import json
import os
import sys
import re
import pandas as pd
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Import temporal scoring logic for verification
from temporal_credibility import compute_temporal_score, _match_source, DECAY_CONSTANTS, SOURCE_REGISTRY
from credibility_config import MIN_CREDIBILITY_THRESHOLD, EVALUATION_YEAR

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "generation_benchmark_results.json")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "generation_benchmark_dataset.json")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chunks", "all_chunks.parquet")

def load_data():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = {item["query_id"]: item for item in json.load(f)}
    chunks_df = pd.read_parquet(CHUNKS_PATH)
    chunk_map = {row["chunk_id"]: row.to_dict() for _, row in chunks_df.iterrows()}
    return results, dataset, chunk_map

def analyze():
    results, dataset, chunk_map = load_data()
    queries = results["query_results"]

    analysis = []

    for q in queries:
        qid = q["query_id"]
        gt_item = dataset[qid]
        gt_chunks = gt_item["relevant_chunk_ids"]
        retrieved = q["retrieved_chunk_ids"]
        path = q["routing_path"]
        q_type = q["predicted_type"]
        answer = q["predicted_answer"]
        score = q["score"]

        # 1. Retrieval Status
        gt_in_retrieved = [cid for cid in gt_chunks if cid in retrieved]
        retrieval_hit = len(gt_in_retrieved) > 0
        retrieved_ranks = {cid: retrieved.index(cid) for cid in gt_in_retrieved}

        # 2. Context supplied to generator
        # Fast path gets chunks[:3], Slow path gets chunks[:5] after temporal decay
        context_chunks_pre_filter = retrieved[:5] if path == "slow" else retrieved[:3]

        # Check temporal credibility for slow path
        temporal_analysis = []
        chunks_after_temporal = []
        context_starved = False

        if path == "slow":
            for cid in retrieved:
                c_meta = chunk_map.get(cid, {})
                sf = c_meta.get("source_file", "")
                src_match = _match_source(sf)
                t_score = compute_temporal_score(src_match, q_type, EVALUATION_YEAR)
                passed = t_score["final_score"] >= MIN_CREDIBILITY_THRESHOLD
                temporal_analysis.append({
                    "chunk_id": cid,
                    "source_file": sf,
                    "matched_src": src_match,
                    "base_score": t_score["base_score"],
                    "age_years": t_score["age_years"],
                    "final_score": t_score["final_score"],
                    "passed": passed
                })
                if passed:
                    chunks_after_temporal.append(cid)

            if len(chunks_after_temporal) == 0:
                context_starved = True
            final_generator_chunks = chunks_after_temporal[:5]
        else:
            final_generator_chunks = retrieved[:3]

        gt_reached_prompt = [cid for cid in gt_chunks if cid in final_generator_chunks]

        # 3. Missing facts analysis
        missing_facts = [f for f in q["facts_details"] if not f["satisfied"]]
        satisfied_facts = [f for f in q["facts_details"] if f["satisfied"]]

        # Check if missing facts were in the supplied context
        context_text = " ".join([chunk_map.get(cid, {}).get("text", "") for cid in final_generator_chunks])
        missing_facts_in_context = []
        for mf in missing_facts:
            # simple check if keywords of missing fact exist in context
            f_text = mf["fact"]
            # extract words > 4 chars
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', f_text)]
            in_ctx = any(w in context_text.lower() for w in words) if words else False
            missing_facts_in_context.append({"fact": f_text, "words_in_context": in_ctx})

        # 4. Refusal detection
        refusal_phrases = [
            "knowledge base doesn't have",
            "not provided in the context",
            "context does not contain",
            "no information",
            "does not specify"
        ]
        is_refusal = any(p in answer.lower() for p in refusal_phrases) or (len(answer.strip()) == 0)

        # 5. Determine Primary Failure Mode
        # Categories: CORRECT, INCOMPLETE_GENERATION, RETRIEVAL_MISS, CONTEXT_STARVATION, UNSUPPORTED_GENERATION, GENERATION_FAILURE, OTHER
        if score == 2:
            primary_failure = "CORRECT"
        elif not retrieval_hit:
            primary_failure = "RETRIEVAL_MISS"
        elif path == "slow" and context_starved:
            primary_failure = "CONTEXT_STARVATION"
        elif len(gt_reached_prompt) == 0 and retrieval_hit:
            # Retrieved, but didn't reach prompt (e.g. rank was 3,4 on fast path which only takes [:3])
            primary_failure = "RERANKING_OR_CUTOFF_DISPLACEMENT"
        elif score == 1:
            primary_failure = "INCOMPLETE_GENERATION"
        elif is_refusal:
            primary_failure = "GENERATION_FAILURE"
        else:
            primary_failure = "GENERATION_FAILURE"

        analysis.append({
            "query_id": qid,
            "category": q["category"],
            "path": path,
            "query_type": q_type,
            "score": score,
            "retrieval_hit": retrieval_hit,
            "gt_chunks": gt_chunks,
            "retrieved_chunks": retrieved,
            "gt_ranks": retrieved_ranks,
            "context_starved": context_starved,
            "final_generator_chunks": final_generator_chunks,
            "gt_reached_prompt": gt_reached_prompt,
            "primary_failure": primary_failure,
            "facts_satisfied": len(satisfied_facts),
            "facts_total": len(q["facts_details"]),
            "facts_details": q["facts_details"],
            "missing_facts": missing_facts,
            "answer": answer,
            "is_refusal": is_refusal,
            "total_latency_ms": q["total_latency_ms"],
            "temporal_analysis": temporal_analysis
        })

    return analysis

def print_deep_trace(records, chunk_map):
    print("\n" + "="*80)
    print("DEEP DIVE INVESTIGATION OF ALL 30 QUERIES")
    print("="*80)

    for r in records:
        qid = r["query_id"]
        print(f"\n--- [{qid}] ({r['category']}) Path: {r['path']} | Type: {r['query_type']} | Score: {r['score']} ---")
        print(f"Query: {r['category']} - Ground Truth Chunks: {r['gt_chunks']}")
        print(f"Retrieved Chunks (5): {r['retrieved_chunks']}")
        print(f"Ground Truth Ranks in Retrieval: {r['gt_ranks']}")
        print(f"Final Generator Chunks: {r['final_generator_chunks']}")
        print(f"Did GT reach generator prompt? {len(r['gt_reached_prompt']) > 0} ({r['gt_reached_prompt']})")
        print(f"Primary Failure: {r['primary_failure']}")
        print(f"Predicted Answer:\n  \"{r['answer']}\"")
        
        if r["path"] == "slow":
            print(f"Temporal Analysis:")
            for t in r["temporal_analysis"]:
                print(f"  Chunk {t['chunk_id']} [{t['matched_src']}]: age={t['age_years']}, score={t['final_score']:.4f}, passed={t['passed']} (file: {t['source_file']})")

        print(f"Fact Coverage: {r['facts_satisfied']}/{r['facts_total']}")
        for f in r["facts_details"]:
            status = "✓ FOUND" if f["satisfied"] else "✗ MISSING"
            print(f"  [{status}] {f['fact']}")
            if not f["satisfied"]:
                print(f"    Reason: {f['reason']}")

if __name__ == "__main__":
    records, dataset, chunk_map = load_data()
    analysis_records = analyze()
    print_deep_trace(analysis_records, chunk_map)
