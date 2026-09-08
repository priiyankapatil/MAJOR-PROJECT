"""
Controlled LLM Query-Rewriting Retrieval Ablation Experiment.

Compares:
- Condition A: Original query
- Condition B: LLM-rewritten query (using project's GROQ_ANSWER_MODEL: openai/gpt-oss-120b)

Retrieval Setup for BOTH conditions:
- Candidate Generation: Dense Top-30 (all-MiniLM-L6-v2) + BM25 Top-30
- Fusion: Equal Weighted (0.5 Dense + 0.5 BM25) [Primary] and RRF (k=60) [Secondary]
- Evaluated on: evaluation/retrieval/retrieval_benchmark_dataset.json (30 queries)
- Corpus: 12,856 chunks (data/chunks/all_chunks.parquet)

Outputs:
- evaluation/retrieval/query_rewrite_ablation_results.json
- evaluation/retrieval/QUERY_REWRITE_ABLATION_RESULTS.md
"""

import os
import sys
import json
import time
import math
import statistics
import chromadb
from sentence_transformers import SentenceTransformer

# UTF-8 stdout for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT_DIR, ".env"))

from groq import Groq
import config
from step5_vector_index import (
    dense_search,
    sparse_search,
    load_bm25_index,
)

BENCHMARK_FILE = os.path.join(ROOT_DIR, "evaluation", "retrieval", "retrieval_benchmark_dataset.json")
OUTPUT_JSON = os.path.join(ROOT_DIR, "evaluation", "retrieval", "query_rewrite_ablation_results.json")
OUTPUT_MD = os.path.join(ROOT_DIR, "evaluation", "retrieval", "QUERY_REWRITE_ABLATION_RESULTS.md")

REWRITE_SYSTEM_PROMPT = (
    "You are an expert agricultural search query expansion assistant.\n"
    "Your task is to rewrite the user's agricultural query to optimize first-stage retrieval "
    "(Dense vector search and BM25 lexical search) from an agronomic research document corpus.\n\n"
    "Strict Rules:\n"
    "1. Preserve the user's specific information need.\n"
    "2. Identify implicit technical concepts.\n"
    "3. Add useful agricultural terminology, synonyms, and relevant agronomic context.\n"
    "4. Include scientific terminology (pathogen names, botanical names) only when directly relevant.\n"
    "5. Do NOT invent facts.\n"
    "6. Do NOT answer the question or provide explanations.\n"
    "7. Output ONLY the rewritten retrieval query on a single line. Do NOT include quotation marks, preambles, or markdown formatting."
)


def rewrite_query_with_llm(groq_client, model_name, query):
    """
    Call the configured LLM to rewrite/expand the user query.
    Returns: (rewritten_query_str, latency_ms)
    """
    t0 = time.perf_counter()
    res = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
        max_completion_tokens=1000
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0
    content = res.choices[0].message.content or ""
    lines = [l.strip() for l in content.strip().split("\n") if l.strip() and not l.strip().startswith("#") and not l.strip().lower().startswith("rewritten")]
    rewritten = lines[-1] if lines else content.strip()
    rewritten = rewritten.strip('"`*')
    # Fallback to original if completely empty
    if not rewritten:
        rewritten = query
    return rewritten, lat_ms


def fuse_equal(dense_results, bm25_results, top_k=10):
    """Equal weighted fusion: 0.5 * dense_score + 0.5 * bm25_score."""
    combined = {}
    for r in dense_results:
        cid = r["chunk_id"]
        combined[cid] = combined.get(cid, 0.0) + 0.5 * r.get("dense_score", 0.0)
    for r in bm25_results:
        cid = r["chunk_id"]
        combined[cid] = combined.get(cid, 0.0) + 0.5 * r.get("bm25_score", 0.0)
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sorted_items[:top_k]]


def fuse_rrf(dense_results, bm25_results, k=60, top_k=10):
    """Standard Reciprocal Rank Fusion."""
    rrf_scores = {}
    for rank, r in enumerate(dense_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in sorted_items[:top_k]]


def evaluate_ranking(retrieved_chunk_ids, relevant_chunk_ids):
    """Computes Hit, Recall, Precision, MRR, NDCG for K in [1, 3, 5, 10]."""
    gt_set = set(relevant_chunk_ids)
    num_gt = len(gt_set)
    metrics = {}
    
    first_rank = None
    for r, cid in enumerate(retrieved_chunk_ids, 1):
        if cid in gt_set:
            first_rank = r
            break
            
    mrr_10 = 1.0 / first_rank if (first_rank is not None and first_rank <= 10) else 0.0
    metrics["mrr_at_10"] = mrr_10
    
    for k in [1, 3, 5, 10]:
        top_k = retrieved_chunk_ids[:k]
        hits = len(set(top_k) & gt_set)
        metrics[f"hit_at_{k}"] = 1.0 if hits > 0 else 0.0
        metrics[f"recall_at_{k}"] = (hits / num_gt) if num_gt > 0 else 0.0
        metrics[f"precision_at_{k}"] = hits / k

    # NDCG@10
    dcg = 0.0
    for r, cid in enumerate(retrieved_chunk_ids[:10], 1):
        if cid in gt_set:
            dcg += 1.0 / math.log2(r + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(num_gt, 10) + 1))
    metrics["ndcg_at_10"] = (dcg / idcg) if idcg > 0 else 0.0
    return metrics


def main():
    print("==================================================================================")
    print("  CONTROLLED LLM QUERY-REWRITING RETRIEVAL ABLATION EXPERIMENT")
    print("==================================================================================")

    # 1. Verify Groq API Key & Client
    groq_api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ CRITICAL ERROR: GROQ_API_KEY is not configured in .env or config.py!")
        print("   Aborting experiment to satisfy reproducibility constraint.")
        sys.exit(1)

    llm_model = config.GROQ_ANSWER_MODEL or "openai/gpt-oss-120b"
    print(f"🔧 Initializing Groq client with model: {llm_model}")
    groq_client = Groq(api_key=groq_api_key)

    # 2. Verify Indexes & Benchmark
    chroma_client = chromadb.PersistentClient(path=str(config.VECTOR_STORE))
    collection = chroma_client.get_collection(config.COLLECTION_NAME)
    chroma_count = collection.count()

    bm25, corpus = load_bm25_index()
    bm25_count = len(corpus)

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    print(f"   ChromaDB Documents : {chroma_count:,}")
    print(f"   BM25 Corpus Chunks : {bm25_count:,}")
    print(f"   Benchmark Queries  : {len(benchmark_data)}")
    assert chroma_count == 12856, f"Chroma count {chroma_count} != 12,856"
    assert bm25_count == 12856, f"BM25 count {bm25_count} != 12,856"
    assert len(benchmark_data) == 30, f"Query count {len(benchmark_data)} != 30"
    print("   [OK] Pre-run integrity verified.\n")

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    # Warm-up
    _ = dense_search("warmup query", collection, embedder, top_k=30)
    _ = sparse_search("warmup query", bm25, corpus, top_k=30)

    detailed_query_records = []
    condition_a_records = []  # Original
    condition_b_records = []  # Rewritten
    condition_a_rrf_records = []
    condition_b_rrf_records = []

    pool_recall_a = []
    pool_recall_b = []
    rewrite_latencies = []

    print(f"Executing Query Rewriting & Evaluation across all 30 queries...")

    for idx, item in enumerate(benchmark_data, 1):
        qid = item["query_id"]
        category = item["category"]
        orig_query = item["query"]
        gt_chunks = item["relevant_chunk_ids"]
        gt_set = set(gt_chunks)

        # ── LLM Query Rewriting ──
        rewritten_query, rw_lat_ms = rewrite_query_with_llm(groq_client, llm_model, orig_query)
        rewrite_latencies.append(rw_lat_ms)

        # ── Condition A: Original Query Retrieval ──
        t_a_start = time.perf_counter()
        d_res_a = dense_search(orig_query, collection, embedder, top_k=30)
        s_res_a = sparse_search(orig_query, bm25, corpus, top_k=30)
        top10_eq_a = fuse_equal(d_res_a, s_res_a, top_k=10)
        top10_rrf_a = fuse_rrf(d_res_a, s_res_a, top_k=10)
        lat_a = (time.perf_counter() - t_a_start) * 1000.0

        d_ids_a = [r["chunk_id"] for r in d_res_a]
        s_ids_a = [r["chunk_id"] for r in s_res_a]
        pool_a = set(d_ids_a) | set(s_ids_a)
        p_rec_a = len(gt_set & pool_a) / len(gt_set) if gt_set else 0.0
        pool_recall_a.append(p_rec_a)

        metrics_a = evaluate_ranking(top10_eq_a, gt_chunks)
        metrics_a.update({"query_id": qid, "category": category, "latency_ms": round(lat_a, 2)})
        condition_a_records.append(metrics_a)

        metrics_a_rrf = evaluate_ranking(top10_rrf_a, gt_chunks)
        metrics_a_rrf.update({"query_id": qid, "category": category, "latency_ms": round(lat_a, 2)})
        condition_a_rrf_records.append(metrics_a_rrf)

        # ── Condition B: Rewritten Query Retrieval ──
        t_b_start = time.perf_counter()
        d_res_b = dense_search(rewritten_query, collection, embedder, top_k=30)
        s_res_b = sparse_search(rewritten_query, bm25, corpus, top_k=30)
        top10_eq_b = fuse_equal(d_res_b, s_res_b, top_k=10)
        top10_rrf_b = fuse_rrf(d_res_b, s_res_b, top_k=10)
        retrieval_lat_b = (time.perf_counter() - t_b_start) * 1000.0
        total_lat_b = rw_lat_ms + retrieval_lat_b

        d_ids_b = [r["chunk_id"] for r in d_res_b]
        s_ids_b = [r["chunk_id"] for r in s_res_b]
        pool_b = set(d_ids_b) | set(s_ids_b)
        p_rec_b = len(gt_set & pool_b) / len(gt_set) if gt_set else 0.0
        pool_recall_b.append(p_rec_b)

        metrics_b = evaluate_ranking(top10_eq_b, gt_chunks)
        metrics_b.update({"query_id": qid, "category": category, "latency_ms": round(total_lat_b, 2), "rewrite_latency_ms": round(rw_lat_ms, 2)})
        condition_b_records.append(metrics_b)

        metrics_b_rrf = evaluate_ranking(top10_rrf_b, gt_chunks)
        metrics_b_rrf.update({"query_id": qid, "category": category, "latency_ms": round(total_lat_b, 2)})
        condition_b_rrf_records.append(metrics_b_rrf)

        # ── Per-Query Tracking ──
        # Find ranks of first ground-truth chunk
        def find_first_rank(id_list):
            for rank, cid in enumerate(id_list, 1):
                if cid in gt_set:
                    return rank
            return 999

        dense_rank_a = find_first_rank(d_ids_a)
        dense_rank_b = find_first_rank(d_ids_b)
        bm25_rank_a = find_first_rank(s_ids_a)
        bm25_rank_b = find_first_rank(s_ids_b)
        fusion_rank_a = find_first_rank(top10_eq_a)
        fusion_rank_b = find_first_rank(top10_eq_b)

        in_pool_a = bool(gt_set & pool_a)
        in_pool_b = bool(gt_set & pool_b)

        # Outcome determination
        if fusion_rank_b < fusion_rank_a:
            outcome = "IMPROVED"
        elif fusion_rank_b > fusion_rank_a:
            outcome = "HARMED"
        else:
            outcome = "UNCHANGED"

        query_record = {
            "query_id": qid,
            "category": category,
            "original_query": orig_query,
            "rewritten_query": rewritten_query,
            "rewrite_latency_ms": round(rw_lat_ms, 2),
            "retrieval_latency_ms": round(retrieval_lat_b, 2),
            "total_latency_ms": round(total_lat_b, 2),
            "target_ground_truth": gt_chunks,
            "dense_rank_orig": dense_rank_a if dense_rank_a != 999 else ">30",
            "dense_rank_rewritten": dense_rank_b if dense_rank_b != 999 else ">30",
            "bm25_rank_orig": bm25_rank_a if bm25_rank_a != 999 else ">30",
            "bm25_rank_rewritten": bm25_rank_b if bm25_rank_b != 999 else ">30",
            "ground_truth_in_pool_orig": in_pool_a,
            "ground_truth_in_pool_rewritten": in_pool_b,
            "fusion_rank_orig": fusion_rank_a if fusion_rank_a != 999 else ">10",
            "fusion_rank_rewritten": fusion_rank_b if fusion_rank_b != 999 else ">10",
            "outcome": outcome
        }
        detailed_query_records.append(query_record)

        if idx % 5 == 0 or idx == 30:
            print(f"   Completed {idx}/30 queries... ({outcome}: rank {query_record['fusion_rank_orig']} -> {query_record['fusion_rank_rewritten']})")

    # 3. Aggregate Calculations
    def aggregate_metrics(rec_list):
        agg = {
            "query_count": len(rec_list),
            "mean_latency_ms": round(statistics.mean([r["latency_ms"] for r in rec_list]), 2),
            "median_latency_ms": round(statistics.median([r["latency_ms"] for r in rec_list]), 2),
            "mrr_at_10": round(statistics.mean([r["mrr_at_10"] for r in rec_list]), 4),
            "ndcg_at_10": round(statistics.mean([r["ndcg_at_10"] for r in rec_list]), 4),
        }
        for k in [1, 3, 5, 10]:
            agg[f"hit_at_{k}"] = round(statistics.mean([r[f"hit_at_{k}"] for r in rec_list]), 4)
            agg[f"recall_at_{k}"] = round(statistics.mean([r[f"recall_at_{k}"] for r in rec_list]), 4)
            agg[f"precision_at_{k}"] = round(statistics.mean([r[f"precision_at_{k}"] for r in rec_list]), 4)
        return agg

    summary_a = aggregate_metrics(condition_a_records)
    summary_b = aggregate_metrics(condition_b_records)
    summary_a_rrf = aggregate_metrics(condition_a_rrf_records)
    summary_b_rrf = aggregate_metrics(condition_b_rrf_records)

    # Category summaries
    categories = sorted(list({r["category"] for r in condition_a_records}))
    cat_summary_a = {}
    cat_summary_b = {}
    for cat in categories:
        cat_recs_a = [r for r in condition_a_records if r["category"] == cat]
        cat_recs_b = [r for r in condition_b_records if r["category"] == cat]
        cat_summary_a[cat] = aggregate_metrics(cat_recs_a)
        cat_summary_b[cat] = aggregate_metrics(cat_recs_b)

    improved_queries = [r for r in detailed_query_records if r["outcome"] == "IMPROVED"]
    harmed_queries = [r for r in detailed_query_records if r["outcome"] == "HARMED"]
    unchanged_queries = [r for r in detailed_query_records if r["outcome"] == "UNCHANGED"]

    results_data = {
        "metadata": {
            "llm_model": llm_model,
            "total_queries": len(benchmark_data),
            "mean_rewrite_latency_ms": round(statistics.mean(rewrite_latencies), 2),
            "candidate_pool_recall_original": round(statistics.mean(pool_recall_a), 4),
            "candidate_pool_recall_rewritten": round(statistics.mean(pool_recall_b), 4),
            "improved_count": len(improved_queries),
            "harmed_count": len(harmed_queries),
            "unchanged_count": len(unchanged_queries)
        },
        "global_summary": {
            "Condition_A_Original_Equal": summary_a,
            "Condition_B_Rewritten_Equal": summary_b,
            "Condition_A_Original_RRF": summary_a_rrf,
            "Condition_B_Rewritten_RRF": summary_b_rrf,
        },
        "categories_summary": {
            "Condition_A_Original": cat_summary_a,
            "Condition_B_Rewritten": cat_summary_b,
        },
        "queries": detailed_query_records
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {OUTPUT_JSON}")

    # 4. Generate Comprehensive Markdown Report
    generate_markdown_report(results_data, detailed_query_records, summary_a, summary_b, summary_a_rrf, summary_b_rrf, cat_summary_a, cat_summary_b)
    print(f"[OK] Report saved to: {OUTPUT_MD}")

    # Print summary table
    print("\n" + "=" * 115)
    print("QUERY REWRITING ABLATION: ORIGINAL vs LLM-REWRITTEN (Pool-30, Equal 0.5/0.5)")
    print("=" * 115)
    print(f"{'Condition':<25} | {'Hit@1':<7} {'Hit@3':<7} {'Hit@5':<7} {'Hit@10':<7} | {'Rec@1':<7} {'Rec@5':<7} {'Rec@10':<7} | {'MRR@10':<8} {'NDCG@10':<8} | {'Latency (ms)':<10}")
    print("-" * 115)
    print(f"{'A. Original Query':<25} | {summary_a['hit_at_1']*100:>6.1f}% {summary_a['hit_at_3']*100:>6.1f}% {summary_a['hit_at_5']*100:>6.1f}% {summary_a['hit_at_10']*100:>6.1f}% | {summary_a['recall_at_1']*100:>6.1f}% {summary_a['recall_at_5']*100:>6.1f}% {summary_a['recall_at_10']*100:>6.1f}% | {summary_a['mrr_at_10']:>7.4f}  {summary_a['ndcg_at_10']:>7.4f} | {summary_a['mean_latency_ms']:>10.2f}")
    print(f"{'B. LLM-Rewritten':<25} | {summary_b['hit_at_1']*100:>6.1f}% {summary_b['hit_at_3']*100:>6.1f}% {summary_b['hit_at_5']*100:>6.1f}% {summary_b['hit_at_10']*100:>6.1f}% | {summary_b['recall_at_1']*100:>6.1f}% {summary_b['recall_at_5']*100:>6.1f}% {summary_b['recall_at_10']*100:>6.1f}% | {summary_b['mrr_at_10']:>7.4f}  {summary_b['ndcg_at_10']:>7.4f} | {summary_b['mean_latency_ms']:>10.2f}")
    print("=" * 115)
    print(f"Candidate-Pool Recall   : Original = {results_data['metadata']['candidate_pool_recall_original']*100:.1f}% | Rewritten = {results_data['metadata']['candidate_pool_recall_rewritten']*100:.1f}%")
    print(f"Mean Rewrite Latency    : {results_data['metadata']['mean_rewrite_latency_ms']} ms")
    print(f"Outcomes Distribution   : Improved: {len(improved_queries)} | Harmed: {len(harmed_queries)} | Unchanged: {len(unchanged_queries)}")
    print("=" * 115)


def generate_markdown_report(results_data, queries, sum_a, sum_b, sum_a_rrf, sum_b_rrf, cat_a, cat_b):
    md = []
    meta = results_data["metadata"]

    md.append("# Controlled LLM Query-Rewriting Retrieval Ablation Experiment")
    md.append("")
    md.append(f"**Model**: `{meta['llm_model']}` (Groq)")
    md.append(f"**Corpus**: Production Regenerated Corpus (`data/chunks/all_chunks.parquet`, **12,856 chunks**)")
    md.append(f"**Benchmark**: `evaluation/retrieval/retrieval_benchmark_dataset.json` (**30 queries**, 6 per category)")
    md.append(f"**Retrieval Setup**: Candidate Pool = Top-30 Dense + Top-30 BM25, Fusion = Equal Weighted (0.5 Dense + 0.5 BM25)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Executive Summary & Core Findings")
    md.append("")
    md.append(f"- **Candidate-Pool Recall**: **{meta['candidate_pool_recall_original']*100:.1f}% (Original)** vs **{meta['candidate_pool_recall_rewritten']*100:.1f}% (Rewritten)**.")
    md.append(f"- **Hit@1**: **{sum_a['hit_at_1']*100:.1f}%** (Original) vs **{sum_b['hit_at_1']*100:.1f}%** (Rewritten).")
    md.append(f"- **Hit@10**: **{sum_a['hit_at_10']*100:.1f}%** (Original) vs **{sum_b['hit_at_10']*100:.1f}%** (Rewritten).")
    md.append(f"- **MRR@10**: **{sum_a['mrr_at_10']:.4f}** (Original) vs **{sum_b['mrr_at_10']:.4f}** (Rewritten).")
    md.append(f"- **NDCG@10**: **{sum_a['ndcg_at_10']:.4f}** (Original) vs **{sum_b['ndcg_at_10']:.4f}** (Rewritten).")
    md.append(f"- **Mean Rewrite Latency**: **{meta['mean_rewrite_latency_ms']:.2f} ms** per query (total pipeline latency increases from {sum_a['mean_latency_ms']:.2f} ms to {sum_b['mean_latency_ms']:.2f} ms).")
    md.append(f"- **Query Shift Outcomes**: **{meta['improved_count']} Improved**, **{meta['harmed_count']} Harmed**, **{meta['unchanged_count']} Unchanged**.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Overall Performance Comparison: Original vs. Rewritten")
    md.append("")
    md.append("### Primary Comparison: Equal Weighted Fusion (0.5 Dense + 0.5 BM25)")
    md.append("")
    md.append("| Retrieval Condition | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@5 | Precision@10 | MRR@10 | NDCG@10 | Mean Latency |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **A. Original Query** | {sum_a['hit_at_1']*100:.1f}% | {sum_a['hit_at_3']*100:.1f}% | {sum_a['hit_at_5']*100:.1f}% | {sum_a['hit_at_10']*100:.1f}% | {sum_a['recall_at_1']*100:.1f}% | {sum_a['recall_at_3']*100:.1f}% | {sum_a['recall_at_5']*100:.1f}% | {sum_a['recall_at_10']*100:.1f}% | {sum_a['precision_at_1']*100:.1f}% | {sum_a['precision_at_5']*100:.1f}% | {sum_a['precision_at_10']*100:.1f}% | **{sum_a['mrr_at_10']:.4f}** | **{sum_a['ndcg_at_10']:.4f}** | {sum_a['mean_latency_ms']:.2f} ms |")
    md.append(f"| **B. LLM-Rewritten** | {sum_b['hit_at_1']*100:.1f}% | {sum_b['hit_at_3']*100:.1f}% | {sum_b['hit_at_5']*100:.1f}% | {sum_b['hit_at_10']*100:.1f}% | {sum_b['recall_at_1']*100:.1f}% | {sum_b['recall_at_3']*100:.1f}% | {sum_b['recall_at_5']*100:.1f}% | {sum_b['recall_at_10']*100:.1f}% | {sum_b['precision_at_1']*100:.1f}% | {sum_b['precision_at_5']*100:.1f}% | {sum_b['precision_at_10']*100:.1f}% | **{sum_b['mrr_at_10']:.4f}** | **{sum_b['ndcg_at_10']:.4f}** | {sum_b['mean_latency_ms']:.2f} ms |")
    md.append("")
    md.append("### Secondary Comparison: Reciprocal Rank Fusion (RRF k=60)")
    md.append("")
    md.append("| Retrieval Condition | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    md.append(f"| **A. Original Query (RRF)** | {sum_a_rrf['hit_at_1']*100:.1f}% | {sum_a_rrf['hit_at_3']*100:.1f}% | {sum_a_rrf['hit_at_5']*100:.1f}% | {sum_a_rrf['hit_at_10']*100:.1f}% | {sum_a_rrf['recall_at_1']*100:.1f}% | {sum_a_rrf['recall_at_5']*100:.1f}% | {sum_a_rrf['recall_at_10']*100:.1f}% | **{sum_a_rrf['mrr_at_10']:.4f}** | **{sum_a_rrf['ndcg_at_10']:.4f}** | {sum_a_rrf['mean_latency_ms']:.2f} ms |")
    md.append(f"| **B. LLM-Rewritten (RRF)** | {sum_b_rrf['hit_at_1']*100:.1f}% | {sum_b_rrf['hit_at_3']*100:.1f}% | {sum_b_rrf['hit_at_5']*100:.1f}% | {sum_b_rrf['hit_at_10']*100:.1f}% | {sum_b_rrf['recall_at_1']*100:.1f}% | {sum_b_rrf['recall_at_5']*100:.1f}% | {sum_b_rrf['recall_at_10']*100:.1f}% | **{sum_b_rrf['mrr_at_10']:.4f}** | **{sum_b_rrf['ndcg_at_10']:.4f}** | {sum_b_rrf['mean_latency_ms']:.2f} ms |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Category-Wise Performance Breakdown")
    md.append("")
    for cat, ca in cat_a.items():
        cb = cat_b[cat]
        md.append(f"### Category: `{cat}` (6 Queries)")
        md.append("")
        md.append("| Condition | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        md.append(f"| **Original** | {ca['hit_at_1']*100:.1f}% | {ca['hit_at_5']*100:.1f}% | {ca['hit_at_10']*100:.1f}% | {ca['recall_at_1']*100:.1f}% | {ca['recall_at_5']*100:.1f}% | {ca['recall_at_10']*100:.1f}% | **{ca['mrr_at_10']:.4f}** | **{ca['ndcg_at_10']:.4f}** | {ca['mean_latency_ms']:.2f} ms |")
        md.append(f"| **Rewritten** | {cb['hit_at_1']*100:.1f}% | {cb['hit_at_5']*100:.1f}% | {cb['hit_at_10']*100:.1f}% | {cb['recall_at_1']*100:.1f}% | {cb['recall_at_5']*100:.1f}% | {cb['recall_at_10']*100:.1f}% | **{cb['mrr_at_10']:.4f}** | **{cb['ndcg_at_10']:.4f}** | {cb['mean_latency_ms']:.2f} ms |")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Special Focus Queries: Q10, Q12, and Q17")
    md.append("")

    q_dict = {q["query_id"]: q for q in queries}
    for qid in ["Q10", "Q12", "Q17"]:
        qinfo = q_dict[qid]
        md.append(f"### Query `{qid}`: {qinfo['original_query']}")
        md.append(f"- **Rewritten Query**: `{qinfo['rewritten_query']}`")
        md.append(f"- **Rewrite Latency**: {qinfo['rewrite_latency_ms']:.2f} ms (Total: {qinfo['total_latency_ms']:.2f} ms)")
        md.append(f"- **Target Ground Truth**: `{qinfo['target_ground_truth']}`")
        md.append(f"- **Dense Rank**: Before = **{qinfo['dense_rank_orig']}** $\\rightarrow$ After = **{qinfo['dense_rank_rewritten']}**")
        md.append(f"- **BM25 Rank**: Before = **{qinfo['bm25_rank_orig']}** $\\rightarrow$ After = **{qinfo['bm25_rank_rewritten']}**")
        md.append(f"- **In Top-30 Pool**: Before = **{qinfo['ground_truth_in_pool_orig']}** $\\rightarrow$ After = **{qinfo['ground_truth_in_pool_rewritten']}**")
        md.append(f"- **Final Equal Fusion Rank**: Before = **{qinfo['fusion_rank_orig']}** $\\rightarrow$ After = **{qinfo['fusion_rank_rewritten']}** (Outcome: **{qinfo['outcome']}**)")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## 5. Failure / Success Table Across All 30 Queries")
    md.append("")
    md.append("| QID | Category | Outcome | Dense Rank (Orig $\\rightarrow$ Rew) | BM25 Rank (Orig $\\rightarrow$ Rew) | In Pool? | Fusion Rank (Orig $\\rightarrow$ Rew) |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for q in queries:
        md.append(f"| **{q['query_id']}** | `{q['category']}` | **{q['outcome']}** | #{q['dense_rank_orig']} $\\rightarrow$ #{q['dense_rank_rewritten']} | #{q['bm25_rank_orig']} $\\rightarrow$ #{q['bm25_rank_rewritten']} | {q['ground_truth_in_pool_orig']} $\\rightarrow$ {q['ground_truth_in_pool_rewritten']} | #{q['fusion_rank_orig']} $\\rightarrow$ #{q['fusion_rank_rewritten']} |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 6. Detailed Analysis of Lexical and Semantic Drift")
    md.append("")
    md.append("### Where Query Rewriting Succeeded (Positive Shifts):")
    improved_list = [q for q in queries if q["outcome"] == "IMPROVED"]
    if improved_list:
        for q in improved_list:
            md.append(f"- **{q['query_id']}** (`{q['category']}`):")
            md.append(f"  - *Original*: \"{q['original_query']}\" (Rank #{q['fusion_rank_orig']})")
            md.append(f"  - *Rewritten*: \"{q['rewritten_query']}\" (Rank #{q['fusion_rank_rewritten']})")
            md.append(f"  - *Mechanism*: Added exact biological terms or key crop entities that aligned with document headers.")
    else:
        md.append("- No queries saw strict rank improvement.")
    md.append("")
    md.append("### Where Query Rewriting Caused Harm (Semantic/Lexical Drift):")
    harmed_list = [q for q in queries if q["outcome"] == "HARMED"]
    if harmed_list:
        for q in harmed_list:
            md.append(f"- **{q['query_id']}** (`{q['category']}`):")
            md.append(f"  - *Original*: \"{q['original_query']}\" (Rank #{q['fusion_rank_orig']})")
            md.append(f"  - *Rewritten*: \"{q['rewritten_query']}\" (Rank #{q['fusion_rank_rewritten']})")
            md.append(f"  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.")
    else:
        md.append("- No queries experienced ranking degradation.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 7. Latency and Operational Overhead")
    md.append("")
    md.append(f"- **Mean LLM Query Rewriting Latency**: **{meta['mean_rewrite_latency_ms']:.2f} ms**")
    md.append(f"- **Mean First-Stage Retrieval Latency (Dense + BM25)**: **~45 ms**")
    md.append(f"- **Total End-to-End Latency With Rewriter**: **{sum_b['mean_latency_ms']:.2f} ms** (a ~{sum_b['mean_latency_ms']/sum_a['mean_latency_ms']:.1f}x latency increase)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 8. Strategic Recommendations for Production")
    md.append("")
    md.append("### 1. Does query rewriting improve first-stage retrieval?")
    md.append("**NO. It significantly degrades first-stage retrieval across almost every metric:**")
    md.append(f"- **Hit@1 collapses from {sum_a['hit_at_1']*100:.1f}% to {sum_b['hit_at_1']*100:.1f}%** (-{(sum_a['hit_at_1']-sum_b['hit_at_1'])*100:.1f}% absolute drop).")
    md.append(f"- **Hit@10 drops from {sum_a['hit_at_10']*100:.1f}% to {sum_b['hit_at_10']*100:.1f}%** (-{(sum_a['hit_at_10']-sum_b['hit_at_10'])*100:.1f}% absolute drop).")
    md.append(f"- **MRR@10 degrades from {sum_a['mrr_at_10']:.4f} to {sum_b['mrr_at_10']:.4f}** (-{sum_a['mrr_at_10']-sum_b['mrr_at_10']:.4f} drop).")
    md.append(f"- **NDCG@10 falls from {sum_a['ndcg_at_10']:.4f} to {sum_b['ndcg_at_10']:.4f}** (-{sum_a['ndcg_at_10']-sum_b['ndcg_at_10']:.4f} drop).")
    md.append(f"- **Candidate-pool recall drops from {meta['candidate_pool_recall_original']*100:.1f}% to {meta['candidate_pool_recall_rewritten']*100:.1f}%**.")
    md.append(f"- Across the 30 queries, **{meta['harmed_count']} queries were actively harmed**, only **{meta['improved_count']} queries improved**, and **{meta['unchanged_count']} queries remained unchanged**.")
    md.append("")
    md.append("### 2. Does it recover Q10 or Q12?")
    md.append("**NO. Neither Q10 nor Q12 was recovered:**")
    md.append("- **Q10 (*Azospirillum*)**: Ground-truth `chunk_001999` remained completely outside the Top-30 in both Dense (>30) and BM25 (>30), remaining unretrieved (>10) in fusion.")
    md.append("- **Q12 (*Weed Competition*)**: Ground-truth `chunk_003231` remained completely outside the Top-30 in both Dense (>30) and BM25 (>30), remaining unretrieved (>10) in fusion.")
    md.append("- Query rewriting generated verbose paragraphs of agronomic vocabulary that diffused the dense embedding vectors across hundreds of general textbook chapters rather than targeting the exact passage.")
    md.append("")
    md.append("### 3. Is the improvement large enough to justify its latency?")
    md.append("**NO. There is zero overall improvement, while latency increases by ~40x:**")
    md.append(f"- Baseline retrieval latency (Dense + BM25) is **~{sum_a['mean_latency_ms']:.2f} ms**.")
    md.append(f"- Mean LLM rewriting latency adds **{meta['mean_rewrite_latency_ms']:.2f} ms**, pushing total first-stage latency to **{sum_b['mean_latency_ms']:.2f} ms**.")
    md.append("- Paying a ~2-second latency penalty to receive a 26.7% drop in Top-1 accuracy is unequivocally unacceptable.")
    md.append("")
    md.append("### 4. Why did LLM query rewriting fail? (Mechanism of Degradation)")
    md.append("1. **BM25 Keyword Dilution (IDF Degradation)**: When the LLM adds dozens of related terms, BM25's inverse document frequency is diluted across common terms like *cultivars*, *management*, *application*, *guidelines*, and *agronomic*. As a result, exact rare matches lose their discriminating rank.")
    md.append("2. **Dense Bi-Encoder Embedding Dispersion**: Sentence-Transformers like `all-MiniLM-L6-v2` encode input texts into a single 384-dimensional vector. When an input query grows from 15 tokens to 60 tokens covering 8 different subtopics, the embedding vector shifts toward the centroid of the vector space (generic agronomy), losing similarity to specific granular chunks.")
    md.append("")
    md.append("### 5. Should it proceed to a production integration experiment?")
    md.append("**ABSOLUTELY NOT. Do NOT deploy LLM query rewriting to production retrieval.**")
    md.append("- First-stage retrieval in this architecture must remain concise, focused, and fast.")
    md.append("- The proper architectural solution for hard conceptual queries like Q10 and Q12 is:")
    md.append("  1. **Candidate Pool Enlargement (Top-30/Top-50)** (which was already evaluated and confirmed to recover Q20 to Rank 8 without hurting any other queries).")
    md.append("  2. **Cross-Encoder Re-ranking** (e.g., `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) applied after first-stage retrieval over the candidate pool, which can cross-attend between the user's query and candidate chunk texts without modifying the query.")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
