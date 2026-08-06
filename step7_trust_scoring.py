# step7_trust_scoring.py
# =============================================
# Layer 4: Trust & Bias Estimation
#
# Part 1 — RA-RAG Re-ranking
#   Uses authority scores from metadata
#   Re-ranks chunks by credibility
#   MODEL: none (pure math)
#
# Part 2 — LCR (MSCP Confidence Scoring)
#   Samples LLM response multiple times
#   Checks answer consistency
#   MODEL: llama-3.3-70b-versatile
#   WHY: Fast sampling, good enough for
#        consistency checks. Save gpt-oss-120b
#        for actual answer generation.
#
# Part 3 — Trust-Weighted Final Selection
#   Combines all scores
#   Returns only trusted chunks
#   MODEL: none (pure math)
# =============================================

import os
import re
import time
import math
import pickle
import numpy as np
import chromadb
from groq import Groq
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    GROQ_API_KEY,
    GROQ_GATE_MODEL,
    GROQ_ANSWER_MODEL,
    VECTOR_STORE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GRAPH_DIR,
    LCR_SAMPLE_COUNT,
    LCR_MSCP_THRESHOLD,
    LCR_MIN_CHUNK_SCORE,
    TRUST_FINAL_THRESHOLD,
)

# Lazy initialize Groq client (will be created in main block)
client = None


# ─────────────────────────────────────────────
# PART 1: RA-RAG RE-RANKING
# Re-orders chunks by authority + bias scores
# MODEL: None — pure mathematical re-ranking
# ─────────────────────────────────────────────

def rerank_by_authority(chunks):
    """
    Re-rank chunks using RA-RAG trust formula:

    trust_weight = authority_score × (1 - bias_penalty)
    final_score  = retrieval_score × trust_weight

    Authority scores (from your config.py):
    Government/University PDFs = 1.0 (all your 6 PDFs)
    News sites = 0.6
    Blogs = 0.3

    Since all your PDFs are from top Indian
    agricultural universities, trust_weight = 1.0
    for everything. This matters more when you
    add diverse sources later.
    """
    reranked = []

    for chunk in chunks:
        # Get existing scores
        retrieval_score = chunk.get("final_score", 0.0)
        authority_score = float(
            chunk.get("authority_score",
            chunk.get("trust_weight", 1.0))
        )
        bias_penalty    = float(
            chunk.get("bias_penalty", 0.0)
        )

        # RA-RAG formula
        trust_weight  = authority_score * (1 - bias_penalty)
        ra_rag_score  = round(retrieval_score * trust_weight, 4)

        reranked.append({
            **chunk,
            "authority_score": authority_score,
            "bias_penalty"   : bias_penalty,
            "trust_weight"   : trust_weight,
            "ra_rag_score"   : ra_rag_score,
        })

    # Sort by RA-RAG score
    reranked.sort(key=lambda x: x["ra_rag_score"], reverse=True)

    return reranked


# ─────────────────────────────────────────────
# PART 2: LCR — MSCP CONFIDENCE SCORING
# Checks if chunk content gives consistent answers
#
# MODEL: llama-3.3-70b-versatile
# WHY: Need fast repeated sampling (5x per chunk)
#      llama-3.3-70b is fast enough and consistent
#      gpt-oss-120b would be too slow and costly
#      for 5 × N chunks
# ─────────────────────────────────────────────

def sample_answer(query, chunk_text, temperature=0.7):
    """
    Get one sampled answer for a query+chunk pair.
    Called multiple times to check consistency.

    MODEL: llama-3.3-70b-versatile
    Temperature: HIGH (0.7) to get varied responses
    This variation reveals if the chunk is consistent
    """
    try:
        response = client.chat.completions.create(
            model    = GROQ_GATE_MODEL,   # llama-3.3-70b
            messages = [
                {
                    "role"   : "system",
                    "content": "Answer the agricultural "
                               "question using ONLY the "
                               "provided context. "
                               "Be concise — 1-2 sentences."
                },
                {
                    "role"   : "user",
                    "content": f"Context: {chunk_text[:600]}"
                               f"\n\nQuestion: {query}"
                }
            ],
            temperature = temperature,
            max_tokens  = 80,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return None


def calculate_mscp(sampled_answers, embedder):
    """
    Calculate Maximum Semantic Cluster Proportion.

    Steps:
    1. Embed all sampled answers
    2. Run K-Means clustering (K=3)
    3. Find largest cluster
    4. MSCP = largest cluster size / total answers

    MSCP > 0.6 → answers agree → chunk is TRUSTWORTHY
    MSCP < 0.6 → answers scattered → chunk UNRELIABLE

    Example:
    5 answers sampled:
    Cluster 1: [ans1, ans2, ans3, ans4] → 4 answers
    Cluster 2: [ans5]                   → 1 answer
    MSCP = 4/5 = 0.8 → TRUSTWORTHY ✅

    Another example:
    Cluster 1: [ans1, ans2] → 2 answers
    Cluster 2: [ans3, ans4] → 2 answers
    Cluster 3: [ans5]       → 1 answer
    MSCP = 2/5 = 0.4 → UNRELIABLE ❌
    """
    # Filter None answers
    valid = [a for a in sampled_answers if a]

    if len(valid) < 2:
        # Not enough samples — assume trustworthy
        return 1.0

    # Embed all answers
    embeddings = embedder.encode(valid)

    # Handle case where we have fewer answers than clusters
    n_clusters = min(3, len(valid))

    if n_clusters < 2:
        return 1.0

    # K-Means clustering
    kmeans  = KMeans(
        n_clusters = n_clusters,
        random_state= 42,
        n_init     = 10
    )
    labels  = kmeans.fit_predict(embeddings)

    # Count cluster sizes
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = \
            cluster_counts.get(label, 0) + 1

    # MSCP = largest cluster / total
    largest_cluster = max(cluster_counts.values())
    mscp = round(largest_cluster / len(valid), 4)

    return mscp


def lcr_score_chunk(query, chunk, embedder):
    """
    Run LCR scoring on a single chunk.

    Process:
    1. Sample LCR_SAMPLE_COUNT answers (default: 5)
    2. Embed all answers
    3. Cluster to find agreement
    4. Calculate MSCP score

    Returns trust assessment for this chunk.
    """
    chunk_text = chunk["text"]
    chunk_id   = chunk["chunk_id"]

    # Only score chunks above minimum retrieval score
    if chunk.get("ra_rag_score", 0) < LCR_MIN_CHUNK_SCORE:
        return {
            "chunk_id"      : chunk_id,
            "mscp_score"    : 1.0,
            "trusted"       : True,
            "skipped"       : True,
            "reason"        : "below_min_score_skipped",
            "sampled_answers": [],
        }

    # Sample multiple answers with high temperature
    sampled_answers = []
    for i in range(LCR_SAMPLE_COUNT):
        answer = sample_answer(query, chunk_text,
                               temperature=0.7)
        if answer:
            sampled_answers.append(answer)
        time.sleep(0.2)   # gentle rate limiting

    if not sampled_answers:
        return {
            "chunk_id"      : chunk_id,
            "mscp_score"    : 0.5,
            "trusted"       : True,
            "skipped"       : False,
            "reason"        : "no_samples_returned",
            "sampled_answers": [],
        }

    # Calculate MSCP
    mscp    = calculate_mscp(sampled_answers, embedder)
    trusted = mscp >= LCR_MSCP_THRESHOLD
    
    # Debug: show MSCP threshold check
    if not trusted:
        print(f"     [debug] mscp={mscp} threshold="
              f"{LCR_MSCP_THRESHOLD} trusted={trusted}")

    return {
        "chunk_id"       : chunk_id,
        "mscp_score"     : mscp,
        "trusted"        : trusted,
        "skipped"        : False,
        "reason"         : (
            "consistent" if trusted
            else "inconsistent_answers"
        ),
        "sampled_answers": sampled_answers,
    }


# ─────────────────────────────────────────────
# PART 3: TRUST-WEIGHTED FINAL SELECTION
# Combines RA-RAG + LCR into final score
# MODEL: None — pure math
# ─────────────────────────────────────────────

def compute_final_trust_score(chunk, lcr_result):
    """
    Combine all trust signals into one final score.

    Formula:
    final_trust = ra_rag_score
                × mscp_score
                × authority_weight

    Where:
    ra_rag_score    = retrieval × authority × (1-bias)
    mscp_score      = answer consistency (0-1)
    authority_weight= source credibility (0-1)
    """
    ra_rag_score     = chunk.get("ra_rag_score", 0.0)
    mscp_score       = lcr_result.get("mscp_score", 1.0)
    authority_weight = chunk.get("authority_score", 1.0)

    final_trust = round(
        ra_rag_score * mscp_score * authority_weight,
        4
    )

    return {
        **chunk,
        "mscp_score"     : mscp_score,
        "lcr_trusted"    : lcr_result["trusted"],
        "lcr_reason"     : lcr_result["reason"],
        "final_trust"    : final_trust,
        "include"        : (
            lcr_result["trusted"] and
            final_trust >= TRUST_FINAL_THRESHOLD
        ),
    }


# ─────────────────────────────────────────────
# MAIN: FULL TRUST SCORING PIPELINE
# ─────────────────────────────────────────────

def run_trust_scoring(query, chunks, embedder,
                      run_lcr=True):
    """
    Complete trust scoring pipeline.

    Args:
        query    : user's question
        chunks   : retrieved chunks from hybrid search
        embedder : sentence-transformers model
        run_lcr  : if False, skip LCR (use for fast path)

    Returns:
        trusted_chunks : list of trusted chunks
        trust_report   : detailed scoring breakdown
    """
    print(f"\n{'─'*55}")
    print(f"🔒 TRUST SCORING PIPELINE")
    print(f"{'─'*55}")
    print(f"   Chunks to evaluate : {len(chunks)}")
    print(f"   Run LCR            : {run_lcr}")

    # ── Step 1: RA-RAG Re-ranking ──
    print(f"\n  📊 Step 1: RA-RAG Re-ranking")
    print(f"     Model: None (authority math)")
    reranked = rerank_by_authority(chunks)

    for i, c in enumerate(reranked):
        print(f"     [{i+1}] {c['source_file'][:35]}"
              f" | authority={c['authority_score']}"
              f" | ra_rag={c['ra_rag_score']}")

    # ── Step 2: LCR Scoring ──
    print(f"\n  🧪 Step 2: LCR Consistency Scoring")

    lcr_results = {}

    if run_lcr:
        print(f"     Model: {GROQ_GATE_MODEL}")
        print(f"     Samples per chunk: {LCR_SAMPLE_COUNT}")
        print(f"     MSCP threshold: {LCR_MSCP_THRESHOLD}")

        for i, chunk in enumerate(reranked):
            cid = chunk["chunk_id"]
            print(f"\n     Chunk {i+1}/{len(reranked)}: "
                  f"{chunk['source_file'][:30]}...")

            lcr = lcr_score_chunk(query, chunk, embedder)
            lcr_results[cid] = lcr

            status = "✅ TRUSTED" if lcr["trusted"] \
                     else "❌ FLAGGED"
            skip   = "(skipped)" if lcr["skipped"] else ""
            print(f"     MSCP: {lcr['mscp_score']} "
                  f"→ {status} {skip}")

            if not lcr["skipped"] and \
               lcr.get("sampled_answers"):
                print(f"     Sample answers:")
                for j, ans in enumerate(
                    lcr["sampled_answers"][:2]
                ):
                    print(f"       [{j+1}] {ans[:70]}...")
    else:
        print(f"     Skipped (fast path)")
        # Default: trust all chunks
        for chunk in reranked:
            lcr_results[chunk["chunk_id"]] = {
                "mscp_score": 1.0,
                "trusted"   : True,
                "skipped"   : True,
                "reason"    : "lcr_skipped_fast_path",
                "sampled_answers": [],
            }

    # ── Step 3: Final Trust Score ──
    print(f"\n  🏆 Step 3: Final Trust Scores")
    print(f"     Model: None (weighted formula)")

    scored_chunks = []
    for chunk in reranked:
        cid = chunk["chunk_id"]
        lcr = lcr_results.get(cid, {
            "mscp_score": 1.0,
            "trusted"   : True,
            "reason"    : "no_lcr",
            "sampled_answers": [],
        })
        scored = compute_final_trust_score(chunk, lcr)
        scored_chunks.append(scored)

    # Display final scores
    print(f"\n     {'Source':<35} "
          f"{'MSCP':<6} "
          f"{'Trust':<7} "
          f"{'Include'}")
    print(f"     {'─'*35} {'─'*6} {'─'*7} {'─'*7}")

    trusted_chunks = []
    excluded       = []

    for c in scored_chunks:
        include = c["include"]
        icon    = "✅" if include else "❌"
        print(f"     {c['source_file'][:35]:<35} "
              f"{c['mscp_score']:<6} "
              f"{c['final_trust']:<7} "
              f"{icon}")

        if include:
            trusted_chunks.append(c)
        else:
            excluded.append(c)

    print(f"\n     Trusted : {len(trusted_chunks)}")
    print(f"     Excluded: {len(excluded)}")

    # ── Fallback: if everything excluded ──
    if len(trusted_chunks) == 0 and len(scored_chunks) > 0:
        print(f"\n  ⚠️  All chunks excluded — "
              f"using top scored chunk as fallback")
        # Sort by final_trust and take the best one
        best = sorted(
            scored_chunks,
            key    = lambda x: x["final_trust"],
            reverse= True
        )[0]
        best["include"]    = True
        best["lcr_reason"] = "fallback_best_available"
        trusted_chunks     = [best]
        print(f"     Fallback chunk: "
              f"{best['source_file'][:35]}"
              f" (trust={best['final_trust']})")

    # Trust report
    trust_report = {
        "total_chunks"   : len(chunks),
        "trusted_count"  : len(trusted_chunks),
        "excluded_count" : len(excluded),
        "lcr_ran"        : run_lcr,
        "scored_chunks"  : scored_chunks,
    }

    return trusted_chunks, trust_report


# ─────────────────────────────────────────────
# INTEGRATED FUNCTION
# Call this from your query gate
# ─────────────────────────────────────────────

def trust_score_and_filter(query, chunks,
                            embedder, path="slow"):
    """
    Main entry point for trust scoring.

    FAST PATH → skip LCR (too slow for fast queries)
    SLOW PATH → run full LCR + RA-RAG

    This is what query_gate.py will call.
    """
    run_lcr = (path == "slow")
    return run_trust_scoring(
        query, chunks, embedder, run_lcr=run_lcr
    )


# ─────────────────────────────────────────────
# TEST TRUST SCORING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("⏳ Loading components...", flush=True)
    sys.stdout.flush()
    
    from sentence_transformers import SentenceTransformer
    
    # Initialize Groq client NOW (before it's used)
    sys.modules[__name__].client = Groq(api_key=GROQ_API_KEY)

    try:
        embedder   = SentenceTransformer(EMBEDDING_MODEL)
        chroma_cli = chromadb.PersistentClient(
            path=VECTOR_STORE
        )
        collection = chroma_cli.get_collection(
            COLLECTION_NAME
        )

        bm25_path   = os.path.join(GRAPH_DIR,
                                   "bm25_index.pkl")
        corpus_path = os.path.join(GRAPH_DIR,
                                   "bm25_corpus.pkl")

        with open(bm25_path,   "rb") as f:
            bm25   = pickle.load(f)
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        print("✅ Ready!\n", flush=True)
        sys.stdout.flush()

        # ── Import retrieval from step6 ──
        from step6_query_gate import retrieve_chunks

        # ── Test 1: Complex query (full LCR) ──
        print("="*55)
        print("TEST 1: Complex query — full LCR scoring")
        print("="*55)

        query1  = ("What organic fertilizer should I use "
                   "for coconut trees?")
        chunks1 = retrieve_chunks(
            query1, embedder, collection, bm25, corpus,
            top_k=4
        )

        trusted1, report1 = run_trust_scoring(
            query1, chunks1, embedder, run_lcr=True
        )

        print(f"\n✅ Trusted chunks for answer: "
              f"{len(trusted1)}/{len(chunks1)}")

        # ── Test 2: Simple query (skip LCR) ──
        print("\n" + "="*55)
        print("TEST 2: Simple query — LCR skipped")
        print("="*55)

        query2  = "What is vermicompost?"
        chunks2 = retrieve_chunks(
            query2, embedder, collection, bm25, corpus,
            top_k=4
        )

        trusted2, report2 = run_trust_scoring(
            query2, chunks2, embedder, run_lcr=False
        )

        print(f"\n✅ Trusted chunks: {len(trusted2)}")

        # ── Summary ──
        print(f"\n{'='*55}")
        print(f"  📊 TRUST SCORING SUMMARY")
        print(f"{'='*55}")
        print(f"  Test 1 (complex + LCR):")
        print(f"    Input chunks  : {report1['total_chunks']}")
        print(f"    Trusted chunks: {report1['trusted_count']}")
        print(f"    Excluded      : {report1['excluded_count']}")
        print(f"    LCR ran       : {report1['lcr_ran']}")
        print(f"\n  Test 2 (simple, no LCR):")
        print(f"    Input chunks  : {report2['total_chunks']}")
        print(f"    Trusted chunks: {report2['trusted_count']}")
        print(f"    LCR ran       : {report2['lcr_ran']}")
        print(f"{'='*55}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)