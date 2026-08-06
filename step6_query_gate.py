# step6_query_gate.py
# =============================================
# Layer 2: Query Gate (QT)
#
# MODEL USAGE:
# llama-3.3-70b-versatile →
#     - Query classification
#     - Entropy measurement
#     - Fast path answers (simple queries)
#
# openai/gpt-oss-120b →
#     - Complex query answers
#     - Diagnostic reasoning
#     - Recommendations
#     - All slow path operations
# =============================================

import os
import math
import json
import time
import pickle
import sys
import chromadb
from groq import Groq
# Lazy load SentenceTransformer to avoid initialization hang

from config import (
    GROQ_API_KEY,
    GROQ_GATE_MODEL,
    GROQ_ANSWER_MODEL,
    QT_ENTROPY_THRESHOLD,
    QT_LOGPROB_TOKENS,
    QT_MIN_CONFIDENCE,
    VECTOR_STORE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GRAPH_DIR,
)

# Adaptive entropy
try:
    from adaptive_entropy import get_current_threshold, record_feedback
    ADAPTIVE_ENTROPY_AVAILABLE = True
except Exception:
    ADAPTIVE_ENTROPY_AVAILABLE = False

# Weather enrichment for recommendations
try:
    from step8_weather_rag import enrich_with_weather, is_weather_query
    WEATHER_ENRICHMENT_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Weather enrichment not available: {e}")
    WEATHER_ENRICHMENT_AVAILABLE = False

# ── Single Groq client (used for both models) ──
client = Groq(api_key=GROQ_API_KEY)
groq_client = client


# ─────────────────────────────────────────────
# PART 1: ENTROPY CALCULATION
# Uses: llama-3.3-70b-versatile
# Why: needs logprobs support + fast response
# ─────────────────────────────────────────────

def calculate_entropy(token_logprobs):
    """
    Calculate Shannon entropy from log probabilities.

    What entropy means:
    - LOW entropy  = model is CONFIDENT = fast path
    - HIGH entropy = model is UNCERTAIN = slow path

    Example:
      "What is NPK?" → tokens very predictable
      → low entropy → fast path ⚡

      "My crop has spots and yellowing at edges" 
      → many possible answers
      → high entropy → slow path 🔍
    """
    if not token_logprobs:
        return 0.0

    entropy = 0.0
    for logprob in token_logprobs:
        prob = math.exp(logprob)      # ln(p) → p
        if prob > 0:
            entropy -= prob * math.log2(prob)

    return round(entropy / len(token_logprobs), 4)


def measure_query_entropy(query):
    """
    Send query to llama-3.3-70b-versatile.
    Capture logprobs of first N tokens.
    Calculate entropy from those logprobs.

    MODEL: llama-3.3-70b-versatile
    REASON: Fast + supports logprobs parameter
    """
    try:
        response = client.chat.completions.create(
            model      = GROQ_GATE_MODEL,   # llama-3.3-70b
            messages   = [
                {
                    "role"   : "system",
                    "content": """You are an expert agricultural 
                    assistant for Indian farmers. Answer questions 
                    about crops, pests, diseases, and farming 
                    practices clearly and concisely."""
                },
                {
                    "role"   : "user",
                    "content": query
                }
            ],
            max_tokens  = QT_LOGPROB_TOKENS,
            temperature = 0,
        )

        # Since logprobs is not supported, use response length 
        # as entropy proxy: shorter response = more confident
        response_text = response.choices[0].message.content
        response_length = len(response_text.split())
        
        # Normalize: typical factual answer = 10-20 words (low entropy)
        # Uncertain answer = 30+ words (high entropy)
        if response_length < 15:
            entropy = 0.8  # Very confident
        elif response_length < 25:
            entropy = 1.2  # Moderately confident
        else:
            entropy = 2.0  # Less confident, more explanation needed

        return {
            "entropy"       : entropy,
            "token_logprobs": [],
            "first_tokens"  : response_text.split()[:5],
            "error"         : None
        }
    

    except Exception as e:
        print(f"   ⚠️  Entropy measure error: {e}")
        return {
            "entropy"       : QT_ENTROPY_THRESHOLD + 0.5,
            "token_logprobs": [],
            "first_tokens"  : [],
            "error"         : str(e)
        }


# ─────────────────────────────────────────────
# PART 2: QUERY CLASSIFICATION
# Uses: llama-3.3-70b-versatile
# Why: Fast instruction-following, JSON output
# ─────────────────────────────────────────────

def classify_query(query):
    """
    Classify the query into one of 6 types.
    Determines which pipeline to use downstream.

    MODEL: llama-3.3-70b-versatile
    REASON: Fast and accurate at classification.
            Saves gpt-oss-120b quota for answers.

    Returns:
        dict with query_type, confidence, topics
    """
    prompt = f"""Classify this agricultural question into 
exactly ONE category.

Categories:
FACTUAL       → simple definitions, basic facts
               e.g. "What is vermicompost?"
DIAGNOSTIC    → identifying problems, diseases, pests
               e.g. "My tomato has yellow leaves"
RECOMMENDATION → asking for advice, what to use/grow
               e.g. "Which fertilizer for coconut?"
PROCEDURAL    → how-to steps, methods, schedules
               e.g. "How to prepare compost?"
CURRENT       → needs latest/recent information
               e.g. "New government farming scheme 2024"
STATISTICAL   → numbers, yield data, area statistics
               e.g. "Rice yield per hectare Tamil Nadu"

Question: "{query}"

Respond in JSON only — no explanation:
{{
  "query_type": "FACTUAL",
  "confidence": 0.95,
  "reasoning": "one line reason",
  "key_topics": ["topic1", "topic2"],
  "complexity": "simple"
}}

complexity must be "simple" or "complex"."""

    try:
        response = client.chat.completions.create(
            model    = GROQ_GATE_MODEL,   # llama-3.3-70b
            messages = [
                {
                    "role"   : "system",
                    "content": "Classify agricultural queries. "
                               "Return only valid JSON."
                },
                {
                    "role"   : "user",
                    "content": prompt
                }
            ],
            temperature     = 0,
            max_tokens      = 200,
            response_format = {"type": "json_object"},
        )

        raw  = response.choices[0].message.content
        data = json.loads(raw)

        # Validate required fields
        data.setdefault("query_type",  "FACTUAL")
        data.setdefault("confidence",  0.7)
        data.setdefault("reasoning",   "")
        data.setdefault("key_topics",  [])
        data.setdefault("complexity",  "simple")

        return data

    except Exception as e:
        print(f"   ⚠️  Classification error: {e}")
        return {
            "query_type" : "FACTUAL",
            "confidence" : 0.5,
            "reasoning"  : f"error: {e}",
            "key_topics" : [],
            "complexity" : "simple"
        }


# ─────────────────────────────────────────────
# PART 3: ROUTING DECISION
# Combines entropy + classification
# ─────────────────────────────────────────────

def make_routing_decision(entropy, classification, threshold=None):
    """
    Combine entropy + classification to decide path.

    FAST PATH conditions (ALL must be true):
    1. entropy < QT_ENTROPY_THRESHOLD
    2. query_type is FACTUAL or STATISTICAL
    3. complexity == "simple"
    4. classification confidence > QT_MIN_CONFIDENCE

    SLOW PATH: everything else
    """
    q_type     = classification.get("query_type",  "FACTUAL")
    confidence = classification.get("confidence",  0.5)
    complexity = classification.get("complexity",  "complex")

    # Fast path query types
    fast_types = {"FACTUAL", "STATISTICAL"}

    # Determine which threshold to use
    threshold_used = threshold if threshold is not None else QT_ENTROPY_THRESHOLD

    # All conditions for fast path
    entropy_ok    = entropy < threshold_used
    type_ok       = q_type in fast_types
    complexity_ok = complexity == "simple"
    confidence_ok = confidence >= QT_MIN_CONFIDENCE

    if entropy_ok and type_ok and complexity_ok and confidence_ok:
        path = "fast"
        reason = (
            f"low entropy ({entropy}) + "
            f"simple {q_type} query"
        )
    else:
        path = "slow"
        # Explain why slow
        reasons = []
        if not entropy_ok:
            reasons.append(
                f"high entropy ({entropy} >= "
                f"{threshold_used})"
            )
        if not type_ok:
            reasons.append(f"complex type: {q_type}")
        if not complexity_ok:
            reasons.append("complex query")
        if not confidence_ok:
            reasons.append(
                f"low confidence ({confidence})"
            )
        reason = " + ".join(reasons)

    return {
        "path"      : path,
        "reason"    : reason,
        "entropy"   : entropy,
        "query_type": q_type,
        "complexity": complexity,
        "confidence": confidence,
    }


# ─────────────────────────────────────────────
# PART 4A: FAST PATH ANSWER
# Uses: llama-3.3-70b-versatile
# Why: Simple queries don't need best model.
#      Saves gpt-oss-120b for complex tasks.
# ─────────────────────────────────────────────

def fast_path_answer(query, chunks):
    """
    Generate quick answer for simple queries.

    MODEL: llama-3.3-70b-versatile
    REASON: Fast, efficient for simple factual
            questions. Saves premium model quota.
    """
    context = "\n\n".join(
        [c["text"] for c in chunks[:3]]
    )
    sources = list({
        c["source_file"] for c in chunks[:3]
    })

    response = client.chat.completions.create(
        model    = GROQ_GATE_MODEL,   # llama-3.3-70b
        messages = [
            {
                "role"   : "system",
                "content": """You are an agricultural expert 
                for Indian farmers. Answer ONLY from context.
                Be concise — 2-4 sentences maximum.
                Include specific numbers/quantities if present."""
            },
            {
                "role"   : "user",
                "content": f"""Context:
{context}

Question: {query}

Give a direct, concise answer."""
            }
        ],
        temperature = 0.1,
        max_tokens  = 250,
    )

    return {
        "answer"    : response.choices[0].message.content,
        "model_used": GROQ_GATE_MODEL,
        "sources"   : sources,
        "path"      : "fast",
    }


# ─────────────────────────────────────────────
# PART 4B: SLOW PATH ANSWER
# Uses: openai/gpt-oss-120b
# Why: Complex queries need best reasoning,
#      diagnostic accuracy, and detailed advice
# ─────────────────────────────────────────────

# Query-type specific prompts for gpt-oss-120b
SLOW_PATH_PROMPTS = {

    "DIAGNOSTIC": """You are an expert plant pathologist 
    and agricultural diagnostician for Indian crops.
    
    Based on the context provided, give a structured answer:
    
    🔍 DIAGNOSIS:
    [Most likely problem with confidence %]
    
    ✅ CONFIRMATION SIGNS:
    [2-3 key symptoms to confirm]
    
    💊 TREATMENT:
    [Specific remedies with dosage/quantities]
    
    🛡️ PREVENTION:
    [How to prevent recurrence]
    
    Be specific with chemical names, dosages, and timing.""",

    "RECOMMENDATION": """You are a senior agricultural 
    advisor for Indian farmers with expertise in 
    organic and conventional farming.
    
    Provide structured recommendations:
    
    ⭐ BEST OPTION:
    [Top recommendation with clear reasoning]
    
    📋 HOW TO APPLY:
    [Specific steps, quantities, timing]
    
    📈 EXPECTED RESULTS:
    [What farmer should see and when]
    
    ⚠️ PRECAUTIONS:
    [Important warnings or considerations]""",

    "PROCEDURAL": """You are an agricultural extension 
    officer explaining farming procedures to Indian farmers.
    
    Provide clear step-by-step instructions:
    [Number each step]
    [Include exact quantities and timing]
    [Mention common mistakes to avoid]
    [Add safety precautions if needed]""",

    "CURRENT": """You are an agricultural policy expert.
    Answer based on your knowledge.
    Clearly indicate if information needs 
    verification from official sources like 
    icar.org.in or agricoop.nic.in""",

    "STATISTICAL": """You are an agricultural data analyst.
    Present the requested statistics clearly:
    [Use bullet points for numbers]
    [Always cite which document the data is from]
    [Provide context for the numbers]""",
}


def slow_path_answer(query, chunks, query_type):
    """
    Generate detailed answer for complex queries.

    MODEL: openai/gpt-oss-120b
    REASON: 120B parameter model with superior
            reasoning for:
            - Disease/pest diagnosis
            - Complex recommendations
            - Multi-step procedures
            - Nuanced agricultural advice

    This is your BEST model — use it for
    anything that needs real intelligence.
    """
    # Build rich context from top 5 chunks
    context_parts = []
    sources       = []

    for chunk in chunks[:5]:
        context_parts.append(
            f"[Source: {chunk['source_file']}]\n"
            f"{chunk['text']}"
        )
        src = chunk.get("source_file", "")
        if src and src not in sources:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts)

    # Get query-type specific prompt
    type_prompt = SLOW_PATH_PROMPTS.get(
        query_type,
        "Answer thoroughly with specific details, "
        "quantities, and practical advice for "
        "Indian farmers."
    )

    try:
        response = client.chat.completions.create(
            model    = GROQ_ANSWER_MODEL,  # gpt-oss-120b
            messages = [
                {
                    "role"   : "system",
                    "content": f"""You are an expert agricultural 
                    consultant specializing in Indian farming 
                    practices, crop protection, and soil health.
                    
                    {type_prompt}
                    
                    CRITICAL RULES:
                    1. Answer ONLY from provided context
                    2. If context lacks the answer, say clearly:
                       "My knowledge base doesn't have specific 
                        information about this"
                    3. Always be practical for Indian farmers
                    4. Include specific quantities when available"""
                },
                {
                    "role"   : "user",
                    "content": f"""Agricultural Context:
{context}

Farmer's Question: {query}

Provide a thorough, practical answer."""
                }
            ],
            temperature = 0.1,
            max_tokens  = 700,
        )

        return {
            "answer"    : response.choices[0].message.content,
            "model_used": GROQ_ANSWER_MODEL,
            "sources"   : sources,
            "path"      : "slow",
        }

    except Exception as e:
        print(f"   ⚠️  Slow path error: {e}")
        # Fallback to fast model if gpt-oss fails
        print(f"   🔄 Falling back to {GROQ_GATE_MODEL}...")
        return fast_path_answer(query, chunks)


# ─────────────────────────────────────────────
# PART 5: RETRIEVAL
# Get relevant chunks for answer generation
# ─────────────────────────────────────────────

def retrieve_chunks(query, embedder, collection,
                    bm25, corpus, top_k=5):
    """
    Hybrid retrieval: Dense + Sparse combined.
    Same as step5 but integrated here.
    """
    import re

    STOP = {
        "the","a","an","and","or","in","on","at",
        "to","for","of","with","is","are","was",
        "be","this","that","it","from","by","as","per"
    }

    def tokenize(text):
        text   = text.lower()
        text   = re.sub(r'(\w+)/(\w+)', r'\1_per_\2', text)
        text   = re.sub(r'[^a-z0-9\s_]', ' ', text)
        return [t for t in text.split()
                if t not in STOP and len(t) >= 2]

    # Dense search
    q_emb  = embedder.encode([query]).tolist()
    d_res  = collection.query(
        query_embeddings = q_emb,
        n_results        = top_k * 2,
        include          = ["documents", "metadatas", "distances"]
    )

    merged = {}
    # ChromaDB returns ids automatically regardless of include param
    doc_ids = d_res.get("ids", [[]])[0] if d_res.get("ids") else []
    docs = d_res.get("documents", [[]])[0] if d_res.get("documents") else []
    metas = d_res.get("metadatas", [[]])[0] if d_res.get("metadatas") else []
    dists = d_res.get("distances", [[]])[0] if d_res.get("distances") else []
    
    for doc_id, doc, meta, dist in zip(doc_ids, docs, metas, dists):
        merged[doc_id] = {
            "chunk_id"    : doc_id,
            "text"        : doc,
            "source_file" : meta.get("source_file", "") if meta else "",
            "trust_weight": float(meta.get("trust_weight", 1.0)) if meta else 1.0,
            "dense_score" : round(1 - dist, 4) if dist is not None else 0.0,
            "sparse_score": 0.0,
        }

    # Sparse BM25
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_i  = sorted(range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True)[:top_k * 2]
    max_s  = max(scores) if max(scores) > 0 else 1.0

    for idx in top_i:
        if scores[idx] <= 0:
            continue
        cid   = corpus[idx]["chunk_id"]
        ss    = round(scores[idx] / max_s, 4)
        if cid in merged:
            merged[cid]["sparse_score"] = ss
        else:
            merged[cid] = {
                "chunk_id"    : cid,
                "text"        : corpus[idx]["text"],
                "source_file" : corpus[idx]["source_file"],
                "trust_weight": corpus[idx]["trust_weight"],
                "dense_score" : 0.0,
                "sparse_score": ss,
            }

    # Final score with trust weight
    for item in merged.values():
        raw = (0.6 * item["dense_score"] +
               0.4 * item["sparse_score"])
        item["final_score"] = round(
            raw * item["trust_weight"], 4
        )

    return sorted(
        merged.values(),
        key    = lambda x: x["final_score"],
        reverse= True
    )[:top_k]


# ─────────────────────────────────────────────
# PART 6: MAIN QUERY GATE FUNCTION
# Ties everything together
# ─────────────────────────────────────────────

def query_gate(query, embedder, collection,
               bm25, corpus):
    """
    Complete Query Gate pipeline.

    Flow:
    1. Classify query     → llama-3.3-70b-versatile
    2. Measure entropy    → llama-3.3-70b-versatile
    3. Make routing decision
    4. Retrieve chunks    → hybrid search
    5a. Fast path answer  → llama-3.3-70b-versatile
    5b. Slow path answer  → openai/gpt-oss-120b

    Returns complete result dict.
    """
    print(f"\n{'='*60}")
    print(f"🌾 AGRICULTURE RAG — QUERY GATE")
    print(f"{'='*60}")
    print(f"❓ Query: {query}")
    print(f"{'─'*60}")

    # ── Step 1: Classify ──
    print(f"\n📊 Step 1: Classifying query...")
    print(f"   Model: {GROQ_GATE_MODEL}")
    classification = classify_query(query)
    q_type     = classification["query_type"]
    confidence = classification["confidence"]
    topics     = classification["key_topics"]
    complexity = classification["complexity"]

    print(f"   Type      : {q_type}")
    print(f"   Complexity: {complexity}")
    print(f"   Confidence: {confidence}")
    print(f"   Topics    : {topics}")
    print(f"   Reason    : {classification['reasoning']}")

    # ── Step 2: Measure entropy ──
    print(f"\n📏 Step 2: Measuring entropy...")
    print(f"   Model: {GROQ_GATE_MODEL}")
    entropy_data = measure_query_entropy(query)
    entropy      = entropy_data["entropy"]
    token_logprobs = entropy_data["token_logprobs"]

    # Keep a named copy for feedback recording
    entropy_score = entropy

    print(f"   Entropy   : {entropy}")
    # Use adaptive threshold when available
    if ADAPTIVE_ENTROPY_AVAILABLE:
        THRESHOLD = get_current_threshold()
    else:
        THRESHOLD = QT_ENTROPY_THRESHOLD
    print(f"   Threshold : {THRESHOLD}  ← adaptive")
    print(f"   Tokens    : {entropy_data['first_tokens']}")
    
    # Debug check: detect if all logprobs are identical
    if token_logprobs and len(set(token_logprobs)) == 1:
        # All logprobs identical — Groq may be
        # returning compressed values
        # Use classification confidence as fallback
        print(f"   ⚠️  Uniform logprobs detected "
              f"— using classification-based routing")

    # ── Step 3: Routing decision ──
    print(f"\n🔀 Step 3: Routing decision...")
    routing = make_routing_decision(entropy, classification, threshold=THRESHOLD)
    path    = routing["path"]

    if path == "fast":
        print(f"   ⚡ FAST PATH selected")
        print(f"   Reason: {routing['reason']}")
        print(f"   Answer model: {GROQ_GATE_MODEL}")
    else:
        print(f"   🔍 SLOW PATH selected")
        print(f"   Reason: {routing['reason']}")
        print(f"   Answer model: {GROQ_ANSWER_MODEL}")

    # ── Step 4: Retrieve chunks ──
    print(f"\n🔍 Step 4: Retrieving chunks...")
    chunks = retrieve_chunks(
        query, embedder, collection, bm25, corpus
    )
    print(f"   Retrieved: {len(chunks)} chunks")
    if chunks:
        print(f"   Top source: "
              f"{chunks[0]['source_file']}")
        print(f"   Top score : "
              f"{chunks[0]['final_score']}")

    # ── Step 4C: Temporal credibility decay (optional) ──
    try:
        from temporal_credibility import (
            score_all_chunks, filter_stale_chunks
        )
        from credibility_config import (
            MIN_CREDIBILITY_THRESHOLD,
            ENABLE_TEMPORAL_DECAY,
            EVALUATION_YEAR,
        )
    except Exception:
        ENABLE_TEMPORAL_DECAY = False

    if ENABLE_TEMPORAL_DECAY:
        print("\n⏳ Step 4C: Applying Temporal Credibility Decay...")
        # Use the classification result as the query type
        scored_chunks = score_all_chunks(chunks, query_type=q_type, current_year=EVALUATION_YEAR)
        filtered_chunks = filter_stale_chunks(scored_chunks, min_score=MIN_CREDIBILITY_THRESHOLD)

        print(f"   Chunks before filter : {len(scored_chunks)}")
        print(f"   Chunks after filter  : {len(filtered_chunks)}")
        for c in scored_chunks:
            label = c.get('freshness_label', 'UNKNOWN')
            score = c.get('temporal_score', 0)
            src   = c.get('source_file', c.get('source', 'unknown'))
            print(f"   [{label}] {src} → score: {score:.4f}")

        chunks_to_use = filtered_chunks
    else:
        chunks_to_use = chunks

    # Replace `chunks` with `chunks_to_use` for downstream steps
    chunks = chunks_to_use

    # ── Step 4D: Phenological Gate (PGRA) ──
    try:
        from phenology_gate import apply_phenological_gate
        PHENO_AVAILABLE = True
    except Exception:
        PHENO_AVAILABLE = False

    if PHENO_AVAILABLE and q_type in ["RECOMMENDATION", "DIAGNOSTIC", "PROCEDURAL"]:
        print("\n🚧 Step 4D: Applying Phenological Gate...")

        # Default coordinates (India center) — weather step may override later
        gate_lat = 20.5937
        gate_lon = 78.9629

        try:
            gate_result = apply_phenological_gate(
                chunks=chunks,
                query=query,
                lat=gate_lat,
                lon=gate_lon
            )

            chunks_to_use_after_gate = gate_result.get("allowed_chunks", [])

            print(f"\n   📋 Gate Summary:")
            print(f"   Stage        : {gate_result.get('stage')} ({gate_result.get('stage_source')})")
            print(f"   Input chunks : {gate_result.get('total_input')}")
            print(f"   After gate   : {gate_result.get('allowed_count')} allowed, {gate_result.get('blocked_count')} blocked")

            if gate_result.get("allowed_count", 0) == 0:
                print("   ⚠️  Gate blocked all chunks — using original set as safety fallback")
                chunks_to_use_after_gate = chunks

            # Use gate-filtered chunks downstream
            chunks = chunks_to_use_after_gate

        except Exception as e:
            print(f"   ⚠️  Phenological gate failed: {e}")
            # keep existing chunks
            chunks = chunks
    else:
        print("\n⏭️  Step 4D: Phenological gate skipped (non-crop or unavailable)")

    # ── Step 4B: Weather enrichment (for recommendations) ──
    print(f"\n🌦️  Step 4B: Checking for weather enrichment...")
    weather_data = None
    weather_enrichment_applied = False
    
    if WEATHER_ENRICHMENT_AVAILABLE and q_type in ["RECOMMENDATION", "DIAGNOSTIC"] and is_weather_query(query):
        print(f"   ✓ Weather-related {q_type} query detected")
        print(f"   Fetching live weather data...")
        
        # Build base context from chunks
        base_context = "\n\n".join(
            [c["text"] for c in chunks[:5]]
        )
        
        # Enrich with weather
        enriched_context, weather_data = enrich_with_weather(
            query, base_context
        )
        
        if weather_data:
            # Optionally build seasonal context for recommendations/diagnostics
            seasonal_injected = False
            combined_context = enriched_context
            # Also run seasonal context when weather was fetched and the query_type is RECOMMENDATION or DIAGNOSTIC
            if weather_data and q_type in ["RECOMMENDATION", "DIAGNOSTIC"]:
                try:
                    from seasonal_context import get_full_seasonal_context

                    # Use lat/lon from weather_data if present, otherwise default to India centre
                    lat = weather_data.get("lat", 20.5937) if isinstance(weather_data, dict) else 20.5937
                    lon = weather_data.get("lon", 78.9629) if isinstance(weather_data, dict) else 78.9629

                    seasonal_ctx = get_full_seasonal_context(
                        query=query,
                        lat=lat,
                        lon=lon
                    )
                    combined_context = enriched_context + "\n\n" + seasonal_ctx["context_string"] + "\n\n" + base_context
                    seasonal_injected = True
                except Exception as e:
                    print(f"   ⚠️  Seasonal context unavailable: {e}")
                    combined_context = enriched_context + "\n\n" + base_context

            # Create weather chunk to prepend (with seasonal context if available)
            weather_chunk = {
                "text": combined_context,
                "source_file": "REAL-TIME WEATHER DATA",
                "final_score": 1.0,  # Highest priority
            }
            # Prepend weather to chunks
            chunks.insert(0, weather_chunk)
            weather_enrichment_applied = True
            print(f"   ✓ Weather context injected")
            if seasonal_injected:
                print(f"   ✓ Seasonal context injected")
        else:
            print(f"   ⚠️  Weather data unavailable")
    else:
        if not WEATHER_ENRICHMENT_AVAILABLE:
            print(f"   ⚠️  Weather enrichment not available")
        elif q_type not in ["RECOMMENDATION", "DIAGNOSTIC"]:
            print(f"   ℹ️  Not a recommendation/diagnostic query ({q_type})")
        elif not is_weather_query(query):
            print(f"   ℹ️  No weather keywords detected")

    # ── Step 5: Generate answer ──
    print(f"\n💬 Step 5: Generating answer...")

    if path == "fast":
        print(f"   Using: {GROQ_GATE_MODEL} (fast path)")
        answer_data = fast_path_answer(query, chunks)
    else:
        print(f"   Using: {GROQ_ANSWER_MODEL} (slow path)")
        answer_data = slow_path_answer(
            query, chunks, q_type
        )

    # ── Step 5B: Sentence-level provenance mapping ──
    try:
        from sentence_provenance import build_provenance_map, print_provenance_report, get_provenance_summary
        answer = answer_data.get('answer', '')
        if answer and answer.strip() and "knowledge base doesn't have" not in answer:
            print("\n🔬 Step 5B: Building Sentence-Level Provenance Map...")

            # Prefer using scored_chunks from temporal step, else construct from available chunks
            try:
                provenance_chunks = scored_chunks
            except NameError:
                provenance_chunks = []
                for c in chunks:
                    provenance_chunks.append({
                        'text': c.get('text'),
                        'source_file': c.get('source_file') or c.get('source'),
                        'temporal_score': c.get('temporal_score', c.get('final_score', 0)),
                        'freshness_label': c.get('freshness_label', 'UNKNOWN')
                    })

            provenance_map = build_provenance_map(
                answer_text=answer,
                scored_chunks=provenance_chunks,
                client=groq_client
            )

            print_provenance_report(provenance_map)

            provenance_summary = get_provenance_summary(provenance_map)

            print(f"\n🗺️  PROVENANCE : {provenance_summary['total_sentences']} sentences mapped")
            print(f"   Avg Confidence  : {provenance_summary['avg_confidence']:.2f}")
            print(f"   Avg Trust Score : {provenance_summary['avg_combined_score']:.4f}")
            print(f"   Sources Traced  : {', '.join(provenance_summary['sources_used'])}")
        else:
            print("\n⏭️  Step 5B: Skipping provenance (no answer generated)")
    except Exception as e:
        print(f"   ⚠️  Provenance step failed: {e}")

    # ── Step 6: Collect routing feedback (optional, interactive) ──
    print("\n📝 Step 6: Recording routing feedback...")
    try:
        feedback = input("   Was this answer accurate? (y/n, press Enter to skip): ").strip().lower()
        if feedback in ['y', 'n']:
            was_accurate = feedback == 'y'
            try:
                if ADAPTIVE_ENTROPY_AVAILABLE:
                    record_feedback(
                        query=query,
                        entropy=entropy_score,
                        path_used=path,
                        was_accurate=was_accurate
                    )
                else:
                    print("   ℹ️  Adaptive entropy not available — feedback not recorded")
            except Exception as e:
                print(f"   ⚠️  Failed to record feedback: {e}")
        else:
            print("   ⏭️  Feedback skipped")
    except Exception:
        print("   ⏭️  Feedback skipped")

    # ── Final output ──
    print(f"\n{'─'*60}")
    print(f"📋 QUERY TYPE  : {q_type}")
    print(f"🔧 PATH        : {path.upper()}")
    print(f"🤖 MODEL USED  : {answer_data['model_used']}")
    if weather_enrichment_applied:
        print(f"🌦️  WEATHER    : Enriched with real-time data")
    print(f"📚 SOURCES     :")
    for src in answer_data["sources"]:
        print(f"   • {src}")
    print(f"\n✅ ANSWER:\n")
    print(answer_data["answer"])
    print(f"{'='*60}")

    return {
        "query"                    : query,
        "query_type"               : q_type,
        "path"                     : path,
        "entropy"                  : entropy,
        "model_used"               : answer_data["model_used"],
        "answer"                   : answer_data["answer"],
        "sources"                  : answer_data["sources"],
        "chunks_used"              : len(chunks),
        "routing_reason"           : routing["reason"],
        "weather_enrichment_applied": weather_enrichment_applied,
        "weather_data"             : weather_data,
    }


# ─────────────────────────────────────────────
# TEST THE QUERY GATE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Load search components
    print("⏳ Loading search components...", flush=True)
    sys.stdout.flush()
    
    try:
        print("  Loading ChromaDB...", flush=True)
        sys.stdout.flush()
        chroma_cli = chromadb.PersistentClient(path=VECTOR_STORE)
        collection = chroma_cli.get_collection(COLLECTION_NAME)
        print("  ✓ ChromaDB loaded", flush=True)
        sys.stdout.flush()

        print("  Loading BM25 indices...", flush=True)
        sys.stdout.flush()
        bm25_path   = os.path.join(GRAPH_DIR, "bm25_index.pkl")
        corpus_path = os.path.join(GRAPH_DIR, "bm25_corpus.pkl")

        with open(bm25_path,   "rb") as f: bm25   = pickle.load(f)
        with open(corpus_path, "rb") as f: corpus = pickle.load(f)
        print("  ✓ BM25 loaded", flush=True)
        sys.stdout.flush()

        print("  Loading embedding model (this may take 30-60 seconds)...", flush=True)
        sys.stdout.flush()
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("  ✓ Embedder loaded", flush=True)
        sys.stdout.flush()

        print("✅ Ready!\n", flush=True)
        sys.stdout.flush()
        
    except Exception as e:
        print(f"\n❌ ERROR during loading: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── Test queries ──
    # Mix of fast and slow expected
    test_queries = [

        # ⚡ EXPECTED: FAST PATH
        # Simple, factual, low entropy
        "What is vermicompost?",
        "What does NPK stand for?",

        # 🔍 EXPECTED: SLOW PATH
        # Complex diagnosis — needs gpt-oss-120b
        ("My tomato plant leaves are turning yellow "
         "from the bottom with brown edges. "
         "What is wrong and how do I fix it?"),

        # 🔍 EXPECTED: SLOW PATH
        # Recommendation — needs gpt-oss-120b
        ("What organic fertilizers should I apply "
         "for coconut trees in coastal Kerala "
         "during the monsoon season?"),

        # 🔍 EXPECTED: SLOW PATH
        # Procedural — needs gpt-oss-120b
        ("What are the complete steps to prepare "
         "neem-based organic pesticide at home "
         "for vegetable crops?"),
    ]

    results = []
    for query in test_queries:
        result = query_gate(
            query, embedder, collection, bm25, corpus
        )
        results.append(result)
        time.sleep(1)  # rate limit safety

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  📊 QUERY GATE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Query':<40} {'Path':<6} {'Model':<25}")
    print(f"  {'─'*40} {'─'*6} {'─'*25}")

    for r in results:
        q     = r["query"][:38] + ".." \
                if len(r["query"]) > 40 else r["query"]
        path  = r["path"].upper()
        model = (r["model_used"]
                 .replace("openai/", "")
                 .replace("-versatile", "")
                 [:24])
        icon  = "⚡" if r["path"] == "fast" else "🔍"
        print(f"  {icon} {q:<40} {path:<6} {model}")

    fast  = sum(1 for r in results if r["path"] == "fast")
    slow  = sum(1 for r in results if r["path"] == "slow")
    total = len(results)

    print(f"\n  Fast path (llama-3.3-70b) : {fast}/{total}")
    print(f"  Slow path (gpt-oss-120b)  : {slow}/{total}")
    print(f"{'='*60}")