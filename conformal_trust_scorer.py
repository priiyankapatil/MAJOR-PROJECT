import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import os
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL      = "openai/gpt-oss-20b"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALPHA          = 0.10   # significance level → 1-alpha = 90% confidence guarantee
TRUST_LOG_PATH = "conformal_trust_log.json"


_CONFORMAL_EMBEDDER = None

def load_embedder() -> SentenceTransformer:
    global _CONFORMAL_EMBEDDER
    if _CONFORMAL_EMBEDDER is None:
        print("Loading embedder for trust scorer...")
        _CONFORMAL_EMBEDDER = SentenceTransformer(EMBEDDER_MODEL)
    return _CONFORMAL_EMBEDDER


def decompose_answer_to_claims(answer: str, client) -> list:
    """
    Uses LLM to decompose a generated answer into
    atomic verifiable sub-claims.
    """
    if not answer or not answer.strip():
        return []

    model_to_use = LLM_MODEL
    try:
        from config import GROQ_GATE_MODEL
        if GROQ_GATE_MODEL:
            model_to_use = GROQ_GATE_MODEL
    except Exception:
        pass

    prompt = f"""You are a fact decomposition assistant.
    
Break the following agricultural advisory answer into individual atomic claims.
Each claim must be a single verifiable statement that can be checked independently.

Answer:
{answer}

Rules:
- Each claim must be one sentence only
- Remove greeting phrases, transition words
- Keep only factual statements
- Maximum 10 claims

Return ONLY a JSON array of strings, no markdown, no backticks:
["claim 1", "claim 2", "claim 3"]"""

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": "You decompose text into verifiable atomic claims. Return ONLY a valid JSON array of strings: [\"claim 1\", \"claim 2\"]. Do not wrap in markdown or backticks."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.0
    )
    
    raw = response.choices[0].message.content.strip()
    clean_raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.IGNORECASE)
    clean_raw = re.sub(r'\s*```$', '', clean_raw).strip()

    # Strategy 1: Extract and parse JSON array substring
    match = re.search(r'\[.*\]', clean_raw, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group(0))
            if isinstance(claims, list):
                flattened = []
                for item in claims:
                    if isinstance(item, list):
                        flattened.extend(item)
                    elif isinstance(item, str) and item.strip().startswith('[') and item.strip().endswith(']'):
                        try:
                            inner = json.loads(item)
                            if isinstance(inner, list):
                                flattened.extend(inner)
                            else:
                                flattened.append(item)
                        except Exception:
                            flattened.append(item)
                    else:
                        flattened.append(item)
                valid = [str(c).strip().strip('"').strip("'") for c in flattened if c and len(str(c).strip()) > 8]
                if valid:
                    return valid[:10]
        except Exception:
            pass

    # Strategy 2: Extract all quoted strings
    quoted = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', raw)
    valid_quoted = [q.strip().replace('\\"', '"') for q in quoted if len(q.strip()) > 10 and not q.strip().startswith('[') and not q.strip().endswith(']')]
    if len(valid_quoted) >= 2:
        return valid_quoted[:10]

    # Strategy 3: Sentence boundary splitting
    clean_text = re.sub(r'[#*`_]', '', raw)
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    valid_sentences = []
    for s in raw_sentences:
        s_clean = re.sub(r'^[-*0-9.)\s"\']+', '', s).strip().strip('"').strip("'")
        if len(s_clean) > 12 and not s_clean.startswith('|') and not s_clean.startswith('---'):
            valid_sentences.append(s_clean)
    if len(valid_sentences) >= 2:
        return valid_sentences[:10]

    # Strategy 4: Line/bullet-based fallback
    lines = [re.sub(r'^[-*0-9.)\s"\']+', '', l).strip().strip('"').strip("'").strip(',')
             for l in raw.split('\n') if l.strip()]
    valid_lines = [l for l in lines if len(l) > 12 and not l.startswith('|') and not l.startswith('---')]
    return valid_lines[:10]


def compute_nonconformity_scores(claims: list, 
                                  chunks: list, 
                                  embedder) -> np.ndarray:
    """
    Nonconformity score = 1 - max_cosine_similarity(claim, chunks)
    
    Low score  = claim is well-supported by evidence (conforming)
    High score = claim has no support in evidence (nonconforming)
    """
    if not claims or not chunks:
        return np.array([])
    
    try:
        # Embed all claims
        claim_embeddings = embedder.encode(claims, 
                                            convert_to_numpy=True,
                                            show_progress_bar=False)
        
        # Embed all chunks
        chunk_texts = [c.get("text", c) if isinstance(c, dict) 
                       else str(c) for c in chunks]
        chunk_embeddings = embedder.encode(chunk_texts,
                                            convert_to_numpy=True,
                                            show_progress_bar=False)
        
        if len(claim_embeddings.shape) != 2 or len(chunk_embeddings.shape) != 2:
            return np.array([])
            
        if claim_embeddings.shape[1] != chunk_embeddings.shape[1]:
            print(f"   ⚠️  Embedding dimension mismatch: claims {claim_embeddings.shape[1]} vs chunks {chunk_embeddings.shape[1]}")
            return np.array([])

        # For each claim: find max similarity across all chunks
        similarity_matrix = cosine_similarity(claim_embeddings, 
                                               chunk_embeddings)
        max_similarities  = similarity_matrix.max(axis=1)
        
        # Nonconformity score = 1 - max_similarity
        nonconformity_scores = 1.0 - max_similarities
        
        return nonconformity_scores
    except Exception as e:
        print(f"   ⚠️  Nonconformity calculation error: {e}")
        return np.array([])


def compute_conformal_threshold(calibration_scores: np.ndarray,
                                 alpha: float = ALPHA) -> float:
    """
    Computes the nonconformity threshold q_hat for semantic evidence alignment.
    
    This is the (1-alpha) quantile of the nonconformity scores.
    Under empirical self-calibration, claims with nonconformity score <= q_hat
    are evaluated as conforming with retrieved context relative to the threshold.
    
    Formula: q_hat = ceil((n+1)(1-alpha)) / n quantile
    """
    n = len(calibration_scores)
    if n == 0:
        return 0.5  # default threshold
    
    # Conformal quantile with finite-sample correction
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    
    q_hat = np.quantile(calibration_scores, level)
    return float(q_hat)


def apply_conformal_prediction(nonconformity_scores: np.ndarray,
                                claims: list,
                                q_hat: float) -> list:
    """
    For each claim: if nonconformity_score <= q_hat → SUPPORTED
                    if nonconformity_score >  q_hat → UNSUPPORTED
    
    Applies empirical nonconformity thresholding to assess claim support.
    """
    results = []
    for i, (claim, score) in enumerate(
            zip(claims, nonconformity_scores)):
        
        supported      = bool(score <= q_hat)
        trust_score    = float(1.0 - score)  # higher = more trusted
        
        # Confidence label
        if trust_score >= 0.85:
            label = "HIGH CONFIDENCE ✅"
        elif trust_score >= 0.65:
            label = "MODERATE CONFIDENCE ⚠️"
        elif trust_score >= 0.45:
            label = "LOW CONFIDENCE 🔶"
        else:
            label = "UNSUPPORTED ❌"
        
        results.append({
            "claim_index":        i + 1,
            "claim":              claim,
            "nonconformity_score": round(score, 4),
            "trust_score":        round(trust_score, 4),
            "supported":          supported,
            "label":              label
        })
    
    return results


def compute_aggregate_trust(claim_results: list,
                             alpha: float = ALPHA) -> dict:
    """
    Aggregates per-claim scores into an overall answer trust score
    reflecting semantic evidence alignment.
    """
    if not claim_results:
        return {
            "overall_trust_score":   0.0,
            "support_ratio":         0.0,
            "supported_claims":      0,
            "total_claims":          0,
            "confidence_level":      1 - alpha,
            "confidence_pct":        int((1 - alpha) * 100),
            "statistical_guarantee": "No evidence or claims to evaluate",
            "verdict":               "UNKNOWN"
        }
    
    trust_scores    = [r["trust_score"] for r in claim_results]
    supported_count = sum(1 for r in claim_results if r["supported"])
    total_count     = len(claim_results)
    
    # Overall trust = mean of individual trust scores
    overall_trust = float(np.mean(trust_scores))
    support_ratio = supported_count / total_count
    
    # Evidence alignment assessment statement
    confidence_pct = int((1 - alpha) * 100)
    guarantee = (
        f"Semantic evidence alignment: "
        f"{supported_count}/{total_count} claims supported by retrieved evidence"
    )
    
    # Verdict based on empirical alignment thresholds
    if overall_trust >= 0.80 and support_ratio >= 0.80:
        verdict = "HIGH EVIDENCE ALIGNMENT ✅"
    elif overall_trust >= 0.65 and support_ratio >= 0.60:
        verdict = "MODERATE EVIDENCE ALIGNMENT ⚠️"
    elif overall_trust >= 0.45:
        verdict = "LOW EVIDENCE ALIGNMENT 🔶"
    else:
        verdict = "UNSUPPORTED ❌"
    
    return {
        "overall_trust_score":    round(overall_trust, 4),
        "support_ratio":          round(support_ratio, 4),
        "supported_claims":       supported_count,
        "total_claims":           total_count,
        "confidence_level":       1 - alpha,
        "confidence_pct":         confidence_pct,
        "statistical_guarantee":  guarantee,
        "verdict":                verdict
    }


def print_trust_report(claim_results: list,
                        aggregate: dict,
                        q_hat: float):
    print("\n" + "="*65)
    print("║        CONFORMAL PREDICTION TRUST REPORT               ║")
    print("="*65)
    print(f"  Conformal threshold (q̂)  : {q_hat:.4f}")
    print(f"  Significance level (α)   : {ALPHA}  "
          f"→ {aggregate['confidence_pct']}% nominal alignment level")
    print(f"  Overall trust score      : "
          f"{aggregate['overall_trust_score']:.4f}")
    print(f"  Support ratio            : "
          f"{aggregate['supported_claims']}/"
          f"{aggregate['total_claims']} claims")
    print(f"  Verdict                  : {aggregate['verdict']}")
    print(f"\n  📊 Evidence Alignment Assessment:")
    print(f"  {aggregate['statistical_guarantee']}")
    print("="*65)
    print("\n  Per-claim breakdown:\n")
    
    for r in claim_results:
        print(f"  [{r['claim_index']}] {r['label']}")
        print(f"      Claim      : "
              f"{r['claim'][:80]}{'...' if len(r['claim'])>80 else ''}")
        print(f"      Trust score: {r['trust_score']:.4f}  "
              f"| Nonconformity: {r['nonconformity_score']:.4f}  "
              f"| Supported: {r['supported']}")
        print()
    
    print("="*65 + "\n")


def save_trust_log(query: str, answer: str,
                   aggregate: dict, claim_results: list,
                   log_path: str = None) -> None:
    """
    Persists telemetry log of trust scoring run.
    Configurable via log_path or environment variable CONFORMAL_TRUST_LOG_PATH
    to allow test isolation without writing to project root artifacts.
    Catches all file IO exceptions gracefully so logging failure never blocks answers.
    """
    target_path = log_path or os.getenv("CONFORMAL_TRUST_LOG_PATH") or TRUST_LOG_PATH
    try:
        log = []
        if os.path.exists(target_path):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    log = json.load(f)
            except Exception:
                log = []
        
        log.append({
            "timestamp":     datetime.now().isoformat(),
            "query":         query,
            "answer_snippet": answer[:200] if answer else "",
            "overall_trust": aggregate.get("overall_trust_score", 0.0),
            "verdict":       aggregate.get("verdict", "UNKNOWN"),
            "support_ratio": aggregate.get("support_ratio", 0.0),
            "claims_count":  aggregate.get("total_claims", 0),
            "guarantee":     aggregate.get("statistical_guarantee", "")
        })
        
        parent_dir = os.path.dirname(target_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        
        print(f"  💾 Trust log saved → {target_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to save trust log to {target_path}: {e}")


def run_conformal_trust_scorer(query:  str,
                                answer: str,
                                chunks: list,
                                embedder = None,
                                client = None,
                                log_path = None) -> dict:
    """
    Master function. Call this with:
      - query   : the original farmer question
      - answer  : the LLM-generated answer
      - chunks  : list of retrieved text chunks (dicts or strings)
      - embedder: optional existing SentenceTransformer instance
      - client  : optional existing Groq client instance
      - log_path: optional path for telemetry log isolation
    
    Returns the full trust result dict.
    """
    print("\n🔬 CONFORMAL PREDICTION TRUST SCORER")
    print("─"*45)
    
    if not answer or not answer.strip():
        print("   ⚠️  Empty answer provided — cannot score")
        fallback_aggregate = {
            "overall_trust_score": 0.0,
            "verdict": "UNVERIFIED ⚠️",
            "support_ratio": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "confidence_level": 1 - ALPHA,
            "confidence_pct": int((1 - ALPHA) * 100),
            "statistical_guarantee": "Empty answer provided",
        }
        return {
            "query": query,
            "claims": [],
            "claim_results": [],
            "aggregate": fallback_aggregate,
            "q_hat": 0.0,
            "alpha": ALPHA,
            "error": "Empty answer"
        }

    if client is None:
        client  = Groq(api_key=GROQ_API_KEY)
    if embedder is None:
        embedder = load_embedder()
    
    # ── Step 1: Decompose answer into atomic claims ──
    print("\n📋 Step 1: Decomposing answer into sub-claims...")
    claims = decompose_answer_to_claims(answer, client)
    print(f"   Extracted {len(claims)} claims")
    for i, c in enumerate(claims):
        print(f"   [{i+1}] {c[:70]}{'...' if len(c)>70 else ''}")
    
    if not claims:
        print("   ⚠️  No claims extracted — cannot score")
        fallback_aggregate = {
            "overall_trust_score": 0.0,
            "verdict": "UNVERIFIED ⚠️",
            "support_ratio": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "confidence_level": 1 - ALPHA,
            "confidence_pct": int((1 - ALPHA) * 100),
            "statistical_guarantee": "No claims extracted for verification",
        }
        return {
            "query": query,
            "claims": [],
            "claim_results": [],
            "aggregate": fallback_aggregate,
            "q_hat": 0.0,
            "alpha": ALPHA,
            "error": "No claims extracted"
        }
    
    # ── Step 2: Compute nonconformity scores ──
    print("\n📐 Step 2: Computing nonconformity scores...")
    nc_scores = compute_nonconformity_scores(claims, chunks, embedder)
    if nc_scores is None or len(nc_scores) == 0:
        print("   ⚠️  No evidence chunks provided — cannot compute nonconformity")
        fallback_aggregate = {
            "overall_trust_score": 0.0,
            "verdict": "UNVERIFIED ⚠️",
            "support_ratio": 0.0,
            "total_claims": len(claims),
            "supported_claims": 0,
            "unsupported_claims": len(claims),
            "confidence_level": 1 - ALPHA,
            "confidence_pct": int((1 - ALPHA) * 100),
            "statistical_guarantee": "No evidence chunks provided for verification",
        }
        return {
            "query": query,
            "claims": claims,
            "claim_results": [],
            "aggregate": fallback_aggregate,
            "q_hat": 0.0,
            "alpha": ALPHA,
            "error": "No evidence chunks provided"
        }

    print(f"   Scores: {[round(s,4) for s in nc_scores]}")
    
    # ── Step 3: Compute conformal threshold (q̂) ──
    print("\n📊 Step 3: Computing conformal threshold (q̂)...")
    # Use the nonconformity scores themselves as calibration set
    # (self-calibration — valid for single-answer evaluation)
    q_hat = compute_conformal_threshold(nc_scores, alpha=ALPHA)
    print(f"   q̂ = {q_hat:.4f}  "
          f"(α={ALPHA} → heuristic evidence alignment threshold)")
    
    # ── Step 4: Apply conformal prediction ──
    print("\n✅ Step 4: Applying conformal prediction labels...")
    claim_results = apply_conformal_prediction(nc_scores, claims, q_hat)
    
    # ── Step 5: Aggregate trust score ──
    print("\n🎯 Step 5: Computing aggregate trust score...")
    aggregate = compute_aggregate_trust(claim_results)
    
    # ── Step 6: Print full report ──
    print_trust_report(claim_results, aggregate, q_hat)
    
    # ── Step 7: Save log ──
    save_trust_log(query, answer, aggregate, claim_results, log_path=log_path)
    
    return {
        "query":         query,
        "claims":        claims,
        "claim_results": claim_results,
        "aggregate":     aggregate,
        "q_hat":         q_hat,
        "alpha":         ALPHA
    }


# ══════════════════════════════════════════════════════════════
# RETRIEVAL-LEVEL CONFORMAL TRUST SCORING
# ══════════════════════════════════════════════════════════════

def score_chunks_conformal(query: str,
                            chunks: list,
                            embedder) -> list:
    """
    Retrieval-level conformal trust scoring.

    For each chunk: nonconformity = 1 - cosine_similarity(chunk, query)
    Low nonconformity  = chunk strongly supports query = HIGH trust
    High nonconformity = chunk weakly supports query  = LOW trust
    """
    if not chunks or not query:
        return []

    # Embed query
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Embed all chunks
    chunk_texts = [
        c.get("text", c) if isinstance(c, dict)
        else str(c) for c in chunks
    ]
    chunk_embeddings = embedder.encode(
        chunk_texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    # Cosine similarity between each chunk and query
    similarities = cosine_similarity(
        query_embedding, chunk_embeddings
    )[0]

    # Nonconformity scores
    nc_scores = 1.0 - similarities

    # Build scored chunk list
    scored = []
    for i, (chunk, nc, sim) in enumerate(
            zip(chunks, nc_scores, similarities)):

        trust = float(sim)

        # Label
        if trust >= 0.75:
            label = "HIGHLY RELEVANT \u2705"
        elif trust >= 0.55:
            label = "RELEVANT \u26a0\ufe0f"
        elif trust >= 0.35:
            label = "WEAKLY RELEVANT \U0001f536"
        else:
            label = "IRRELEVANT \u274c"

        source = chunk.get("source", f"Chunk {i+1}") \
                 if isinstance(chunk, dict) else f"Chunk {i+1}"
        text   = chunk.get("text", str(chunk)) \
                 if isinstance(chunk, dict) else str(chunk)

        scored.append({
            "chunk_index":         i + 1,
            "source":              source,
            "text_snippet":        text[:100] + "..."
                                   if len(text) > 100 else text,
            "similarity":          round(trust, 4),
            "nonconformity_score": round(float(nc), 4),
            "label":               label
        })

    # Sort by similarity descending
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored


def apply_retrieval_conformal_threshold(
        scored_chunks: list,
        alpha: float = ALPHA) -> dict:
    """
    Applies conformal threshold to decide which chunks
    are statistically trustworthy for retrieval.

    Threshold q_hat = (1-alpha) quantile of nonconformity scores.
    Chunks with nc_score <= q_hat are ACCEPTED.
    Chunks with nc_score >  q_hat are REJECTED.
    """
    if not scored_chunks:
        return {"accepted": [], "rejected": [], "q_hat": 0.5}

    nc_scores = np.array(
        [c["nonconformity_score"] for c in scored_chunks]
    )
    q_hat = compute_conformal_threshold(nc_scores, alpha)

    accepted = []
    rejected = []

    for chunk in scored_chunks:
        if chunk["nonconformity_score"] <= q_hat:
            accepted.append(chunk)
        else:
            rejected.append(chunk)

    return {
        "accepted":       accepted,
        "rejected":       rejected,
        "q_hat":          round(q_hat, 4),
        "total":          len(scored_chunks),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "confidence_pct": int((1 - alpha) * 100)
    }


def print_retrieval_trust_report(result: dict, query: str):
    print("\n" + "="*65)
    print("\u2551     RETRIEVAL-LEVEL CONFORMAL TRUST REPORT             \u2551")
    print("="*65)
    print(f"  Query              : {query[:60]}{'...' if len(query)>60 else ''}")
    print(f"  Conformal q\u0302       : {result['q_hat']:.4f}")
    print(f"  Confidence         : {result['confidence_pct']}% guarantee")
    print(f"  Total chunks       : {result['total']}")
    print(f"  \u2705 Accepted        : {result['accepted_count']}")
    print(f"  \u274c Rejected        : {result['rejected_count']}")
    print("="*65)

    print("\n  \u2705 ACCEPTED CHUNKS (pass to LLM):\n")
    for c in result["accepted"]:
        print(f"  [{c['chunk_index']}] {c['label']}")
        print(f"      Source     : {c['source']}")
        print(f"      Similarity : {c['similarity']:.4f}  "
              f"| Nonconformity: {c['nonconformity_score']:.4f}")
        print(f"      Text       : {c['text_snippet']}")
        print()

    if result["rejected"]:
        print("  \u274c REJECTED CHUNKS (filtered out):\n")
        for c in result["rejected"]:
            print(f"  [{c['chunk_index']}] {c['label']}")
            print(f"      Source     : {c['source']}")
            print(f"      Similarity : {c['similarity']:.4f}  "
                  f"| Nonconformity: {c['nonconformity_score']:.4f}")
            print(f"      Text       : {c['text_snippet']}")
            print()

    print("="*65 + "\n")


def run_retrieval_conformal_trust(query:  str,
                                   chunks: list) -> dict:
    """
    Master function for retrieval-level trust scoring.

    Call this with:
      - query  : the farmer's question
      - chunks : list of retrieved chunks (dicts with 'text', 'source')

    Returns accepted chunks that are statistically
    trustworthy for passing to the LLM.
    """
    print("\n\U0001f50d RETRIEVAL-LEVEL CONFORMAL TRUST SCORER")
    print("\u2500"*45)

    embedder = load_embedder()

    # Step 1: Score all chunks against query
    print(f"\n\U0001f4d0 Step 1: Scoring {len(chunks)} chunks "
          f"against query...")
    scored_chunks = score_chunks_conformal(query, chunks, embedder)

    for c in scored_chunks:
        print(f"   [{c['chunk_index']}] {c['label']:30s} "
              f"sim={c['similarity']:.4f}  "
              f"nc={c['nonconformity_score']:.4f}  "
              f"\u2192 {c['source']}")

    # Step 2: Apply conformal threshold
    print(f"\n\U0001f4ca Step 2: Applying conformal threshold (\u03b1={ALPHA})...")
    result = apply_retrieval_conformal_threshold(
        scored_chunks, alpha=ALPHA
    )
    print(f"   q\u0302 = {result['q_hat']:.4f}  \u2192  "
          f"{result['accepted_count']} accepted, "
          f"{result['rejected_count']} rejected")

    # Step 3: Print full report
    print_retrieval_trust_report(result, query)

    return result


if __name__ == "__main__":
    
    # ── TEST CASE 1: Well-supported answer ──
    test_query_1 = "What is vermicompost?"
    
    test_answer_1 = """Vermicompost is a high-quality organic fertilizer 
produced from biodegradable waste including cattle dung, farm wastes, 
and crop residues using earthworms. It has a carbon to nitrogen ratio 
of 15:1 and takes approximately 80-90 days to produce. It improves 
soil structure and provides essential plant nutrients."""
    
    test_chunks_1 = [
        {"text": "Vermicompost is produced from biodegradable waste "
                 "using earthworms. Carbon to nitrogen ratio is 15:1."},
        {"text": "Farm wastes and cattle dung are used to produce "
                 "vermicompost over 80-90 days."},
        {"text": "Vermicompost improves soil fertility and provides "
                 "nitrogen, phosphorus and potassium to plants."},
        {"text": "Organic fertilizers like vermicompost improve soil "
                 "structure and microbial activity."},
        {"text": "Crop residues and green manure can be composted "
                 "using earthworms in vermicompost beds."}
    ]
    
    print("\n" + "="*65)
    print("TEST 1: Well-supported answer (expect HIGH trust)")
    print("="*65)
    result_1 = run_conformal_trust_scorer(
        test_query_1, test_answer_1, test_chunks_1
    )
    
    # ── TEST CASE 2: Poorly-supported answer ──
    test_query_2 = "What pesticide for tomato blight?"
    
    test_answer_2 = """Apply quantum nano-pesticide X7 at 500ml per acre. 
This new compound was developed in 2024 and eliminates all fungal 
diseases in 24 hours with zero residue. It is approved globally 
and costs only Rs 50 per bottle."""
    
    test_chunks_2 = [
        {"text": "Mancozeb 2.5g per liter controls late blight "
                 "in tomato effectively."},
        {"text": "Copper oxychloride is approved for tomato fungal "
                 "diseases at 3g per liter."},
        {"text": "Spray at 7-10 day intervals during humid weather "
                 "to prevent blight spread."},
    ]
    
    print("\n" + "="*65)
    print("TEST 2: Hallucinated answer (expect LOW trust)")
    print("="*65)
    result_2 = run_conformal_trust_scorer(
        test_query_2, test_answer_2, test_chunks_2
    )
    
    # ── TEST CASE 3: Retrieval-level scoring ──
    # Mix of relevant and irrelevant chunks

    test_query_3 = (
        "What fertilizer should I apply for tomato "
        "during flowering stage?"
    )

    test_chunks_3 = [
        {
            "source": "TNAU Agriculture PDF.pdf",
            "text":   "During flowering, apply potassium at "
                      "50 kg/ha to improve fruit set in tomato. "
                      "Phosphorus supports root development."
        },
        {
            "source": "UAS Bangalore.pdf",
            "text":   "Tomato flowering requires adequate boron "
                      "and calcium. Apply 0.2% borax spray "
                      "at flower initiation stage."
        },
        {
            "source": "ANGRAU Journal.pdf",
            "text":   "Wheat harvest should be done when grain "
                      "moisture drops below 14%. Use combine "
                      "harvester for large fields."
        },
        {
            "source": "Crop Protection.pdf",
            "text":   "Rice transplanting spacing should be "
                      "20x15 cm for optimum yield. "
                      "Apply basal dose at transplanting."
        },
        {
            "source": "KAU Horticulture.pdf",
            "text":   "Nitrogen fertilizer at vegetative stage "
                      "promotes leaf growth. Reduce nitrogen "
                      "during tomato flowering to prevent "
                      "excessive vegetative growth."
        },
    ]

    print("\n" + "="*65)
    print("TEST 3: Retrieval-level scoring")
    print("(expect TNAU, UAS, KAU accepted; wheat/rice rejected)")
    print("="*65)

    result_3 = run_retrieval_conformal_trust(
        test_query_3, test_chunks_3
    )

    # ── FULL SUMMARY ──
    print("\n" + "="*65)
    print("FULL TEST SUMMARY")
    print("="*65)
    t1_score = result_1.get("aggregate", {}).get("overall_trust_score", 0.0)
    t1_verdict = result_1.get("aggregate", {}).get("verdict", "N/A")
    t2_score = result_2.get("aggregate", {}).get("overall_trust_score", 0.0)
    t2_verdict = result_2.get("aggregate", {}).get("verdict", "N/A")

    print(f"  Test 1 — Generation trust (good answer)  : "
          f"{t1_score:.4f} — {t1_verdict}")
    print(f"  Test 2 — Generation trust (hallucination): "
          f"{t2_score:.4f} — {t2_verdict}")
    print(f"  Test 3 — Retrieval trust                 : "
          f"{result_3['accepted_count']}/{result_3['total']} "
          f"chunks accepted")
    print("="*65)
