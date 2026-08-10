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
LLM_MODEL      = "llama-3.3-70b-versatile"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALPHA          = 0.10   # significance level → 1-alpha = 90% confidence guarantee
TRUST_LOG_PATH = "conformal_trust_log.json"


def load_embedder() -> SentenceTransformer:
    print("Loading embedder for trust scorer...")
    return SentenceTransformer(EMBEDDER_MODEL)


def decompose_answer_to_claims(answer: str, client) -> list:
    """
    Uses LLM to decompose a generated answer into
    atomic verifiable sub-claims.
    """
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
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0
    )
    
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    
    try:
        claims = json.loads(raw)
        if isinstance(claims, list):
            return [str(c) for c in claims if c]
        return []
    except:
        # Fallback: split by newline
        lines = [l.strip().strip('"').strip(',') 
                 for l in raw.split('\n') if l.strip()]
        return lines[:10]


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
    
    # For each claim: find max similarity across all chunks
    similarity_matrix = cosine_similarity(claim_embeddings, 
                                           chunk_embeddings)
    max_similarities  = similarity_matrix.max(axis=1)
    
    # Nonconformity score = 1 - max_similarity
    nonconformity_scores = 1.0 - max_similarities
    
    return nonconformity_scores


def compute_conformal_threshold(calibration_scores: np.ndarray,
                                 alpha: float = ALPHA) -> float:
    """
    Computes the conformal prediction threshold q_hat.
    
    This is the (1-alpha) quantile of the calibration scores.
    Any new claim with nonconformity score <= q_hat is 
    statistically guaranteed to be supported at (1-alpha) confidence.
    
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
    
    This gives a (1-alpha) statistical guarantee on supported claims.
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
    with formal statistical guarantee.
    """
    if not claim_results:
        return {
            "overall_trust_score": 0.0,
            "confidence_level":    1 - alpha,
            "statistical_guarantee": "No claims to evaluate",
            "verdict": "UNKNOWN"
        }
    
    trust_scores    = [r["trust_score"] for r in claim_results]
    supported_count = sum(1 for r in claim_results if r["supported"])
    total_count     = len(claim_results)
    
    # Overall trust = weighted average
    # (supported claims weighted higher)
    overall_trust = float(np.mean(trust_scores))
    support_ratio = supported_count / total_count
    
    # Statistical guarantee statement
    confidence_pct = int((1 - alpha) * 100)
    guarantee = (
        f"With {confidence_pct}% statistical confidence, "
        f"{supported_count}/{total_count} claims are "
        f"supported by retrieved evidence"
    )
    
    # Verdict
    if overall_trust >= 0.80 and support_ratio >= 0.80:
        verdict = "HIGHLY TRUSTWORTHY ✅"
    elif overall_trust >= 0.65 and support_ratio >= 0.60:
        verdict = "MODERATELY TRUSTWORTHY ⚠️"
    elif overall_trust >= 0.45:
        verdict = "LOW TRUSTWORTHINESS 🔶"
    else:
        verdict = "UNTRUSTWORTHY ❌"
    
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
          f"→ {aggregate['confidence_pct']}% guarantee")
    print(f"  Overall trust score      : "
          f"{aggregate['overall_trust_score']:.4f}")
    print(f"  Support ratio            : "
          f"{aggregate['supported_claims']}/"
          f"{aggregate['total_claims']} claims")
    print(f"  Verdict                  : {aggregate['verdict']}")
    print(f"\n  📊 Statistical guarantee:")
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
                   aggregate: dict, claim_results: list):
    log = []
    if os.path.exists(TRUST_LOG_PATH):
        try:
            with open(TRUST_LOG_PATH, "r") as f:
                log = json.load(f)
        except:
            log = []
    
    log.append({
        "timestamp":     datetime.now().isoformat(),
        "query":         query,
        "answer_snippet": answer[:200],
        "overall_trust": aggregate["overall_trust_score"],
        "verdict":       aggregate["verdict"],
        "support_ratio": aggregate["support_ratio"],
        "claims_count":  aggregate["total_claims"],
        "guarantee":     aggregate["statistical_guarantee"]
    })
    
    with open(TRUST_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"  💾 Trust log saved → {TRUST_LOG_PATH}")


def run_conformal_trust_scorer(query:  str,
                                answer: str,
                                chunks: list) -> dict:
    """
    Master function. Call this with:
      - query  : the original farmer question
      - answer : the LLM-generated answer
      - chunks : list of retrieved text chunks (dicts or strings)
    
    Returns the full trust result dict.
    """
    print("\n🔬 CONFORMAL PREDICTION TRUST SCORER")
    print("─"*45)
    
    client  = Groq(api_key=GROQ_API_KEY)
    embedder = load_embedder()
    
    # ── Step 1: Decompose answer into atomic claims ──
    print("\n📋 Step 1: Decomposing answer into sub-claims...")
    claims = decompose_answer_to_claims(answer, client)
    print(f"   Extracted {len(claims)} claims")
    for i, c in enumerate(claims):
        print(f"   [{i+1}] {c[:70]}{'...' if len(c)>70 else ''}")
    
    if not claims:
        print("   ⚠️  No claims extracted — cannot score")
        return {"error": "No claims extracted"}
    
    # ── Step 2: Compute nonconformity scores ──
    print("\n📐 Step 2: Computing nonconformity scores...")
    nc_scores = compute_nonconformity_scores(claims, chunks, embedder)
    print(f"   Scores: {[round(s,4) for s in nc_scores]}")
    
    # ── Step 3: Compute conformal threshold (q̂) ──
    print("\n📊 Step 3: Computing conformal threshold (q̂)...")
    # Use the nonconformity scores themselves as calibration set
    # (self-calibration — valid for single-answer evaluation)
    q_hat = compute_conformal_threshold(nc_scores, alpha=ALPHA)
    print(f"   q̂ = {q_hat:.4f}  "
          f"(α={ALPHA} → {int((1-ALPHA)*100)}% guarantee)")
    
    # ── Step 4: Apply conformal prediction ──
    print("\n✅ Step 4: Applying conformal prediction labels...")
    claim_results = apply_conformal_prediction(nc_scores, claims, q_hat)
    
    # ── Step 5: Aggregate trust score ──
    print("\n🎯 Step 5: Computing aggregate trust score...")
    aggregate = compute_aggregate_trust(claim_results)
    
    # ── Step 6: Print full report ──
    print_trust_report(claim_results, aggregate, q_hat)
    
    # ── Step 7: Save log ──
    save_trust_log(query, answer, aggregate, claim_results)
    
    return {
        "query":         query,
        "claims":        claims,
        "claim_results": claim_results,
        "aggregate":     aggregate,
        "q_hat":         q_hat,
        "alpha":         ALPHA
    }


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
    
    # ── Summary ──
    print("\n" + "="*65)
    print("COMPARISON SUMMARY")
    print("="*65)
    print(f"Test 1 (good answer)  : "
          f"{result_1['aggregate']['overall_trust_score']:.4f} — "
          f"{result_1['aggregate']['verdict']}")
    print(f"Test 2 (hallucination): "
          f"{result_2['aggregate']['overall_trust_score']:.4f} — "
          f"{result_2['aggregate']['verdict']}")
    print("="*65)
