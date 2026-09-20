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
import re
import math
import json
import time
import pickle
import sys
import chromadb
from groq import Groq
# Lazy load SentenceTransformer to avoid initialization hang

# Prevent UnicodeEncodeError on Windows consoles with restricted codepages
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass


def _safe_print(*args, **kwargs):
    """
    Encoding-safe print wrapper for console output.
    Prevents UnicodeEncodeError on Windows consoles or redirected streams (e.g. cp1252, cp437, ascii)
    without globally mutating the user's console or environment.
    """
    file = kwargs.get("file") or sys.stdout
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    flush = kwargs.get("flush", False)
    text = sep.join(str(a) for a in args)
    try:
        file.write(text + end)
        if flush:
            file.flush()
    except UnicodeEncodeError:
        encoding = getattr(file, "encoding", None) or "ascii"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        file.write(safe_text + end)
        if flush:
            file.flush()
    except Exception:
        try:
            safe_text = text.encode("ascii", errors="replace").decode("ascii")
            file.write(safe_text + end)
            if flush:
                file.flush()
        except Exception:
            pass

print = _safe_print


def _load_environment_variables() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except Exception:
        pass

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except Exception:
        pass


_load_environment_variables()

if os.environ.get("RUN_REGULATORY_UPDATE_ON_STARTUP") == "1":
    try:
        from regulatory_updater import run_startup_update
        run_startup_update(borderline_pesticides=None)
    except Exception as e:
        print(f"⚠️ Regulatory update skipped: {e}")

import config
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

from compliance_scanner import scan_answer_for_compliance, build_compliance_warning_box

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
    Send query to LLM to measure entropy / confidence.
    """
    try:
        response = client.chat.completions.create(
            model      = GROQ_ANSWER_MODEL,   # openai/gpt-oss-120b
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
        response_text = response.choices[0].message.content or ""
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
# ─────────────────────────────────────────────

def classify_query(query):
    """
    Classify the query into one of the categories.
    Determines which pipeline to use downstream.
    """
    prompt = f"""Classify this agricultural query. Reply with JSON only.

Query: {query}

Reply format:
{{"type": "FACTUAL", "complexity": "simple", "confidence": 0.95, "topics": ["topic1"], "reason": "brief reason"}}

Type must be one of: FACTUAL, DIAGNOSTIC, RECOMMENDATION, PROCEDURAL"""

    try:
        response = client.chat.completions.create(
            model    = GROQ_ANSWER_MODEL,   # openai/gpt-oss-120b
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
            max_tokens      = 800,
            response_format = {"type": "json_object"},
        )

        raw  = response.choices[0].message.content
        data = json.loads(raw)

        # Validate and normalize required fields
        query_type = data.get("type") or data.get("query_type") or "FACTUAL"
        data["query_type"] = query_type
        data["type"] = query_type
        data.setdefault("confidence",  data.get("confidence", 0.95))
        data.setdefault("reasoning",   data.get("reason", ""))
        data.setdefault("key_topics",  data.get("topics", []))
        data.setdefault("complexity",  data.get("complexity", "simple"))

        return data

    except Exception as e:
        print(f"   ⚠️  Classification error: {e}")
        # Intelligent keyword heuristic fallback when LLM classification is unavailable
        q_lower = query.lower()
        if any(w in q_lower for w in ["why", "yellow", "disease", "pest", "damage", "symptom", "blight", "rot", "wilt", "borer", "spot"]):
            fb_type = "DIAGNOSTIC"
        elif any(w in q_lower for w in ["how to", "control", "manage", "management", "recommend", "spray", "fertilizer", "dose", "dosage", "treatment"]):
            fb_type = "RECOMMENDATION"
        elif any(w in q_lower for w in ["steps", "step by step", "procedure", "how do i prepare"]):
            fb_type = "PROCEDURAL"
        else:
            fb_type = "FACTUAL"

        return {
            "query_type" : fb_type,
            "type"       : fb_type,
            "confidence" : 0.5,
            "reasoning"  : f"heuristic fallback (error: {e})",
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

    MODEL: openai/gpt-oss-20b (GROQ_GATE_MODEL)
    REASON: Fast, efficient for simple factual
            questions. Saves premium model quota.
    """
    if not chunks:
        return {
            "answer": (
                "⚠️ Insufficient verified context: No verified agricultural documents were available to answer this query. "
                "Please consult your local Krishi Vigyan Kendra (KVK) or State Agricultural University extension."
            ),
            "model_used": "system/insufficient-context",
            "sources": [],
            "path": "fast",
        }

    context = "\n\n".join(
        [c["text"] for c in chunks[:3] if c.get("text")]
    )
    sources = []
    for c in chunks[:3]:
        src = c.get("source_file") or c.get("source")
        if src and src not in sources:
            sources.append(src)

    try:
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
            max_tokens  = 350,
        )

        content = response.choices[0].message.content or ""
        if not content and hasattr(response.choices[0].message, "reasoning"):
            content = response.choices[0].message.reasoning or ""
    except Exception as e:
        print(f"   ⚠️  Fast path generation error: {e}")
        content = f"⚠️ Generation temporarily unavailable: {e}"

    return {
        "answer"    : content,
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

    "DIAGNOSTIC": """You are an expert plant pathologist and agricultural diagnostician for Indian crops.

CRITICAL DIAGNOSTIC RULES:
1. NEVER output invented probability percentages (such as "70% confidence" or "30% confidence") unless calibrated statistical data is explicitly provided in the context.
2. Present differential diagnoses as potential possibilities (e.g. "Primary Suspected Cause", "Alternative Differential Diagnosis"), NOT absolute certainties.
3. Explicitly list CRITICAL FIELD OBSERVATIONS NEEDED from the farmer to confirm (e.g., examination of stem base/roots, underside of leaves, pattern across field, recent weather, soil drainage, crop stage).
4. Tie all recommendations strictly to retrieved context.
5. DO NOT recommend specific chemical insecticides, fungicides, or dosages unless the retrieved context contains explicit, verified guidance for that exact crop and problem.
6. REGULATORY SAFETY MANDATE (OFFLINE_STATIC_KB / UNVERIFIED STATUS):
   - When regulatory verification is OFFLINE_STATIC_KB or live checks are unavailable, you MUST NOT emit synthetic pesticide brand names, active ingredients, or dosages (e.g., Dimethoate, Carbaryl, Chlorpyrifos, or spray percentages) as actionable spray instructions.
   - Any chemical treatments mentioned in historical literature (e.g., KAU) are unverified and non-actionable; do not quote specific chemical dosages or spray instructions. Synthetic chemical recommendations and dosages are strictly withheld under OFFLINE_STATIC_KB.
   - Under CHEMICAL INTERVENTION, state clearly: "⚠️ Live CIB&RC regulatory verification is currently unavailable (offline static database). Synthetic chemical treatments and dosages are withheld without live regulatory verification. Consult your nearest Krishi Vigyan Kendra (KVK) or State Agricultural University extension officer for current, legally registered options."
7. TREATMENT RATE & STALE SOURCE MANDATE (BIOLOGICAL & FERTILIZER):
   - Do NOT output specific numerical application rates, dosages, concentrations, or quantities for biological controls (e.g., do NOT output '1 lakh/ha', '10 g/L', '10 g L⁻¹', or '20 g/l') or blanket chemical fertilizer rates from historical, unverified, or stale sources (such as KAU or older literature).
   - Biological agents and cultural methods must be recommended QUALITATIVELY only while explicitly directing the farmer to their local KVK or State Agricultural University extension office for current formulation-specific application rates, viable spore counts (CFU), and local registration.
   - Blanket fertilizer quantities must not be prescribed without soil test verification; direct the farmer to soil testing and current state Package of Practices (POP).
   - A disclaimer (such as 'verify with KVK' or 'historical reference') does NOT make an unverified numerical rate acceptable. The numerical rate itself must be omitted.
8. AGRONOMIC INTEGRITY & STRICT CONTEXT GROUNDING:
   - NEVER recommend Trichogramma for Hemipteran sucking pests (Brown Plant Hopper / Nilaparvata lugens, rice mealybug, aphids, whiteflies) or nematodes. Trichogramma is strictly an egg parasitoid of Lepidopteran borers and caterpillars.
   - DO NOT present biological control agents (such as Beauveria bassiana, Metarhizium, mirid bugs, or spiders) or cultural measures (such as removing tillers or crop rotation) as source-verified unless explicitly documented in the retrieved text for that specific pest. In the supplied excerpts, BPH and mealybug management does not contain verified biological control agents. If mentioning potential natural enemies as general knowledge, you MUST explicitly label them: "⚠️ Unverified in Retrieved Context — Consult Local KVK: Specific biological agents or cultural controls are not documented in retrieved excerpts; consult your local KVK for verified options."
   - For Nutrient Deficiencies (Nitrogen and Iron): The symptom descriptions are grounded in TNAU text, but specific fertilizer products, foliar sprays (e.g. iron chelates), or green manure mulches are unverified in context. Advise soil testing and KVK prescription.
   - For Water & Drainage: Crop Protection text notes that poor drainage can cause seedling yellowing. Do NOT invent numerical water depths (e.g. 5–7 cm, 10–12 cm) or ungrounded practices like Alternate Wetting and Drying (AWD) as verified. Advise maintaining proper field drainage and avoiding stagnant water, directing the farmer to state Package of Practices and KVK.
   - NEVER recommend destructive burning of crop residues. Recommend field sanitation and clearing crop residues.
   - Do NOT label the answer or unverified claims as factually validated or proven.

Structure your answer clearly:
🔍 POTENTIAL CAUSES (SOURCE-GROUNDED):
[List plausible causes directly identified in retrieved context: Nitrogen deficiency (TNAU), Iron deficiency (TNAU), Brown plant hopper (KAU), Rice mealybug (KAU), Rice root nematode (KAU), poor drainage (Crop Protection)]

📋 CONFIRMATORY OBSERVATIONS NEEDED:
[2-4 specific signs the farmer should observe in the field to differentiate causes based on the symptoms in the text]

🛡️ IMMEDIATE PRACTICAL & CULTURAL MANAGEMENT:
[Safe cultural, sanitation, or drainage practices supported by evidence. For unverified management areas (BPH/mealybug biocontrol, specific fertilizer rates, soil amendments), clearly label them: "⚠️ Unverified in Retrieved Context — Consult Local KVK: Specific biological agents, soil amendments, and cultural controls are not documented in the retrieved evidence. Consult local KVK / SAU extension services for approved field practices."]

💊 CHEMICAL INTERVENTION:
[State clearly that live regulatory verification is unavailable and that synthetic chemical treatments and dosages are strictly withheld under OFFLINE_STATIC_KB. Direct farmer to KVK/extension guidance for current registered options; prioritize cultural and biological practices.]""",

    "RECOMMENDATION": """You are a senior agricultural advisor for Indian farmers with expertise in organic and conventional farming.

CRITICAL RECOMMENDATION RULES:
1. All recommendations must be grounded strictly in the provided agricultural context.
2. Require crop-specific, stage-specific evidence before recommending chemical names, dosages, or timing.
3. NEVER invent or extrapolate chemical dosages, sprays, or application intervals without explicit evidence. If specific chemical dosage is not in context, state: "Consult your local KVK or agricultural university package of practices for approved dosage."
4. When regulatory verification is OFFLINE_STATIC_KB or live checks are unavailable, DO NOT prescribe synthetic chemical pesticides as actionable advice. Clearly disclose that live registration is unverified and direct the farmer to local KVK/extension officers.
5. TREATMENT RATE & STALE SOURCE MANDATE:
   - Do NOT output specific numerical application rates, dosages, concentrations, or quantities for biological controls (e.g., do not output '1 lakh/ha' or '10 g/L') or blanket chemical fertilizer rates from historical, unverified, or stale sources.
   - Recommend biological agents and cultural methods QUALITATIVELY only and instruct the farmer to obtain current formulation-specific application rates and CFU counts from their local KVK or state extension office.
   - Prioritize integrated practices (soil health, water management, cultural and biological control).
6. PEST-AGENT & PRACTICE INTEGRITY:
   - Ensure biological controls match pest biology (never recommend Trichogramma for sucking pests).
   - Do not invent numerical water depths or recommend burning crop residues.
7. Output complete, well-formed answers. Do not leave tables or sentences incomplete.

Provide structured recommendations:
⭐ BEST OPTION:
[Top recommendation with clear reasoning grounded in context]

📋 HOW TO APPLY:
[Step-by-step practical advice with quantities and timing ONLY if supported by context]

📈 EXPECTED RESULTS:
[What farmer should see and when]

⚠️ PRECAUTIONS & OBSERVATIONS:
[Important warnings, safety intervals, or conditions to monitor]""",

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



def _get_synthetic_pesticide_names():
    try:
        from regulatory_kb import PESTICIDE_DB
        names = []
        for p_key, p_data in PESTICIDE_DB.items():
            if p_data.get("organic_status") not in {"NPOP_APPROVED", "PGS_APPROVED"}:
                names.append(p_key.lower())
                for c_name in p_data.get("common_names", []):
                    names.append(c_name.lower())
        return sorted(list(set(names)), key=lambda x: -len(x))
    except Exception:
        return [
            "dimethoate", "carbaryl", "chlorpyrifos", "monocrotophos",
            "endosulfan", "imidacloprid", "rogor", "tafgor", "dimecron",
            "confidor", "admire", "gaucho", "dursban", "lorsban", "radar",
            "nuvacron", "azodrin", "sevin", "ravyon", "tilt", "bumper",
            "thiodan", "beosit", "streptocycline", "streptomycin", "kcycline",
            "tricyclazole", "beam", "blascide", "buprofezin", "applaud", "courier",
            "glyphosate", "roundup", "sweep", "acephate", "asataf", "orthene",
            "cartap hydrochloride", "padan", "caldan", "critap", "fipronil", "regent"
        ]


_SYNTHETIC_PESTICIDES_CACHE = None

def _get_all_competing_treatment_items():
    global _SYNTHETIC_PESTICIDES_CACHE
    if _SYNTHETIC_PESTICIDES_CACHE is None:
        base_items = [
            "urea", "dap", "npk", "mop", "potash", "nitrogen", "phosphorus", "potassium",
            "zinc", "boron", "gypsum", "compost", "fym", "neem cake", "neem oil", "nske",
            "ferrous sulphate", "iron chelate", "fe-eddha", "zinc sulphate", "borax",
            "ammonium sulphate", "calcium nitrate", "ssp",
            "trichogramma", "beauveria", "metarhizium", "pseudomonas", "bacillus", "bt",
            "trichoderma", "verticillium", "lecanicillium", "paecilomyces", "chrysoperla",
            "dimethoate", "chlorpyrifos", "carbaryl", "monocrotophos", "endosulfan",
            "imidacloprid", "cartap", "fipronil", "tricyclazole", "carbendazim", "mancozeb",
            "propiconazole", "hexaconazole", "copper oxychloride", "streptocycline",
            "glyphosate", "paraquat", "2,4-d", "butachlor", "pretilachlor", "pendimethalin",
            "acephate", "buprofezin", "quinalphos", "profenofos", "cypermethrin", "thiamethoxam",
            "malathion", "dichlorvos", "phosphamidon", "triazophos", "methyl demeton",
            "flubendiamide", "chlorantraniliprole", "spinetoram", "indoxacarb", "novaluron"
        ]
        synth = _get_synthetic_pesticide_names()
        _SYNTHETIC_PESTICIDES_CACHE = sorted(list(set(base_items + synth)), key=lambda x: -len(x))
    return _SYNTHETIC_PESTICIDES_CACHE


OTHER_TREATMENT_ITEMS = [
    "urea", "dap", "npk", "mop", "potash", "nitrogen", "phosphorus", "potassium",
    "zinc", "boron", "gypsum", "compost", "fym", "neem cake", "neem oil", "nske",
    "dimethoate", "chlorpyrifos", "carbaryl", "monocrotophos", "endosulfan",
    "imidacloprid", "cartap", "fipronil", "tricyclazole", "carbendazim", "mancozeb",
    "propiconazole", "hexaconazole", "copper oxychloride", "streptocycline",
    "glyphosate", "paraquat", "2,4-d", "butachlor", "pretilachlor", "pendimethalin",
    "trichogramma", "beauveria", "metarhizium", "pseudomonas", "bacillus", "bt"
]


def _is_rate_verified_in_context(item_name: str, rate_snippet: str, chunks: list = None, target_crop: str = "rice", target_pest: str = None) -> bool:
    """
    Checks whether a specific treatment item, formulation, and its exact numerical rate and units
    are explicitly supported by a verified, current, context-appropriate source in chunks
    (e.g., FRESH or ACCEPTABLE status, temporal score >= 0.70).

    Validates:
      1. Source Freshness / Temporal credibility >= 0.70 (excludes STALE/OUTDATED/UNKNOWN)
      2. Non-rate exclusion: immediately rejects waiting periods, temperatures, and unrelated quantities
      3. Formulation consistency: if formulation (e.g. 20 EC, 50 WP) is specified, verifies it in window
      4. Bound Rate & Unit verification: exact numerical rate and unit must appear as a bound unit phrase.
         Fails closed if no recognized agronomic dosage unit is present.
      5. Syntactic Association: verifies treatment item and rate belong to the same sentence/clause/table-cell
         (rejects cross-treatment association errors where rate belongs to another nearby chemical/fertilizer
         or appears across sentences, table rows, or multiple table columns)
      6. Crop & Pest consistency: rejects cross-crop false positives and cross-pest mismatches
    """
    if not chunks:
        return False

    norm_item = re.sub(r"[\*\_]+", "", item_name.lower()).strip()
    norm_rate = rate_snippet.lower().strip()

    # Reject non-rate quantities (waiting periods, temperatures, intervals, plant counts, yields)
    if re.search(r"\b(?:days?|hrs?|hours?|weeks?|months?|celsius|°c|°f|deg(?:ree)?s?|temperature|interval|phi|waiting\s*period|safety\s*interval)\b", norm_rate):
        return False

    combined_query = f"{norm_item} {norm_rate}"
    formulation_match = re.search(
        r"\b(\d+\s*(?:ec|wp|sc|sp|wg|sl|gr|dp|fs|cs))\b|\b(ec|wp|sc|sp|wg|sl|gr|dp|fs|cs)\b",
        combined_query
    )
    req_formulation = None
    if formulation_match:
        raw_form = formulation_match.group(1) or formulation_match.group(2)
        req_formulation = re.sub(r"\s+", "", raw_form)

    # Strip formulation before finding rate digits so formulation digits (e.g. 20 EC) are not treated as rate
    rate_only = re.sub(r"\b\d+\s*(?:ec|wp|sc|sp|wg|sl|gr|dp|fs|cs)\b", "", norm_rate)
    digits = re.findall(r"\d+(?:\.\d+)?", rate_only)
    if not digits:
        return False

    unit_defs = [
        (r"kg\s*(?:\/|\s*per\s*)\s*ha|kg\s*ha⁻¹", r"(?:kg\s*(?:\/|\s*per\s*)\s*ha|kg\s*ha⁻¹|kg\s*per\s*hectare)"),
        (r"l\s*(?:\/|\s*per\s*)\s*ha|l\s*ha⁻¹|litres?\s*(?:\/|\s*per\s*)\s*ha", r"(?:l\s*(?:\/|\s*per\s*)\s*ha|l\s*ha⁻¹|l\s*per\s*ha|litres?\s*(?:\/|\s*per\s*)\s*ha)"),
        (r"g(?:m)?\s*(?:\/|\s*per\s*)\s*l(?:itre)?|g(?:m)?\s*l⁻¹", r"(?:g(?:m)?\s*(?:\/|\s*per\s*)\s*l(?:itre)?|g(?:m)?\s*l⁻¹|grams?\s*per\s*litre)"),
        (r"ml\s*(?:\/|\s*per\s*)\s*l(?:itre)?|ml\s*l⁻¹", r"(?:ml\s*(?:\/|\s*per\s*)\s*l(?:itre)?|ml\s*l⁻¹|ml\s*per\s*litre)"),
        (r"cm|centimeters|centimetres", r"(?:cm|centimeters?|centimetres?)"),
        (r"%", r"%"),
        (r"ppm", r"ppm"),
        (r"lakh|million", r"(?:lakh|million)(?:\s*(?:\/|\s*per\s*)\s*ha)?"),
        (r"cfu(?:\s*(?:\/|\s*per\s*)\s*(?:g|ml))?", r"cfu(?:\s*(?:\/|\s*per\s*)\s*(?:g|ml))?")
    ]

    matched_unit_regex = None
    for pat, u_regex in unit_defs:
        if re.search(pat, norm_rate):
            matched_unit_regex = u_regex
            break

    # Fail closed: a valid dosage rate MUST match a recognized agronomic unit
    if not matched_unit_regex:
        return False

    item_words = [w for w in re.split(r"\W+", norm_item) if len(w) > 3]

    # Crop aliases and competing crop definitions
    other_crops = [
        "cotton", "sugarcane", "tea", "coffee", "apple", "grape", "wheat",
        "maize", "soybean", "groundnut", "mustard", "potato", "tomato", "chilli",
        "cabbage", "cauliflower", "pulses", "chickpea", "pigeonpea"
    ]
    crop_aliases = {
        "rice": ["rice", "paddy", "oryza"],
        "cotton": ["cotton"],
        "wheat": ["wheat"],
        "maize": ["maize", "corn"],
    }
    target_aliases = crop_aliases.get(target_crop.lower(), [target_crop.lower()]) if target_crop else []
    all_competing_treatments = _get_all_competing_treatment_items()

    for c in chunks:
        freshness = c.get("freshness_label", "UNKNOWN")
        temp_score = c.get("temporal_score", c.get("final_score", 0))
        if freshness in ["STALE", "OUTDATED", "UNKNOWN"] or temp_score < 0.70:
            continue

        doc_text = (c.get("text") or c.get("content") or "").lower()
        if not doc_text:
            continue

        # 1. Target crop consistency: reject if chunk clearly belongs to another crop without target crop reference
        if target_crop:
            doc_source = (c.get("source_file") or c.get("source") or "").lower()
            doc_has_target = any(alias in doc_text or alias in doc_source for alias in target_aliases)
            doc_has_other = any(re.search(r"\b" + re.escape(oc) + r"\b", doc_text) for oc in other_crops if oc != target_crop.lower())
            if doc_has_other and not doc_has_target:
                continue

        search_terms = item_words if item_words else [norm_item]
        for term in search_terms:
            for m_item in re.finditer(r"\b" + re.escape(term) + r"\b", doc_text):
                start = max(0, m_item.start() - 140)
                end = min(len(doc_text), m_item.end() + 140)
                window = doc_text[start:end]

                # 2. Local window must not attribute treatment to another crop
                if target_crop:
                    win_has_target = any(alias in window for alias in target_aliases)
                    win_has_other = any(re.search(r"\b" + re.escape(oc) + r"\b", window) for oc in other_crops if oc != target_crop.lower())
                    if win_has_other and not win_has_target:
                        continue

                # 3. Formulation validation
                if req_formulation:
                    compact_window = re.sub(r"\s+", "", window)
                    if req_formulation not in compact_window:
                        continue

                # 4. Bound Rate & Unit validation with Syntactic Association
                if len(digits) >= 2:
                    bound_pat = rf"\b{re.escape(digits[0])}\s*(?:-|–|to)\s*{re.escape(digits[1])}\s*{matched_unit_regex}"
                else:
                    bound_pat = rf"\b{re.escape(digits[0])}\b(?:\s*(?:-|–|to)\s*\d+(?:\.\d+)?)?\s*{matched_unit_regex}"

                rate_matches = list(re.finditer(bound_pat, window))
                if not rate_matches:
                    continue

                # Association check: ensure matched rate directly associates with treatment item
                rate_associated = False
                for m_rate in rate_matches:
                    abs_rate_start = start + m_rate.start()
                    abs_rate_end = start + m_rate.end()

                    span_start = min(m_item.start(), abs_rate_start)
                    span_end = max(m_item.end(), abs_rate_end)
                    between_text = doc_text[span_start:span_end]

                    # Reject if sentence boundary, newline, semicolon, or 3+ table pipes intervene
                    has_boundary = bool(re.search(r"(?:\n|\;|\.(?!\d)\s+)", between_text)) or between_text.count("|") >= 3
                    if has_boundary:
                        continue

                    # Reject if another competing chemical/fertilizer/biocontrol treatment item intervenes
                    intervening_treatment = False
                    for oth in all_competing_treatments:
                        if oth != term and oth not in norm_item and not any(pw in oth for pw in item_words):
                            if re.search(r"\b" + re.escape(oth) + r"\b", between_text):
                                intervening_treatment = True
                                break
                    if intervening_treatment:
                        continue

                    rate_associated = True
                    break

                if not rate_associated:
                    continue

                # 5. Crop / Pest context consistency
                if target_pest:
                    norm_pest = target_pest.lower()
                    pest_words = [w for w in re.split(r"\W+", norm_pest) if len(w) > 3]
                    has_target_pest_near = any(pw in window for pw in pest_words)
                    if not has_target_pest_near:
                        continue

                    distinct_pests = [
                        "stem borer", "leaf folder", "gall midge", "mealy bug", "mealybug",
                        "brown plant hopper", "bph", "root nematode", "hispa", "blast"
                    ]
                    conflicting_pests_in_window = [
                        dp for dp in distinct_pests
                        if dp in window and not any(pw in dp for pw in pest_words)
                    ]
                    if conflicting_pests_in_window:
                        continue

                return True

    return False


BIO_TERMS = [
    "trichogramma", r"t\.?\s*chilonis", r"t\.?\s*japonicum", "beauveria", r"b\.?\s*bassiana",
    "metarhizium", "verticillium", "lecanicillium",
    "pseudomonas", "bacillus", "bt", "hanpv", "slnpv", "npv",
    "chrysoperla", "bracon", "eocanthecona", "platygaster"
]


def _scrub_pest_parasitoid_mismatches(text: str, chunks: list = None) -> str:
    """
    Qualifies or removes pest-parasitoid mismatches (e.g. Trichogramma / T. chilonis recommended for
    Hemipteran sucking pests like Brown Plant Hopper, rice mealybug, green leafhopper,
    or nematodes).
    """
    lines = text.splitlines()
    new_lines = []

    sucking_pest_terms = [
        "brown plant hopper", "brown planthopper", "bph", "nilaparvata lugens",
        "rice mealy bug", "rice mealybug", "mealy bug", "mealybug", "brevennia rehi",
        "green leaf hopper", "green leafhopper", "glh", "nephotettix",
        "sucking pest", "nematode", "hirschmanniella"
    ]

    current_pest_context = None

    for line in lines:
        lower_line = line.lower()

        # If line is already qualified, keep as is
        if "not recommended for sucking pests" in lower_line or "unsuited for sucking pests" in lower_line:
            new_lines.append(line)
            continue

        # Check if line sets a pest context
        for sp in sucking_pest_terms:
            if sp in lower_line:
                current_pest_context = sp
                break
        # Lepidopteran resets context
        if any(lp in lower_line for lp in ["stem borer", "leaf folder", "lepidoptera", "caterpillar"]):
            current_pest_context = None

        has_tricho = bool(re.search(r"(?i)\b(?:trichogramma|t\.?\s*chilonis|t\.?\s*japonicum)\b", line))
        line_has_sucking = any(sp in lower_line for sp in sucking_pest_terms)
        is_mismatch = has_tricho and (line_has_sucking or current_pest_context is not None)

        if is_mismatch:
            if _is_rate_verified_in_context("Trichogramma", line, chunks):
                new_lines.append(line)
                continue

            # Table row mismatch: qualify Trichogramma specifically while preserving other organisms in cell
            if line.strip().startswith("|") and line.strip().endswith("|"):
                line = re.sub(
                    r"(?i)(?:release\s+)?[\*\_]*(?:trichogramma|t\.?\s*chilonis|t\.?\s*japonicum)[\*\_]*(?:\s+spp\.?|\s+japonicum|\s+chilonis)?(?:\s*\([^\)]*\))?",
                    "Trichogramma (note: egg parasitoid of lepidopteran borers, not recommended for sucking pests; consult KVK)",
                    line
                )
            else:
                # Bullet or text line: qualify Trichogramma
                line = re.sub(
                    r"(?i)(?:release\s+|apply\s+)?[\*\_]*(?:trichogramma|t\.?\s*chilonis|t\.?\s*japonicum)[\*\_]*(?:\s+spp\.?|\s+japonicum|\s+chilonis)?(?:[^\(\,\.\;]*\([^\)]*\))?",
                    "Consult local KVK for verified biological control options (unverified possibility: general university references mention predatory mirid bugs or spiders, but these are not verified in retrieved context; note: Trichogramma is an egg parasitoid of lepidopteran borers and unsuited for sucking pests)",
                    line
                )
        new_lines.append(line)

    return "\n".join(new_lines)


def _scrub_unverified_agronomic_claims(text: str, chunks: list = None) -> str:
    """
    Clearly labels or qualifies unverified agronomic practices (e.g. AWD, specific iron chelates,
    crop rotations, and tiller destruction) when not explicitly verified in context.
    Does not silently substitute unsupported treatments. Preserves claims if explicitly present in chunks.
    """
    if not text:
        return text

    # Check whether any retrieved chunk actually contains verified documentation
    awd_in_context = any(
        bool(re.search(r"\b(?:alternate\s+wetting\s+and\s+drying|awd)\b", (c.get("text") or c.get("content") or "").lower()))
        for c in (chunks or [])
    )
    chelate_in_context = any(
        bool(re.search(r"\b(?:iron\s+chelates?|fe-eddha|moringa\s+leaf\s+mulch)\b", (c.get("text") or c.get("content") or "").lower()))
        for c in (chunks or [])
    )
    # Sentence-bound tiller destruction verification: requires same-sentence directive to remove/destroy tillers
    tiller_actions = r"(?:remove|destroy|cutting|cut|uproot|pull|clip|clipping|pulling|uprooting|dispose\s+of)"
    tiller_targets = r"(?:tillers?|infested\s+tillers?|affected\s+tillers?)"
    tiller_pat_in_context = (
        rf"\b{tiller_actions}\b[^\.\;\n\|]{{0,60}}\b{tiller_targets}\b|"
        rf"\b{tiller_targets}\b[^\.\;\n\|]{{0,60}}\b(?:should\s+be|must\s+be|are|be)\s+(?:removed|destroyed|cut|clipped|uprooted|pulled)\b"
    )
    tiller_in_context = any(
        bool(re.search(tiller_pat_in_context, c.get("text") or c.get("content") or "", re.IGNORECASE))
        for c in (chunks or [])
    )
    rotation_in_context = any(
        bool(re.search(r"\b(?:crop\s+rotation|rotate\s+rice\s+with)\b", (c.get("text") or c.get("content") or "").lower()))
        for c in (chunks or [])
    )

    # 3. Tiller removal / destruction for mealybug (including line-wrapped and synonym variants)
    if not tiller_in_context:
        action_verbs = r"(?:remove|destroy|uproot|pull|cut|clip|removing|destroying|uprooting|pulling|cutting|clipping)"
        tiller_pat = (
            rf"(?i)\b{action_verbs}(?:\s+(?:and|or)\s+{action_verbs})?"
            rf"(?:[^\.\;\:\?!\|\n]{{0,35}}(?:\n\s*[^\.\;\:\?!\|\n]{{0,35}})?)\s*"
            rf"\b(?:the\s+|all\s+|any\s+)?(?:heavily\s+|severely\s+)?(?:infested|affected|damaged)?\s*tillers\b"
        )
        tiller_rev_pat = (
            r"(?i)\b(?:the\s+|all\s+|any\s+)?(?:heavily\s+|severely\s+)?(?:infested|affected|damaged)?\s*tillers\s+"
            r"(?:should|must|can)?\s*be\s+(?:removed|destroyed|uprooted|pulled|cut|clipped)\b"
        )
        tiller_replacement = (
            "practice field sanitation and clear crop residues "
            "(tiller removal is unverified in retrieved context; consult KVK for mealybug management)"
        )
        text = re.sub(tiller_pat, tiller_replacement, text)
        text = re.sub(tiller_rev_pat, tiller_replacement, text)

    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if "unverified in retrieved context" in line.lower():
            new_lines.append(line)
            continue

        # 1. Alternate Wetting and Drying (AWD)
        if not awd_in_context:
            awd_pat = r"(?i)\b(?:practice\s+)?(?:alternate\s+wetting\s+and\s+drying(?:\s*\(\s*AWD\s*\))?|AWD)(?:\s+or\s+other\s+water-saving\s+regimes)?(?:\s+as\s+advised\s+in\s+the\s+state\s+Package\s+of\s+Practices)?"
            line = re.sub(
                awd_pat,
                "maintain proper field drainage and avoid prolonged stagnant water (consult local KVK for water management practices; note: AWD is not verified in retrieved context)",
                line
            )

        # 2. Specific iron chelates / Moringa mulch
        if not chelate_in_context:
            chelate_pat = r"(?i)(?:apply\s+|spray\s+)?(?:iron\s+chelates?|Fe-EDDHA|moringa\s+leaf\s+mulch)(?:\s*\([^\)]*(?:fe-eddha|chelates?)[^\)]*\))?(?:\s+(?:foliar\s+)?(?:spray|application))?(?:\s*(?:at|@)\s*[\d\.\-–]+\s*(?:g|gm|kg|ml|%)[^\,\.\;\n\|]*)?"
            line = re.sub(
                chelate_pat,
                "consult local KVK for soil-test-based nutrient amendments (specific iron chelate formulations are not documented in retrieved context)",
                line
            )

        # 4. Crop rotation with pulses/millets
        if not rotation_in_context:
            rotation_pat = r"(?i)\b(?:rotate\s+rice\s+with\s+(?:non-host\s+crops\s*\(?e\.g\.,\s*)?(?:pulses|millets|oilseeds)[^\.\;\n]*\)?)"
            line = re.sub(
                rotation_pat,
                "consult local KVK for approved nematode-suppressive crop rotation options (crop rotation sequences are unverified in retrieved context)",
                line
            )
        new_lines.append(line)

    return "\n".join(new_lines)


def _scrub_water_depths(text: str, chunks: list = None) -> str:
    """
    Suppresses unverified numerical water depth recommendations (e.g. 5–7 cm, 10–12 cm).
    Explicitly restricted to measurements describing water depth, standing water,
    submergence, or water level. Preserves non-water distance/spacing/height measurements
    (e.g., '15–20 cm spacing', '30 cm above canopy', '5 cm water pipe').
    """
    water_replacement = (
        "maintain proper field drainage and recommended water levels based on state Package of Practices "
        "(consult local KVK for field-specific irrigation guidance; avoid prolonged standing water)"
    )

    # 1. Columnar table scrubber: detects column indices with "water depth" or "standing water"
    # in markdown table header rows and scrubs invalid values in matching column cells.
    lines = text.split("\n")
    new_lines = []
    target_cols = set()
    in_table = False

    water_header_pat = re.compile(
        r"(?i)\b(?:standing\s+water|water\s+depth|depth\s+of\s+water|water\s+level|submergence)\b"
    )
    val_pat = re.compile(
        r"(?i)\b(\d+\s*(?:[–\-]|to)\s*\d+|\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|inches)\b"
    )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            parts = line.split("|")
            cells = parts[1:-1]
            is_sep = all(re.match(r"^\s*:?-+:?\s*$", c) for c in cells) if cells else False

            if not in_table and not is_sep:
                in_table = True
                target_cols = set()
                for idx, cell in enumerate(cells):
                    if water_header_pat.search(cell):
                        target_cols.add(idx)
                new_lines.append(line)
            elif is_sep:
                new_lines.append(line)
            else:
                if target_cols:
                    new_cells = list(cells)
                    for c_idx in target_cols:
                        if c_idx < len(new_cells):
                            cell_val = new_cells[c_idx]
                            m = val_pat.search(cell_val)
                            if m:
                                rate_phrase = f"{m.group(1)} {m.group(2)}"
                                if not _is_rate_verified_in_context("water", rate_phrase, chunks):
                                    new_cells[c_idx] = val_pat.sub(
                                        "recommended levels (consult KVK; ensure proper drainage)",
                                        cell_val
                                    )
                    new_lines.append("|" + "|".join(new_cells) + "|")
                else:
                    new_lines.append(line)
        else:
            in_table = False
            target_cols = set()
            new_lines.append(line)

    text = "\n".join(new_lines)

    # 2. Key-value table cell pattern: table cell with water depth label followed by adjacent measurement cell
    # e.g. "| Water Depth | 5–7 cm |" or "| Standing Water | 5 cm |"
    p_table = (
        r"(?i)(\|\s*(?:(?!consult\s+KVK)[^\n\|]*?\b(?:standing\s+water|water\s+depth|depth\s+of\s+water|water\s+level|submergence)\b[^\n\|]*?)\|\s*)"
        r"([^\n\|]*?\b(\d+\s*(?:[–\-]|to)\s*\d+|\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|inches)\b[^\n\|]*?)"
        r"(\s*\|)"
    )

    def _repl_table(m):
        prefix = m.group(1)
        raw_cell = m.group(2)
        num = m.group(3)
        unit = m.group(4)
        suffix = m.group(5)
        rate_phrase = f"{num} {unit}"
        if _is_rate_verified_in_context("water", rate_phrase, chunks):
            return m.group(0)
        new_cell = re.sub(
            rf"\b{re.escape(num)}\s*{re.escape(unit)}\b",
            "recommended levels (consult KVK; ensure proper drainage)",
            raw_cell
        )
        return f"{prefix}{new_cell}{suffix}"

    # 3. Prefix pattern: explicit water context preceding numerical measurement
    # e.g. "maintain standing water of 5-7 cm", "water depth of 10 cm", "maintain water depth at 5 cm"
    p1 = (
        r"(?i)\(?\s*(?:\b(?:maintain|keep|ensure|hold|provide)\s+)?\b"
        r"(?:standing\s+water|water\s+depth|depth\s+of\s+water|water\s+level|submergence|irrigation\s+water)"
        r"(?:\s+(?:of|at|around|to))?\s+"
        r"(\d+\s*(?:[–\-]|to)\s*\d+|\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|inches)\b\s*\)?"
    )

    # 4. Suffix pattern: numerical measurement followed by explicit water context
    # e.g. "5-7 cm of standing water", "10 cm water depth", "5 cm submergence", "5 cm depth of water"
    # Excludes non-depth water phrases: pipe, piping, tube, tubing, trap, channel, pump, furrow, etc.
    p2 = (
        r"(?i)\(?\s*\b(\d+\s*(?:[–\-]|to)\s*\d+|\d+(?:\.\d+)?)\s*(cm|centimeters?|centimetres?|inches)\s+"
        r"(?:depth\s+of\s+water|(?:\s*of\s+)?(?:standing\s+water|water\s+depth|submergence|water\s+level|"
        r"standing\s+irrigation(?:\s+water)?|water\s+column|"
        r"water(?!\s*(?:spacing|gap|apart|height|tall|distance|interval|deep\s+furrow|deep\s+trench|"
        r"ploughing|plowing|pipe|piping|tube|tubing|trap|channel|furrow|trench|line|pump|source|table|harvest|harvesting))))\b\s*\)?"
    )

    def _repl_depth(m):
        raw = m.group(0)
        num = m.group(1)
        unit = m.group(2)
        rate_phrase = f"{num} {unit}"
        if _is_rate_verified_in_context("water", rate_phrase, chunks):
            return raw
        return water_replacement

    text = re.sub(p_table, _repl_table, text)
    text = re.sub(p1, _repl_depth, text)
    text = re.sub(p2, _repl_depth, text)
    return text


def _scrub_destructive_practices(text: str, chunks: list = None) -> str:
    """
    Suppresses unsupported destructive burning advice, converting to safe field sanitation.
    """
    burn_pat = (
        r"(?i)\b(?:remove\s+and\s+burn|burn)\s+((?:(?:infected|infested|affected|the|all)\s+)*(?:plants|stubbles|straw|residues|leaves|tillers|crops|debris)(?:\s+and\s+(?:tillers|plants|leaves|residues|stubbles))?)"
    )
    def _repl_burn(m):
        target = m.group(1).strip()
        raw = m.group(0)
        if _is_rate_verified_in_context("burn", raw, chunks):
            return raw
        return f"remove and safely dispose of {target} through field sanitation (avoid burning crop residues)"

    text = re.sub(burn_pat, _repl_burn, text)
    text = re.sub(
        r"(?i)\b(?:remove\s+and\s+burn|burn\s+them)\b",
        "remove and safely clear affected plants through field sanitation (avoid burning)",
        text
    )
    text = re.sub(
        r"(?i)\b(?:open\s+)?burning\s+of\s+(?:crop\s+residues|stubbles|straw|plants|tillers)\b",
        "safely clearing crop residues through field sanitation and composting (avoid burning)",
        text
    )
    return text


def _scrub_bio_rates(line: str, chunks: list = None) -> str:
    """
    Deterministically sanitizes unverified/stale numerical biocontrol rates
    both inside and outside parentheses, while preserving organism names and safe guidance.
    """
    bio_regex = r"(?i)([\*\_]*\b(?:" + "|".join(BIO_TERMS) + r")(?:[\*\_]*\s+[\*\_]*(?:spp\.?|sp\.?|(?!(?:at|in|on|by|for|is|to|with)\b)[a-z]+))?[\*\_]*)"

    def _repl_paren(m):
        organism = m.group(1).strip()
        rate_part = m.group(2).strip()
        if _is_rate_verified_in_context(organism, rate_part, chunks):
            return m.group(0)
        if any(tok in organism.lower() for tok in ["tricho", "chilonis", "japonicum"]):
            return f"{organism} (consult local KVK for current recommended release rate)"
        return f"{organism} (consult local KVK for approved formulation and application rate)"

    paren_pat = bio_regex + r"\s*\(\s*(?:[≈~@]|at\s+)?\s*([0-9][^\)]*?(?:lakh|million|thousand|k|g|gm|ml|kg|l⁻¹|%|cfu)[^\)]*?)\)"
    line = re.sub(paren_pat, _repl_paren, line)

    def _repl_outside(m):
        organism = m.group(1).strip()
        sep = m.group(2)
        rate_part = m.group(3).strip()
        if _is_rate_verified_in_context(organism, rate_part, chunks):
            return m.group(0)
        is_table = "|" in sep
        if any(tok in organism.lower() for tok in ["tricho", "chilonis", "japonicum"]):
            ref = "consult local KVK for recommended release rate"
        else:
            ref = "consult local KVK for approved application rate"
        if is_table:
            return f"{organism} | {ref}"
        return f"{organism} ({ref})"

    outside_pat = (
        bio_regex
        + r"(\s*\|\s*|\s+(?:at|@|with a dose of|with a rate of)\s*|\s+)"
        + r"([0-9][^\s,\.;\|)]*(?:\s*(?:lakh|million|thousand|g|gm|ml|kg|l⁻¹|%))?(?:\s*(?:per|\/|\s*)\s*(?:ha⁻¹|l⁻¹|ha|hectare|acre|l|liter|litre|kg))?)"
    )
    line = re.sub(outside_pat, _repl_outside, line)

    line = re.sub(
        r"(?i)\s*[-–—]?\s*note that the exact rates are from historical references[;,]?\s*verify current recommended rates with your kvk\.?",
        "",
        line
    )
    return line


def _scrub_fertilizer_rates(line: str, chunks: list = None) -> str:
    """
    Sanitizes blanket unverified fertilizer rates from historical or stale sources
    into soil-test based recommendations.
    """
    npk_pat = (
        r"(?i)(?:apply\s+)?\b\d+[\d,\.]*\s*kg\s*N(?:,\s*and\s+|,\s*|\s+and\s+)\d+[\d,\.]*\s*kg\s*P2O5"
        r"(?:,\s*and\s+|,\s*|\s+and\s+)\d+[\d,\.]*\s*kg\s*K2O(?:\s*per\s*(?:ha|hectare|acre))?"
    )
    m = re.search(npk_pat, line)
    if m:
        if not _is_rate_verified_in_context("NPK", m.group(0), chunks):
            line = re.sub(
                npk_pat,
                "apply balanced NPK fertilizer based on soil testing and current state Package of Practices (POP) (consult local KVK for field-specific rates)",
                line
            )
    return line


def reconcile_chemical_recommendations(answer_text: str, is_offline_reg: bool = True, chunks: list = None) -> str:
    """
    Ensures that under OFFLINE_STATIC_KB or unverified regulatory status:
    1. Synthetic chemical recommendations (by common name, brand name, dosage, or spray rate)
       are never presented as actionable instructions, ensuring 100% agreement with compliance scanner.
       Actionable advice is removed or blocked, rather than merely appending a warning.
    2. Unverified or stale biological treatment rates (e.g., Trichogramma, Beauveria) and blanket
       fertilizer rates are suppressed into qualitative guidance with KVK/extension referral,
       while preserving verified rates from current authoritative context.
    3. Pest-parasitoid mismatches (Trichogramma for BPH/mealybugs/sucking pests/nematodes) are
       corrected or qualified with natural predator/KVK guidance.
    4. Unsupported numerical water depths (e.g. 5–7 cm, 10–12 cm) are suppressed unless verified.
    5. Destructive burning advice is converted to safe field sanitation and composting.
    """
    if not answer_text:
        return answer_text

    synthetic_terms = _get_synthetic_pesticide_names()

    lines = answer_text.splitlines()
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        lower_line = line.lower()
        contains_synthetic = any(
            re.search(rf"\b{re.escape(term)}\b", lower_line) for term in synthetic_terms
        ) if is_offline_reg else False

        # Lookahead for line-break evasion: line i has chemical, line i+1 has dosage/rate
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        lower_next = next_line.lower()
        next_has_rate = any(
            kw in lower_next for kw in [
                "spray", "apply", "dose", "dosage", "rate", "%", "l/ha", "kg/ha", "ml/l", "g/l", "gm/l", "ppm", "@"
            ]
        ) if next_line else False

        # 1. Table row formatting: if row contains synthetic chemical or chemical spray dosage
        if line.strip().startswith("|") and line.strip().endswith("|"):
            if is_offline_reg:
                has_dosage_pattern = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:ml(?:\s*\/\s*l)?|g(?:m)?(?:\s*\/\s*l)?|kg(?:\s*\/\s*ha)?|l(?:\s*\/\s*ha)?|%|ppm)\b", lower_line))
                has_pesticide_keyword = any(kw in lower_line for kw in ["spray", "pesticide", "insecticide", "fungicide", "chemical", "ec", "wp", "sc", "drench"])
                has_table_dosage = has_dosage_pattern and has_pesticide_keyword
                if contains_synthetic or has_table_dosage:
                    parts = [p.strip() for p in line.strip().split("|")[1:-1]]
                    num_cols = len(parts)
                    if num_cols == 1:
                        new_parts = ["⛔ UNVERIFIED / BLOCKED (Chemical treatment withheld under OFFLINE_STATIC_KB; consult KVK)"]
                    elif num_cols == 2:
                        new_parts = [parts[0], "⛔ UNVERIFIED / BLOCKED (Chemical treatment withheld under OFFLINE_STATIC_KB; consult KVK)"]
                    elif num_cols == 3:
                        new_parts = [
                            parts[0],
                            "⛔ UNVERIFIED / BLOCKED",
                            "Live CIB&RC registration is unverified under OFFLINE_STATIC_KB. Actionable chemical sprays are blocked. Consult local KVK for approved products."
                        ]
                    elif num_cols == 4:
                        new_parts = [
                            parts[0],
                            "⛔ UNVERIFIED / BLOCKED",
                            "Chemical treatment withheld (OFFLINE_STATIC_KB)",
                            "Consult local KVK for approved products."
                        ]
                    else:
                        # num_cols >= 5: Preserve exact column count
                        new_parts = [parts[0], "⛔ UNVERIFIED / BLOCKED", "Chemical treatment withheld (OFFLINE_STATIC_KB)"]
                        while len(new_parts) < num_cols - 1:
                            new_parts.append("Dosage withheld (unverified live)")
                        new_parts.append("Consult local KVK for approved products.")

                    line = "| " + " | ".join(new_parts) + " |"

            # Apply biological and fertilizer rate reconciliation to table cells
            line = _scrub_bio_rates(line, chunks=chunks)
            line = _scrub_fertilizer_rates(line, chunks=chunks)
            line = _scrub_pest_parasitoid_mismatches(line, chunks=chunks)
            line = _scrub_water_depths(line, chunks=chunks)
            line = _scrub_destructive_practices(line, chunks=chunks)
            line = _scrub_unverified_agronomic_claims(line, chunks=chunks)
            new_lines.append(line)
            i += 1
            continue

        blocked_msg = (
            "• **Chemical treatment blocked (OFFLINE_STATIC_KB):** Live CIB&RC registration is unverified. "
            "Synthetic chemical prescriptions and dosages are withheld and blocked without live regulatory verification. "
            "Consult local Krishi Vigyan Kendra (KVK) for current label-approved options."
        )

        # 2. Check for actionable prescription keywords
        is_prescription = any(
            kw in lower_line for kw in [
                "spray", "apply", "dose", "dosage", "rate", "application", "drench", "dust",
                "%", "l/ha", "kg/ha", "ml/l", "g/l", "gm/l", "ppm", "@", "ec", "wp", "sc"
            ]
        )

        # 3. Line break evasion: chemical on line i, prescription rate on line i+1 (or both)
        if contains_synthetic and next_has_rate:
            new_lines.append(blocked_msg)
            i += 2  # consume both chemical and rate line
            continue

        if contains_synthetic:
            # If an actionable prescription is present, block the entire prescription line.
            # Disclaimers like 'unverified', 'historical', or 'withheld' must NOT prevent blocking.
            if is_prescription:
                line = blocked_msg
            else:
                # Purely historical or descriptive reference: remove specific brand/chemical names and dosages
                line = re.sub(
                    r"(?i)\*{0,2}(?:spray\s+)?dimethoate\*{0,2}[\s\u202f\xa0]*(?:\*{0,2}0?\.05\s*%\*{0,2})?",
                    "historical chemical treatments (withheld under OFFLINE_STATIC_KB)",
                    line
                )
                for term in synthetic_terms:
                    line = re.sub(rf"(?i)\b{re.escape(term)}\b", "[chemical treatment withheld]", line)

        # 4. Line break for biological rates: line i has bio organism, line i+1 has rate
        has_bio_on_line = any(re.search(rf"(?i)\b{re.escape(bt)}\b", lower_line) for bt in BIO_TERMS)
        bio_rate_next_match = re.match(
            r"^\s*(\(?\s*(?:[≈~@]|at\s+)?\s*\d+[\d,\.]*\s*(?:lakh|million|thousand|k|g|gm|ml|kg|l⁻¹|%)[^\)]*?\)?)(.*)$",
            next_line,
            re.IGNORECASE
        )
        if has_bio_on_line and bio_rate_next_match:
            rate_candidate = bio_rate_next_match.group(1)
            remainder = bio_rate_next_match.group(2)
            if not _is_rate_verified_in_context(line, rate_candidate, chunks):
                lines[i + 1] = f"  (consult local KVK for current recommended rate){remainder}"

        # 5. Sanitize biological and fertilizer rates on the current line
        line = _scrub_bio_rates(line, chunks=chunks)
        line = _scrub_fertilizer_rates(line, chunks=chunks)

        new_lines.append(line)
        i += 1

    sanitized_text = "\n".join(new_lines)
    # Apply full-context agronomic integrity scrubbers
    sanitized_text = _scrub_pest_parasitoid_mismatches(sanitized_text, chunks=chunks)
    sanitized_text = _scrub_water_depths(sanitized_text, chunks=chunks)
    sanitized_text = _scrub_destructive_practices(sanitized_text, chunks=chunks)
    sanitized_text = _scrub_unverified_agronomic_claims(sanitized_text, chunks=chunks)
    return sanitized_text


reconcile_treatment_recommendations = reconcile_chemical_recommendations


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
    if not chunks:
        return {
            "answer": (
                "⚠️ Insufficient verified context: No verified agricultural documents were available to answer this query. "
                "Please consult your local Krishi Vigyan Kendra (KVK) or State Agricultural University extension."
            ),
            "model_used": "system/insufficient-context",
            "sources": [],
            "path": "slow",
        }

    # Build rich context from top 5 chunks with explicit freshness tags
    context_parts = []
    sources       = []

    for chunk in chunks[:5]:
        freshness = chunk.get("freshness_label", "UNKNOWN")
        src = chunk.get("source_file") or chunk.get("source", "unknown")
        if freshness == "STALE":
            tag = f"[Source: {src} | STATUS: HISTORICAL / STALE (published ~{chunk.get('pub_year', 'prior years')}) - DISCLOSED AS HISTORICAL REFERENCE ONLY; DO NOT USE FOR ACTIVE CHEMICAL PRESCRIPTION]"
        elif freshness == "OUTDATED":
            tag = f"[Source: {src} | STATUS: OUTDATED - HISTORICAL ARCHIVE ONLY]"
        else:
            tag = f"[Source: {src} | Freshness: {freshness}]"
        chunk_text = chunk.get('text', '')
        if getattr(config, "ENABLE_CONTEXT_COMPRESSION", False):
            try:
                from components.context.compression import distill_chunk_text
                chunk_text, _, _ = distill_chunk_text(query, chunk_text)
            except Exception:
                chunk_text = chunk.get('text', '')

        context_parts.append(f"{tag}\n{chunk_text}")
        if src and src not in sources:
            sources.append(src)

    context = "\n\n---\n\n".join(context_parts)

    # Check regulatory verification status
    try:
        from regulatory_updater import get_regulatory_verification_status
        reg_status = get_regulatory_verification_status()
    except Exception:
        reg_status = {"status": "OFFLINE_STATIC_KB", "is_offline": True}
    is_offline_reg = reg_status.get("status") == "OFFLINE_STATIC_KB" or reg_status.get("is_offline", True)

    regulatory_prompt_injection = ""
    if is_offline_reg:
        regulatory_prompt_injection = (
            "\n\nCRITICAL REGULATORY SAFETY MANDATE (OFFLINE_STATIC_KB):\n"
            "Live CIB&RC and EU SANTE regulatory verification is currently UNAVAILABLE (operating in offline static database mode).\n"
            "- You MUST NOT prescribe synthetic chemical pesticide products or dosages (e.g., Dimethoate, Carbaryl, Chlorpyrifos, or spray percentages) as actionable spray instructions.\n"
            "- In particular, for rice mealy-bug or any insect pest, do NOT present Dimethoate or other chemical sprays as actionable treatments.\n"
            "- Under chemical intervention, state clearly: '⚠️ Live CIB&RC regulatory verification is currently unavailable (offline static database). Synthetic chemical treatments and dosages are withheld without live regulatory verification. Consult your local Krishi Vigyan Kendra (KVK) or State Agricultural University extension officer for legally registered options.'\n"
            "- Emphasize non-chemical cultural, physical, and biological management."
        )

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
                    {regulatory_prompt_injection}
                    
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
            max_tokens  = 2200,
        )

        content = response.choices[0].message.content or ""
        if not content and hasattr(response.choices[0].message, "reasoning"):
            content = response.choices[0].message.reasoning or ""

        # Validate complete markdown output: remove dangling table separators or unfinished sections
        if content:
            lines = content.splitlines()
            while lines:
                last_line = lines[-1].strip()
                if not last_line:
                    lines.pop()
                elif last_line.startswith("|") and not last_line.endswith("|"):
                    lines.pop()
                elif last_line.startswith("|") and last_line.replace("|", "").replace("-", "").replace(":", "").strip() == "":
                    lines.pop()
                elif last_line.startswith("**💊 CHEMICAL INTERVENTION") or last_line == "---":
                    lines.pop()
                else:
                    break
            content = "\n".join(lines).strip()

            # Ensure agreement between regulatory status and answer
            content = reconcile_chemical_recommendations(content, is_offline_reg=is_offline_reg, chunks=chunks)
        else:
            content = (
                "⚠️ Generation incomplete: The model was unable to generate a complete answer within token constraints. "
                "Please re-run or specify your query with more field context."
            )

        return {
            "answer"    : content,
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

    # Dense search (top_k * 3 candidate pool)
    candidate_k = top_k * 3
    q_emb  = embedder.encode([query]).tolist()
    d_res  = collection.query(
        query_embeddings = q_emb,
        n_results        = candidate_k,
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

    # Sparse BM25 (top_k * 3 candidate pool)
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_i  = sorted(range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True)[:candidate_k]
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

    # Baseline Equal 0.5/0.5 fusion with trust weight
    for item in merged.values():
        raw = (0.5 * item["dense_score"] +
               0.5 * item["sparse_score"])
        trust = item["trust_weight"] if item["trust_weight"] > 0 else 1.0
        item["fusion_score"] = round(raw * trust, 4)
        item["final_score"]  = item["fusion_score"]

    # Stage 2: Cross-Encoder Reranking
    candidate_list = list(merged.values())
    if candidate_list:
        from step5_vector_index import get_cross_encoder
        ce = get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidate_list]
        ce_scores = ce.predict(pairs, batch_size=32)
        for c, score in zip(candidate_list, ce_scores):
            c["cross_encoder_score"] = round(float(score), 4)

        ranked = sorted(
            candidate_list,
            key    = lambda x: x["cross_encoder_score"],
            reverse= True
        )[:top_k]
    else:
        ranked = []

    return ranked


# ─────────────────────────────────────────────
# PART 6: MAIN QUERY GATE FUNCTION
# Ties everything together
# ─────────────────────────────────────────────

def query_gate(query, embedder=None, collection=None,
               bm25=None, corpus=None,
               enable_interactive_feedback=False):
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
    if not query or not str(query).strip():
        return {
            "status": "error",
            "error": "Empty or invalid query provided.",
            "model_used": "system/input-validation-guard",
            "answer": "Please provide a valid query describing the crop, symptom, or agricultural question."
        }

    if embedder is None or collection is None or bm25 is None or corpus is None:
        c_embedder, c_collection, c_bm25, c_corpus = get_or_load_components()
        embedder = embedder if embedder is not None else c_embedder
        collection = collection if collection is not None else c_collection
        bm25 = bm25 if bm25 is not None else c_bm25
        corpus = corpus if corpus is not None else c_corpus
    print(f"\n{'='*60}")
    print(f"🌾 AGRICULTURE RAG — QUERY GATE")
    print(f"{'='*60}")
    print(f"❓ Query: {query}")
    print(f"{'─'*60}")
    # ── STEP 0: Rural-to-Scientific Semantic Bridge ──
    try:
        from semantic_bridge import apply_semantic_bridge
        bridge_result = apply_semantic_bridge(query)
        if bridge_result.get("bridged"):
            query = bridge_result.get("enriched", query)
            print("   ✓ Query enriched with scientific taxonomy")
    except Exception:
        # Bridge unavailable — continue with original query
        pass

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
    is_fast_path = (path == "fast")

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

    # ── Step 4C: Temporal credibility decay ──
    retrieval_insufficient = False
    insufficient_reason = None
    scored_chunks = []

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

    if ENABLE_TEMPORAL_DECAY and chunks and not is_fast_path and q_type in ["RECOMMENDATION", "DIAGNOSTIC", "PROCEDURAL"]:
        print("\n⏳ Step 4C: Applying Temporal Credibility Decay...")
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
        if len(chunks_to_use) == 0 and len(chunks) > 0:
            print("   ⚠️  All candidate chunks rejected as outdated/stale under temporal credibility decay.")
            retrieval_insufficient = True
            insufficient_reason = "All retrieved documents were rejected as outdated or stale under temporal credibility evaluation."
    else:
        chunks_to_use = chunks

    # Replace `chunks` with `chunks_to_use` (never silently reintroduce rejected stale chunks)
    chunks = chunks_to_use

    # ── Step 4D: Phenological Gate (PGRA) ──
    try:
        from phenology_gate import apply_phenological_gate
        PHENO_AVAILABLE = True
    except Exception:
        PHENO_AVAILABLE = False

    if PHENO_AVAILABLE and chunks and q_type in ["RECOMMENDATION", "DIAGNOSTIC", "PROCEDURAL"]:
        print("\n🚧 Step 4D: Applying Phenological Gate...")

        # Extract location coordinates only if explicitly mentioned in query
        gate_lat, gate_lon = None, None
        if WEATHER_ENRICHMENT_AVAILABLE:
            try:
                from step8_weather_rag import LOCATION_PATTERNS
                for pat in LOCATION_PATTERNS:
                    m = re.search(pat, query)
                    if m:
                        explicit_loc = m.group(1)
                        from weather_fetcher import get_coordinates
                        coords = get_coordinates(explicit_loc)
                        if coords.get("found"):
                            gate_lat, gate_lon = coords["lat"], coords["lon"]
                        break
            except Exception:
                gate_lat, gate_lon = None, None

        try:
            gate_result = apply_phenological_gate(
                chunks=chunks,
                query=query,
                lat=gate_lat,
                lon=gate_lon
            )

            # Consolidate phenological stage detection: prioritize seasonal context / crop calendar
            # and bypass mock SATELLITE fallback that outputs contradictory winter/dormancy stages
            stage_val = str(gate_result.get("stage", ""))
            if (
                gate_result.get("stage_source") == "SATELLITE"
                or "winter" in stage_val.lower()
                or "dormancy" in stage_val.lower()
            ):
                try:
                    from crop_calendar import get_crop_stage
                    cal_info = get_crop_stage(query)
                    if cal_info and cal_info.get("stage") and cal_info.get("stage") != "unknown":
                        gate_result["stage"] = cal_info.get("stage")
                        gate_result["stage_source"] = "CALENDAR"
                except Exception:
                    pass

            chunks_to_use_after_gate = gate_result.get("allowed_chunks", [])

            print(f"\n   📋 Gate Summary:")
            print(f"   Stage        : {gate_result.get('stage')} ({gate_result.get('stage_source')})")
            print(f"   Input chunks : {gate_result.get('total_input')}")
            print(f"   After gate   : {gate_result.get('allowed_count')} allowed, {gate_result.get('blocked_count')} blocked")

            if len(chunks) > 0 and gate_result.get("allowed_count", 0) == 0:
                print("   ⚠️  All candidate chunks blocked by phenological stage incompatibility.")
                retrieval_insufficient = True
                insufficient_reason = f"All candidate chunks were incompatible with current crop stage ({gate_result.get('stage')})."
                chunks_to_use_after_gate = []

            # Use gate-filtered chunks downstream (NEVER fallback to blocked chunks)
            chunks = chunks_to_use_after_gate

        except Exception as e:
            print(f"   ⚠️  Phenological gate failed: {e}")
            chunks = chunks
    else:
        print("\n⏭️  Step 4D: Phenological gate skipped (non-crop or unavailable)")

    # ── Step 4B: Weather enrichment (for recommendations) ──
    weather_data = None
    weather_enrichment_applied = False

    if is_fast_path:
        print(f"\n⏭️  Step 4B: Skipped (FAST path)")
    elif WEATHER_ENRICHMENT_AVAILABLE and q_type in ["RECOMMENDATION", "DIAGNOSTIC"] and is_weather_query(query):
        print(f"\n🌦️  Step 4B: Checking for weather enrichment...")
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
        print(f"\n🌦️  Step 4B: Checking for weather enrichment...")
        if not WEATHER_ENRICHMENT_AVAILABLE:
            print(f"   ⚠️  Weather enrichment not available")
        elif q_type not in ["RECOMMENDATION", "DIAGNOSTIC"]:
            print(f"   ℹ️  Not a recommendation/diagnostic query ({q_type})")
        elif not is_weather_query(query):
            print(f"   ℹ️  No weather keywords detected")

    # ── Step 5: Generate answer ──
    print(f"\n💬 Step 5: Generating answer...")

    if retrieval_insufficient or len(chunks) == 0:
        if not insufficient_reason:
            insufficient_reason = "No relevant verified agricultural documents were available for this query."
        print(f"   ⚠️  Generation skipped — {insufficient_reason}")
        answer_data = {
            "answer": (
                f"⚠️ Insufficient verified evidence: {insufficient_reason}\n\n"
                f"To receive a reliable diagnosis or recommendation, please provide additional field details "
                f"(such as crop stage, days after sowing, field location, and specific plant symptoms), "
                f"or consult your local Krishi Vigyan Kendra (KVK) or State Agricultural University extension officer."
            ),
            "model_used": "system/insufficient-evidence",
            "sources": [],
            "path": path,
        }
    elif path == "fast":
        print(f"   Using: {GROQ_GATE_MODEL} (fast path)")
        answer_data = fast_path_answer(query, chunks)
    else:
        print(f"   Using: {GROQ_ANSWER_MODEL} (slow path)")
        answer_data = slow_path_answer(
            query, chunks, q_type
        )

    answer = answer_data.get('answer', '')

    # Reconcile chemical recommendations with regulatory status (both fast and slow path)
    try:
        from regulatory_updater import get_regulatory_verification_status
        reg_status = get_regulatory_verification_status()
    except Exception:
        reg_status = {"status": "OFFLINE_STATIC_KB", "is_offline": True}
    is_offline_reg = reg_status.get("status") == "OFFLINE_STATIC_KB" or reg_status.get("is_offline", True)

    if answer:
        answer = reconcile_chemical_recommendations(answer, is_offline_reg=is_offline_reg, chunks=chunks)
        answer_data["answer"] = answer

    core_generated_answer = answer

    answer_failed = (
        not answer or not answer.strip() or
        answer.startswith("⚠️") or
        answer_data.get("model_used") == "system/insufficient-evidence"
    )

    provenance_summary = None

    # ── Step 5B: Sentence-level provenance mapping ──
    if is_fast_path:
        print("\n⏭️  Step 5B: Skipped (FAST path)")
        answer_data["provenance_summary"] = None
        answer_data["is_factually_proven"] = False
        answer_data["factually_validated"] = False
    elif os.environ.get("ENABLE_PROVENANCE_LOGGING") != "1":
        answer_data["provenance_summary"] = None
        answer_data["is_factually_proven"] = False
        answer_data["factually_validated"] = False
    elif answer_failed:
        print("\n⏭️  Step 5B: Skipped (generation empty, failed, or insufficient evidence)")
    else:
        try:
            from sentence_provenance import build_provenance_map, print_provenance_report, get_provenance_summary
            if answer and answer.strip() and "knowledge base doesn't have" not in answer:
                print("\n🔬 Step 5B: Building Sentence-Level Provenance Map...")

                provenance_chunks = scored_chunks if scored_chunks else []
                if not provenance_chunks:
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
                answer_data["provenance_summary"] = provenance_summary

                print(f"\n🗺️  PROVENANCE : {provenance_summary['total_sentences']} sentences mapped")
                print(f"   Entailed (strict proof)  : {provenance_summary['entailed_count']}")
                print(f"   Candidate Matches (lex)  : {provenance_summary['candidate_match_count']}")
                print(f"   Unsupported / unverified : {provenance_summary['unsupported_count']}")
                print(f"   Avg Lexical Overlap      : {provenance_summary['avg_lexical_overlap']:.3f}")
                print(f"   Sources Traced           : {', '.join(provenance_summary['sources_used'])}")

                entailed = provenance_summary['entailed_count']
                candidates = provenance_summary['candidate_match_count']
                unsupported = provenance_summary['unsupported_count']
                total = provenance_summary['total_sentences']

                if entailed == 0:
                    print("   ⚠️  Validation Notice     : 0 claims strictly entailed. Statements are candidate matches or unverified against retrieved documents; answer is NOT factually proven.")
                    answer_data["is_factually_proven"] = False
                    unverified_notice = (
                        "> ⚠️ **UNVERIFIED ANSWER NOTICE — NOT FACTUALLY PROVEN:**\n"
                        f"> None of the factual claims in this response were strictly entailed by retrieved documents "
                        f"(0 strictly entailed, {candidates} candidate lexical matches, {unsupported} unverified statements out of {total} analyzed).\n"
                        "> The points below represent unverified diagnostic possibilities based on general literature patterns, NOT confirmed facts for your field.\n"
                        "> Verify all symptoms, soil/leaf nutrient tests, and pest identifications with your local Krishi Vigyan Kendra (KVK) or State Agricultural University extension before taking action.\n\n"
                    )
                    answer = re.sub(
                        r"(?i)POTENTIAL\s+CAUSES\s*\([^\)]*SOURCE[‑\-]GROUNDED[^\)]*\)",
                        "POSSIBLE CAUSES (UNVERIFIED HYPOTHESES / LITERATURE PATTERNS)",
                        answer
                    )
                    answer = re.sub(r"(?i)\bSOURCE[‑\-]GROUNDED\b", "UNVERIFIED HYPOTHESES", answer)
                    answer = unverified_notice + answer
                    answer_data["answer"] = answer
                else:
                    answer_data["is_factually_proven"] = True
                    provenance_notice = (
                        f"> ℹ️ **PROVENANCE VERIFICATION NOTICE:** {entailed}/{total} claims strictly entailed by retrieved context; "
                        f"{unsupported} statements are candidate matches or unverified against current documents.\n\n"
                    )
                    answer = provenance_notice + answer
                    answer_data["answer"] = answer
            else:
                print("\n⏭️  Step 5B: Skipping provenance (no answer generated)")
        except Exception as e:
            print(f"   ⚠️  Provenance step failed: {e}")

    # ── Step 5C: Regulatory compliance scan ──
    if answer_failed:
        print("\n⏭️  Step 5C: Skipped (generation empty, failed, or insufficient evidence)")
    elif core_generated_answer and core_generated_answer.strip() and "knowledge base doesn't have" not in core_generated_answer:
        print("\n⚖️  Step 5C: Running Regulatory Compliance Scan...")

        compliance_result = scan_answer_for_compliance(answer)

        if compliance_result["pesticides_found"]:
            print(f"   Pesticides detected : {len(compliance_result['pesticides_found'])}")
            print(f"   Overall compliance  : {compliance_result['overall_compliance']:.4f}")
            print(f"   Warning needed      : {compliance_result['warning_needed']}")
            print(f"   Any India banned    : {compliance_result['any_banned']}")
            print(f"   Any EU banned       : {compliance_result['any_eu_banned']}")

            if compliance_result.get("borderline_pesticides"):
                print(f"\n   🔄 Borderline pesticides detected — triggering background verification...")
                print(f"   Pesticides: {compliance_result['borderline_pesticides']}")

                import threading

                def background_verify():
                    run_startup_update(
                        borderline_pesticides=compliance_result["borderline_pesticides"]
                    )

                thread = threading.Thread(target=background_verify, daemon=True)
                thread.start()
                print(f"   ✅ Background verification started (non-blocking)")

            warning_box = build_compliance_warning_box(compliance_result)
            print(warning_box)

            answer = answer + "\n\n" + warning_box
            answer_data["answer"] = answer
        else:
            print("   ✅ No regulated pesticides detected in answer")
    else:
        print("\n⏭️  Step 5C: Compliance scan skipped (no answer)")

    # ── Step 5D: Conformal Trust Scoring (Optional / Guarded) ──
    trust_score = None
    trust_verdict = None
    trust_data = None

    ENABLE_CONFORMAL_TRUST = getattr(config, "ENABLE_CONFORMAL_TRUST", False)

    if is_fast_path:
        print("\n⏭️  Step 5D: Skipped (FAST path)")
    elif not ENABLE_CONFORMAL_TRUST:
        print("\n⏭️  Step 5D: Skipped (ENABLE_CONFORMAL_TRUST=False)")
    elif answer_failed:
        print("\n⏭️  Step 5D: Skipped (generation empty, failed, or insufficient evidence)")
    elif core_generated_answer and core_generated_answer.strip() and "knowledge base doesn't have" not in core_generated_answer:
        print("\n🛡️  Step 5D: Running Conformal Trust Scoring (Evidence Alignment)...")
        try:
            from conformal_trust_scorer import run_conformal_trust_scorer
            trust_result = run_conformal_trust_scorer(
                query=query,
                answer=core_generated_answer,
                chunks=chunks,
                embedder=embedder,
                client=groq_client
            )
            if trust_result and isinstance(trust_result, dict) and "aggregate" in trust_result:
                agg = trust_result["aggregate"]
                trust_score = agg.get("overall_trust_score")
                trust_verdict = agg.get("verdict")
                trust_data = trust_result
        except Exception as e:
            print(f"   ⚠️  Conformal trust scoring failed: {e}")
            trust_score = None
            trust_verdict = None
            trust_data = None
    else:
        print("\n⏭️  Step 5D: Skipped (no answer generated)")

    # ── Final output (DISPLAY COMPLETE ANSWER AND STATUS BEFORE FEEDBACK) ──
    print(f"\n{'─'*60}")
    print(f"📋 QUERY TYPE  : {q_type}")
    print(f"🔧 PATH        : {path.upper()}")
    print(f"🤖 MODEL USED  : {answer_data['model_used']}")
    if weather_enrichment_applied:
        print(f"🌦️  WEATHER    : Enriched with real-time data")
    print(f"📚 SOURCES     :")
    if answer_data["sources"]:
        source_freshness = {}
        for c in chunks:
            s_name = c.get("source_file") or c.get("source")
            if s_name and s_name not in source_freshness:
                source_freshness[s_name] = c.get("freshness_label")
        for src in answer_data["sources"]:
            f_label = source_freshness.get(src)
            if f_label == "STALE":
                print(f"   • {src} (⚠️ HISTORICAL / STALE - Reference Only)")
            elif f_label == "OUTDATED":
                print(f"   • {src} (⛔ OUTDATED)")
            elif f_label:
                print(f"   • {src} ({f_label})")
            else:
                print(f"   • {src}")
    else:
        print("   • None (Insufficient verified context)")
    if trust_score is not None:
        print(f"🛡️  TRUST SCORE : {trust_score:.4f} ({trust_verdict})")
    print(f"\n✅ ANSWER:\n")
    print(answer_data["answer"])
    print(f"{'='*60}")

    # ── Step 6: Collect routing feedback (optional, interactive, AFTER answer shown) ──
    print("\n📝 Step 6: Recording routing feedback...")
    if answer_failed or retrieval_insufficient:
        print("   ℹ️  Skipping routing feedback adaptation (answer was empty, failed, or insufficient context)")
    else:
        try:
            should_prompt = (
                enable_interactive_feedback or
                os.getenv("ENABLE_INTERACTIVE_FEEDBACK") == "1"
            ) and os.getenv("NO_INTERACTIVE_FEEDBACK") != "1"
            is_interactive = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

            if not should_prompt or not is_interactive:
                print("   ℹ️  Skipping interactive feedback (non-interactive by default)")
            else:
                feedback = input("   Was this answer accurate? (y/n, press Enter to skip): ").strip().lower()
                if feedback in ['y', 'n']:
                    was_accurate = (feedback == 'y')
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

    return {
        "query"                    : query,
        "query_type"               : q_type,
        "path"                     : path,
        "entropy"                  : entropy,
        "model_used"               : answer_data["model_used"],
        "answer"                   : answer_data["answer"],
        "sources"                  : answer_data["sources"],
        "chunks_used"              : len(chunks),
        "source_chunks"            : chunks,
        "routing_reason"           : routing["reason"],
        "weather_enrichment_applied": weather_enrichment_applied,
        "weather_data"             : weather_data,
        "trust_score"              : trust_score,
        "trust_verdict"            : trust_verdict,
        "trust_data"               : trust_data,
        "provenance_summary"       : provenance_summary,
        "is_factually_proven"      : bool(answer_data.get("is_factually_proven", False) and provenance_summary and provenance_summary.get('entailed_count', 0) > 0),
        "factually_validated"      : bool(provenance_summary and provenance_summary.get('entailed_count', 0) > 0),
    }


# ─────────────────────────────────────────────

def load_components():
    """
    Load search components (ChromaDB, BM25 indices, SentenceTransformer embedder).
    """
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

        print("  Loading Cross-Encoder reranker...", flush=True)
        sys.stdout.flush()
        from step5_vector_index import get_cross_encoder
        _ = get_cross_encoder()
        print("  ✓ Cross-Encoder loaded", flush=True)
        sys.stdout.flush()

        print("✅ Ready!\n", flush=True)
        sys.stdout.flush()
        return embedder, collection, bm25, corpus
        
    except Exception as e:
        print(f"\n❌ ERROR during loading: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        raise e


_GLOBAL_COMPONENTS = None


def get_or_load_components():
    """
    Get cached search components or load them once.
    """
    global _GLOBAL_COMPONENTS
    if _GLOBAL_COMPONENTS is None:
        _GLOBAL_COMPONENTS = load_components()
    return _GLOBAL_COMPONENTS


class QueryGate:
    """
    Object-oriented interface for the Agricultural Query Gate pipeline.
    Maintains compatibility with tests and application callers.
    """
    def __init__(self, embedder=None, collection=None, bm25=None, corpus=None):
        if embedder is None or collection is None or bm25 is None or corpus is None:
            self.embedder, self.collection, self.bm25, self.corpus = get_or_load_components()
        else:
            self.embedder = embedder
            self.collection = collection
            self.bm25 = bm25
            self.corpus = corpus

    def process_query(self, query: str, location: str = None) -> dict:
        return query_gate(
            query=query,
            embedder=self.embedder,
            collection=self.collection,
            bm25=self.bm25,
            corpus=self.corpus
        )


# ─────────────────────────────────────────────
# RUN THE QUERY GATE (CLI / VS CODE EXECUTION)
# ─────────────────────────────────────────────

def _run_benchmarks(embedder, collection, bm25, corpus):
    test_queries = [
        "What is vermicompost?",
        "What does NPK stand for?",
        ("My tomato plant leaves are turning yellow "
         "from the bottom with brown edges. "
         "What is wrong and how do I fix it?"),
        ("What organic fertilizers should I apply "
         "for coconut trees in coastal Kerala "
         "during the monsoon season?"),
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
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"  📊 QUERY GATE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Query':<40} {'Path':<6} {'Model':<25}")
    print(f"  {'─'*40} {'─'*6} {'─'*25}")

    for r in results:
        q     = r["query"][:38] + ".." if len(r["query"]) > 40 else r["query"]
        path  = r["path"].upper()
        model = (r["model_used"].replace("openai/", "").replace("-versatile", "")[:24])
        icon  = "⚡" if r["path"] == "fast" else "🔍"
        print(f"  {icon} {q:<40} {path:<6} {model}")

    fast  = sum(1 for r in results if r["path"] == "fast")
    slow  = sum(1 for r in results if r["path"] == "slow")
    total = len(results)

    print(f"\n  Fast path : {fast}/{total}")
    print(f"  Slow path : {slow}/{total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    embedder, collection, bm25, corpus = get_or_load_components()

    # Case 1: Command line arguments provided
    if len(sys.argv) > 1:
        args = [a for a in sys.argv[1:]]
        interactive_fb = False
        if "--interactive-feedback" in args:
            args.remove("--interactive-feedback")
            interactive_fb = True
        cli_query = " ".join(args).strip()
        if cli_query:
            query_gate(cli_query, embedder, collection, bm25, corpus, enable_interactive_feedback=interactive_fb)

    # Case 2: Interactive terminal (e.g. running in VS Code terminal)
    elif hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
        print("=" * 60)
        print("🌾 Agricultural RAG Assistant (VS Code Interactive Mode)")
        print("Enter an agricultural query below, 'test' to run benchmark, or 'exit' to quit.")
        print("=" * 60)
        while True:
            try:
                user_input = input("\n🌾 Enter agricultural query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting. Goodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting. Goodbye!")
                break
            if user_input.lower() == "test":
                _run_benchmarks(embedder, collection, bm25, corpus)
                continue

            query_gate(user_input, embedder, collection, bm25, corpus, enable_interactive_feedback=True)

    # Case 3: Non-interactive automated execution
    else:
        _run_benchmarks(embedder, collection, bm25, corpus)