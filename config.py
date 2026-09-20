# config.py
# =============================================
# Central configuration for your RAG project
# =============================================
from dotenv import load_dotenv
load_dotenv()
import os

# Add this inside SOURCE_METADATA = { ... }


# ── Paths ──────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_PDF_DIR     = os.path.join(BASE_DIR, "data", "raw pdf")
EXTRACTED_DIR   = os.path.join(BASE_DIR, "data", "extracted_text")
CHUNKS_DIR      = os.path.join(BASE_DIR, "data", "chunks")
METADATA_DIR    = os.path.join(BASE_DIR, "data", "metadata")
VECTOR_STORE    = os.path.join(BASE_DIR, "vector_store")
GRAPH_DIR       = os.path.join(BASE_DIR, "knowledge_graph")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
# Add to config.py

REGISTRY_FILE = os.path.join(BASE_DIR, "data", "metadata", "registry.json")

# chunk_id counter file — so new chunks never clash with old ones
CHUNK_COUNTER_FILE = os.path.join(BASE_DIR, "data", "metadata", "chunk_counter.json")

# ── Chunk storage settings ─────────────────
CHUNKS_STORE_FORMAT = "parquet"  # supported: parquet, pickle
PARQUET_ENGINE     = "pyarrow"
ALL_CHUNKS_FILE    = os.path.join(
    CHUNKS_DIR,
    "all_chunks.parquet" if CHUNKS_STORE_FORMAT == "parquet" else "all_chunks.pkl"
)

# ── Chunking settings ──────────────────────
NSP_THRESHOLD   = 0.85   # BERT NSP boundary threshold
MAX_CHUNK_SIZE  = 500    # Max characters per chunk
MIN_CHUNK_SIZE  = 50     # Ignore tiny chunks below this

# ── Semantic Bridge Settings ───────────────
ENABLE_SB2_GUARDRAILS = True   # Set True to enable guarded SB2 alignment

# ── Conformal Trust Scoring Settings ────────
ENABLE_CONFORMAL_TRUST = False  # Set True to enable slow-path conformal trust scoring

# ── Context Compression Settings ─────────────
ENABLE_CONTEXT_COMPRESSION = False  # Set True to enable selective context distillation in slow path

# ── Embedding model ────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── ChromaDB collection name ───────────────
COLLECTION_NAME = "agriculture_knowledge"

# ── Source metadata for your 6 PDFs ────────
SOURCE_METADATA = {
    "ANGRAU Journal of Research.pdf": {
        "title": "ANGRAU Journal of Research",
        "source": "Acharya N.G. Ranga Agricultural University",
        "publisher": "ANGRAU, Andhra Pradesh",
        "year": 2023,
        "authority_score": 1.0,
        "bias_penalty": 0.0,
        "domain": "agriculture_research",
        "region": "Andhra Pradesh, India",
    },
    "Crop Protection.pdf": {
        "title": "Crop Protection Manual",
        "source": "Agricultural University",
        "publisher": "India",
        "year": 2023,
        "authority_score": 1.0,
        "bias_penalty": 0.0,
        "domain": "crop_protection",
        "region": "India",
    },
    "KAU (Kerala Agricultural University).pdf": {
        "title": "KAU Package of Practices",
        "source": "Kerala Agricultural University",
        "publisher": "KAU, Thrissur, Kerala",
        "year": 2023,
        "authority_score": 1.0,
        "bias_penalty": 0.0,
        "domain": "horticulture_agronomy",
        "region": "Kerala, India",
    },
    "The TNAU Agriculture PDF.pdf": {
        "title": "TNAU Agritech Package of Practices",
        "source": "Tamil Nadu Agricultural University",
        "publisher": "TNAU, Coimbatore",
        "year": 2023,
        "authority_score": 1.0,
        "bias_penalty": 0.0,
        "domain": "agronomy",
        "region": "Tamil Nadu, India",
    },
    "The TNAU Horticulture PDF.pdf": {
        "title": "Organic Package of Practices - Horticultural Crops",
        "source": "Tamil Nadu Agricultural University",
        "publisher": "TNAU / NCOF, Ministry of Agriculture, GoI",
        "year": 2022,
        "authority_score": 1.0,
        "bias_penalty": 0.0,
        "domain": "horticulture",
        "region": "Tamil Nadu, India",
    },
    "UAS Bangalore.pdf": {
        "title": "UAS Bangalore Package of Practices",
        "source": "University of Agricultural Sciences Bangalore",
        "publisher": "UAS, Bengaluru, Karnataka",
        "year": 2023,
        "authority_score": 1.0,
        "bias_penalty": 0.0,
        "domain": "agronomy_horticulture",
        "region": "Karnataka, India",
    },
    # ── Crop-specific guides ───────────────────

    "Apple.pdf": {
        "title"          : "Apple Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / State Agricultural University",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "horticulture",
        "region"         : "India",
    },
    "Banana.pdf": {
        "title"          : "Banana Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / State Agricultural University",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "horticulture",
        "region"         : "India",
    },
    "Blackgram.pdf": {
        "title"          : "Blackgram Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / State Agricultural University",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "pulse_crops",
        "region"         : "India",
    },
    "Cardamom.pdf": {
        "title"          : "Cardamom Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research / Spices Board",
        "publisher"      : "ICAR / Spices Board India",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "spices_horticulture",
        "region"         : "India",
    },
    "Chickpea.pdf": {
        "title"          : "Chickpea Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / State Agricultural University",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "pulse_crops",
        "region"         : "India",
    },
    "Citrus.pdf": {
        "title"          : "Citrus Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / State Agricultural University",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "horticulture",
        "region"         : "India",
    },
    "Cotton.pdf": {
        "title"          : "Cotton Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / Central Institute for Cotton Research",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "cash_crops",
        "region"         : "India",
    },
    "Ginger and Turmeric.pdf": {
        "title"          : "Ginger and Turmeric Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research / Spices Board",
        "publisher"      : "ICAR / Spices Board India",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "spices_horticulture",
        "region"         : "India",
    },
    "Grapes.pdf": {
        "title"          : "Grapes Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / National Research Centre for Grapes",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "horticulture",
        "region"         : "India",
    },
    "Millets.pdf": {
        "title"          : "Millets Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / State Agricultural University",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "cereal_crops",
        "region"         : "India",
    },
    "Mustard.pdf": {
        "title"          : "Mustard Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / Directorate of Rapeseed-Mustard Research",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "oilseed_crops",
        "region"         : "India",
    },
    "Pepper.pdf": {
        "title"          : "Black Pepper Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research / Spices Board",
        "publisher"      : "ICAR / Spices Board India",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "spices_horticulture",
        "region"         : "India",
    },
    "Redgram.pdf": {
        "title"          : "Redgram (Pigeonpea) Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / ICRISAT",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "pulse_crops",
        "region"         : "India",
    },
    "Soybean.pdf": {
        "title"          : "Soybean Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / IISR",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "oilseed_crops",
        "region"         : "India",
    },
    "Sugarcane.pdf": {
        "title"          : "Sugarcane Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / IISR",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "cash_crops",
        "region"         : "India",
    },
    "Sunflower.pdf": {
        "title"          : "Sunflower Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / DRMR",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "oilseed_crops",
        "region"         : "India",
    },
    "mango.pdf": {
        "title"          : "Mango Cultivation Package of Practices",
        "source"         : "Indian Council of Agricultural Research",
        "publisher"      : "ICAR / NRC Mango",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "horticulture",
        "region"         : "India",
    },

    # ── Government / Regulatory documents ──────

    "Crop-Management_1.pdf": {
        "title"          : "Crop Management Guidelines",
        "source"         : "Ministry of Agriculture and Farmers Welfare, Government of India",
        "publisher"      : "GoI / DACFW",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "crop_management",
        "region"         : "India",
    },
    "Nutrient cards.pdf": {
        "title"          : "Soil Nutrient Cards - Reference Guide",
        "source"         : "Ministry of Agriculture and Farmers Welfare, Government of India",
        "publisher"      : "GoI / Soil Health Card Scheme",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "soil_fertility",
        "region"         : "India",
    },
    "Document-1-Guidelines-for-Soil-Health-Fertility-Scheme.pdf": {
        "title"          : "Guidelines for Soil Health and Fertility Scheme",
        "source"         : "Ministry of Agriculture and Farmers Welfare, Government of India",
        "publisher"      : "GoI / DACFW",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "soil_fertility",
        "region"         : "India",
    },
    "mplementation Guidelines for Soil Health & Fertility (RKVY).pdf": {
        "title"          : "Implementation Guidelines for Soil Health and Fertility (RKVY)",
        "source"         : "Ministry of Agriculture and Farmers Welfare, Government of India",
        "publisher"      : "GoI / RKVY",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "soil_fertility",
        "region"         : "India",
    },
    "operational_guidelines_pmfby .pdf": {
        "title"          : "Operational Guidelines - Pradhan Mantri Fasal Bima Yojana (PMFBY)",
        "source"         : "Ministry of Agriculture and Farmers Welfare, Government of India",
        "publisher"      : "GoI / PMFBY",
        "year"           : 2024,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "agricultural_insurance",
        "region"         : "India",
    },
    "list_of_pesticides_as_on_31.03.2026.pdf": {
        "title"          : "List of Registered Pesticides (as on 31.03.2026)",
        "source"         : "Central Insecticides Board and Registration Committee (CIBRC), Government of India",
        "publisher"      : "GoI / CIBRC / Ministry of Agriculture",
        "year"           : 2026,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "pesticide_regulation",
        "region"         : "India",
    },
    "list_of_pesticides_which_are_banned_refused_registration_and_restricted_in_use.pdf": {
        "title"          : "List of Banned, Refused Registration and Restricted Pesticides",
        "source"         : "Central Insecticides Board and Registration Committee (CIBRC), Government of India",
        "publisher"      : "GoI / CIBRC / Ministry of Agriculture",
        "year"           : 2026,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "pesticide_regulation",
        "region"         : "India",
    },
    "list_pf_pesticide_formulations_registered_as_on_31.03.2026.pdf": {
        "title"          : "List of Registered Pesticide Formulations (as on 31.03.2026)",
        "source"         : "Central Insecticides Board and Registration Committee (CIBRC), Government of India",
        "publisher"      : "GoI / CIBRC / Ministry of Agriculture",
        "year"           : 2026,
        "authority_score": 1.0,
        "bias_penalty"   : 0.0,
        "domain"         : "pesticide_regulation",
        "region"         : "India",
    },
}
# ── Groq LLM Settings ──────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fast model — simple tasks, classification,
# entropy check, fast path answers
GROQ_GATE_MODEL   = "openai/gpt-oss-20b"

# Best model — complex answers, diagnosis,
# recommendations, validation
GROQ_ANSWER_MODEL = "openai/gpt-oss-120b"

# ── Query Gate Thresholds ──────────────────
# Entropy below this = confident = fast path
QT_ENTROPY_THRESHOLD = 1.5

# How many tokens to sample for entropy
QT_LOGPROB_TOKENS    = 10

# Fast path confidence minimum
QT_MIN_CONFIDENCE    = 0.6
# ── Trust Scoring Settings ─────────────────

# LCR: how many times to sample per chunk
LCR_SAMPLE_COUNT     = 5

# MSCP threshold: below this = untrusted
# 0.55 means 55% of answers must agree
# (lowered from 0.6 — university PDFs naturally lower)
LCR_MSCP_THRESHOLD   = 0.55

# Only run LCR on chunks above this
# retrieval score (saves API calls)
# (lowered from 0.4 — match university PDF score range)
LCR_MIN_CHUNK_SCORE  = 0.35

# Final trust weight formula:
# final = retrieval_score × authority × (1-bias)
# Chunks below this get excluded
# (lowered from 0.3 — better match for university sources)
TRUST_FINAL_THRESHOLD = 0.22