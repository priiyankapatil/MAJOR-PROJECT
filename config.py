# config.py
# =============================================
# Central configuration for your RAG project
# =============================================

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
    # Add this inside SOURCE_METADATA = { ... }

    "Your New PDF Name.pdf": {
        "title"          : "Title of the document",
        "source"         : "University or Organization name",
        "publisher"      : "Publisher",
        "year"           : 2024,
        "authority_score": 1.0,   # 1.0 for government/university
        "bias_penalty"   : 0.0,
        "domain"         : "agriculture",
        "region"         : "India",
    },
}
# ── Groq LLM Settings ──────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Model Assignment ───────────────────────
# Fast model — simple tasks, classification,
# entropy check, fast path answers
GROQ_GATE_MODEL   = "llama-3.3-70b-versatile"

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