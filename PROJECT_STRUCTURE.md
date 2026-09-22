# AgriRAG 4-Folder Architecture

The entire project has been consolidated into **exactly 4 organized folders**:

```
MAJOR-PROJECT/
│
├── 🎨 ui/                     # 1. FRONTEND INTERFACE
│   ├── agrirag_ui.html        # Main 13-module interactive single-page UI
│   └── pipeline_explainer.html# Visual architecture diagram
│
├── 🌾 pipeline/               # 2. CORE RAG PIPELINE & INTELLIGENCE
│   ├── __init__.py
│   ├── step1_extract.py       # Document & table extraction
│   ├── step2_clean.py         # Text cleaning & normalization
│   ├── step3_chunk.py         # Semantic chunking (BERT NSP)
│   ├── step4_store.py         # Storage & parquet manager
│   ├── step5_vector_index.py  # Vector index & Cross-Encoder
│   ├── step6_query_gate.py    # Main Query Gate & Routing Controller
│   ├── step8_weather_rag.py   # Real-time agro-weather integration
│   ├── config.py              # Central configuration & paths
│   ├── credibility_config.py  # Credibility threshold settings
│   ├── registry.py            # Corpus document registry
│   ├── registry.json
│   ├── semantic_bridge.py     # Step 0: Rural-to-Scientific bridge
│   ├── dialectal_alignment.py # Dialect mapping & normalization
│   ├── folk_taxonomy.py       # Agronomic folk taxonomy
│   ├── adaptive_entropy.py    # Step 2: Shannon entropy & adaptive threshold
│   ├── temporal_credibility.py# Step 4C: Half-life decay modeling
│   ├── phenology_gate.py      # Step 4D: Crop growth stage gating
│   ├── stage_compatibility.py # Phenological stage rules
│   ├── crop_calendar.py       # Sowing/harvesting calendar
│   ├── seasonal_context.py    # Step 4B: Agro-meteorological context
│   ├── soil_moisture.py       # Satellite soil moisture proxy
│   ├── pest_alerts.py         # Agro-climatic pest risk alerts
│   ├── weather_fetcher.py     # Weather API integration
│   ├── sentence_provenance.py # Step 5B: NLI evidential sentence mapping
│   ├── compliance_scanner.py  # Step 5C: CIBRC & WHO pesticide safety scan
│   ├── regulatory_kb.py       # Pesticide regulatory knowledge base
│   ├── regulatory_updater.py  # Online gazette crawler & updater
│   ├── conformal_trust_scorer.py # Step 12: Split-conformal risk bounds
│   ├── bias_mitigation.py     # Step 13: DebiasRAG fairness re-ranking
│   └── components/            # Guardrail sub-packages
│
├── 📁 data/                   # 3. DATA, DATABASES & DATA SCRIPTS
│   ├── raw pdf/               # Original agricultural reference PDFs
│   ├── extracted_text/        # Text extraction JSON outputs
│   ├── chunks/                # Parquet chunk stores (all_chunks.parquet)
│   ├── metadata/              # Metadata & chunk counter registries
│   ├── chroma_db/             # ChromaDB vector database
│   ├── vector_store/          # Vector persistence
│   ├── knowledge_graph/       # BM25 pickle indices
│   ├── backups/               # Intermediate backups & scratch
│   ├── logs/                  # JSON runtime execution logs
│   └── scripts/               # Data maintenance & re-extraction scripts
│
├── 📊 evaluation/             # 4. BENCHMARKS, TESTS & PILOT EVALUATIONS
│   ├── chunking/              # Chunk quality & boundary evaluations
│   ├── extraction/            # OCR & layout benchmarks
│   ├── generation/            # Generation ablation & grounding diagnostics
│   ├── retrieval/             # Dense/Sparse hybrid retrieval evaluations
│   ├── tests/                 # All unit test files (test_*.py)
│   └── docs/                  # Annotator pilot studies & presentations
│
├── api.py                     # Main FastAPI server entry point
├── README.md                  # Project documentation
├── .env                       # Environment variables
└── .gitignore                 # Git ignore rules
```
