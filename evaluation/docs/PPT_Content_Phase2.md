# PPT CONTENT — PHASE 2 PRESENTATION
## Project: Trustworthy Agricultural RAG System for Indian Farmers

---

## SLIDE 1 — Problem Statement and Objective

### Problem Statement

| # | Problem |
|---|---------|
| 1 | Indian farmers rely on **unverified online advice** — wrong pesticide use, crop loss |
| 2 | Existing agricultural chatbots have **no fact-checking** — hallucinations go undetected |
| 3 | **Biased sources** (commercial ads, outdated manuals) treated equally as university research |
| 4 | No system checks if advice is **safe, legal, or region-specific** for India |
| 5 | LLMs alone **cannot cite sources** or guarantee trustworthiness of answers |

### Why It Matters
- 60%+ of India's population depends on agriculture
- Wrong advice = crop failure, financial loss, health hazards
- No existing RAG system targets verifiability + bias control + regulatory compliance together

### Key Objectives

| # | Objective |
|---|-----------|
| 1 | Build a **multi-layer RAG pipeline** grounded on authoritative Indian agricultural PDFs |
| 2 | Implement **Conformal Prediction Trust Scoring** — 90% statistical confidence guarantee |
| 3 | Detect and filter **3 types of bias**: Commercial, Temporal, Demographic |
| 4 | Route queries via **Entropy-based Query Gate** (fast vs. slow path) |
| 5 | Add **Real-time Regulatory Compliance** scanning (banned pesticides via CIBRC) |
| 6 | Ensure answers are **explainable, sourced, and verifiable** |

---

## SLIDE 2 — Flow / Methodology (Part 1 — Data Pipeline)

### Data Sources

| Source | Type | Authority Score |
|--------|------|-----------------|
| TNAU Agriculture and Horticulture PDFs | University | 1.0 |
| KAU (Kerala Agricultural University) | University | 1.0 |
| UAS Bangalore | University | 1.0 |
| ANGRAU Journal of Research | University | 1.0 |
| CIBRC Pesticide Lists (Registered / Banned) | Government | 1.0 |
| PMFBY Operational Guidelines | Government | 1.0 |
| Soil Health and Fertility Scheme Documents | Government | 1.0 |
| Crop-specific PDFs (Mango, Cotton, Millets, etc.) | ICAR | 1.0 |

**Total: 31 PDFs → 10,857+ Chunks stored**

### 4-Step Data Pipeline

```
[Step 1] PDF Extraction
         PyMuPDF + pdfplumber + OCR fallback
         |
[Step 2] Text Cleaning
         Noise removal, normalization
         |
[Step 3] HDR Chunking  (BERT NSP Model on GPU)
         Hybrid Domain-aware chunking
         Agricultural section headers + BERT NSP scoring
         |
[Step 4] Storage
         ChromaDB (Vector) + Parquet (Chunks)
         all-MiniLM-L6-v2 Embeddings on GPU
```

### HDR Chunking Logic
- **H** = Hard boundary detection (crop headers, section titles)
- **D** = Domain-aware splits (Pest, Disease, Irrigation, Fertilizer sections)
- **R** = BERT NSP score less than 0.85 triggers sentence split

---

## SLIDE 3 — Flow / Methodology (Part 2 — Query Pipeline)

### 6-Layer Query Architecture

```
User Query
    |
[Layer 1] Hybrid Retrieval
          Dense (ChromaDB cosine) + Sparse (BM25 Okapi) + Knowledge Graph
    |
[Layer 2] Query Gate   (llama-3.3-70b-versatile)
          Entropy Measurement → Fast Path / Slow Path
          Query Classification into 6 types
          Adaptive Threshold (self-learning from feedback)
    |
    Fast Path (Entropy < 1.5)        Slow Path (Entropy >= 1.5)
    llama-3.3-70b                    gpt-oss-120b
    |
[Layer 3] RA-RAG Re-Ranking
          Trust Weight = Authority Score x (1 - Bias Penalty)
          Final Score = Retrieval Score x Trust Weight
    |
[Layer 4] Bias Mitigation
          Commercial Bias (weight 0.40)
          Temporal Bias   (weight 0.35)
          Demographic Bias (weight 0.25)
    |
[Layer 5] Conformal Trust Scoring
          Decompose answer into Atomic Claims
          Nonconformity Score = 1 - max_cosine_similarity(claim, evidence)
          90% Statistical Confidence Guarantee (alpha = 0.10)
    |
[Layer 6] Regulatory Compliance Scanner
          Real-time CIBRC scraping
          Pesticide ban checking
          Weather enrichment (contextual)
    |
Final Answer + Trust Score + Source Citation
```

### Query Types and Models Used

| Query Type | Model Used |
|------------|------------|
| Simple factual (NPK, seed rate) | llama-3.3-70b (Fast Path) |
| Disease diagnosis | gpt-oss-120b (Slow Path) |
| Pest control recommendation | gpt-oss-120b (Slow Path) |
| Pesticide query | gpt-oss-120b + Compliance Scanner |
| Weather + crop advisory | gpt-oss-120b + Weather API |
| Regulatory question | gpt-oss-120b + CIBRC Live Data |

---

## SLIDE 4 — Demo of Project

### Demo Query 1: Trusted Answer
- **Query:** "What is vermicompost and how to apply it?"
- Trust Score: **0.8596 — HIGHLY TRUSTWORTHY**
- Source: KAU / TNAU University PDF
- Claims checked: 6/6 supported by retrieved evidence
- Guarantee: 90% confidence, all claims evidence-backed

### Demo Query 2: Untrustworthy Answer Detected
- **Query:** "Apply quantum nano-pesticide X7 at 500ml per acre"
- Trust Score: **0.2593 — UNTRUSTWORTHY**
- System flags and warns user automatically
- No supporting evidence found in verified corpus

### Demo Query 3: Bias Detection in Action
- **Query:** "What fertilizer for tomato during flowering stage?"
- Sources re-ranked: TNAU at rank 1, UAS Bangalore at rank 2
- "AgroChemCorp Advertisement" pushed to last position
- "1985 Farming Manual" flagged as temporally outdated

### Key System Features

| Feature | Status |
|---------|--------|
| Source citation with every answer | Live |
| Trust score badge (0.0 to 1.0) | Live |
| Bias filter and re-ranking | Live |
| Regulatory compliance warning | Live |
| Fast / Slow path routing | Live |
| Weather-enriched advisory | Live |

---

## SLIDE 5 — Results / Output

### Trust Scoring Results (from conformal_trust_log.json)

| Query | Trust Score | Verdict | Claims Checked | Confidence |
|-------|------------|---------|----------------|------------|
| "What is vermicompost?" | **0.8596** | HIGHLY TRUSTWORTHY | 6/6 | 90% |
| "Nano-pesticide X7 at 500ml" | **0.2593** | UNTRUSTWORTHY | 6/6 | 90% |

### Bias Mitigation Results (from bias_mitigation_log.json)

| Query | Total Sources | Reranking Applied | Top Source |
|-------|--------------|-------------------|------------|
| Tomato fertilizer at flowering | 5 chunks | Yes — Commercial + Temporal flagged | TNAU moved to Rank 1 |

### Knowledge Base Scale

| Metric | Value |
|--------|-------|
| Total PDFs processed | 31 PDFs |
| Total chunks created | 10,857 chunks |
| Vector DB (ChromaDB) size | ~50 MB |
| Embedding model | all-MiniLM-L6-v2 |
| Chunk counter (next ID) | 12,577 |

### Bias Detection Weights

| Bias Type | Weight | High Penalty | Medium Penalty | Low Penalty |
|-----------|--------|-------------|----------------|-------------|
| Commercial Bias | 40% | 0.80 | 0.45 | 0.20 |
| Temporal Bias | 35% | 0.80 | 0.45 | 0.20 |
| Demographic Bias | 25% | 0.80 | 0.45 | 0.20 |

### Entropy-Based Query Routing

| Entropy Value | Path | Model | Speed |
|--------------|------|-------|-------|
| Less than 1.5 (confident) | Fast Path | llama-3.3-70b | ~1-2 sec |
| 1.5 or above (uncertain) | Slow Path | gpt-oss-120b | ~3-5 sec |
| Threshold | Self-adapts | Feedback-driven | Dynamic |

---

## SLIDE 6 — Status of Paper (Part 1)

### Paper Title
**"A Multi-Layer Trust-Aware Retrieval-Augmented Generation System for Agricultural Advisory Using Conformal Prediction and Bias Mitigation"**

### Paper Details

| Field | Detail |
|-------|--------|
| Domain | AI / NLP / Agricultural Informatics |
| Keywords | RAG, Conformal Prediction, Bias Mitigation, LLM, Agriculture, Trust Scoring |
| Type | Research Paper |
| Status | Under Preparation / Submitted |

### Novel Contributions
- First RAG system to apply **Conformal Prediction** for agricultural advisory trust scoring
- Novel **3-category bias detection framework** (Commercial + Temporal + Demographic)
- **Entropy-based adaptive query routing** with self-learning feedback threshold
- Real-time **CIBRC regulatory compliance integration**
- **HDR Chunking** — domain-aware hybrid chunking method for agricultural text
- Dual-model architecture: separate models for fast classification vs. deep reasoning

---

## SLIDE 7 — Status of Paper (Part 2 — Proof)

> ACTION REQUIRED: Insert your actual submission proof here

### If Submitted to a Conference or Journal
- Screenshot of submission confirmation email
- Screenshot of submission portal with paper ID
- Screenshot of review acknowledgement email

### If Under Internal Review
- Screenshot of guide / supervisor approval email
- Internal submission form or receipt

### Suggested Target Venues

| Venue | Type | Focus |
|-------|------|-------|
| IEEE ICACCS 2026 | Conference | AI + Computing |
| Springer LNCS | Journal | NLP / ML |
| Elsevier — Computers and Electronics in Agriculture | Journal | AgriTech AI |
| IJARCCE | Journal | Computer Engineering |

> Note to Presenter: Replace this slide with your actual submission screenshot or email as proof of paper status.

---

## SLIDE 8 — Proposed Outcome vs Actual Outcome (Part 1)

### Main Deliverables Comparison

| # | Proposed Outcome | Actual Outcome | Status |
|---|-----------------|----------------|--------|
| 1 | Build RAG pipeline on agricultural PDFs | 31 PDFs processed, 10,857 chunks in ChromaDB | EXCEEDED |
| 2 | Implement trust scoring for answers | Conformal Prediction with 90% statistical confidence guarantee | ACHIEVED |
| 3 | Detect and filter biased content | 3-type bias detection: Commercial (40%), Temporal (35%), Demographic (25%) | ACHIEVED |
| 4 | Route queries by complexity | Entropy-based gate: Fast path vs Slow path with adaptive threshold | ACHIEVED |
| 5 | Real-time regulatory compliance | Automated CIBRC live scraping for banned pesticides | ACHIEVED |
| 6 | Knowledge graph for reasoning | NetworkX graph with crop-pest-disease triplets | ACHIEVED |
| 7 | Hybrid search | BM25 Sparse + ChromaDB Dense + Knowledge Graph traversal | ACHIEVED |
| 8 | GPU acceleration | RTX 3050 GPU — BERT and Embedding on CUDA 12.4 | ACHIEVED |

---

## SLIDE 9 — Proposed Outcome vs Actual Outcome (Part 2)

### Trust Scoring — Expected vs Actual

| Metric | Expected | Actual |
|--------|----------|--------|
| Trust score for verified university facts | Greater than 0.75 | **0.8596** |
| Trust score for hallucinated claims | Less than 0.40 | **0.2593** |
| Statistical confidence level | 90% | **90% (alpha = 0.10)** |
| Claims verified per answer | 5 to 10 | **6 claims** |

### System Scale — Expected vs Actual

| Component | Expected | Actual |
|-----------|----------|--------|
| KB size | ~5,000 chunks | **10,857 chunks** |
| Number of sources | 6 PDFs | **31 PDFs** |
| Regulatory check | Manual | **Automated (CIBRC live)** |
| Adaptive learning | Optional | **Implemented with feedback loop** |

### Bias Mitigation — Expected vs Actual

| Bias Type | Expected | Actual Result |
|-----------|----------|--------------|
| Commercial ads deprioritized | Yes | "AgroChemCorp Ad" pushed to rank 5 |
| University sources top-ranked | Yes | TNAU and UAS Bangalore ranked 1 and 2 |
| Outdated content penalized | Yes | "1985 Farming Manual" penalized by temporal bias |

### Gaps and Challenges

| Challenge | Current Status |
|-----------|---------------|
| Multilingual support (Hindi / Telugu) | Not yet implemented |
| Real farmer evaluation (user study) | Planned |
| Voice interface for low-literacy farmers | Future work |
| Weather API rate limits | Partially handled |

---

## SLIDE 10 — Conclusion: Deliverables and Impact

### Key Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | Multi-layer RAG Pipeline | 8-step pipeline from PDF ingestion to trusted answer |
| 2 | Conformal Trust Scorer | 90% statistically guaranteed per-claim verification |
| 3 | 3-Type Bias Mitigator | Commercial / Temporal / Demographic bias detection and filtering |
| 4 | Adaptive Query Gate | Entropy-based fast/slow routing with self-learning threshold |
| 5 | Knowledge Graph | Crop-pest-disease-chemical relationships (NetworkX) |
| 6 | Hybrid Search Engine | BM25 + Dense Retrieval + Graph Traversal |
| 7 | Regulatory Scanner | Real-time CIBRC pesticide compliance check |
| 8 | Agricultural Knowledge Base | 31 PDFs, 10,857 chunks from top Indian institutions |

### Impact

- **Farmer Safety** — Prevents harmful and illegal pesticide advice from reaching farmers
- **Knowledge Democratization** — University-grade advice accessible to all farmers
- **Scientific Reliability** — First conformal prediction-based agricultural advisory system
- **Policy Aligned** — Sources from ICAR, TNAU, KAU, CIBRC, Government of India

### Future Work

- Multilingual support: Hindi, Tamil, Telugu, Kannada
- Voice interface for low-literacy farmers
- Mobile app integration
- Real-time satellite crop monitoring integration

---

## SLIDE 11 — Individual Responsibilities

> ACTION REQUIRED: Fill in your actual group members names

| Member | Role | Responsibilities |
|--------|------|-----------------|
| [Member 1 Name] | Pipeline Engineer | Step 1-4: PDF extraction, cleaning, HDR chunking, ChromaDB storage |
| [Member 2 Name] | RAG Architect | Step 5: Hybrid retrieval (BM25 + Dense + Graph), Knowledge graph |
| [Member 3 Name] | Trust and Bias | Step 6 Query Gate, Step 7 Trust scoring, Conformal Prediction |
| [Member 4 Name] | Compliance and Integration | Bias mitigation, CIBRC regulatory updater, Weather enrichment, Adaptive entropy |
| [Member 5 Name] | Evaluation and Docs | Testing, results analysis, paper writing, presentation |

> Adjust based on your actual team size and division of work.

---

## SLIDE 12 — Gantt Chart

### Project Timeline (April 2026 to August 2026)

| Task | Apr | May | Jun | Jul | Aug |
|------|-----|-----|-----|-----|-----|
| Literature Review and Problem Definition | XX |  |  |  |  |
| Data Collection (31 PDFs) | XX | XX |  |  |  |
| Step 1-2: PDF Extraction and Cleaning |  | XX |  |  |  |
| Step 3: HDR Chunking (BERT NSP) |  | XX | XX |  |  |
| Step 4: ChromaDB Storage and Embedding |  |  | XX |  |  |
| Step 5: Hybrid Retrieval + Knowledge Graph |  |  | XX | XX |  |
| Step 6: Query Gate (Entropy + Classification) |  |  |  | XX |  |
| Step 7: Trust Scoring (Conformal Prediction) |  |  |  | XX |  |
| Bias Mitigation Module |  |  |  | XX | XX |
| Regulatory Compliance Scanner |  |  |  |  | XX |
| Adaptive Entropy Feedback Loop |  |  |  |  | XX |
| Testing and Evaluation |  |  |  | XX | XX |
| Paper Writing |  |  |  |  | XX |
| Phase 2 Presentation |  |  |  |  | Aug 18 |

### Milestones

| Milestone | Date | Status |
|-----------|------|--------|
| Data pipeline complete (26 PDFs) | April 2026 | DONE |
| RAG retrieval functional | June 2026 | DONE |
| Trust scoring + Bias mitigation live | August 2026 | DONE |
| Phase 2 Presentation | August 18, 2026 | TARGET |

---

## SLIDE 13 — References (IEEE Style)

1. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459-9474, 2020.

2. G. Shafer and V. Vovk, "A Tutorial on Conformal Prediction," *Journal of Machine Learning Research*, vol. 9, pp. 371-421, 2008.

3. A. Angelopoulos and S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," *arXiv preprint arXiv:2107.07511*, 2021.

4. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, pp. 4171-4186, 2019.

5. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP*, pp. 3982-3992, 2019.

6. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009.

7. Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in *Proc. ICML*, pp. 1050-1059, 2016.

8. Touvron, H. et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," *arXiv preprint arXiv:2307.09288*, 2023.

9. CIBRC, "List of Banned / Restricted Pesticides in India," Ministry of Agriculture and Farmers Welfare, Government of India, 2026. [Online]. Available: https://cibrc.nic.in

10. TNAU Agritech Portal, "Package of Practices for Crops of Tamil Nadu," Tamil Nadu Agricultural University, Coimbatore, 2023.

11. J. Johnson, M. Douze, and H. Jegou, "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535-547, 2019.

12. A. Vaswani et al., "Attention Is All You Need," in *Proc. NeurIPS*, pp. 5998-6008, 2017.

---
*End of PPT Content — 13 Slides Total*
