# Retrieval Evaluation Framework (Step 4 Preparation)

This directory contains the ground-truth benchmark dataset, query definitions, and validation tooling for evaluating retrieval quality in the Agricultural RAG pipeline.

---

## 1. Dataset Overview

The retrieval evaluation dataset ([retrieval_benchmark_dataset.json](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/retrieval_benchmark_dataset.json)) consists of **30 verified agricultural queries** linked to authoritative ground-truth chunk IDs from the 12,856-chunk production corpus ([data/chunks/all_chunks.parquet](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data/chunks/all_chunks.parquet)).

### Query Categories (6 Queries Each):

1. **`exact_lexical`** (Q01 – Q06):
   - Specific technical queries containing exact botanical terms, pesticide chemical names, dosages, and machinery specifications where exact keyword matching is critical.
2. **`conceptual_paraphrased`** (Q07 – Q12):
   - Broader agronomic questions formulated in natural language without verbatim keywords, testing semantic generalization and conceptual association.
3. **`crop_specific`** (Q13 – Q18):
   - Targeted crop-level inquiries (Tomato, Coffee, Cashew, Sugarcane, Rice) concerning specific variety performance, seed rates, and climatological management.
4. **`diagnostic`** (Q19 – Q24):
   - Symptom-to-cause diagnostic scenarios (dead hearts/white ears, chlorosis, loose smut powdery masses, shot-holes, frass) testing whether the retrieval system can identify the underlying pathogen or pest from symptom descriptions alone.
5. **`procedural_recommendation`** (Q25 – Q30):
   - Multi-step procedural and dosage instructions (vermicompost dimensions, nursery transplanting depth, organic manure rates, spray preparation, godown disinfection).

---

## 2. Schema Specification

Each query record in `retrieval_benchmark_dataset.json` contains:

```json
{
  "query_id": "Q01",
  "query": "What is the recommended malathion and sugar concentration to control Bacterocera fruit fly?",
  "category": "exact_lexical",
  "relevant_chunk_ids": [
    "chunk_003966",
    "chunk_003736"
  ],
  "source_files": [
    "KAU (Kerala Agricultural University).pdf"
  ],
  "relevance_note": "Provides exact chemical concentration (malathion 0.1% emulsion + 2% sugar) and mechanical bag protection against Bacterocera fruit fly."
}
```

- **`query_id`**: Unique string identifier (`Q01` through `Q30`).
- **`query`**: Natural query text.
- **`category`**: One of the 5 benchmark categories.
- **`relevant_chunk_ids`**: 1 to 3 verified chunk IDs from `all_chunks.parquet` containing the direct answer.
- **`source_files`**: Origin document(s) verified in the corpus.
- **`relevance_note`**: Ground-truth justification citing why the chunk satisfies the query intent.

---

## 3. Dataset Validation

Run the ground-truth validation script:

```bash
python evaluation/retrieval/validate_ground_truth.py
```

The script verifies:
1. All required fields are present and non-empty.
2. Every `query_id` is unique.
3. Categories are strictly balanced (exactly 6 queries per category).
4. No duplicate chunk IDs exist within any query.
5. Every single `chunk_id` exists in `all_chunks.parquet`.

---

## 4. Next Step: Full Retrieval Benchmark

Once verified, this dataset serves as the standard evaluation harness for comparing:
- **Pure Dense Retrieval** (ChromaDB + `all-MiniLM-L6-v2`)
- **Pure Sparse Retrieval** (BM25Okapi)
- **Hybrid Weighted Retrieval** ($0.6 \times \text{Dense} + 0.4 \times \text{Sparse} \times \text{Trust}$)
- **Reciprocal Rank Fusion (RRF)**
