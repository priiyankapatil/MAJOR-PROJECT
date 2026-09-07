# Agricultural RAG Research System

A domain-specialized Retrieval-Augmented Generation (RAG) system for agricultural advisory and research in India. The pipeline integrates authoritative Indian agricultural university publications (ICAR, TNAU, ANGRAU, KAU, UAS Bangalore), semantic document chunking, trust-weighted hybrid retrieval, and conformal uncertainty estimation.

---

## Current Project Status

### Completed
- PDF extraction and layout-aware preprocessing
- Text cleaning and normalization
- HDR (Hierarchical Document Representation) semantic chunking
- Dense + sparse hybrid retrieval (ChromaDB + BM25)
- Query routing and gating
- Trust-weighted retrieval
- Step 3 chunking evaluation and optimization

### Step 3 Chunking Evaluation

The chunking pipeline was evaluated using controlled failure-case testing and full-corpus validation.

Key results after optimization:

- **Baseline chunks**: 14,766
- **Regenerated chunks**: 12,856
- **Maximum chunk size**: reduced from 14,916 to 553 characters
- **Chunks >1,000 characters**: reduced from 129 to 0
- **Chunks below 100 characters**: reduced from 418 to 155
- **Trust-weight consistency**: 100%
- **Document processing failures**: 0
- **Corpus text coverage**: 100% (all 4,809,603 characters covered; 0 text loss detected)

The evaluation identified and corrected:
- **Oversized unpunctuated chunks**: Bypassed size limits in unpunctuated advisory lists; fixed via recursive intra-sentence hard size capping.
- **False section-header splitting**: Narrative sentences beginning with header keywords (e.g., *"Transplanting is done to regulate moisture retention"*) were being artificially severed; resolved with standalone heading syntax matching.
- **False crop-tag assignment**: Substring matching caused insect pests like "Tea Mosquito Bug" on cashew to trigger false "tea" crop tags; resolved with regex word boundaries and compound-term filtering.

Detailed evaluation reports and artifacts are available in:
- [evaluation/chunking/](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/chunking/)
  - [FULL_CORPUS_REGENERATION_RESULTS.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/chunking/FULL_CORPUS_REGENERATION_RESULTS.md)
  - [STEP3_REGRESSION_RESULTS.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/chunking/STEP3_REGRESSION_RESULTS.md)
  - [chunk_root_cause_analysis.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/chunking/chunk_root_cause_analysis.md)
  - [full_corpus_before_after.csv](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/chunking/full_corpus_before_after.csv)

---

## Pipeline Architecture

```
[Raw PDFs] ──> [Step 1: Extract] ──> [Step 2: Clean] ──> [Step 3: HDR Chunk]
                                                               │
                                                               ▼
[Evaluation Framework] <── [Hybrid Retrieval] <── [all_chunks.parquet]
(Dense + BM25 + Gate)
```

- **Extraction**: Dual-engine extraction (PyMuPDF for complex tables/multi-column, pdfplumber fallback).
- **Cleaning**: Unicode normalization, ligatures correction, and non-content filtering.
- **Chunking**: spaCy sentence boundary detection + BERT Next Sentence Prediction (NSP) semantic cohesion + intra-sentence natural boundary hard capping.
- **Retrieval**: ChromaDB dense vector search + BM25 sparse keyword search + authority/bias trust weighting.