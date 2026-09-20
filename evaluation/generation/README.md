# Agricultural RAG: End-to-End Generation Evaluation & Benchmark Suite

This directory contains the assets, evaluation scripts, dataset specifications, and benchmark results for assessing the generation layer of the Agricultural RAG system.

---

## 1. Benchmark Purpose

While retrieval benchmarks measure chunk-level recall and ranking precision (MRR, Hit@k), an end-to-end generation benchmark evaluates the agricultural advice actually delivered to the farmer. The purpose of this benchmark is to:
1. Objectively quantify factual precision and completeness (expected-fact coverage) across diverse agricultural query types.
2. Verify strict adherence to statutory safety constraints (CIB&RC bans, restrictive pesticide schedules, application warnings).
3. Evaluate answer correctness using a transparent, deterministic grading standard without subjective bias.
4. Profile real end-to-end system latency (routing, candidate retrieval, cross-encoder reranking, LLM generation, sentence provenance, compliance scanning).
5. Diagnose failure modes across the entire RAG pipeline to isolate whether errors originate in retrieval, temporal decay filtering, phenology gating, or LLM generation.

---

## 2. Dataset Specification

The evaluation runs against the validated, 100% source-grounded dataset:
- **File**: [`generation_benchmark_dataset.json`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/generation_benchmark_dataset.json)
- **Size**: 30 agricultural queries spanning 5 representative categories:
  - `exact_lexical` (6 queries)
  - `conceptual_paraphrased` (6 queries)
  - `crop_specific` (6 queries)
  - `diagnostic` (6 queries)
  - `procedural_recommendation` (6 queries)
- **Expected Key Facts**: 114 atomic, source-grounded facts extracted directly from the underlying 12,856-chunk corpus (`all_chunks.parquet`).
- **Active Safety Constraints**: 7 statutory constraints strictly grounded in the reference corpus (unsupported external textbook constraints were eliminated during the ground-truth audit).
- **Validation**: Every record is verified via [`validate_generation_benchmark_dataset.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/validate_generation_benchmark_dataset.py).

---

## 3. Production Path Tested

The benchmark invokes the real production pipeline via `step6_query_gate.query_gate(query)`:

```
                          ┌──────────────────────────┐
                          │     User Query (q)       │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │  Rural-to-Scientific     │
                          │      Semantic Bridge     │
                          └─────────────┬────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐               ┌───────────────────────┐
        │ Query Classification  │               │  Query Entropy (H)    │
        │ (Type, Complexity,    │               │  (Token logprobs      │
        │  Confidence, Topics)  │               │   via gpt-oss-20b)    │
        └───────────┬───────────┘               └───────────┬───────────┘
                    └───────────────────┬───────────────────┘
                                        ▼
                          ┌──────────────────────────┐
                          │     Routing Decision     │
                          │  (Entropy < 1.5 & Simple │
                          │   FACTUAL/STATISTICAL)   │
                          └─────┬──────────────┬─────┘
                   Fast Path    │              │    Slow Path
              ┌─────────────────┘              └─────────────────┐
              ▼                                                  ▼
┌──────────────────────────┐                       ┌──────────────────────────┐
│ First-Stage + Cross-Enc  │                       │ First-Stage + Cross-Enc  │
│ Retrieval (Top-3 Chunks) │                       │ Retrieval (Top-5 Chunks) │
└─────────────┬────────────┘                       └─────────────┬────────────┘
              │                                                  │
              │                                                  ▼
              │                                    ┌──────────────────────────┐
              │                                    │ Temporal Decay Filtering │
              │                                    └─────────────┬────────────┘
              │                                                  │
              │                                                  ▼
              │                                    ┌──────────────────────────┐
              │                                    │ Phenological Gate (PGRA) │
              │                                    └─────────────┬────────────┘
              │                                                  │
              │                                                  ▼
              │                                    ┌──────────────────────────┐
              │                                    │ Weather / Context Inject │
              │                                    └─────────────┬────────────┘
              │                                                  │
              ▼                                                  ▼
┌──────────────────────────┐                       ┌──────────────────────────┐
│ Fast Path Generator      │                       │ Slow Path Generator      │
│ Model: gpt-oss-20b       │                       │ Model: gpt-oss-120b      │
│ Context: Top-3 chunks    │                       │ Context: Top-5 chunks    │
│ Max Tokens: 250          │                       │ Max Tokens: 700          │
└─────────────┬────────────┘                       └─────────────┬────────────┘
              │                                                  │
              │                                                  ▼
              │                                    ┌──────────────────────────┐
              │                                    │ Sentence Provenance Map  │
              │                                    └─────────────┬────────────┘
              │                                                  │
              │                                                  ▼
              │                                    ┌──────────────────────────┐
              │                                    │ Compliance Scanner       │
              │                                    └─────────────┬────────────┘
              │                                                  │
              └───────────────────┬──────────────────────────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │       Final Answer       │
                    │   (Output + Latency)     │
                    └──────────────────────────┘
```

### Models & Hardware
- **Fast-Path Generator**: `openai/gpt-oss-20b` (fallback: `llama-3.3-70b-versatile`).
- **Slow-Path Generator**: `openai/gpt-oss-120b`.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Vector Search**: ChromaDB (`sentence-transformers/all-MiniLM-L6-v2`) + BM25 (`rank_bm25`).

---

## 4. Evaluation Metrics & Grading Methodology

Evaluation is conducted deterministically without subjective LLM judge variance:

### Level A: Expected-Fact Coverage
- Each ground-truth expected fact is decomposed into critical lexical keywords, technical chemicals, botanical entities, and numerical dosage tokens.
- An expected fact is marked **satisfied** if all required keywords and entities are explicitly stated in the generated text.
- $$\text{Fact Coverage} = \frac{\sum \text{Facts Satisfied}}{\sum \text{Total Expected Facts}} \times 100\%$$

### Level B: Safety-Constraint Adherence
- Evaluates only queries with non-empty `safety_constraints` in the remediated dataset (7 active constraints).
- Checks whether the model recommended a banned chemical, contradicted an application precaution, or violated pre-harvest intervals.
- Queries with empty constraint lists are marked `NOT_APPLICABLE` and never penalized.

### Level C: Answer-Level Correctness Score
- **Score = 2 (Fully Correct)**:
  - $\ge 70\%$ of expected key facts satisfied.
  - Zero safety violations.
  - No generation refusal.
- **Score = 1 (Partially Correct)**:
  - At least one expected key fact satisfied ($> 0\%$), but $< 70\%$ coverage.
  - Zero safety violations.
- **Score = 0 (Incorrect / Unsupported)**:
  - 0% fact coverage, model refusal, hallucination, or safety violation.

### Normalization Rules & Grader Versions (V1 vs V2)
The evaluator supports two reproducible evaluation modes in [`run_generation_benchmark.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/run_generation_benchmark.py):
- **V1 (Original Strict Baseline)**: Strict lexical matching on raw model output strings.
- **V2 (Validated Normalization Baseline)**: Transparent typographical and formatting normalizations that prevent false negatives without loosening semantic rigor:
  1. *Unicode Typography Normalization*: Maps Unicode non-breaking hyphens (`\u2011`), en-dashes (`\u2013`), and em-dashes (`\u2014`) to standard ASCII hyphens (`-`). Normalizes non-breaking spaces (`\u00a0`, `\u202f`) to standard spaces.
  2. *Standard SI Metric Unit Symbols*: Accepts standard metric SI notation with optional spacing (e.g., `200 L` alongside `200 litres`).
  3. *Unambiguous Domain Acronyms*: Recognizes established entomological/botanical acronyms (e.g., `CSRB` for Cashew Stem and Root Borer).

---

## 5. Benchmark Results Summary (30 Queries)

Full evaluation was executed via [`run_generation_benchmark.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/run_generation_benchmark.py). Detailed per-query outputs are recorded in [`GENERATION_BENCHMARK_RESULTS.md`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/GENERATION_BENCHMARK_RESULTS.md), [`GENERATION_BASELINE_V2.md`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/GENERATION_BASELINE_V2.md), and [`generation_benchmark_results.json`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/generation_benchmark_results.json).

### Overall Metrics Comparison

| Metric | Original Baseline (V1) | Validated Normalized (V2) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Strict Fully-Correct Accuracy (Score=2)** | **26.7%** (8 / 30) | **30.0%** (9 / 30) | **+3.3%** (+1 query: `Q29`) |
| **Partial Credit Rate (Score=1)** | **36.7%** (11 / 30) | **33.3%** (10 / 30) | **-3.3%** |
| **Incorrect / Unsupported Rate (Score=0)** | **36.7%** (11 / 30) | **36.7%** (11 / 30) | **0.0%** |
| **Overall Expected-Fact Coverage** | **37.72%** (43 / 114) | **40.35%** (46 / 114) | **+2.63%** (+3 facts) |
| **Safety-Constraint Adherence** | **100.0%** (7 / 7) | **100.0%** (7 / 7) | **0.0%** |
| **Mean End-to-End Latency** | **6,148.5 ms** | **6,148.5 ms** | — |
| **Median Latency** | **4,441.4 ms** | **4,441.4 ms** | — |
| **API / Generation Failures** | **0** | **0** | — |

---

## 6. Root-Cause Failure Analysis & Limitations

The benchmark revealed three distinct bottlenecks in the production architecture:

### 1. Temporal Credibility Decay vs. University Packages of Practices (Generation Failure)
- **Symptom**: In queries routed to the slow path (`DIAGNOSTIC`, `PROCEDURAL`, `RECOMMENDATION`), such as Q01, Q22, Q23, and Q30, the retrieval engine successfully retrieved ground-truth chunks from university publications (e.g., TNAU Agritech Portal, PAU Ludhiana, KAU Kerala, Crop Protection).
- **Mechanism**: In `step6_query_gate.py` Step 4C, `temporal_credibility.py` applies exponential publication-date decay. Documents published prior to 2024 received credibility scores $< 0.55$ (`MIN_CREDIBILITY_THRESHOLD = 0.55`).
- **Impact**: Step 4C discarded all retrieved chunks, reducing the LLM context to empty (`chunks = []`). Consequently, the LLM followed its slow-path system prompt and truthfully responded:
  > *"My knowledge base doesn't have specific information about this."*
- **Takeaway**: Static agronomic principles (tine spacing, seed rates, pruning geometry) do not decay like market prices or seasonal weather advisories. Filtering foundational agronomic literature through a uniform temporal decay filter causes severe context starvation.

### 2. Incomplete Fact Extraction & Brevity Constraints (Incomplete Answers: 33.3%)
- **Symptom**: The model answered the primary question but omitted 1 or 2 secondary parameters (e.g., Q03 stated 8-inch tine spacing but omitted the specific seed drill model; Q09, Q10, Q20, Q24, Q25 captured 1–2 facts out of 3–4).
- **Mechanism**: Fast path prompts emphasize extreme conciseness (2–4 sentences max), forcing the LLM to truncate secondary details to respect the brevity rule.

### 3. Reasoning-Token Budget Exhaustion
- **Symptom**: In `Q04`, `Q13`, and `Q26`, `openai/gpt-oss-20b` returned empty strings `""`. In `Q16`, `openai/gpt-oss-120b` was cut off mid-sentence after 46 characters.
- **Mechanism**: Both models are reasoning models. Internal reasoning consumed the small token budget (`max_tokens = 250` on fast path, `700` on slow path) before outputting text in `content`.

### 4. First-Stage Recall Drops on Complex Multi-Hop Queries (Retrieval Failure: 16.7%)
- **Symptom**: Q08, Q10, Q12, Q17, and Q28 failed because the relevant ground-truth chunks were not within the top-15 candidate pool.
- **Mechanism**: Queries with complex colloquial phrasing or multiple overlapping crop entities diluted the lexical BM25 and dense semantic fusion pool before the Cross-Encoder could score them.

---

## 7. How to Reproduce

Run the benchmark script directly from the project root:

```bash
# Verify environment and inspect pipeline
python -X utf8 evaluation/generation/inspect_generation_pipeline.py

# Validate ground-truth dataset integrity
python -X utf8 evaluation/generation/validate_generation_benchmark_dataset.py

# Run unit tests for verified grader normalization
python -X utf8 -m unittest evaluation/generation/test_grader_normalization.py

# Regrade existing saved benchmark results in V2 mode (0 API calls)
python -X utf8 evaluation/generation/run_generation_benchmark.py --regrade

# Regrade existing saved benchmark results in V1 mode (0 API calls)
python -X utf8 evaluation/generation/run_generation_benchmark.py --regrade-v1

# Run live benchmark dry run (2 queries)
python -X utf8 evaluation/generation/run_generation_benchmark.py --dry-run
```
