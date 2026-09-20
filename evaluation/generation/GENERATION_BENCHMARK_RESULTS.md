# Generation Benchmark Results: Production Pipeline Evaluation

**Date**: 2026-09-19 12:14:31  
**Mode**: DRY RUN (2 Queries)  
**Gate Model**: `openai/gpt-oss-20b` | **Answer Model**: `openai/gpt-oss-120b`  
**Corpus**: 12,856 Chunks (`all_chunks.parquet`) | **Reranker**: `ms-marco-MiniLM-L-6-v2`  

---

## 1. Executive Summary & Overall Metrics

| Metric | Result | Target Benchmark Standard |
| :--- | :---: | :---: |
| **Strict Fully-Correct Accuracy (Score=2)** | **50.0%** (1/2) | $\ge 70.0\%$ |
| **Partial Credit Rate (Score=1)** | **0.0%** (0/2) | $\le 20.0\%$ |
| **Incorrect / Unsupported Rate (Score=0)** | **50.0%** (1/2) | $\le 10.0\%$ |
| **Overall Expected-Fact Coverage** | **42.9%** (3/7) | $\ge 80.0\%$ |
| **Safety-Constraint Adherence** | **100.0%** (0/0) | **100.0%** |
| **Mean Latency (E2E)** | **6895.2 ms** | $< 3,000\text{ ms}$ |
| **Median Latency (E2E)** | **6895.2 ms** | $< 2,500\text{ ms}$ |
| **Min / Max Latency** | **2195.3 ms / 11595.1 ms** | — |
| **API / Generation Failures** | **0** | **0** |

---

## 2. Category-Level Performance Breakdown

| Category | Queries | Strict Accuracy | Partial Rate | Incorrect Rate | Fact Coverage | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_lexical` | 2 | **50.0%** | 0.0% | 50.0% | **42.9%** | 6895.2 ms |

---

## 3. Per-Query Evaluation Table

| QID | Category | Correctness | Fact Coverage | Safety Adherence | Latency (ms) | Failure Classification & Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `Q01` | `exact_lexical` | ❌ Incorrect (0) | 0/4 (0%) | — N/A | 11595.1 | **GENERATION_FAILURE**: Model generated refusal or failed to extract from retrieved context. |
| `Q02` | `exact_lexical` | ✅ Fully Correct (2) | 3/3 (100%) | — N/A | 2195.3 | **CORRECT_ANSWER**: All essential facts covered accurately. |

---

## 4. Failure Mode Analysis

| Failure Mode | Count | % of Benchmark | Primary Root Cause |
| :--- | :---: | :---: | :--- |
| **`GENERATION_FAILURE`** | **1** | **50.0%** | Model failed to synthesize facts despite chunk presence in context. |
| **`CORRECT_ANSWER`** | **1** | **50.0%** | Answer fully captured all ground-truth facts. |

---

## 5. Production Pipeline Integrity Confirmation

1. **Real Production Code Tested**: Queries executed through `step6_query_gate.query_gate()`, triggering real query classification, entropy routing, cross-encoder retrieval, and Groq LLM generation.
2. **Zero Production Changes**: No pipeline files were altered.
3. **Zero Leaks**: All evaluation logic, fact checking, and grading scripts are contained inside `evaluation/generation/`.
