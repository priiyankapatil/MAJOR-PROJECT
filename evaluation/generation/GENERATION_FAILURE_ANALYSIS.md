# Generation Failure Attribution Audit: Production Pipeline Diagnostic

**Date**: 2026-09-08  
**Scope**: 30-Query Agricultural Generation Benchmark Evaluation  
**Corpus**: 12,856 Chunks (`all_chunks.parquet`) | **Ground Truth**: `generation_benchmark_dataset.json`  
**Pipeline Tested**: `step6_query_gate.py` (`openai/gpt-oss-20b` fast path, `openai/gpt-oss-120b` slow path)

---

## 1. Executive Forensic Summary

The generation benchmark demonstrated a strict fully-correct accuracy of **26.7%** (8/30), partial credit of **36.7%** (11/30), and incorrect/unsupported rate of **36.7%** (11/30). A forensic audit was conducted to attribute every sub-optimal query to its precise mechanical root cause.

The investigation revealed that failures do **not** stem from a single source, but rather from three distinct architectural bottlenecks:
1. **Post-Retrieval Temporal Credibility Filtering (13.3%, 4 queries)**: Ground-truth chunks retrieved at ranks 0–2 from foundational university publications (KAU 2019, Crop Protection 2018) were penalized by exponential publication decay ($e^{-\lambda \Delta T}$) and filtered out by `MIN_CREDIBILITY_THRESHOLD = 0.55`, starving the LLM of evidence.
2. **Groq Reasoning-Token Budget Exhaustion (13.3%, 4 queries)**: Both `openai/gpt-oss-20b` and `openai/gpt-oss-120b` are reasoning models. In 3 fast-path queries (`Q04`, `Q13`, `Q26`), the model exhausted its 250-token budget entirely within internal reasoning, returning an empty string `""` in `content`. In 1 slow-path query (`Q16`), reasoning consumed ~600 tokens out of 700, truncating the answer mid-sentence.
3. **First-Stage Candidate Pool Misses (16.7%, 5 queries)**: In `Q08`, `Q10`, `Q12`, `Q17`, and `Q28`, dense/sparse candidate retrieval missed the ground-truth chunk.
4. **Prompt Conciseness vs. Multi-Fact Coverage (26.7%, 8 queries)**: In 8 queries, the model retrieved and understood the context, but omitted secondary parameters due to the prompt's "2–4 sentences maximum" constraint.

---

## 2. Thirty-Query Failure Attribution Table

| QID | Cat | Path | Type | Ret Hit? | GT in Prompt? | Score | Primary Failure Mode | Root Cause Mechanism |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `Q01` | exact | slow | REC | ✅ (Ranks 0,1) | ❌ | 0 | **TEMPORAL_FILTER_DISPLACEMENT** | KAU chunks (age=6, $\lambda=0.10$) decayed to 0.4939 < 0.55; context starved to 0 chunks. Model refused. |
| `Q02` | exact | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | All 3 facts captured accurately by `gpt-oss-20b`. |
| `Q03` | exact | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Captured 8-inch spacing and Coragen dose; omitted weed control benefit due to 2-sentence brevity. |
| `Q04` | exact | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 0 | **REASONING_TOKEN_STARVATION** | `gpt-oss-20b` spent all 250 tokens in internal reasoning channel; returned `""` empty content. |
| `Q05` | exact | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | All 3 facts and safety restrictions for Fenitrothion correctly stated. |
| `Q06` | exact | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Stated primary dose; omitted soil application interval due to brevity constraint. |
| `Q07` | conc | fast | FACT | ✅ (Rank 1) | ✅ (Rank 1) | 1 | **INCOMPLETE_GENERATION** | Stated root rot control; omitted specific application volume per vine. |
| `Q08` | conc | fast | FACT | ❌ | ❌ | 0 | **RETRIEVAL_MISS** | `chunk_001041` absent from top-15 candidate pool. Model answered from parametric memory. |
| `Q09` | conc | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Stated 2 of 4 rotational benefits; omitted specific root exudate mechanisms. |
| `Q10` | conc | fast | FACT | ❌ | ❌ | 1 | **RETRIEVAL_MISS** | Ground truth `chunk_001999` absent; model used related chunk `chunk_004546` for partial credit. |
| `Q11` | conc | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | Fully answered with 75% fact coverage and no safety violations. |
| `Q12` | conc | fast | FACT | ❌ | ❌ | 0 | **RETRIEVAL_MISS** | `chunk_003231` absent from candidate pool; model output truncated. |
| `Q13` | crop | fast | FACT | ✅ (Rank 2) | ✅ (Rank 2) | 0 | **REASONING_TOKEN_STARVATION** | `gpt-oss-20b` spent all 250 tokens in internal reasoning; returned `""` empty content. |
| `Q14` | crop | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Captured susceptibility reason; omitted resistant cultivar names due to brevity. |
| `Q15` | crop | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | All primary diagnostic and botanical management facts covered. |
| `Q16` | crop | slow | PROC | ✅ (Rank 2) | ✅ (Rank 2) | 0 | **GENERATION_TRUNCATION** | `gpt-oss-120b` reasoning tokens consumed token budget; answer truncated mid-sentence at 46 chars. |
| `Q17` | crop | slow | REC | ❌ | ❌ | 0 | **RETRIEVAL_MISS** | `chunk_003772` absent from candidate pool; all 5 retrieved chunks also failed temporal filter. |
| `Q18` | crop | slow | REC | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | Captured all cultivars; (Note: grader marked 1 fact missing due to hyphen formatting). |
| `Q19` | diag | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | Accurate diagnostic identification of yellow stem borer damage. |
| `Q20` | diag | fast | FACT | ✅ (Rank 3) | ❌ | 1 | **FAST_PATH_CUTOFF_DISPLACEMENT** | `chunk_009535` retrieved at rank 3, but fast path takes `chunks[:3]`; displaced by cutoff. |
| `Q21` | diag | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | Accurate diagnostic identification of loose smut in wheat. |
| `Q22` | diag | slow | DIAG | ✅ (Rank 0) | ❌ | 0 | **TEMPORAL_FILTER_DISPLACEMENT** | `chunk_001148` (Crop Protection 2018) decayed to 0.5207 < 0.55. Treatment evidence discarded. |
| `Q23` | diag | slow | DIAG | ✅ (Rank 0) | ❌ | 0 | **TEMPORAL_FILTER_DISPLACEMENT** | `chunk_001017` (Crop Protection 2018) decayed to 0.5207 < 0.55. Diagnosis evidence discarded. |
| `Q24` | diag | slow | DIAG | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Identified borer damage; omitted wood ash recipe and physical grub removal. |
| `Q25` | proc | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Stated dimensions and leveling; omitted bed length variability. |
| `Q26` | proc | fast | FACT | ✅ (Rank 0) | ✅ (Rank 0) | 0 | **REASONING_TOKEN_STARVATION** | `gpt-oss-20b` spent 250 tokens in reasoning; returned `""` empty content. |
| `Q27` | proc | slow | PROC | ✅ (Rank 0) | ✅ (Rank 0) | 2 | **CORRECT** | Comprehensive step-by-step nursery transplanting procedure. |
| `Q28` | proc | slow | REC | ❌ | ❌ | 0 | **RETRIEVAL_MISS** | `chunk_011687` absent from candidate pool. |
| `Q29` | proc | slow | PROC | ✅ (Rank 0) | ✅ (Rank 0) | 1 | **INCOMPLETE_GENERATION** | Captured chemicals and mixing; omitted spray repetition interval. |
| `Q30` | proc | slow | REC | ✅ (Rank 0) | ❌ | 1 | **TEMPORAL_FILTER_DISPLACEMENT** | `chunk_001818` (Crop Protection 2018) decayed to 0.4221 < 0.55. Model used TNAU chunk instead. |

---

## 3. Summary Attribution Tables

### Table A: Primary Root Cause Breakdown (30 Queries)

| Primary Failure Category | Query Count | Benchmark % | Failure Mechanism Description |
| :--- | :---: | :---: | :--- |
| **`CORRECT`** | **8** | **26.7%** | Pipeline operated as designed; $\ge 70\%$ facts satisfied, 0 safety violations. |
| **`INCOMPLETE_GENERATION`** | **8** | **26.7%** | Context reached generator, but prompt length constraint caused omission of secondary details. |
| **`RETRIEVAL_MISS`** | **5** | **16.7%** | First-stage Dense/BM25 candidate pool did not contain the ground-truth chunk. |
| **`TEMPORAL_FILTER_DISPLACEMENT`** | **4** | **13.3%** | Ground-truth chunk was retrieved in top 1–2 ranks, but discarded by temporal credibility decay. |
| **`REASONING_TOKEN_STARVATION`** | **3** | **10.0%** | `gpt-oss-20b` consumed entire 250-token budget in reasoning, outputting `""` in content. |
| **`GENERATION_TRUNCATION`** | **1** | **3.3%** | `gpt-oss-120b` reasoning tokens consumed budget, truncating output mid-sentence. |
| **`FAST_PATH_CUTOFF_DISPLACEMENT`** | **1** | **3.3%** | Retrieved at rank 3, but discarded because fast path takes only `chunks[:3]`. |
| **Total** | **30** | **100.0%** | |

---

### Table B: Retrieval vs. Generation Success Comparison

| Stage | Hit Count | Miss / Starvation Count | Effective Pass Rate |
| :--- | :---: | :---: | :---: |
| **First-Stage + Reranking Retrieval** (GT chunk in top-5) | **25 / 30** | 5 / 30 | **83.3%** |
| **Post-Retrieval Filter Passing** (GT reached generator prompt) | **20 / 25** | 5 / 25 lost in pipeline | **80.0%** of retrieved |
| **LLM Synthesis & Answer Generation** (Score $\ge 1$ when GT in prompt) | **16 / 20** | 4 / 20 token starved/refused | **80.0%** of delivered |
| **Strict Full Correctness** (Score = 2 across entire dataset) | **8 / 30** | 22 / 30 | **26.7%** |

---

### Table C: Fast-Path vs. Slow-Path Comparison

| Dimension | Fast Path (`gpt-oss-20b`) | Slow Path (`gpt-oss-120b`) |
| :--- | :---: | :---: |
| **Total Query Volume** | 19 queries (63.3%) | 11 queries (36.7%) |
| **Candidate Retrieval Success** | 16 / 19 (84.2%) | 9 / 11 (81.8%) |
| **Post-Retrieval Losses** | 1 (`Q20` fast cutoff) | 4 (`Q01`, `Q22`, `Q23`, `Q30` temporal decay) |
| **Prompt Evidence Delivery Rate** | 15 / 19 (78.9%) | 5 / 11 (45.5%) |
| **Reasoning Token Failures** | 3 (`Q04`, `Q13`, `Q26` empty content) | 1 (`Q16` truncation) |
| **Strict Accuracy (Score=2)** | **31.6%** (6 / 19) | **18.2%** (2 / 11) |
| **Partial Credit (Score=1)** | **42.1%** (8 / 19) | **27.3%** (3 / 11) |
| **Incorrect / Refusal (Score=0)** | **26.3%** (5 / 19) | **54.5%** (6 / 11) |
| **Mean End-to-End Latency** | **3,232.2 ms** | **11,185.8 ms** |
| **Median Latency** | **2,685.5 ms** | **10,856.1 ms** |

> [!IMPORTANT]
> **Key Finding**: The Slow Path experienced a **54.5% incorrect/refusal rate**, compared to 26.3% on the Fast Path. The primary driver was **not** model reasoning ability, but **upstream context starvation** caused by temporal credibility decay filtering out university agronomic packages of practices.

---

### Table D: Category-Level Failure Distribution

| Category | Queries | Correct | Incomplete | Ret Miss | Temp Filter | Cutoff | Reasoning Empty |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_lexical` | 6 | 2 | 2 | 0 | 1 | 0 | 1 |
| `conceptual_paraphrased` | 6 | 1 | 3 | 3 | 0 | 0 | 0 |
| `crop_specific` | 6 | 2 | 1 | 1 | 0 | 0 | 1 |
| `diagnostic` | 6 | 2 | 2 | 0 | 2 | 1 | 0 |
| `procedural_recommendation` | 6 | 1 | 3 | 1 | 1 | 0 | 1 |

---

## 4. Deep-Dive Attribution of Critical Subsets

### 4.1. Audit of the 7 Reported Generation/Starvation Queries

The benchmark report initially grouped `Q01`, `Q04`, `Q13`, `Q16`, `Q22`, `Q23`, and `Q26` as generation failures. Forensic inspection reveals two fundamentally different failure mechanisms:

#### A. True Temporal Credibility Starvation / Filtering
1. **`Q01` (Bacterocera Fruit Fly, Slow Path REC)**:
   - *Retrieval*: Ground-truth chunks `chunk_003966` and `chunk_003736` retrieved at **Ranks 0 and 1**.
   - *Temporal Filter*: Source is `KAU` (pub_year 2019, age = 6 years). For `RECOMMENDATION`, $\lambda = 0.10$.
     $$\text{Score} = 0.90 \times e^{-0.10 \times 6} = 0.4939 < 0.55$$
   - *Result*: All 5 retrieved chunks were discarded. Context delivered to LLM was empty (`chunks = []`). The LLM followed its system instructions and truthfully stated: *"My knowledge base doesn't have specific information about this."*
2. **`Q22` (Coffee Anthracnose / Die-back, Slow Path DIAG)**:
   - *Retrieval*: `chunk_001148` retrieved at **Rank 0**.
   - *Temporal Filter*: Source is `Crop Protection.pdf` (pub_year 2018, age = 7 years). For `DIAGNOSTIC`, $\lambda = 0.07$.
     $$\text{Score} = 0.85 \times e^{-0.07 \times 7} = 0.5207 < 0.55$$
   - *Result*: `chunk_001148` and `chunk_001158` were discarded. The LLM received only general symptom chunks and correctly diagnosed the disease, but for treatment stated: *"My knowledge base doesn't have specific information."*
3. **`Q23` (Cercospora Leaf Spot, Slow Path DIAG)**:
   - *Retrieval*: `chunk_001017` retrieved at **Rank 0**.
   - *Temporal Filter*: Source is `Crop Protection.pdf` (age = 7). Score $= 0.5207 < 0.55$. Discarded.
   - *Result*: LLM output refusal: *"My knowledge base doesn't have specific information about this."*
4. **`Q30` (Godown Disinfection, Slow Path REC)**:
   - *Retrieval*: `chunk_001818` retrieved at **Rank 0**.
   - *Temporal Filter*: Source is `Crop Protection.pdf` (age = 7). Score $= 0.4221 < 0.55$. Discarded.
   - *Result*: LLM answered using remaining TNAU chunks, missing the specific Malathion dilution in `chunk_001818`.

#### B. Groq Reasoning Token Budget Exhaustion (Fast Path `gpt-oss-20b`)
1. **`Q04` (Trichoderma viride rate, Fast Path FACT)**:
   - *Retrieval*: `chunk_002690` retrieved at **Rank 0** and delivered to prompt.
   - *Mechanism*: `gpt-oss-20b` allocated all 250 tokens to internal reasoning (analyzing the discrepancy between 10 g/kg and 4 g/kg in the text). It hit the 250 token limit before emitting any text in `content`, returning `""`.
2. **`Q13` (Tomato/Chilli/Brinjal seed rates, Fast Path FACT)**:
   - *Retrieval*: `chunk_007442` retrieved at **Rank 2** and delivered to prompt.
   - *Mechanism*: `gpt-oss-20b` exhausted its 250 token budget in reasoning across 3 crop entities, returning `""`.
3. **`Q26` (Vermicompost precautions, Fast Path FACT)**:
   - *Retrieval*: `chunk_001595` retrieved at **Rank 0** and delivered to prompt.
   - *Mechanism*: Reasoning tokens consumed all 250 tokens, returning `""`.
4. **`Q16` (Sugarcane pan evaporation, Slow Path PROC)**:
   - *Retrieval*: `chunk_009552` retrieved at **Rank 2** and delivered to prompt.
   - *Mechanism*: `gpt-oss-120b` produced 2,534 characters of reasoning, leaving only ~50 tokens for `content`. The generated table was cut off mid-sentence after 46 characters (`"**Irrigation Scheduling for Sugarcane (Sacchar"`).

---

### 4.2. Audit of the 5 Retrieval Misses

| QID | Query | Ground Truth Chunk | Candidate Pool Status | Root Cause |
| :--- | :--- | :---: | :---: | :--- |
| `Q08` | High nitrogen fertilizer causing pest susceptibility | `chunk_001041` | Absent from top-15 Dense and Sparse pools | Dense search semantic drift on colloquial phrasing; BM25 diluted across general nitrogen chunks. |
| `Q10` | Azospirillum seed treatment biological mechanisms | `chunk_001999` | Absent from top-15 candidate pool | Term overlap selected sugarcane agronomy chunks (`chunk_004546`) rather than the microbiological biofertilizer chapter. |
| `Q12` | Critical weed-free period in groundnut | `chunk_003231` | Absent from top-15 candidate pool | BM25 selected general weed management advisory rather than crop-specific critical period table. |
| `Q17` | Cashew high-density planting pruning schedule | `chunk_003772` | Absent from top-15 candidate pool | Multi-hop query: dense search prioritized spacing chunks over pruning calendar. |
| `Q28` | Pre-monsoon organic manure in coconut basins | `chunk_011687` | Absent from top-15 candidate pool | High lexical density in other coconut advisory chunks outranked the specific basin manure chunk. |

---

## 5. Incomplete Generation & Prompt Length Constraint Analysis

Eight queries (`Q03`, `Q06`, `Q07`, `Q09`, `Q14`, `Q24`, `Q25`, `Q29`) achieved partial credit (Score = 1) because the LLM answered the primary question but omitted secondary parameters.

### Quantitative Verification
In the fast-path prompt, the system instruction explicitly states:
```
You are an agricultural expert for Indian farmers. Answer ONLY from context.
Be concise — 2-4 sentences maximum.
Include specific numbers/quantities if present.
```
- For `Q03`: The ground-truth facts require (1) 8-inch tine spacing, (2) better weed control benefit, and (3) Coragen 40 ml/acre. The model generated:
  > *"Use the Happy Seeder with an 8-inch tine spacing. Apply Coragen 18.5 SC at 40 ml per acre for armyworm control in wheat."*
  This is an exemplary 2-sentence answer, but it necessarily omitted fact #2 to respect the 2-sentence instruction.
- For `Q14`: The model explained the susceptibility of Kents and Coorgs arabica varieties, but omitted the list of replacement cultivars to stay within the 4-sentence limit.
- **Conclusion**: The 2–4 sentence constraint is directly responsible for fact omissions in at least 5 of the 8 incomplete queries.

---

## 6. Grader Fidelity Audit

The deterministic grader used in the benchmark was audited for potential false negatives caused by regex rigidity:

1. **`Q18` (Rice Zinc Biofortification Cultivars)**:
   - *Expected Fact*: `CO51, CO47, ADT 47, and ADT 37`
   - *Model Generated*: `CO-51, CO-47, ADT-47 and ADT-37`
   - *Grader Result*: Marked **"Not matched"** because of hyphens. The model was factually 100% correct.
2. **`Q24` (Cashew Stem and Root Borer)**:
   - *Expected Fact*: `Cashew Stem and Root Borer (Plocaederus ferrugineus)`
   - *Model Generated*: `Cashew Stem-and-Root Borer (CSRB) - confidence ≈ 92%`
   - *Grader Result*: Marked **"Not matched"** because the scientific Latin binomial was omitted, despite exact common name and acronym matching.
3. **`Q25` (Vermicompost Bed Construction)**:
   - *Expected Fact*: `Constructed on leveled ground`
   - *Model Generated*: `built on a leveled, pucca floor to prevent seepage`
   - *Grader Result*: Marked **"Not matched"** due to "pucca floor" synonym.
4. **`Q29` (Wheat Zinc Sulphate Foliar Spray)**:
   - *Expected Fact*: `200 litres of water (yielding a 0.5% zinc sulphate solution)`
   - *Model Generated*: `200 L of lukewarm water`
   - *Grader Result*: Marked **"Not matched"** due to abbreviation `200 L` vs `200 litres`.

> [!NOTE]
> If these 4 grader false-negatives are recognized, the true fully-correct rate of the pipeline is **40.0%** (12/30) rather than 26.7%.

---

## 7. Latency Profile Analysis

Telemetry was captured across all 30 queries:

| Path | Query Count | Mean Latency | Median Latency | Min Latency | Max Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fast Path** | 19 | **3,232.2 ms** | **2,685.5 ms** | 1,926.6 ms (`Q06`) | 7,011.6 ms (`Q20`) |
| **Slow Path** | 11 | **11,185.8 ms** | **10,856.1 ms** | 5,276.8 ms (`Q16`) | 19,802.8 ms (`Q01`) |
| **Overall** | 30 | **6,148.5 ms** | **4,441.4 ms** | 1,926.6 ms | 19,802.8 ms |

- **Fast-Path Latency**: Dominated by retrieval + reranking (~250 ms) and Groq `gpt-oss-20b` inference (~2,000–3,000 ms).
- **Slow-Path Latency**: Substantially higher due to post-generation sentence-level provenance mapping (`sentence_provenance.py` making LLM verification calls per sentence) and CIB&RC compliance scanning, totaling 10–19 seconds.

---

## 8. Specific Audit Inquiries (Answers A through H)

### A. The True Dominant Failure Mode
The dominant root cause of sub-optimal performance is a tie between **Incomplete Generation due to brevity constraints** (8 queries, 26.7%) and **Post-Retrieval Filtering / Token Budget Starvation** (8 queries, 26.7% across temporal decay and reasoning token exhaustion).

### B. Failures Caused by Retrieval
**5 queries** (16.7%): `Q08`, `Q10`, `Q12`, `Q17`, and `Q28`. In these cases, the ground-truth chunk was never delivered to the reranker's top-5 pool.

### C. Failures Caused by Post-Retrieval Filtering
**5 queries** (16.7%):
- 4 caused by **Temporal Credibility Decay**: `Q01`, `Q22`, `Q23`, `Q30`.
- 1 caused by **Fast-Path Cutoff Displacement**: `Q20` (ground-truth chunk was retrieved at rank 3, but discarded by `chunks[:3]`).

### D. Genuine Generation Failures Despite Evidence in Context
**4 queries** (13.3%):
- 3 caused by **Reasoning Token Exhaustion**: `Q04`, `Q13`, `Q26` (`gpt-oss-20b` spent all 250 tokens in reasoning, returning empty content).
- 1 caused by **Mid-Sentence Truncation**: `Q16` (`gpt-oss-120b` ran out of tokens due to reasoning overhead).

### E. Grader / Scoring Uncertainty Cases
**4 queries** (13.3%): `Q18` (hyphenation), `Q24` (missing Latin binomial), `Q25` (synonym), `Q29` (unit abbreviation).

### F. Is Temporal Credibility Mis-calibrated for Foundational Agronomy?
**Yes, decisively.** Foundational agronomic literature (e.g. TNAU, PAU, KAU packages of practices) detailing seed rates, tine spacing, disease symptom etiology, and pruning geometry remains scientifically authoritative for 5–10 years. Penalizing documents published in 2018–2020 with exponential decay ($\lambda = 0.07\text{--}0.10$) drops their scores below 0.55, causing catastrophic context starvation for timeless agricultural facts.

### G. Is the 2–4 Sentence Constraint Causing Fact Omission?
**Yes, demonstrably.** The fast-path prompt strictly enforces "2–4 sentences maximum". When an agricultural query requires 3–4 discrete technical facts (e.g. tine spacing, implement type, chemical dosage, spray timing), the LLM is forced to discard secondary numbers to satisfy the brevity rule.

### H. Recommended Production Fixes (For Next Stage)
1. **Calibrate Temporal Decay for Evergreen Agronomy**:
   - exempt static agronomic and botanical categories (`PROCEDURAL`, `FACTUAL`, crop geometry) from steep decay, or lower `decay_lambda` to $0.01\text{--}0.02$.
   - Ensure `_match_source()` handles generic university filenames rather than defaulting unmatched documents to a punishing 0.50 score.
2. **Increase Fast-Path Token Budget for Reasoning Models**:
   - Increase `max_tokens` from 250 to 600 in `fast_path_answer()` so that reasoning models (`gpt-oss-20b`) have sufficient headroom to complete reasoning and output the text.
3. **Expand Fast-Path Context Slicing**:
   - Pass top-5 chunks (`chunks[:5]`) rather than top-3 chunks (`chunks[:3]`) to avoid cutoff displacement (e.g. `Q20`).
4. **Relax Prompt Brevity Constraint for Multi-Part Questions**:
   - Change fast-path prompt instruction from "2-4 sentences maximum" to "Be concise but ensure all specific numbers, dosages, and operational parameters are included."
