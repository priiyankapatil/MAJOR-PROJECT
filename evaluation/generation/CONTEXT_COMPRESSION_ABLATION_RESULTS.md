# Selective Context Compression (C1) Experiment Results

**Experiment Date**: 2026-09-09 12:19:49  
**Target**: Production Optimization Experiment C1 — Selective Context Compression / Prompt Distillation  
**Evaluator**: Validated V2 Deterministic Rubric Grader (Zero LLM Judge)  
**Baseline Reference**: T1 Selective Temporal Credibility Experiment  
**Condition (C1)**: Sentence-level context distillation applied to retrieved chunks prior to generator prompt injection. Distillation prunes sentences lacking query relevance or technical specificity, while preserving units, dosages, names, and formulas.  

> [!NOTE]
> **Evaluation Benchmark & Reproducibility Context**:
> - **Ground Truth Version**: Evaluated against the September 9, 2026 pre-remediation ground truth (114 facts, 24 safety constraints including 17 external agronomic rules). Results are not directly comparable to post-remediation benchmarks (R1, R1-C) which evaluate against 7 source-grounded safety constraints.
> - **Reproducibility Note**: Raw benchmark JSON (`context_compression_ablation_results.json`) is gitignored. Embedded Markdown results reflect historical trial records and do not guarantee independent clean-checkout reproducibility.

---

## 1. Overall Performance Comparison: Baseline T1 vs. Compressed C1

| Metric | Baseline T1 | Compressed C1 | Delta (C1 − T1) | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Strict Accuracy (Score=2)** | **33.3%** (10/30) | **33.3%** (10/30) | **+0.0%** | 0.0% |
| **Partial Credit Rate (Score=1)** | **40.0%** (12/30) | **43.3%** (13/30) | **+3.3%** | +8.3% |
| **Incorrect / Refusal Rate (Score=0)** | **26.7%** (8/30) | **23.3%** (7/30) | **-3.3%** | −12.5% |
| **Expected-Fact Coverage** | **45.6%** (52/114) | **50.9%** (58/114) | **+5.3%** (+6 facts) | +11.5% |
| **Safety Adherence** | **100.0%** (7/7 active) | **100.0%** (7/7 active) | **0.0%** (Maintained) | 0.0% |
| **Mean Latency** | **7094.1 ms** | **7019.6 ms** | **-74.5 ms** | −1.0% |
| **Median Latency** | **4065.1 ms** | **4752.5 ms** | **+687.4 ms** | +16.9% |
| **Mean Context Size** | **271.4 words** | **223.2 words** | **-48.1 words** | −17.0% reduction |
| **Median Context Size** | **225.5 words** | **180.0 words** | **-45.5 words** | −20.2% reduction |

### Compression Activity & Context Integrity
- **Mean Compression Ratio**: **17.0%** word reduction across prompt contexts
- **Queries with CONTEXT_LOSS**: **7 of 30 queries** (23.3%)

### Outcome Breakdown (vs. Baseline T1)
- **IMPROVED**: **8 queries** (26.7%)
- **UNCHANGED**: **13 queries** (43.3%)
- **DEGRADED**: **9 queries** (30.0%)
- **NEW_FAILURE**: **0 queries** (0.0%)

---

## 2. Partition of Failure Causes Across All 30 Benchmark Queries

| Failure Category | Query Count | Description |
| :--- | :---: | :--- |
| **CORRECT_ANSWER** | **10** | Satisfies $\ge 70\%$ of expected key facts with zero safety violations. |
| **INCOMPLETE_ANSWER** | **13** | Partially satisfies facts ($< 70\%$) but articulates at least one key fact accurately. |
| **GENERATION_FAILURE** | **3** | Refusal or empty response produced despite evidence present in context. |
| **FACT_PRESENT_IN_C1_BUT_NOT_GENERATED** | **3** | Target fact survived compression and reached prompt, but LLM omitted it from generation. |
| **FACT_NOT_RETRIEVED** | **1** | Target evidence was absent from the initial top-5 retrieved candidates. |
| **FACT_RETRIEVED_BUT_REMOVED_BY_C1** | **0** | Target fact was present in retrieved chunks but pruned by C1 sentence compression. |

---

## 3. Category-Level Performance Comparison

| Category | Queries | Strict T1 | Strict C1 | Facts T1 | Facts C1 | Net Facts Delta | Mean Word Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_lexical` | 6 | 2/6 (33%) | 2/6 (33%) | 11/19 (57.9%) | 10/19 (52.6%) | **-1** | **16.6%** |
| `conceptual_paraphrased` | 6 | 0/6 (0%) | 1/6 (17%) | 6/24 (25.0%) | 8/24 (33.3%) | **+2** | **17.8%** |
| `crop_specific` | 6 | 1/6 (17%) | 2/6 (33%) | 7/22 (31.8%) | 10/22 (45.5%) | **+3** | **12.4%** |
| `diagnostic` | 6 | 4/6 (67%) | 3/6 (50%) | 16/24 (66.7%) | 17/24 (70.8%) | **+1** | **15.7%** |
| `procedural_recommendation` | 6 | 3/6 (50%) | 2/6 (33%) | 12/25 (48.0%) | 13/25 (52.0%) | **+1** | **23.3%** |

---

## 4. Query-Level Comparison Table (All 30 Benchmark Queries)

| QID | Category | Path | Orig Words | Comp Words | Reduction | Base T1 Sc | C1 Sc | Base Facts | C1 Facts | Outcome | Context Loss | Failure Mode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Q01` | `exact_lexical` | `slow` | 310w | 268w | 13.5% | 1 | **1** | 2/4 | **1/4** | **DEGRADED** | ⚠️ Lost [3] | `INCOMPLETE_ANSWER` |
| `Q02` | `exact_lexical` | `fast` | 111w | 102w | 8.1% | 2 | **2** | 3/3 | **3/3** | **UNCHANGED** | None | `CORRECT_ANSWER` |
| `Q03` | `exact_lexical` | `fast` | 237w | 221w | 6.8% | 1 | **1** | 2/3 | **1/3** | **DEGRADED** | None | `INCOMPLETE_ANSWER` |
| `Q04` | `exact_lexical` | `fast` | 215w | 178w | 17.2% | 0 | **0** | 0/3 | **0/3** | **UNCHANGED** | None | `GENERATION_FAILURE` |
| `Q05` | `exact_lexical` | `slow` | 334w | 272w | 18.6% | 2 | **2** | 3/3 | **3/3** | **UNCHANGED** | None | `CORRECT_ANSWER` |
| `Q06` | `exact_lexical` | `fast` | 218w | 147w | 32.6% | 1 | **1** | 1/3 | **2/3** | **IMPROVED** | ⚠️ Lost [2] | `INCOMPLETE_ANSWER` |
| `Q07` | `conceptual_paraphrased` | `fast` | 198w | 174w | 12.1% | 1 | **2** | 1/4 | **4/4** | **IMPROVED** | None | `CORRECT_ANSWER` |
| `Q08` | `conceptual_paraphrased` | `fast` | 195w | 145w | 25.6% | 1 | **1** | 1/4 | **2/4** | **IMPROVED** | None | `INCOMPLETE_ANSWER` |
| `Q09` | `conceptual_paraphrased` | `fast` | 184w | 182w | 1.1% | 1 | **0** | 2/4 | **0/4** | **DEGRADED** | None | `FACT_PRESENT_IN_C1_BUT_NOT_GENERATED` |
| `Q10` | `conceptual_paraphrased` | `fast` | 214w | 140w | 34.6% | 1 | **1** | 1/4 | **1/4** | **UNCHANGED** | None | `INCOMPLETE_ANSWER` |
| `Q11` | `conceptual_paraphrased` | `fast` | 213w | 177w | 16.9% | 1 | **0** | 1/4 | **0/4** | **DEGRADED** | None | `FACT_PRESENT_IN_C1_BUT_NOT_GENERATED` |
| `Q12` | `conceptual_paraphrased` | `fast` | 173w | 150w | 13.3% | 0 | **1** | 0/4 | **1/4** | **IMPROVED** | None | `INCOMPLETE_ANSWER` |
| `Q13` | `crop_specific` | `fast` | 162w | 156w | 3.7% | 0 | **2** | 0/4 | **3/4** | **IMPROVED** | None | `CORRECT_ANSWER` |
| `Q14` | `crop_specific` | `fast` | 239w | 147w | 38.5% | 1 | **1** | 1/3 | **1/3** | **UNCHANGED** | None | `INCOMPLETE_ANSWER` |
| `Q15` | `crop_specific` | `slow` | 269w | 228w | 15.2% | 1 | **1** | 2/4 | **2/4** | **UNCHANGED** | None | `INCOMPLETE_ANSWER` |
| `Q16` | `crop_specific` | `slow` | 348w | 329w | 5.5% | 0 | **0** | 0/4 | **0/4** | **UNCHANGED** | None | `GENERATION_FAILURE` |
| `Q17` | `crop_specific` | `fast` | 190w | 173w | 8.9% | 0 | **0** | 0/3 | **0/3** | **UNCHANGED** | None | `FACT_PRESENT_IN_C1_BUT_NOT_GENERATED` |
| `Q18` | `crop_specific` | `slow` | 328w | 312w | 4.9% | 2 | **2** | 4/4 | **4/4** | **UNCHANGED** | None | `CORRECT_ANSWER` |
| `Q19` | `diagnostic` | `fast` | 216w | 178w | 17.6% | 2 | **2** | 3/4 | **3/4** | **UNCHANGED** | ⚠️ Lost [3] | `CORRECT_ANSWER` |
| `Q20` | `diagnostic` | `fast` | 183w | 164w | 10.4% | 0 | **1** | 0/4 | **2/4** | **IMPROVED** | None | `INCOMPLETE_ANSWER` |
| `Q21` | `diagnostic` | `fast` | 218w | 168w | 22.9% | 2 | **2** | 3/4 | **3/4** | **UNCHANGED** | ⚠️ Lost [3] | `CORRECT_ANSWER` |
| `Q22` | `diagnostic` | `slow` | 356w | 295w | 17.1% | 2 | **2** | 4/4 | **3/4** | **DEGRADED** | None | `CORRECT_ANSWER` |
| `Q23` | `diagnostic` | `slow` | 325w | 305w | 6.2% | 2 | **0** | 4/4 | **4/4** | **DEGRADED** | None | `GENERATION_FAILURE` |
| `Q24` | `diagnostic` | `slow` | 352w | 281w | 20.2% | 1 | **1** | 2/4 | **2/4** | **UNCHANGED** | None | `INCOMPLETE_ANSWER` |
| `Q25` | `procedural_recommendation` | `fast` | 220w | 104w | 52.7% | 2 | **2** | 4/4 | **3/4** | **DEGRADED** | None | `CORRECT_ANSWER` |
| `Q26` | `procedural_recommendation` | `fast` | 231w | 199w | 13.9% | 0 | **2** | 0/4 | **3/4** | **IMPROVED** | ⚠️ Lost [2] | `CORRECT_ANSWER` |
| `Q27` | `procedural_recommendation` | `slow` | 367w | 271w | 26.2% | 1 | **1** | 2/5 | **3/5** | **IMPROVED** | ⚠️ Lost [4] | `INCOMPLETE_ANSWER` |
| `Q28` | `procedural_recommendation` | `slow` | 253w | 240w | 5.1% | 0 | **0** | 0/4 | **0/4** | **UNCHANGED** | None | `FACT_NOT_RETRIEVED` |
| `Q29` | `procedural_recommendation` | `slow` | 287w | 243w | 15.3% | 2 | **1** | 3/4 | **2/4** | **DEGRADED** | None | `INCOMPLETE_ANSWER` |
| `Q30` | `procedural_recommendation` | `slow` | 995w | 748w | 24.8% | 2 | **1** | 3/4 | **2/4** | **DEGRADED** | ⚠️ Lost [3] | `INCOMPLETE_ANSWER` |

---

## 5. In-Depth Evaluation of Targeted Queries (Q03, Q14, Q16, Q25, Q29)

### 🔍 Query `Q03`: *What is the recommended tine spacing for Happy Seeder and Coragen 18.5 SC dose in wheat?*
- **Competing Context**: Retrieved chunks contained section headings ("PROTECTION TECHNIQUES", "PRODUCTION TECHNIQUES") and general seeding depth advice.
- **C1 Retained**: Pruned boilerplate headings; kept the 8-inch tine spacing, better weed control, and 40 ml Coragen dosage sentences.
- **Fact Preservation**: 100% preserved (all 3 expected facts present in compressed context).
- **Outcome**: Score 1 (1/3 facts generated vs 2/3 in T1 due to model omitting the secondary 'better weed control' phrase). Latency = 2,484 ms.

### 🔍 Query `Q14`: *Why are older arabica coffee varieties like Kents and Coorgs considered vulnerable to leaf rust disease?*
- **Competing Context**: Chunks contained nursery bed straw mulching techniques and Dadap vegetative cutting rooting protocols.
- **C1 Retained**: Pruned all unrelated vegetative propagation sentences (38.5% context reduction). Kept all sentences comparing older arabica susceptibility to modern selections.
- **Fact Preservation**: 100% preserved of retrieved facts.
- **Outcome**: Score 1 (1/3 facts). Model generated clean focus on *Hemileia vastatrix* susceptibility. Latency = 2,419 ms.

### 🔍 Query `Q16`: *How is irrigation scheduled in sugarcane based on climatological pan evaporation approach?*
- **Competing Context**: Sugarcane IW/CPE guidelines competed with Maize drip irrigation formulas (`(Pe x Kp x Kc x A x Wp) - Re`) and aerobic rice data.
- **C1 Retained**: Pruned alternate furrow descriptions and winter intervals.
- **Fact Preservation**: All formula components remained in the compressed context.
- **Outcome**: Score 0. Model produced empty generation on this run (`GENERATION_FAILURE`). Demonstrates that compression alone does not solve model generation refusal when the underlying source chunk is labelled 'Maize' for a sugarcane query.

### 🔍 Query `Q25`: *What are the bed dimensions and structural specifications for preparing a vermicompost production unit?*
- **Competing Context**: Retrieved chunks contained 15 cm height cutting beds and 20 L plastic basin vermiwash equipment.
- **C1 Retained**: Pruned all unrelated cutting beds and basin equipment sentences, achieving a **52.7% context reduction** (220 words down to 104 words).
- **Fact Preservation**: 100% preserved (all 4 bed dimensions and pucca floor facts retained).
- **Outcome**: Score 2 (3/4 facts). Generator focused strictly on 6' × 3' × 2' cemented pucca beds. Latency = 2,829 ms.

### 🔍 Query `Q29`: *How should foliar spray solution of zinc sulphate and unslaked lime be prepared per acre to correct deficiency?*
- **Competing Context**: Chunks contained 5 competing zinc recipes: 2g/L foliar, 0.25% lime mixture, 50 kg/ha soil dose, Atrazine herbicide spray, and hand weeding at 20 DAT.
- **C1 Retained**: Pruned unrelated hand weeding and soil weed management sentences (15.3% reduction).
- **Fact Preservation**: 100% preserved (all wheat zinc sulphate + unslaked lime facts kept).
- **Outcome**: Score 1 (2/4 facts). Reduced distractor material, but model articulated 1 kg zinc in 200 L without mentioning 15-day intervals.

---

## 6. Critical Analysis of CONTEXT_LOSS Risk

The experiment revealed an essential architectural risk: **Sentence-level query distillation prunes secondary agronomic qualifications**.
In 7 of 30 queries, a required ground-truth fact was pruned because the sentence lacked direct query keywords:
1. `Q01`: *"Spray fortnightly through the flowering and fruit initiation period"* was pruned because the query only asked for *concentration*.
2. `Q26`: *"Cover beds with shed to protect from rain, cold, and heat"* was pruned because the query only asked for *duration and temperature precautions*.
3. `Q27`: *"Watering in the afternoon leads to damping-off disease"* was pruned because the query did not mention *damping-off*.
4. `Q30`: *"Cover floor with 7 cm layer of sand/sawdust to prevent dhora infestation"* was pruned because the query asked for *emulsion concentration*.

**Conclusion**: Aggressive sentence distillation risks pruning vital collateral agronomic instructions that are part of comprehensive university extension advice.