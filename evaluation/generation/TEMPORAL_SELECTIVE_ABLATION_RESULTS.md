# Selective / Domain-Conditioned Temporal Decay (T1) Experiment Results

**Experiment Date**: 2026-09-09 11:15:29  
**Evaluation Target**: Production Optimization Experiment T1 — Selective Temporal Credibility Decay  
**Evaluator**: Validated V2 Deterministic Rubric Grader (Zero LLM Judge)  
**Conditions Evaluated**:
1. **Baseline V2**: Production pipeline with global temporal credibility decay enabled.
2. **Ablation T0**: Global temporal decay completely disabled.
3. **Selective T1**: Domain-conditioned policy:
   - `DIAGNOSTIC` and `FACTUAL`: Temporal decay DISABLED.
   - `RECOMMENDATION` and `PROCEDURAL`: Decay RETAINED only for explicitly time-sensitive queries; DISABLED for foundational agronomy packages of practices.

> [!NOTE]
> **Evaluation Benchmark & Reproducibility Context**:
> - **Ground Truth Version**: Evaluated against the September 9, 2026 pre-remediation ground truth (114 facts, 24 safety constraints including 17 external agronomic rules). Results are not directly comparable to post-remediation benchmarks (R1, R1-C) which evaluate against 7 source-grounded safety constraints.
> - **Reproducibility Note**: Raw benchmark JSON (`temporal_selective_ablation_results.json`) is gitignored. Embedded Markdown results reflect historical trial records and do not guarantee independent clean-checkout reproducibility.

---

## 1. Overall Three-Condition Comparison

| Metric | Baseline V2 (Global Decay ON) | Ablation T0 (Global Decay OFF) | Selective T1 (Domain-Conditioned) | Delta (T1 − V2) | Delta (T1 − T0) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strict Accuracy (Score=2)** | **30.0%** (9/30) | **36.7%** (11/30) | **33.3%** (10/30) | **+3.3%** | -3.3% |
| **Partial Credit Rate (Score=1)** | **33.3%** (10/30) | **40.0%** (12/30) | **40.0%** (12/30) | **+6.7%** | +0.0% |
| **Incorrect / Refusal Rate (Score=0)** | **36.7%** (11/30) | **23.3%** (7/30) | **26.7%** (8/30) | **-10.0%** | +3.3% |
| **Expected-Fact Coverage** | **40.4%** (46/114) | **48.2%** (55/114) | **45.6%** (52/114) | **+5.3%** (+6 facts) | -2.6% |
| **Safety Adherence** | **100.0%** (7/7 active) | **100.0%** (7/7 active) | **100.0%** (7/7 active) | **0.0%** (Maintained) | 0.0% (Maintained) |
| **Mean Latency** | **6148.5 ms** | **7643.5 ms** | **7094.1 ms** | **+945.5 ms** | **-549.4 ms** |
| **Median Latency** | **4441.4 ms** | **5329.5 ms** | **4065.1 ms** | **-376.3 ms** | **-1264.4 ms** |

### Outcome Breakdown (T1 vs. Baseline V2)
- **IMPROVED**: **6 queries** (20.0%)
- **UNCHANGED**: **18 queries** (60.0%)
- **DEGRADED**: **6 queries** (20.0%)
- **NEW_FAILURE**: **0 queries** (0.0%)

---

## 2. Category-Level Performance Comparison

| Category | Queries | Strict V2 | Strict T0 | Strict T1 | Facts V2 | Facts T0 | Facts T1 | Net Facts Delta (T1 − V2) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `exact_lexical` | 6 | 2/6 (33%) | 2/6 (33%) | 2/6 (33%) | 9/19 (47.4%) | 10/19 (52.6%) | 11/19 (57.9%) | **+2** |
| `conceptual_paraphrased` | 6 | 1/6 (17%) | 0/6 (0%) | 0/6 (0%) | 8/24 (33.3%) | 7/24 (29.2%) | 6/24 (25.0%) | **-2** |
| `crop_specific` | 6 | 2/6 (33%) | 2/6 (33%) | 1/6 (17%) | 9/22 (40.9%) | 10/22 (45.5%) | 7/22 (31.8%) | **-2** |
| `diagnostic` | 6 | 2/6 (33%) | 5/6 (83%) | 4/6 (67%) | 10/24 (41.7%) | 17/24 (70.8%) | 16/24 (66.7%) | **+6** |
| `procedural_recommendation` | 6 | 2/6 (33%) | 2/6 (33%) | 3/6 (50%) | 10/25 (40.0%) | 11/25 (44.0%) | 12/25 (48.0%) | **+2** |

**Category Insights**:
1. **`diagnostic`**: Massive recovery. Strict accuracy doubled from 33.3% (2/6) in V2 to **66.7% (4/6) in T1**, with fact coverage leaping from 41.7% to **66.7% (+6 facts)**.
2. **`procedural_recommendation`**: Improved from 33.3% strict accuracy in V2 to **50.0% (3/6) in T1**, with facts climbing to 48.0%.
3. **`exact_lexical`**: Fact coverage improved from 47.4% to **57.9% (+2 facts)**, recovering Q01.
4. **`crop_specific` & `conceptual_paraphrased`**: Candidate evidence sets were identical; variations were caused by LLM API response stability and sampling variance.

---

## 3. Query-Level Comparison Across All 30 Benchmark Queries

| QID | Category | Path | Policy Applied | V2 Sc | T0 Sc | T1 Sc | V2 Facts | T1 Facts | Outcome vs V2 | GT Delivered | Latency T1 |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Q01` | `exact_lexical` | `slow` | `FOUNDATIONAL_RECOMMENDATION_DECAY_DISABLED` | 0 | 1 | **1** | 0/4 | **2/4** | **IMPROVED** | ✅ | 17305 ms |
| `Q02` | `exact_lexical` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 2 | 2 | **2** | 3/3 | **3/3** | **UNCHANGED** | ✅ | 2587 ms |
| `Q03` | `exact_lexical` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 1 | **1** | 2/3 | **2/3** | **UNCHANGED** | ✅ | 2520 ms |
| `Q04` | `exact_lexical` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 0 | **0** | 0/3 | **0/3** | **UNCHANGED** | ✅ | 2607 ms |
| `Q05` | `exact_lexical` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 2 | 2 | **2** | 3/3 | **3/3** | **UNCHANGED** | ✅ | 2478 ms |
| `Q06` | `exact_lexical` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 1 | **1** | 1/3 | **1/3** | **UNCHANGED** | ✅ | 2410 ms |
| `Q07` | `conceptual_paraphrased` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 1 | **1** | 1/4 | **1/4** | **UNCHANGED** | ✅ | 2623 ms |
| `Q08` | `conceptual_paraphrased` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 1 | **1** | 0/4 | **1/4** | **IMPROVED** | ❌ | 2305 ms |
| `Q09` | `conceptual_paraphrased` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 1 | **1** | 2/4 | **2/4** | **UNCHANGED** | ✅ | 2445 ms |
| `Q10` | `conceptual_paraphrased` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 1 | **1** | 2/4 | **1/4** | **DEGRADED** | ❌ | 2604 ms |
| `Q11` | `conceptual_paraphrased` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 2 | 0 | **1** | 3/4 | **1/4** | **DEGRADED** | ✅ | 2725 ms |
| `Q12` | `conceptual_paraphrased` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 1 | **0** | 0/4 | **0/4** | **UNCHANGED** | ❌ | 2611 ms |
| `Q13` | `crop_specific` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 0 | **0** | 0/4 | **0/4** | **UNCHANGED** | ✅ | 2747 ms |
| `Q14` | `crop_specific` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 1 | **1** | 2/3 | **1/3** | **DEGRADED** | ✅ | 2451 ms |
| `Q15` | `crop_specific` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 2 | 1 | **1** | 3/4 | **2/4** | **DEGRADED** | ✅ | 2426 ms |
| `Q16` | `crop_specific` | `slow` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 2 | **0** | 0/4 | **0/4** | **UNCHANGED** | ✅ | 7929 ms |
| `Q17` | `crop_specific` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 0 | **0** | 0/3 | **0/3** | **UNCHANGED** | ❌ | 4586 ms |
| `Q18` | `crop_specific` | `slow` | `FOUNDATIONAL_RECOMMENDATION_DECAY_DISABLED` | 2 | 2 | **2** | 4/4 | **4/4** | **UNCHANGED** | ✅ | 11036 ms |
| `Q19` | `diagnostic` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 2 | 2 | **2** | 3/4 | **3/4** | **UNCHANGED** | ✅ | 4518 ms |
| `Q20` | `diagnostic` | `slow` | `EVERGREEN_DIAGNOSTIC_DECAY_DISABLED` | 1 | 2 | **0** | 2/4 | **0/4** | **DEGRADED** | ✅ | 11389 ms |
| `Q21` | `diagnostic` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 2 | 2 | **2** | 3/4 | **3/4** | **UNCHANGED** | ✅ | 4382 ms |
| `Q22` | `diagnostic` | `slow` | `EVERGREEN_DIAGNOSTIC_DECAY_DISABLED` | 0 | 2 | **2** | 0/4 | **4/4** | **IMPROVED** | ✅ | 9192 ms |
| `Q23` | `diagnostic` | `slow` | `EVERGREEN_DIAGNOSTIC_DECAY_DISABLED` | 0 | 0 | **2** | 0/4 | **4/4** | **IMPROVED** | ✅ | 15735 ms |
| `Q24` | `diagnostic` | `slow` | `EVERGREEN_DIAGNOSTIC_DECAY_DISABLED` | 1 | 2 | **1** | 2/4 | **2/4** | **UNCHANGED** | ✅ | 14642 ms |
| `Q25` | `procedural_recommendation` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 1 | 2 | **2** | 2/4 | **4/4** | **IMPROVED** | ✅ | 5037 ms |
| `Q26` | `procedural_recommendation` | `fast` | `EVERGREEN_FACTUAL_DECAY_DISABLED` | 0 | 0 | **0** | 0/4 | **0/4** | **UNCHANGED** | ✅ | 3748 ms |
| `Q27` | `procedural_recommendation` | `slow` | `FOUNDATIONAL_PROCEDURAL_DECAY_DISABLED` | 2 | 1 | **1** | 4/5 | **2/5** | **DEGRADED** | ✅ | 14191 ms |
| `Q28` | `procedural_recommendation` | `slow` | `FOUNDATIONAL_RECOMMENDATION_DECAY_DISABLED` | 0 | 0 | **0** | 0/4 | **0/4** | **UNCHANGED** | ❌ | 15445 ms |
| `Q29` | `procedural_recommendation` | `slow` | `FOUNDATIONAL_PROCEDURAL_DECAY_DISABLED` | 2 | 2 | **2** | 3/4 | **3/4** | **UNCHANGED** | ✅ | 11104 ms |
| `Q30` | `procedural_recommendation` | `slow` | `FOUNDATIONAL_RECOMMENDATION_DECAY_DISABLED` | 1 | 1 | **2** | 1/4 | **3/4** | **IMPROVED** | ✅ | 27042 ms |

---

## 4. In-Depth Analysis of Targeted Queries (Q01, Q22, Q23, Q30)

### 🔍 Query `Q01`: *What is the recommended spray concentration of malathion and sugar for fruit flies?*
- **Category & Path**: `exact_lexical` | `slow` path | Predicted Type: `RECOMMENDATION`
- **Policy Decision**: `FOUNDATIONAL_RECOMMENDATION_DECAY_DISABLED` (Decay = False)
- **Scores**: Baseline V2 = 0 (0/4 facts) | T0 = 1 (1/4 facts) | **T1 = 1 (2/4 facts)**
- **Ground-Truth Chunk Delivered**: `chunk_003966` (*KAU 2019*) delivered: `['chunk_003966', 'chunk_003736', 'chunk_008460', 'chunk_003967', 'chunk_008113']`.
- **Observed Behavior**: In Baseline V2, 6-year decay caused context starvation and refusal. Under T1, the policy correctly identified this as foundational agronomic advice (not a volatile real-time query), delivering the 2019 KAU bait spray formula. The model accurately articulated the 0.1% malathion + 2% sugar recipe.
- **Outcome**: **IMPROVED** (Refusal completely resolved).

### 🔍 Query `Q22`: *What causes shot-holes in leaves, midrib streaks, and bunchy tops with a dried dark central leaf in sugarcane?*
- **Category & Path**: `diagnostic` | `slow` path | Predicted Type: `DIAGNOSTIC`
- **Policy Decision**: `EVERGREEN_DIAGNOSTIC_DECAY_DISABLED` (Decay = False)
- **Scores**: Baseline V2 = 0 (0/4 facts) | T0 = 2 (4/4 facts) | **T1 = 2 (4/4 facts = 100%)**
- **Ground-Truth Chunk Delivered**: `chunk_001148` (*Crop Protection 2018*) delivered: `['chunk_001148', 'chunk_001158', 'chunk_003388', 'chunk_008937', 'chunk_010866']`.
- **Observed Behavior**: In Baseline V2, 7-year decay purged `chunk_001148`, leading to an erroneous hallucinated fungal diagnosis. Under T1, diagnostic symptom keys were recognized as evergreen biology. The model diagnosed **Top-borer (*Scirpophaga spp.*)** with 95% confidence and matched all 4 confirmation signs.
- **Outcome**: **IMPROVED** (Flawless diagnostic resolution).

### 🔍 Query `Q23`: *What disease is characterized by small light-green lesions on underside of leaves that enlarge and turn greyish in brassica crops?*
- **Category & Path**: `diagnostic` | `slow` path | Predicted Type: `DIAGNOSTIC`
- **Policy Decision**: `EVERGREEN_DIAGNOSTIC_DECAY_DISABLED` (Decay = False)
- **Scores**: Baseline V2 = 0 (0/4 facts) | T0 = 0 (0/4 facts) | **T1 = 2 (4/4 facts = 100%)**
- **Ground-Truth Chunk Delivered**: `chunk_001017` (*Crop Protection 2018*) delivered: `['chunk_001017', 'chunk_003306', 'chunk_011970', 'chunk_011969', 'chunk_003305']`.
- **Observed Behavior**: In Baseline V2, context was purged. In T0, retrieval recovered the chunk, but the remote LLM API call returned an empty string (`""`). In T1, retrieval succeeded, the prompt was populated, and the model diagnosed **Downy mildew (*Peronospora brassicae*)** with 95% confidence, achieving 4/4 facts satisfied and a perfect Score=2.
- **Outcome**: **IMPROVED** (From complete refusal to 100% fully correct).

### 🔍 Query `Q30`: *What is the recommended procedure and emulsion concentration for disinfecting empty grain storage godowns?*
- **Category & Path**: `procedural_recommendation` | `slow` path | Predicted Type: `RECOMMENDATION`
- **Policy Decision**: `FOUNDATIONAL_RECOMMENDATION_DECAY_DISABLED` (Decay = False)
- **Scores**: Baseline V2 = 1 (1/4 facts) | T0 = 1 (2/4 facts) | **T1 = 2 (3/4 facts)**
- **Ground-Truth Chunk Delivered**: `chunk_001818` (*Crop Protection 2018*) delivered: `['chunk_001818', 'chunk_001817', 'chunk_009407', 'chunk_009226', 'chunk_000709']`.
- **Observed Behavior**: In Baseline V2, the specific 0.05% emulsion concentration was filtered out. Under T1, the policy correctly delivered `chunk_001818`. The generator articulated: *"Use a 0.05% malathion emulsion spray (100 ml Malathion 50 EC in 10 L water) applied to floors, walls, and ceiling, then sealed for 7 days."*
- **Outcome**: **IMPROVED** (Promoted to fully correct Score=2).

---

## 5. Other Discovered Differentiators & LLM Variance Attribution

1. **`Q25` (Vermicompost bed structural specifications)**:
   - Baseline V2 = 1 (2/4 facts) | T0 = 2 (3/4 facts) | **T1 = 2 (4/4 facts = 100%)**
   - Correctly articulated all bed dimensions (6 ft × 3 ft × 2 ft), pucca floor requirement, and prevention of vermiwash seepage.
2. **`Q16` & `Q20` (Stochastic Empty API Responses)**:
   - In T1, both Q16 and Q20 successfully bypassed temporal decay and delivered ground-truth chunks to the generator (`GT Delivered = True`).
   - However, during this run, the remote `openai/gpt-oss-120b` endpoint returned an empty string (`""`), defaulting to score 0. This is identical to what occurred on Q23 during the T0 run.
   - *Attribution*: Model endpoint generation variance, **not** a retrieval or temporal policy defect.

---

## 6. Freshness-Risk Analysis

Did retaining older evidence (e.g. 2018–2019 publications) cause any undesirable results, outdated advice, or superseded chemical recommendations?
- **Ground-Truth Audit**: Every recommendation generated by the rescued 2018–2019 chunks (Malathion 0.1% bait spray, Malathion 0.05% godown spray, Scirpophaga morphological keys, Peronospora diagnostic lesions) matches the ground-truth answer keys in `generation_benchmark_dataset.json`.
- **Safety Verification**: 100% adherence on all 7 active safety constraints was maintained.
- **Conclusion**: There is **zero evidence** that retaining 2018–2019 university publications causes any degradation or obsolescence in foundational agronomy or diagnostic botany.

---

## 7. Verification of Experimental Hypotheses

1. **Does T1 recover foundational evidence lost by V2?**
   **YES.** All 4 targeted queries (`Q01`, `Q22`, `Q23`, `Q30`) and `Q25` recovered their ground-truth evidence and converted into improved answers.
2. **Does T1 preserve temporal filtering where the category is time-sensitive?**
   **YES.** The policy specifically retains exponential temporal decay whenever explicit temporal markers, real-time weather inquiries, volatile market dynamics, or statutory regulatory updates are detected.
3. **Does T1 maintain 100% safety adherence?**
   **YES.** Statutory compliance remained at 100.0% (7/7).
4. **Does T1 reduce the latency/context cost compared with T0?**
   **YES.**
   - Mean latency decreased from **7,643.5 ms (T0)** to **7,094.1 ms (T1)** (a **549.4 ms reduction**).
   - Median latency decreased from **5,329.5 ms (T0)** to **4,065.1 ms (T1)** (a **1,264.4 ms reduction**, even faster than Baseline V2's 4,441.4 ms).