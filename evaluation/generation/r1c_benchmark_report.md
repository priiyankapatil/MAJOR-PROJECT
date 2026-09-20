# Experiment R1-C: Optimal Synthesis Benchmark Report

## 1. Executive Summary & Configuration

Experiment **R1-C (Optimal Synthesis)** combines the validated advantages of **R1-A** (450-token fast-path budget on `openai/gpt-oss-20b`) and **R1-B** (routing multi-entity and procedural queries to `openai/gpt-oss-120b`), while raising slow-path `max_tokens` from 700 to **800** to eliminate length truncations on detailed recipes.

### R1-C Architecture:
- **Fast Path**: `openai/gpt-oss-20b`, `max_tokens = 450`, `E1_FAST_PATH_SYSTEM` prompt.
- **Slow Path**: `openai/gpt-oss-120b`, `max_tokens = 800`, `E1_SLOW_PATH_SYSTEM` prompt.
- **Complexity-Aware Routing Policy**:
  - Procedural recommendations (`procedural_recommendation`: Q25, Q26, Q27, Q28, Q29, Q30) routed to **Slow Path (120B)**.
  - Multi-entity comparative/procedural queries (`Q09`, `Q13`) routed to **Slow Path (120B)**.
  - Standard lexical, conceptual, and diagnostic queries routed to **Fast Path (20B)**.
- **Grading**: Validated V2 Deterministic Rubric Grader (114 key facts, 7 safety rules).

> [!NOTE]
> **Evaluation Assessment & Reproducibility Context**:
> - **Accuracy & Optimality**: "Optimal Synthesis" reflects architectural intent balancing latency and token length, not empirical production superiority. Observed strict fully-correct accuracy (Score=2) for R1-C is **36.67%** (11/30 queries), which is lower than R1-A at **46.67%** (14/30 queries). While R1-C achieved higher partial credit (50.0%) and reduced token consumption, it cannot be claimed as empirically optimal for production deployment.
> - **Reproducibility Note**: Raw benchmark JSON (`r1c_benchmark_results.json`) is gitignored. Embedded Markdown results reflect historical trial records and do not guarantee independent clean-checkout reproducibility.

---

## 2. Benchmark Progress: T1 vs E1 vs R1-A vs R1-B vs R1-C

| Metric | Baseline T1 | E1 Hardened | R1-A (Budget=450) | R1-B (Complexity Routing) | **R1-C (Optimal Synthesis)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Strict Accuracy (Score 2)** | 33.33% (10/30) | 33.33% (10/30) | 46.67% (14/30) | 36.67% (11/30) | **36.67% (11/30)** |
| **Partial Answer Rate (Score 1)** | 40.0% (12/30) | 53.33% (16/30) | 43.33% (13/30) | 46.67% (14/30) | **50.0% (15/30)** |
| **Incorrect / Failure Rate (Score 0)** | 26.67% (8/30) | 13.33% (4/30) | 10.0% (3/30) | 16.67% (5/30) | **13.33% (4/30)** |
| **Fact Coverage (%)** | 45.61% | 56.14% | 62.28% | 56.14% | **57.02%** |
| **Total Facts Satisfied** | 52 / 114 | 64 / 114 | 71 / 114 | 64 / 114 | **65 / 114** |
| **Safety Adherence (%)** | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Unsupported Claims** | 0 | 0 | 0 | 0 | **0** |
| **Mean Latency (ms)** | 7094.1 ms | 1834.7 ms | 2691.7 ms | 3010.8 ms | **2335.9 ms** |
| **Median Latency (ms)** | 4065.1 ms | 1075.2 ms | 1090.0 ms | 1704.8 ms | **949.4 ms** |
| **Mean Answer Length (words)** | ~140.0 | 73.4 | 70.8 | 81.6 | **83.9** |
| **Median Answer Length (words)** | ~120.0 | 59.0 | 61.0 | 64.0 | **67.5** |
| **Finish Reason Distribution** | - | - | {"length": 3, "stop": 27} | {"length": 5, "stop": 25} | **{"length": 4, "stop": 26}** |
| **Token-Starved / Truncated Answers** | 0 | 0 | 3 | 5 | **4** |

---

## 3. Transition Breakdown vs. E1

- **Improved queries vs E1 (3)**: Q09, Q26, Q29
- **Unchanged queries vs E1 (25)**: Q02, Q03, Q04, Q05, Q06, Q07, Q08, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20, Q21, Q22, Q23, Q24, Q25, Q27, Q28
- **Degraded queries vs E1 (2)**: Q01, Q30
- **Converted Score 0 → 1/2**: None
- **Converted Score 1 → 2**: Q09, Q26
- **Score Regressions**: Q30
- **Generation Failures (Score 0)**: Q04, Q11, Q17, Q28

---

## 4. Mandatory Query-Level Audit

### `Q09`: How do green manure crops like cowpea, dhaincha, or sunnhemp contribute to organic nitrogen enrichment in rice?
- **Category**: `conceptual_paraphrased` | **Path**: `slow` (MULTI_ENTITY_COMPARATIVE_PROCEDURE) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 1 (2/4 facts)
- **E1 Hardened**: Score 1 (2/4 facts)
- **R1-C**: Score 2 (3/4 facts, 75.0%) | Finish: `stop` | Latency: 1350.1 ms
- **Facts Recovered vs E1**: [2] | **Facts Lost vs E1**: []
- **Answer Excerpt**: `**How cowpea, dhaincha or sunn hemp enrich rice with organic nitrogen**

- **Grow the green‑manure crop**  
  - Seed rate: **20 kg per acre** (cowpea, dhaincha or sunn hemp).  
  - Plant after wheat h...`

### `Q13`: What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?
- **Category**: `crop_specific` | **Path**: `slow` (MULTI_ENTITY_COMPARATIVE_PROCEDURE) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 0 (0/4 facts)
- **E1 Hardened**: Score 1 (2/4 facts)
- **R1-C**: Score 1 (2/4 facts, 50.0%) | Finish: `stop` | Latency: 1320.5 ms
- **Facts Recovered vs E1**: [] | **Facts Lost vs E1**: []
- **Answer Excerpt**: `**Seed‑rate (per hectare)**  

- **Tomato**  
  - Normal (open‑pollinated) : 400 – 500 g / ha  
  - Hybrid : 125 – 150 g / ha  

- **Chili**  
  - Normal : 500 g / ha  
  - Hybrid : 200 g / ha  

- **...`

### `Q25`: What are the bed dimensions and structural specifications for preparing a vermicompost production unit?
- **Category**: `procedural_recommendation` | **Path**: `slow` (CATEGORY_PROCEDURAL_RECOMMENDATION) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 2 (4/4 facts)
- **E1 Hardened**: Score 2 (4/4 facts)
- **R1-C**: Score 2 (4/4 facts, 100.0%) | Finish: `stop` | Latency: 1382.0 ms
- **Facts Recovered vs E1**: [] | **Facts Lost vs E1**: []
- **Answer Excerpt**: `- **Bed size:** 6 ft (L) × 3 ft (B) × 2 ft (H) (length can be adjusted to suit available space).  
- **Construction:** Cemented (concrete) structure built on a **leveled ground**.  
- **Floor:** Must...`

### `Q26`: What are the duration and temperature precautions required before vermicompost is fully mature and ready?
- **Category**: `procedural_recommendation` | **Path**: `slow` (CATEGORY_PROCEDURAL_RECOMMENDATION) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 0 (0/4 facts)
- **E1 Hardened**: Score 1 (2/4 facts)
- **R1-C**: Score 2 (3/4 facts, 75.0%) | Finish: `stop` | Latency: 6670.6 ms
- **Facts Recovered vs E1**: [2, 3] | **Facts Lost vs E1**: [0]
- **Answer Excerpt**: `**Duration to maturity**
- Typical vermicompost becomes ready in **60 – 70 days** of composting (general practice).  
- When using leaf litter or cashew‑apple residue, maturity may take up to **95 day...`

### `Q27`: How should coffee nursery seedlings be transplanted to prevent taproot deformation and damping-off disease?
- **Category**: `procedural_recommendation` | **Path**: `slow` (ORIGINAL_SLOW_ROUTING) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 1 (2/5 facts)
- **E1 Hardened**: Score 2 (5/5 facts)
- **R1-C**: Score 2 (5/5 facts, 100.0%) | Finish: `stop` | Latency: 9387.9 ms
- **Facts Recovered vs E1**: [] | **Facts Lost vs E1**: []
- **Answer Excerpt**: `**Transplanting steps to avoid tap‑root deformation and damping‑off**

1. **Prepare the transplant hole**  
   - Make a **vertical hole 5 cm deep** at the centre of each nursery basket (or secondary b...`

### `Q28`: What organic manures and coir pith application practices are recommended for maintaining coconut palms?
- **Category**: `procedural_recommendation` | **Path**: `slow` (ORIGINAL_SLOW_ROUTING) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 0 (0/4 facts)
- **E1 Hardened**: Score 0 (0/4 facts)
- **R1-C**: Score 0 (0/4 facts, 0.0%) | Finish: `stop` | Latency: 9953.3 ms
- **Facts Recovered vs E1**: [] | **Facts Lost vs E1**: []
- **Answer Excerpt**: `**Recommended organic‑manure / coir‑pith practice (as given in the source)**  

- **Rate:** Spread **12.5 t ha⁻¹** of good‑quality farmyard manure (FYM), compost **or** composted coir pith.  
- **Timi...`

### `Q29`: How should foliar spray solution of zinc sulphate and unslaked lime be prepared per acre to correct deficiency?
- **Category**: `procedural_recommendation` | **Path**: `slow` (ORIGINAL_SLOW_ROUTING) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 2 (3/4 facts)
- **E1 Hardened**: Score 2 (3/4 facts)
- **R1-C**: Score 2 (4/4 facts, 100.0%) | Finish: `length` | Latency: 9969.9 ms
- **Facts Recovered vs E1**: [3] | **Facts Lost vs E1**: []
- **Answer Excerpt**: `**Foliar spray preparation for one acre (wheat)**  

1. **Standard schedule (early‑season deficiency)**
   - **Zinc source:** Zinc sulphate (21 % Zn) – 1 kg  
   - **Lime source:** Unslaked (quick) li...`

### `Q30`: What is the recommended procedure and emulsion concentration for disinfecting empty grain storage godowns?
- **Category**: `procedural_recommendation` | **Path**: `slow` (ORIGINAL_SLOW_ROUTING) | **Model**: `openai/gpt-oss-120b` (max_tokens=800)
- **Baseline T1**: Score 2 (3/4 facts)
- **E1 Hardened**: Score 2 (3/4 facts)
- **R1-C**: Score 1 (2/4 facts, 50.0%) | Finish: `stop` | Latency: 8719.3 ms
- **Facts Recovered vs E1**: [] | **Facts Lost vs E1**: [2]
- **Answer Excerpt**: `**Procedure to disinfect empty grain‑storage godowns**

1. **Prepare a 0.05 % malathion emulsion**  
   - Mix **100 ml of Malathion 50 EC** (50 % EC) with **10 L of water**.  
   - This gives the requ...`

---

## 5. Complete Query-by-Query Evaluation Table

| ID | Category | T1 Score (Facts) | E1 Score (Facts) | R1-C Score (Facts) | Path | Model | Finish | Latency | Outcome vs E1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Q01 | exact_lexical | 1 (2/4) | 1 (2/4) | **1 (1/4)** | slow | gpt-oss-120b | length | 2671.1ms | DEGRADED |
| Q02 | exact_lexical | 2 (3/3) | 2 (3/3) | **2 (3/3)** | fast | gpt-oss-20b | stop | 804.9ms | UNCHANGED |
| Q03 | exact_lexical | 1 (2/3) | 1 (2/3) | **1 (2/3)** | fast | gpt-oss-20b | stop | 828.1ms | UNCHANGED |
| Q04 | exact_lexical | 0 (0/3) | 0 (0/3) | **0 (0/3)** | fast | gpt-oss-20b | length | 941.9ms | UNCHANGED |
| Q05 | exact_lexical | 2 (3/3) | 2 (3/3) | **2 (3/3)** | fast | gpt-oss-20b | stop | 1025.8ms | UNCHANGED |
| Q06 | exact_lexical | 1 (1/3) | 1 (2/3) | **1 (2/3)** | fast | gpt-oss-20b | stop | 634.3ms | UNCHANGED |
| Q07 | conceptual_paraphrased | 1 (1/4) | 1 (2/4) | **1 (2/4)** | fast | gpt-oss-20b | stop | 804.5ms | UNCHANGED |
| Q08 | conceptual_paraphrased | 1 (1/4) | 1 (1/4) | **1 (1/4)** | fast | gpt-oss-20b | stop | 765.1ms | UNCHANGED |
| Q09 | conceptual_paraphrased | 1 (2/4) | 1 (2/4) | **2 (3/4)** | slow | gpt-oss-120b | stop | 1350.1ms | IMPROVED |
| Q10 | conceptual_paraphrased | 1 (1/4) | 1 (1/4) | **1 (1/4)** | fast | gpt-oss-20b | stop | 471.0ms | UNCHANGED |
| Q11 | conceptual_paraphrased | 1 (1/4) | 0 (0/4) | **0 (0/4)** | fast | gpt-oss-20b | stop | 789.6ms | UNCHANGED |
| Q12 | conceptual_paraphrased | 0 (0/4) | 2 (3/4) | **2 (3/4)** | fast | gpt-oss-20b | stop | 802.0ms | UNCHANGED |
| Q13 | crop_specific | 0 (0/4) | 1 (2/4) | **1 (2/4)** | slow | gpt-oss-120b | stop | 1320.5ms | UNCHANGED |
| Q14 | crop_specific | 1 (1/3) | 1 (2/3) | **1 (2/3)** | fast | gpt-oss-20b | stop | 749.4ms | UNCHANGED |
| Q15 | crop_specific | 1 (2/4) | 1 (2/4) | **1 (2/4)** | fast | gpt-oss-20b | stop | 793.6ms | UNCHANGED |
| Q16 | crop_specific | 0 (0/4) | 1 (2/4) | **1 (2/4)** | slow | gpt-oss-120b | length | 2101.0ms | UNCHANGED |
| Q17 | crop_specific | 0 (0/3) | 0 (0/3) | **0 (0/3)** | fast | gpt-oss-20b | stop | 631.4ms | UNCHANGED |
| Q18 | crop_specific | 2 (4/4) | 2 (4/4) | **2 (4/4)** | slow | gpt-oss-120b | stop | 956.9ms | UNCHANGED |
| Q19 | diagnostic | 2 (3/4) | 1 (1/4) | **1 (1/4)** | fast | gpt-oss-20b | stop | 614.6ms | UNCHANGED |
| Q20 | diagnostic | 0 (0/4) | 1 (2/4) | **1 (2/4)** | slow | gpt-oss-120b | stop | 846.9ms | UNCHANGED |
| Q21 | diagnostic | 2 (3/4) | 1 (2/4) | **1 (2/4)** | fast | gpt-oss-20b | stop | 672.4ms | UNCHANGED |
| Q22 | diagnostic | 2 (4/4) | 2 (3/4) | **2 (3/4)** | slow | gpt-oss-120b | stop | 1181.0ms | UNCHANGED |
| Q23 | diagnostic | 2 (4/4) | 2 (4/4) | **2 (4/4)** | slow | gpt-oss-120b | stop | 1198.3ms | UNCHANGED |
| Q24 | diagnostic | 1 (2/4) | 1 (2/4) | **1 (2/4)** | slow | gpt-oss-120b | stop | 1039.1ms | UNCHANGED |
| Q25 | procedural_recommendation | 2 (4/4) | 2 (4/4) | **2 (4/4)** | slow | gpt-oss-120b | stop | 1382.0ms | UNCHANGED |
| Q26 | procedural_recommendation | 0 (0/4) | 1 (2/4) | **2 (3/4)** | slow | gpt-oss-120b | stop | 6670.6ms | IMPROVED |
| Q27 | procedural_recommendation | 1 (2/5) | 2 (5/5) | **2 (5/5)** | slow | gpt-oss-120b | stop | 9387.9ms | UNCHANGED |
| Q28 | procedural_recommendation | 0 (0/4) | 0 (0/4) | **0 (0/4)** | slow | gpt-oss-120b | stop | 9953.3ms | UNCHANGED |
| Q29 | procedural_recommendation | 2 (3/4) | 2 (3/4) | **2 (4/4)** | slow | gpt-oss-120b | length | 9969.9ms | IMPROVED |
| Q30 | procedural_recommendation | 2 (3/4) | 2 (3/4) | **1 (2/4)** | slow | gpt-oss-120b | stop | 8719.3ms | DEGRADED |
