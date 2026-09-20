# Controlled Generation Experiment: R1 Routing & Token Budget Realignment Report

## 1. Executive Summary & Benchmark Overview

This controlled evaluation examines the isolated and interactive effects of **fast-path token budget expansion (R1-A)** and **complexity-aware routing (R1-B)** on the Agronomic RAG generation pipeline.

- **Benchmark Baseline**: T1 (Domain-Conditioned Selective Temporal Decay, Strict Acc: 33.33%, Fact Cov: 45.61%).
- **Prompt Predecessor**: E1 (Prompt Instruction Hardening, Strict Acc: 33.33%, Fact Cov: 56.14%).
- **Variant R1-A**: Fast-path token budget raised from 250 to 450 tokens (`openai/gpt-oss-20b`); slow path unchanged (700 tokens, `openai/gpt-oss-120b`). Original E1 routing preserved.
- **Variant R1-B**: Complexity-aware routing policy routing procedural recommendations (`procedural_recommendation`) and multi-fact procedural queries (`Q09`, `Q13`) to 120B slow path. Token budgets kept at baseline (fast=250, slow=700).
- **Grading**: Validated V2 Deterministic Rubric Grader (114 key facts, 7 safety rules).

> [!NOTE]
> **Reproducibility Note**: Raw benchmark JSON (`r1_benchmark_results.json`) is gitignored. Embedded Markdown results reflect historical trial records and do not guarantee independent clean-checkout reproducibility.

---

## 2. Consolidated Comparison Table: T1 vs E1 vs R1-A vs R1-B

| Metric | Baseline T1 | E1 Hardened | R1-A (Budget=450) | R1-B (Complexity Routing) |
| :--- | :---: | :---: | :---: | :---: |
| **Strict Accuracy (Score 2)** | 33.33% (10/30) | 33.33% (10/30) | **46.67% (14/30)** | **36.67% (11/30)** |
| **Partial Answer Rate (Score 1)** | 40.0% (12/30) | 53.33% (16/30) | 43.33% (13/30) | 46.67% (14/30) |
| **Incorrect / Failure Rate (Score 0)** | 26.67% (8/30) | 13.33% (4/30) | 10.0% (3/30) | 16.67% (5/30) |
| **Fact Coverage (%)** | 45.61% | 56.14% | **62.28%** | **56.14%** |
| **Total Facts Satisfied** | 52 / 114 | 64 / 114 | **71 / 114** | **64 / 114** |
| **Safety Adherence (%)** | 100.0% | 100.0% | 100.0% | 100.0% |
| **Unsupported Claims** | 0 | 0 | 0 | 0 |
| **Mean Latency (ms)** | 7094.1 ms | 1834.7 ms | 2691.7 ms | 3010.8 ms |
| **Median Latency (ms)** | 4065.1 ms | 1075.2 ms | 1090.0 ms | 1704.8 ms |
| **Mean Answer Length (words)** | ~140.0 | 73.4 | 70.8 | 81.6 |
| **Median Answer Length (words)** | ~120.0 | 59.0 | 61.0 | 64.0 |
| **Finish Reason Distribution** | - | - | {"length": 3, "stop": 27} | {"length": 5, "stop": 25} |
| **Token-Starved / Truncated Answers** | 0 | 0 | 3 | 5 |

---

## 3. Transition Breakdown vs. E1

### Variant R1-A (Token Budget 450):
- **Improved vs E1**: 5 (Q01, Q04, Q16, Q21, Q26)
- **Unchanged vs E1**: 24
- **Degraded vs E1**: 1 (Q30)
- **Converted Score 0 → 1/2**: Q04
- **Converted Score 1 → 2**: Q01, Q16, Q21, Q26
- **Score Regressions**: Q30
- **Generation Failures (Score 0)**: Q11, Q17, Q28

### Variant R1-B (Complexity Routing):
- **Improved vs E1**: 5 (Q09, Q13, Q16, Q22, Q26)
- **Unchanged vs E1**: 20
- **Degraded vs E1**: 5 (Q05, Q08, Q12, Q14, Q30)
- **Converted Score 0 → 1/2**: None
- **Converted Score 1 → 2**: Q09, Q13, Q16, Q26
- **Score Regressions**: Q05, Q08, Q12, Q30
- **Generation Failures (Score 0)**: Q04, Q08, Q11, Q17, Q28

---

## 4. Deep-Dive Audit on Key Specific Queries (Q09, Q13, Q26, Q29)

### `Q09`: How do green manure crops like cowpea, dhaincha, or sunnhemp contribute to organic nitrogen enrichment in rice?
- **Category**: `conceptual_paraphrased` | **Total Facts**: 4
- **Baseline T1**: Score 1 (2/4 facts)
- **E1 Hardened**: Score 1 (2/4 facts)
- **R1-A**: Score 1 (2/4 facts) | Path: `fast` | Finish: `stop` | Latency: 913.2 ms
- **R1-B**: Score 2 (3/4 facts) | Path: `slow` (MULTI_ENTITY_COMPARATIVE_PROCEDURE) | Finish: `stop` | Latency: 1479.1 ms
- **R1-A Answer Excerpt**: `- Grow cowpea, dhaincha, or sunnhemp at 20 kg / acre; let the crop reach ~50 days old before rice transplanting.
- Incorporate the green manure crop into the soil (e.g., by tilla...`
- **R1-B Answer Excerpt**: `**How cowpea, dhaincha or sunn‑hemp enrich rice with organic nitrogen**

- **Green‑manure principle** – The growing crop fixes atmospheric nitrogen (legumes) and builds nitrogen‑ri...`

### `Q13`: What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?
- **Category**: `crop_specific` | **Total Facts**: 4
- **Baseline T1**: Score 0 (0/4 facts)
- **E1 Hardened**: Score 1 (2/4 facts)
- **R1-A**: Score 1 (2/4 facts) | Path: `fast` | Finish: `stop` | Latency: 4766.0 ms
- **R1-B**: Score 2 (3/4 facts) | Path: `slow` (MULTI_ENTITY_COMPARATIVE_PROCEDURE) | Finish: `stop` | Latency: 1255.2 ms
- **R1-A Answer Excerpt**: `- **Tomato**
  - Normal variety: 400 – 500 g per hectare.
  - Hybrid variety: 125 – 150 g per hectare.

- **Chili**
  - Normal variety: 500 g per hectare.
  - Hybrid vari...`
- **R1-B Answer Excerpt**: `**Seed‑rate (per hectare) for normal vs. hybrid varieties**

- **Tomato (Lycopersicon esculentum)**
  - Normal (open‑pollinated) – 400 – 500 g / ha
  - Hybrid – 125 – 150 g / ha...`

### `Q26`: What are the duration and temperature precautions required before vermicompost is fully mature and ready?
- **Category**: `procedural_recommendation` | **Total Facts**: 4
- **Baseline T1**: Score 0 (0/4 facts)
- **E1 Hardened**: Score 1 (2/4 facts)
- **R1-A**: Score 2 (4/4 facts) | Path: `fast` | Finish: `stop` | Latency: 753.0 ms
- **R1-B**: Score 2 (4/4 facts) | Path: `slow` (CATEGORY_PROCEDURAL_RECOMMENDATION) | Finish: `stop` | Latency: 5668.4 ms
- **R1-A Answer Excerpt**: `- **Maturity period**: 60–70 days of composting (or 80–90 days if terminating early).
- **Temperature precaution**:
  - Do **not** use fresh animal dung, as it generates high t...`
- **R1-B Answer Excerpt**: `**Duration to maturity**
- Typical vermicompost becomes ready in **60‑70 days** (general practice).
- When using leaf litter or cashew‑apple residue, maturity may take up to **95...`

### `Q29`: How should foliar spray solution of zinc sulphate and unslaked lime be prepared per acre to correct deficiency?
- **Category**: `procedural_recommendation` | **Total Facts**: 4
- **Baseline T1**: Score 2 (3/4 facts)
- **E1 Hardened**: Score 2 (3/4 facts)
- **R1-A**: Score 2 (3/4 facts) | Path: `slow` | Finish: `length` | Latency: 10253.7 ms
- **R1-B**: Score 2 (3/4 facts) | Path: `slow` (CATEGORY_PROCEDURAL_RECOMMENDATION) | Finish: `length` | Latency: 9167.2 ms
- **R1-A Answer Excerpt**: `**Foliar spray recipe for one acre (wheat) to correct Zn deficiency**

1. **Materials per acre**
   - Zinc sulphate (21 % Zn, any grade – heptahydrate or monohydrate) – **1 kg**...`
- **R1-B Answer Excerpt**: `**Foliar spray preparation for one acre (wheat)**

1. **Standard (early‑season) recipe** – 0.5 % ZnSO₄ solution
   - Zinc sulphate (21 % Zn) – **1 kg**
   - Unslaked lime – *...`

---

## 5. Query-by-Query Comparison Table

| ID | Cat | T1 Sc (Facts) | E1 Sc (Facts) | R1-A Sc (Facts) | R1-B Sc (Facts) | R1-A Path | R1-B Path | R1-A Lat | R1-B Lat |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Q01 | exact_lexical | 1 (2/4) | 1 (2/4) | 2 (3/4) | 1 (2/4) | slow | slow | 2356.7ms | 1643.0ms |
| Q02 | exact_lexical | 2 (3/3) | 2 (3/3) | 2 (3/3) | 2 (3/3) | fast | fast | 692.0ms | 320.3ms |
| Q03 | exact_lexical | 1 (2/3) | 1 (2/3) | 1 (2/3) | 1 (2/3) | fast | fast | 736.3ms | 699.7ms |
| Q04 | exact_lexical | 0 (0/3) | 0 (0/3) | 2 (3/3) | 0 (0/3) | fast | fast | 1196.3ms | 709.6ms |
| Q05 | exact_lexical | 2 (3/3) | 2 (3/3) | 2 (3/3) | 1 (1/3) | fast | fast | 795.8ms | 683.2ms |
| Q06 | exact_lexical | 1 (1/3) | 1 (2/3) | 1 (2/3) | 1 (2/3) | fast | fast | 539.1ms | 572.0ms |
| Q07 | conceptual_paraphrased | 1 (1/4) | 1 (2/4) | 1 (2/4) | 1 (2/4) | fast | fast | 702.4ms | 668.1ms |
| Q08 | conceptual_paraphrased | 1 (1/4) | 1 (1/4) | 1 (1/4) | 0 (0/4) | fast | fast | 663.5ms | 721.6ms |
| Q09 | conceptual_paraphrased | 1 (2/4) | 1 (2/4) | 1 (2/4) | 2 (3/4) | fast | slow | 913.2ms | 1479.1ms |
| Q10 | conceptual_paraphrased | 1 (1/4) | 1 (1/4) | 1 (1/4) | 1 (1/4) | fast | fast | 861.3ms | 3762.7ms |
| Q11 | conceptual_paraphrased | 1 (1/4) | 0 (0/4) | 0 (0/4) | 0 (0/4) | fast | fast | 5830.8ms | 2748.8ms |
| Q12 | conceptual_paraphrased | 0 (0/4) | 2 (3/4) | 2 (3/4) | 1 (2/4) | fast | fast | 5805.9ms | 2740.1ms |
| Q13 | crop_specific | 0 (0/4) | 1 (2/4) | 1 (2/4) | 2 (3/4) | fast | slow | 4766.0ms | 1255.2ms |
| Q14 | crop_specific | 1 (1/3) | 1 (2/3) | 1 (2/3) | 1 (1/3) | fast | fast | 2802.0ms | 4186.6ms |
| Q15 | crop_specific | 1 (2/4) | 1 (2/4) | 1 (2/4) | 1 (2/4) | fast | fast | 4715.9ms | 2720.5ms |
| Q16 | crop_specific | 0 (0/4) | 1 (2/4) | 2 (3/4) | 2 (3/4) | slow | slow | 2126.0ms | 1935.4ms |
| Q17 | crop_specific | 0 (0/3) | 0 (0/3) | 0 (0/3) | 0 (0/3) | fast | fast | 740.6ms | 1766.5ms |
| Q18 | crop_specific | 2 (4/4) | 2 (4/4) | 2 (4/4) | 2 (4/4) | slow | slow | 1545.2ms | 1146.5ms |
| Q19 | diagnostic | 2 (3/4) | 1 (1/4) | 1 (1/4) | 1 (1/4) | fast | fast | 1747.0ms | 1876.9ms |
| Q20 | diagnostic | 0 (0/4) | 1 (2/4) | 1 (2/4) | 1 (2/4) | slow | slow | 722.0ms | 737.8ms |
| Q21 | diagnostic | 2 (3/4) | 1 (2/4) | 2 (3/4) | 1 (2/4) | fast | fast | 587.7ms | 4528.4ms |
| Q22 | diagnostic | 2 (4/4) | 2 (3/4) | 2 (3/4) | 2 (4/4) | slow | slow | 1145.7ms | 1262.9ms |
| Q23 | diagnostic | 2 (4/4) | 2 (4/4) | 2 (4/4) | 2 (4/4) | slow | slow | 822.5ms | 1187.4ms |
| Q24 | diagnostic | 1 (2/4) | 1 (2/4) | 1 (2/4) | 1 (2/4) | slow | slow | 1034.2ms | 6104.5ms |
| Q25 | procedural_recommendation | 2 (4/4) | 2 (4/4) | 2 (4/4) | 2 (4/4) | fast | slow | 869.2ms | 1556.6ms |
| Q26 | procedural_recommendation | 0 (0/4) | 1 (2/4) | 2 (4/4) | 2 (4/4) | fast | slow | 753.0ms | 5668.4ms |
| Q27 | procedural_recommendation | 1 (2/5) | 2 (5/5) | 2 (5/5) | 2 (5/5) | slow | slow | 8155.5ms | 12084.4ms |
| Q28 | procedural_recommendation | 0 (0/4) | 0 (0/4) | 0 (0/4) | 0 (0/4) | slow | slow | 9344.2ms | 9024.3ms |
| Q29 | procedural_recommendation | 2 (3/4) | 2 (3/4) | 2 (3/4) | 2 (3/4) | slow | slow | 10253.7ms | 9167.2ms |
| Q30 | procedural_recommendation | 2 (3/4) | 2 (3/4) | 1 (2/4) | 1 (2/4) | slow | slow | 7526.8ms | 7365.6ms |
