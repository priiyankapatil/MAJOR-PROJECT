# Experiment SB2: Multi-Crop Comparative Balance & Retrieval Precision

**Date**: 2026-09-19T04:09:51Z  
**Scope**: Controlled Retrieval Evaluation across 30 Canonical Queries and 5 Vernacular Queries  
**Conditions Evaluated**: Raw Baseline vs. Current Production Bridge vs. Guarded SB1 vs. Guarded SB2  
**Models**: `all-MiniLM-L6-v2` (Dense) + BM25 (Sparse) + `BAAI/bge-reranker-base` (Cross-Encoder)  

> [!NOTE]
> **Evaluation Status & Reproducibility Context**:
> - **Production Readiness Status**: Any statements or assumptions regarding SB2 being "ready for production" represent initial historical trial findings. This status is superseded by the committed production review [sb2_production_integration_review.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_production_integration_review.md), which ruled **NEEDS TARGETED FIXES BEFORE INTEGRATION TEST** due to plural inflection blindness (*tomatoes*, *chillies*) and risk of chemical table entity bleeding. Full hardening was subsequently evaluated in [sb2_hardened_integration_review.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_hardened_integration_review.md).
> - **Reproducibility Note**: Raw benchmark JSON (`sb2_benchmark_results.json`) is gitignored. Embedded Markdown results reflect historical trial records and do not guarantee independent clean-checkout reproducibility.

---

## 1. Executive Summary

Experiment SB2 tested whether adding a **Multi-Crop Balance Guardrail** to the semantic bridge eliminates the single regression observed in SB1 (`Q13`, which dropped from #4 to #6) while fully preserving the retrieval breakthroughs on `Q28` (Rank #1) and `Q17` (Rank #5).

### Key Empirical Breakthroughs
1. **`Q13` Full Recovery**: In SB1, asymmetric botanical Latin binomial injection on tomato pushed multi-crop seed rate table `chunk_007442` down to Rank #6. In SB2, suppressing asymmetric binomial injection on multi-crop lists restores `chunk_007442` to **Rank #2** (Recall@5 = 1.00, NDCG@5 = 0.6309).
2. **`Q17` Gain Preserved**: Single-crop query `Q17` maintains its botanical enrichment `tomato (Solanum lycopersicum)`, holding **Rank #5** (Recall@5 = 1.00, inside Top-5).
3. **`Q28` Breakthrough Preserved**: Single-crop query `Q28` maintains specificity suppression on generic `"organic manures"` and clean deduplication on `coconut (Cocos nucifera)`, holding **Rank #1** (Recall@5 = 1.00, NDCG@5 = 1.0000).
4. **Overall Retrieval Ceiling**: Macro Recall@5 achieves **0.9000** (90.0%), outperforming the Current Production Bridge by **+0.0667** and matching the highest possible ceiling with **0 regressions across all 30 benchmark queries**.

---

## 2. Head-to-Head Macro Benchmark Metrics (30 Queries)

| Metric | Condition 1 (Baseline) | Condition 2 (Current Bridge) | Condition 3 (SB1) | Condition 4 (Guarded SB2) | Δ (SB2 vs Current) | Δ (SB2 vs SB1) | Δ (SB2 vs Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Recall@1** | 0.7000 | 0.6667 | 0.7000 | **0.7000** | +0.0333 | +0.0000 | +0.0000 |
| **Recall@3** | 0.8167 | 0.7500 | 0.7833 | **0.8167** | +0.0667 | +0.0334 | +0.0000 |
| **Recall@5** | 0.8667 | 0.8333 | 0.8667 | **0.9000** | +0.0667 | +0.0333 | +0.0333 |
| **Recall@10** | 0.9333 | 0.9333 | 0.9333 | **0.9333** | +0.0000 | +0.0000 | +0.0000 |
| **Hit@5** | 0.8667 | 0.8333 | 0.8667 | **0.9000** | +0.0667 | +0.0333 | +0.0333 |
| **NDCG@5** | 0.8154 | 0.7710 | 0.8073 | **0.8283** | +0.0573 | +0.0210 | +0.0129 |
| **NDCG@10** | 0.8373 | 0.8026 | 0.8292 | **0.8383** | +0.0357 | +0.0091 | +0.0010 |
| **MRR** | 0.8176 | 0.7737 | 0.8076 | **0.8187** | +0.0450 | +0.0111 | +0.0011 |
| **Mean Latency** | 142.05 ms | 151.51 ms | 138.11 ms | **135.09 ms** | -16.42 ms | -3.02 ms | -6.96 ms |

---

## 3. Targeted Triad Verification: Q13, Q17, Q28

### Query `Q13`: Seed Rate for Normal vs Hybrid (Tomato, Chilli, Brinjal)
- **Raw Query**: *"What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?"*
- **Current Bridge Enriched**: *"What is the seed rate requirement for normal vs hybrid varieties in tomato (Tomato, Solanum lycopersicum), chilli, and brinjal?"*
- **SB1 Enriched**: *"What is the seed rate requirement for normal vs hybrid varieties in tomato (Solanum lycopersicum), chilli, and brinjal?"*
- **SB2 Enriched**: *"What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?"*
- **Rank Progression**: Baseline **#2** → Current **#4** → SB1 **#6** (Regression) → SB2 **#2** (Fully Restored)
- **Recall@5 Progression**: Baseline `1.0` → Current `1.0` → SB1 `0.0` → SB2 **`1.0`**
- **NDCG@5 Progression**: Baseline `0.6309` → Current `0.4307` → SB1 `0.0` → SB2 **`0.6309`**

### Query `Q17`: Bacterial Wilt Resistant Tomato Varieties (KAU)
- **Raw Query**: *"Which bacterial wilt resistant varieties of tomato are recommended by Kerala Agricultural University?"*
- **Current Bridge Enriched**: *"Which bacterial wilt resistant varieties of tomato (Tomato, Solanum lycopersicum) are recommended by Kerala Agricultural University?"*
- **SB1 Enriched**: *"Which bacterial wilt resistant varieties of tomato (Solanum lycopersicum) are recommended by Kerala Agricultural University?"*
- **SB2 Enriched**: *"Which bacterial wilt resistant varieties of tomato (Solanum lycopersicum) are recommended by Kerala Agricultural University?"*
- **Rank Progression**: Baseline **#6** → Current **#6** → SB1 **#5** → SB2 **#5** (Top-5 Preserved)
- **Recall@5 Progression**: Baseline `0.0` → Current `0.0` → SB1 `1.0` → SB2 **`1.0`**

### Query `Q28`: Coconut Palms Organic Manure & Coir Pith
- **Raw Query**: *"What organic manures and coir pith application practices are recommended for maintaining coconut palms?"*
- **Current Bridge Enriched**: *"What organic manures (Farmyard Manure, Farm Yard Manure (FYM)) and coir pith application practices are recommended for maintaining coconut (Coconut, Cocos nucifera) palms?"*
- **SB1 Enriched**: *"What organic manures and coir pith application practices are recommended for maintaining coconut (Cocos nucifera) palms?"*
- **SB2 Enriched**: *"What organic manures and coir pith application practices are recommended for maintaining coconut (Cocos nucifera) palms?"*
- **Rank Progression**: Baseline **#1** → Current **#10** (Drift Failure) → SB1 **#1** → SB2 **#1** (Top-1 Preserved)
- **Recall@5 Progression**: Baseline `1.0` → Current `0.0` → SB1 `1.0` → SB2 **`1.0`**

---

## 4. Vernacular Smoke Test Results Across Bridge Variants

| ID | Vernacular Query | Current Bridge | SB1 Guarded | SB2 Guarded | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `V01` | *dhan mein khaira rog ka lakshan aur upchar kya hai?* | *dhan (Rice / Paddy, Oryza sativa) mein khaira rog ka lakshan aur upchar kya hai?* | *dhan (Rice / Paddy, Oryza sativa) mein khaira rog ka lakshan aur upchar kya hai?* | *dhan (Rice / Paddy, Oryza sativa) mein khaira rog ka lakshan aur upchar kya hai?* | ✓ PASS |
| `V02` | *arhar ki fasal mein sundli ke niyantran ke liye kaun si dawai use karein?* | *arhar (Pigeon Pea, Cajanus cajan) ki fasal mein sundli ke niyantran ke liye kaun si dawai use karein?* | *arhar (Pigeon Pea, Cajanus cajan) ki fasal mein sundli ke niyantran ke liye kaun si dawai use karein?* | *arhar (Pigeon Pea, Cajanus cajan) ki fasal mein sundli ke niyantran ke liye kaun si dawai use karein?* | ✓ PASS |
| `V03` | *chana mein ukhta rog lagne par kya karein?* | *chana (Chickpea, Cicer arietinum) mein ukhta rog lagne par kya karein?* | *chana (Chickpea, Cicer arietinum) mein ukhta rog lagne par kya karein?* | *chana (Chickpea, Cicer arietinum) mein ukhta rog lagne par kya karein?* | ✓ PASS |
| `V04` | *tamatar mein safed makhi ka ilaj batayein* | *tamatar (Tomato, Solanum lycopersicum) mein safed makhi (Whitefly, Bemisia tabaci) ka ilaj batayein* | *tamatar (Tomato, Solanum lycopersicum) mein safed makhi (Whitefly, Bemisia tabaci) ka ilaj batayein* | *tamatar (Tomato, Solanum lycopersicum) mein safed makhi (Whitefly, Bemisia tabaci) ka ilaj batayein* | ✓ PASS |
| `V05` | *ganna ki kheti mein gobar khad kitni matra mein daalna chahiye?* | *ganna (Sugarcane, Saccharum officinarum) ki kheti mein gobar (Farmyard Manure, Farm Yard Manure (FYM)) khad kitni matra mein daalna chahiye?* | *ganna (Sugarcane, Saccharum officinarum) ki kheti mein gobar (Farm Yard Manure (FYM)) khad kitni matra mein daalna chahiye?* | *ganna (Sugarcane, Saccharum officinarum) ki kheti mein gobar (Farm Yard Manure (FYM)) khad kitni matra mein daalna chahiye?* | ✓ PASS |

---

## 5. Complete 30-Query Retrieval Breakdown (Condition 1 vs 2 vs 3 vs 4)

| QID | Category | Baseline | Current | SB1 | SB2 | Recall@5 (Base/Curr/SB1/SB2) | Status (SB2 vs Current) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Q01` | `exact_lexical` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q02` | `exact_lexical` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q03` | `exact_lexical` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q04` | `exact_lexical` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q05` | `exact_lexical` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q06` | `exact_lexical` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q07` | `conceptual_paraphrased` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q08` | `conceptual_paraphrased` | #9 | #9 | #9 | **#9** | 0.00 / 0.00 / 0.00 / **0.00** | STABLE |
| `Q09` | `conceptual_paraphrased` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q10` | `conceptual_paraphrased` | >10 | >10 | >10 | **>10** | 0.00 / 0.00 / 0.00 / **0.00** | STABLE |
| `Q11` | `conceptual_paraphrased` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q12` | `conceptual_paraphrased` | >10 | >10 | >10 | **>10** | 0.00 / 0.00 / 0.00 / **0.00** | STABLE |
| `Q13` | `crop_specific` | #2 | #4 | #6 | **#2** | 1.00 / 1.00 / 0.00 / **1.00** | 🟢 IMPROVED |
| `Q14` | `crop_specific` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q15` | `crop_specific` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q16` | `crop_specific` | #2 | #3 | #2 | **#2** | 1.00 / 1.00 / 1.00 / **1.00** | 🟢 IMPROVED |
| `Q17` | `crop_specific` | #6 | #6 | #5 | **#5** | 0.00 / 0.00 / 1.00 / **1.00** | 🟢 IMPROVED |
| `Q18` | `crop_specific` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q19` | `diagnostic` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q20` | `diagnostic` | #4 | #4 | #4 | **#4** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q21` | `diagnostic` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q22` | `diagnostic` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q23` | `diagnostic` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q24` | `diagnostic` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q25` | `procedural_recommendation` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q26` | `procedural_recommendation` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q27` | `procedural_recommendation` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q28` | `procedural_recommendation` | #1 | #10 | #1 | **#1** | 1.00 / 0.00 / 1.00 / **1.00** | 🟢 IMPROVED |
| `Q29` | `procedural_recommendation` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |
| `Q30` | `procedural_recommendation` | #1 | #1 | #1 | **#1** | 1.00 / 1.00 / 1.00 / **1.00** | STABLE |

