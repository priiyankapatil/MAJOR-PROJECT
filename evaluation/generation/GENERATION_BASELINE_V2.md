# Generation Benchmark Baseline V2: Validated Evaluator Normalization

**Date**: 2026-09-08  
**Scope**: 30-Query Generation Benchmark Regrading & Validation  
**Evaluator Version**: V2 (Deterministic Text & Pattern Normalization)  
**Execution Type**: Pure Offline Regrading (0 Generation Calls, 0 API Calls)  
**Baseline Artifacts Evaluated**: `generation_benchmark_results.json`, `generation_benchmark_dataset.json`

---

## 1. Executive Summary & Metric Evolution

Following the forensic failure analysis and grader validation audit, strict deterministic normalizations were integrated into the generation benchmark evaluator (`run_generation_benchmark.py`). These adjustments resolve typographical and formatting edge cases (Unicode non-breaking hyphens, standard SI metric spacing, and standardized pest acronyms) without relaxing semantic thresholds or introducing LLM judge variance.

| Metric | Original Baseline (V1) | Validated Baseline (V2) | Net Delta |
| :--- | :---: | :---: | :---: |
| **Strict Fully-Correct Accuracy (Score = 2)** | **26.7%** (8 / 30) | **30.0%** (9 / 30) | **+3.3%** (+1 query) |
| **Partial Credit Rate (Score = 1)** | **36.7%** (11 / 30) | **33.3%** (10 / 30) | **-3.3%** (-1 query) |
| **Incorrect / Refusal Rate (Score = 0)** | **36.7%** (11 / 30) | **36.7%** (11 / 30) | **0.0%** (unchanged) |
| **Overall Expected-Fact Coverage** | **37.72%** (43 / 114) | **40.35%** (46 / 114) | **+2.63%** (+3 facts) |
| **Safety-Constraint Adherence** | **100.0%** (7 / 7 active) | **100.0%** (7 / 7 active) | **0.0%** (unchanged) |
| **API / Generation Model Calls** | 0 | 0 | **Pure Offline Regrade** |

---

## 2. Exact Queries Whose Scores Changed

Across all 30 queries, exactly **one** query experienced a score change:

| Query ID | Category | Original Score (V1) | Validated Score (V2) | Fact Coverage Delta | Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Q29`** | `procedural_recommendation` | **1 (Partial)** | **2 (Fully Correct)** | 2/4 (50.0%) $\rightarrow$ **3/4 (75.0%)** | Recognition of `200 L` as standard SI symbol for `200 litres` satisfies Fact 1. Meeting the $\ge 70.0\%$ threshold with 0 safety violations elevates `Q29` to full credit. |

*Note on Other Flagged Queries*:
- **`Q18`**: Fact satisfaction increased from 3/4 to 4/4 (100.0%). Because 3/4 (75.0%) had already exceeded the 70.0% threshold in V1, its score was already **Score = 2** and remains **Score = 2**.
- **`Q24`**: Fact satisfaction increased from 1/4 (25.0%) to 2/4 (50.0%). Because 50.0% remains below the 70.0% threshold, its score remains **Score = 1 (Partial)**.
- **`Q25`**: Remains at 2/4 (50.0%) and **Score = 1**. Fact 1 requires both leveled ground and length variability; because length variability was omitted, it remains unsatisfied (True Negative).

---

## 3. Exact Facts Whose Matching Status Changed

Exactly **three** false-negative facts were corrected:

### 1. `Q18` Fact 0 (Rice Zinc Biofortification Cultivars)
- **Expected Fact**: `Biofortification rice cultivars: CO51, CO47, ADT 47, and ADT 37`
- **Original Status (V1)**: `False` ("Not matched")
- **Validated Status (V2)**: `True` ("Pattern matched")
- **Generated Answer**: `Grow the zinc‑efficient rice cultivars **CO‑51, CO‑47, ADT‑47 and ADT‑37**...`
- **Normalization Rule**: Normalizing Unicode non-breaking hyphens (`\u2011`) to ASCII hyphens (`-`) allows `r"co[\s-]?(51|47)"` and `r"adt[\s-]?(47|37)"` to match the exact cultivar designations.

### 2. `Q24` Fact 0 (Cashew Stem and Root Borer Identification)
- **Expected Fact**: `Pest identification: Cashew Stem and Root Borer (Plocaederus ferrugineus)`
- **Original Status (V1)**: `False` ("Not matched")
- **Validated Status (V2)**: `True` ("Pattern matched")
- **Generated Answer**: `**🔍 DIAGNOSIS:** **Cashew Stem‑and‑Root Borer (CSRB)** – confidence ≈ 92 %`
- **Normalization Rule**: Normalizing hyphens and recognizing the standard entomological acronym `CSRB` alongside `Cashew Stem-and-Root Borer`.

### 3. `Q29` Fact 1 (Foliar Spray Water Volume)
- **Expected Fact**: `Water volume: 200 litres of water (yielding a 0.5% zinc sulphate solution)`
- **Original Status (V1)**: `False` ("Not matched")
- **Validated Status (V2)**: `True` ("Pattern matched")
- **Generated Answer**: `| Water | **200 L** (≈ 200 kg) |` and `pour 200 L of lukewarm water into the drum.`
- **Normalization Rule**: Standard metric unit abbreviation `r"\b200\s*(l|litres?|liters?|lit)\b"` recognizes `200 L` with whitespace as equivalent to `200 litres`.

---

## 4. Normalization Rules Implemented in `run_generation_benchmark.py`

1. **`normalize_text_for_grading(text)`**:
   - Maps Unicode hyphens and dashes (`\u2010`, `\u2011`, `\u2012`, `\u2013`, `\u2014`, `\u2015`) to standard ASCII hyphen `-`.
   - Maps Unicode non-breaking spaces (`\u00a0`, `\u202f`) to standard space `' '`.
2. **Standard Metric SI Unit Matching**:
   - Calibrated volume patterns to accept standard SI symbols with optional boundary whitespace (e.g. `200 L`, `200l`, `200 litres`, `200 liters`).
3. **Established Acronym Matching**:
   - Permits official institutional/entomological acronyms (e.g. `CSRB` for Cashew Stem and Root Borer) where domain context is unambiguous.
4. **Dual Evaluator Preservation**:
   - `FACT_PATTERNS_V1` and `FACT_PATTERNS_V2` are both preserved in `run_generation_benchmark.py`.
   - Running `python run_generation_benchmark.py --regrade-v1` reproduces the original 8/30 (26.7%) score.
   - Running `python run_generation_benchmark.py --regrade` executes the validated V2 evaluation yielding 9/30 (30.0%).

---

## 5. Verification and Regression Testing

The unit test suite [`test_grader_normalization.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/test_grader_normalization.py) was executed:
- `test_unicode_hyphen_normalization`: **PASS**
- `test_q18_cultivars_matching`: **PASS**
- `test_q24_stem_and_root_borer_matching`: **PASS**
- `test_q29_si_unit_matching`: **PASS**
- `test_q25_true_negative_preservation`: **PASS** (confirms `Q25` length-variability omission is not falsely credited)
- `test_q27_true_negative_preservation`: **PASS** (confirms `Q27` afternoon watering omission is not falsely credited)

Full regression over all 30 queries confirmed that **zero unintended facts changed**, and **zero scores changed** other than `Q29`.

---

## 6. Confirmation of Production Code & Ground Truth Integrity

1. **Zero LLM / API Calls**: This regrading was performed purely offline on saved prediction strings in `generation_benchmark_results.json`.
2. **Production Code Untouched**: `step1` through `step6`, `config.py`, prompts, routing, and models were not modified.
3. **Ground-Truth Dataset Untouched**: `generation_benchmark_dataset.json` was not modified.
4. **Benchmark Results JSON Unmodified by Hand**: `generation_benchmark_results.json` remains the exact raw artifact produced during the live benchmark run.
