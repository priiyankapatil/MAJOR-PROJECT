# Score & Grader Validation Audit: Deterministic Matching Verification

**Date**: 2026-09-08<br>
**Scope**: Grader Fidelity and Normalization Verification across 30 Benchmark Queries<br>
**Evaluation Assets**: `generation_benchmark_results.json`, `generation_benchmark_dataset.json`, `verify_generation_grader.py`<br>
**Constraint Compliance**: Zero production code changes; zero edits to benchmark dataset or results JSON; no LLM judge used.

---

## 1. Executive Summary

During the generation failure attribution audit, four specific queries (`Q18`, `Q24`, `Q25`, `Q29`) were flagged where strict deterministic regex rules potentially marked valid answers as "Not matched".

A comprehensive grader validation audit was conducted across all 30 queries to:
1. Forensically verify whether the generated answers semantically and factually expressed the expected key facts.
2. Isolate the exact regex/string-matching mechanisms that caused discrepancies.
3. Test conservative, transparent text normalization rules (Unicode hyphens, unit spacing, and standard agronomic acronyms) without relaxing semantic rigor.

### Key Audit Findings
- **Confirmed Grader False Negatives**: **3 facts** across 3 queries (`Q18` Fact 0, `Q24` Fact 0, `Q29` Fact 1).
- **Compound Fact Omission Confirmed**: **1 query** (`Q25` Fact 1) was confirmed as a **True Negative** for the compound fact.
- **Score Impact**: Exactly **1 query** (`Q29`) changed its overall correctness score from **1 (Partial)** to **2 (Fully Correct)**.
- **Original Strict Accuracy**: **8 / 30 (26.7%)**
- **Validated Strict Accuracy**: **9 / 30 (30.0%)**
- **Original Fact Coverage**: **43 / 114 (37.72%)**
- **Validated Fact Coverage**: **46 / 114 (40.35%)**
- **Safety Scoring**: **Unchanged** (100.0% adherence, 7/7 active constraints satisfied, 0 violations).

---

## 2. Detailed Forensic Audit of Flagged Queries

### 2.1. Query `Q18` (Crop-Specific: Rice Zinc Biofortification)

#### 1. Exact Generated Answer Excerpt
```markdown
**⭐ BEST OPTION:**
Grow the zinc‑efficient rice cultivars **CO‑51, CO‑47, ADT‑47 and ADT‑37**. These varieties have been identified by TNAU as responding well to zinc bio‑fortification and will give the highest grain Zn enrichment when the recommended Zn management is followed.
```

#### 2. Expected Key Facts
- **Fact 0**: `Biofortification rice cultivars: CO51, CO47, ADT 47, and ADT 37`
- **Fact 1**: `Basal soil application: 50 kg ZnSO₄ (per hectare)` [Satisfied: True]
- **Fact 2**: `Foliar spray: 0.50% ZnSO₄ applied thrice` [Satisfied: True]
- **Fact 3**: `Foliar spray stages: 50% flowering, milky, and dough stages` [Satisfied: True]

#### 3. Deterministic Matching Rule in Grader
```python
"Q18": [
    [r"co51", r"co47", r"adt\s*47", r"adt\s*37", r"cultivar"], # requires >= 2 matches
    ...
]
```

#### 4. Discrepancy Mechanism
Groq's LLM outputs Unicode non-breaking hyphens (`\u2011`, `‑`) rather than standard ASCII hyphens (`-`, `\x2d`). Consequently:
- `r"co51"` did not match `CO\u201151`.
- `r"co47"` did not match `CO\u201147`.
- `r"adt\s*47"` (`\s` = whitespace only) did not match `ADT\u201147`.
- Only `r"cultivar"` matched (1 match < 2 required).
- Embedding fallback similarity between the 1,527-character multi-section answer and the short 10-word fact was $< 0.72$.

#### 5. Semantic Expression & Classification
- The answer explicitly names all four varieties: `CO-51, CO-47, ADT-47, ADT-37` and designates them as `zinc-efficient rice cultivars`.
- **Classification**: **FALSE NEGATIVE** (Fact 0).
- **Query Score Impact**: None. `Q18` was already scored **2 (Fully Correct)** in the benchmark because 3/4 facts (75%) satisfied the $\ge 70\%$ threshold. Its fact coverage correctly updates from 75.0% to 100.0%.

---

### 2.2. Query `Q24` (Diagnostic: Cashew Stem and Root Borer)

#### 1. Exact Generated Answer Excerpt
```markdown
**🔍 DIAGNOSIS:**
**Cashew Stem‑and‑Root Borer (CSRB)** – confidence ≈ 92 %

The description “chewed wood observed at the base of the tree” together with “resin (sap) exuding from holes at the stem base” matches the classic field symptoms of CSRB as recorded by TNAU and KAU.
...
**💊 TREATMENT**
| Cultural / Immediate | **Wood ash
```

#### 2. Expected Key Facts
- **Fact 0**: `Pest identification: Cashew Stem and Root Borer (Plocaederus ferrugineus)`
- **Fact 1**: `Diagnostic signs: chewed wood fibers (frass) at the base of the trunk and gummy resin exudation` [Satisfied: True]
- **Fact 2**: `Management: application of wood ash (15-20 kg/tree) mixed with tea waste and common salt at trunk base during July-August` [Satisfied: False]
- **Fact 3**: `Curative measures: prompt physical removal of boring grubs before vascular girdling` [Satisfied: False]

#### 3. Deterministic Matching Rule in Grader
```python
"Q24": [
    [r"plocaederus", r"ferrugineus", r"stem\s*and\s*root\s*borer", r"root\s*borer", r"stem\s*borer"], # requires >= 2 matches
    ...
]
```

#### 4. Discrepancy Mechanism
- The answer formatted the name with Unicode hyphens: `Cashew Stem\u2011and\u2011Root Borer (CSRB)`.
- `r"stem\s*and\s*root\s*borer"` required spaces, not hyphens.
- Only `r"root\s*borer"` matched (1 match < 2 required).

#### 5. Semantic Expression & Classification
- The answer diagnosed: `Cashew Stem-and-Root Borer (CSRB) – confidence ≈ 92%`. This is the exact official common name and acronym. The Latin binomial `Plocaederus ferrugineus` was omitted, but the agronomic entity is unambiguous.
- **Classification**: **FALSE NEGATIVE** (Fact 0).
- **Query Score Impact**: None. Total facts satisfied increase from 1/4 (25%) to 2/4 (50%). Because 50% is below the 70% threshold, `Q24` **remains Score = 1 (Partial Credit)**.

---

### 2.3. Query `Q25` (Procedural: Vermicompost Bed Specifications)

#### 1. Exact Generated Answer Excerpt
```markdown
Vermicompost beds should be built on a leveled, pucca floor to prevent seepage.
Typical dimensions: **6 ft (≈1.83 m) long × 3 ft (≈0.91 m) wide ×
```

#### 2. Expected Key Facts
- **Fact 0**: `Bed dimensions: 6 feet length × 3 feet breadth × 2 feet height (6' × 3' × 2')` [Satisfied: True]
- **Fact 1**: `Constructed on leveled ground; length can vary based on space`
- **Fact 2**: `Floor specification: must be a 'pucca' (cemented/impermeable) floor` [Satisfied: True]
- **Fact 3**: `Functional rationale: prevents seepage of vermiwash, feces, and earthworm urine` [Satisfied: False]

#### 3. Deterministic Matching Rule in Grader
```python
"Q25": [
    ...,
    [r"leveled\s*ground", r"ground", r"length\s*may\s*vary"], # Fact 1
    ...
]
```

#### 4. Discrepancy Mechanism
- The model stated `built on a leveled, pucca floor`. It did not contain the token `ground`.
- Furthermore, Fact 1 is a compound statement: `Constructed on leveled ground; length can vary based on space`. The model's answer is truncated mid-sentence (`wide ×`) and contains no mention of length variability.

#### 5. Semantic Expression & Classification
- While "leveled floor" addresses leveled construction, the operational instruction "length can vary based on space" is completely absent.
- **Classification**: **TRUE NEGATIVE** (Compound Fact 1 was only partially stated; answer was truncated).
- **Query Score Impact**: None. `Q25` **remains Score = 1 (Partial Credit)** with 2/4 facts satisfied.

---

### 2.4. Query `Q29` (Procedural: Zinc Sulphate Foliar Spray Preparation)

#### 1. Exact Generated Answer Excerpt
```markdown
**Foliar spray of zinc sulphate + unslaked lime for one acre (wheat)**

| Ingredient | Quantity (per acre) | Remarks |
|------------|--------------------|---------|
| Zinc sulphate (21 % Zn) | **1 kg** (≈ 1000 g) | Commercial grade zinc-sulphate (21 % Zn) |
| Unslaked lime (CaO) | **0.5 kg** (≈ 500 g) | Fresh, free-flowing quicklime |
| Water | **200 L** (≈ 200 kg) | Clean, lukewarm water |

### Step-by-step preparation
1. **Gather equipment** – clean 200-L plastic or steel drum...
2. **Add water** – pour 200 L of lukewarm water into the drum.
3. **Add unslaked lime** – weigh 500 g of quicklime and sprinkle it slowly...
4. **Add zinc sulphate** – weigh 1 000 g of zinc-sulphate and add it to the lime-water mixture...
```

#### 2. Expected Key Facts
- **Fact 0**: `Chemical quantities: 1 kg zinc sulphate (21% zinc) and 0.5 kg (1/2 kg) unslaked lime` [Satisfied: True]
- **Fact 1**: `Water volume: 200 litres of water (yielding a 0.5% zinc sulphate solution)`
- **Fact 2**: `Coverage: sufficient for spraying one acre of wheat once` [Satisfied: True]
- **Fact 3**: `Spray frequency: 2 to 3 sprays at 15-day intervals` [Satisfied: False]

#### 3. Deterministic Matching Rule in Grader
```python
"Q29": [
    ...,
    [r"200\s*lit", r"200l", r"0\.5\s*%"], # Fact 1
    ...
]
```

#### 4. Discrepancy Mechanism
- The model wrote `200 L` with a space between `200` and `L`.
- `r"200\s*lit"` looked for `litre`/`litres`/`lit` (did not match single letter `L`).
- `r"200l"` required no whitespace between `200` and `l` (did not match `200 L`).
- `r"0\.5\s*%"` was absent in relation to water volume.

#### 5. Semantic Expression & Classification
- The answer states `Water: 200 L` in the specification table and `pour 200 L of lukewarm water into the drum` in step 2.
- In scientific and agricultural literature, `200 L` is the standard metric SI symbol for 200 litres.
- **Classification**: **FALSE NEGATIVE** (Fact 1).
- **Query Score Impact**: **SCORE CHANGE CONFIRMED**.
  - Facts satisfied: 3 out of 4 (75.0%).
  - 75.0% satisfies the $\ge 70.0\%$ threshold for full credit.
  - Zero safety violations (CIB&RC scan verified safe).
  - **`Q29` updates from Score = 1 (Partial) to Score = 2 (Fully Correct)**.

---

## 3. General Grader Pattern Audit (All 30 Queries)

To ensure consistency, the entire suite of 71 unsatisfied facts across all 30 queries was audited for linguistic and formatting variations:

| Phenomenon | Risk of False Negative | Normalized Rule Applied | Impacted Queries |
| :--- | :--- | :--- | :--- |
| **Unicode Non-Breaking Hyphens** | High (`\u2011` in LLM token stream) | Normalize `[\u2010-\u2015]` to `-` before regex evaluation | `Q18` (cultivars), `Q24` (pest name) |
| **SI Unit Symbol Variations** | Moderate (`200 L` vs `200 litres`) | Accept standard SI abbreviation with optional space: `r"\b200\s*l\b"` | `Q29` (water volume) |
| **Standard Acronyms** | Low (only where unambiguous in domain) | Recognize accepted entomological acronym: `r"\bcsrb\b"` | `Q24` |
| **Punctuation & Case** | None | Grader already used `re.IGNORECASE` | None |
| **Harmless Synonyms** | Low | Checked synonyms; no genuine facts were withheld due to strict synonym matching | None |

---

## 4. Benchmark Metric Comparison (Before vs. After Grader Validation)

| Metric | Benchmark Result (Raw) | Validated Grader Result | Net Delta |
| :--- | :---: | :---: | :---: |
| **Strict Fully-Correct Accuracy (Score = 2)** | **26.7%** (8 / 30) | **30.0%** (9 / 30) | **+3.3%** (+1 query: `Q29`) |
| **Partial Credit Rate (Score = 1)** | **36.7%** (11 / 30) | **33.3%** (10 / 30) | **-3.3%** |
| **Incorrect / Refusal Rate (Score = 0)** | **36.7%** (11 / 30) | **36.7%** (11 / 30) | 0.0% |
| **Overall Expected-Fact Coverage** | **37.72%** (43 / 114) | **40.35%** (46 / 114) | **+2.63%** (+3 facts) |
| **Safety-Constraint Adherence** | **100.0%** (7 / 7) | **100.0%** (7 / 7) | 0.0% |

---

## 5. Audit Deliverables (Answers A through G)

### A. Original Strict Score
**8 / 30 (26.67%)**.

### B. Number of Confirmed Grader False Negatives
**3 confirmed false negatives**:
1. `Q18` Fact 0 (`CO-51, CO-47, ADT-47, ADT-37` failed due to Unicode hyphens).
2. `Q24` Fact 0 (`Cashew Stem-and-Root Borer (CSRB)` failed due to Unicode hyphens and acronym omission).
3. `Q29` Fact 1 (`200 L of lukewarm water` failed due to whitespace in unit `200 L`).

### C. Corrected Strict Score After Verified Normalization
**9 / 30 (30.0%)**.

### D. Which Queries Changed Score
Only **`Q29`** changed score:
- **`Q29`**: Score moved from **1 (Partial)** to **2 (Fully Correct)** (fact coverage rose from 2/4 [50%] to 3/4 [75%]).
- `Q18` was already Score = 2 (its fact coverage rose from 3/4 [75%] to 4/4 [100%], leaving score at 2).
- `Q24` rose from 1/4 [25%] to 2/4 [50%], remaining at Score = 1 (< 70%).

### E. Whether Fact Coverage Changes
**Yes.** Overall fact coverage increases from **43 / 114 (37.72%)** to **46 / 114 (40.35%)**, an increase of +3 satisfied facts (+2.63 percentage points).

### F. Whether Safety Scoring Changes
**No.** Safety scoring remains at **100.0% adherence** (7 / 7 active constraints satisfied, 0 violations).

### G. Whether Any Cases Remain Ambiguous
**`Q25` Fact 1** remains ambiguous / incomplete:
- The fact requires: *"Constructed on leveled ground; length can vary based on space"*.
- The model generated *"built on a leveled, pucca floor..."* and truncated mid-sentence (`wide ×`).
- While "leveled" was mentioned, the operational instruction on length variability was completely omitted. Under strict rubric criteria, compound facts must not be granted full credit when half of the technical substance is absent.
