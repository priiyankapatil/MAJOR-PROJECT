# Deep-Dive Audit: Retrieval Deficit & Grader Mismatch Investigation
**Target Queries**: `Q04`, `Q11`, `Q13`, `Q17`, `Q28`  
**Evaluation Scope**: Agronomic RAG Generation & Retrieval Pipeline  
**Date**: September 19, 2026  
**Status**: Completed (Evidence-Backed Root Cause Identification)

> [!NOTE]
> **Audit Scope & Reproducibility Notice**:
> - **Investigated Query Set**: This audit investigated exactly five target queries: `Q04`, `Q11`, `Q13`, `Q17`, and `Q28`. Conclusions regarding source chunk presence and vector store adequacy are strictly limited to these five audited queries and must not be extrapolated as a proof of complete domain information across the entire corpus.
> - **Reproducibility Note**: Raw benchmark JSON (`retrieval_deficit_audit.json`) is gitignored. Embedded Markdown results reflect historical trial records and do not guarantee independent clean-checkout reproducibility.

---

## 1. Executive Summary

Following the completion of **Experiment R1-C**, a targeted architectural audit was conducted on the four persistent generation failures (`Q04`, `Q11`, `Q17`, `Q28`) and the strict-grader spelling mismatch on `Q13`.

The initial hypothesis was that these failures represented gaps in the retrieval index or ChromaDB vector store. **Our empirical investigation disproves this assumption.** The vector store possesses high-fidelity source chunks for all five queries. The real root causes span three distinct failure categories:

| Query ID | Domain / Question | Apparent Symptom | True Root Cause Category | Evidence-Backed Finding |
| :--- | :--- | :---: | :--- | :--- |
| **`Q04`** | *Trichoderma viride* seed treatment rate | Score 0 (0/3 facts) | **Generator Reasoning Exhaustion** (Not Retrieval) | Target chunk `chunk_002690` was retrieved and delivered. `gpt-oss-20b` exhausted its token limit inside internal `<think>` reasoning, emitting an empty string (`answer: ""`, `completion: 450`, `finish_reason: length`). |
| **`Q11`** | Mechanical destruction of congregating insects | Score 0 (0/4 facts) | **Grader Pattern Over-Specification & Boundary** (Not Retrieval) | Target chunk `chunk_011830` was delivered. Model generated a valid agronomic answer, but rubric regex demanded `["mechanical", "shaking", "destruction", "collect", "kill", "egg", "nymph"]` simultaneously. "Shaking" is absent from the source chunk. Semantic fallback reached 0.711 (threshold is 0.72). |
| **`Q17`** | Bacterial wilt resistant tomato varieties (KAU) | Score 0 (0/3 facts) | **Ground-Truth Annotation Defect: Cross-Crop Contamination** (Not Retrieval) | Retrieval delivered `chunk_003784` (Tomato section). Generator correctly output tomato varieties: *Sakthi, Mukthi, Anagha*. Rubric failed it because ground truth erroneously copied chilli varieties (*Ujwala, Anugraha*) from `chunk_003772`. |
| **`Q28`** | Coconut palm coir pith & organic manure | Score 0 (0/4 facts) | **Query Expansion Corruption (Semantic Bridge Pollution)** | Target chunks `chunk_011687` and `chunk_003146` rank **#1 and #2** in pure retrieval! However, `semantic_bridge.py` fuzzy-mapped "organic manures" to "Farmyard Manure (Farm Yard Manure (FYM))", causing BM25 and dense retrieval to swamp the pool with generic vegetable FYM chunks. |
| **`Q13`** | Seed rates for tomato, chilli, and brinjal | Score 1 (2/4 facts) | **Grader Pattern Unit Conflict & Single-Letter Sensitivity** | Generator correctly stated: *"Chili: Normal: 500 g/ha, Hybrid: 200 g/ha"*. Grader regex expected `"1 kg"` (from KAU) instead of `"500 g"` (from TNAU) and required double 'l' (`"chilli"`). Semantic fallback scored 0.7114 vs 0.72 threshold (missed by 0.0086). |

---

## 2. Evidence-Backed Query Investigations

### Investigation 1: `Q04` — Talc-Based *Trichoderma viride* Seed Treatment
* **Query**: `"What is the application rate of talc-based formulation of Trichoderma viride for seed treatment?"`
* **Expected Facts**:
  1. Seed treatment rate: 10 g per kg of seed (10 g kg⁻¹)
  2. Talc-based formulation of *Trichoderma viride*
  3. Used prophylactically for disease management such as sheath blight in rice
* **Delivered Chunks in Benchmark**:
  * [chunk_002690](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data) (*KAU (Kerala Agricultural University).pdf*):  
    > *"Prophylactic application of talc based formulation of Trichoderma viride as - Seed treatment (10 g kg-1 seed) + Soil application (2.5 kg ha-1) one week after transplanting + Foliar spray (10g l-1) after one month is effective for the management of sheath blight for upland rice."*
  * `chunk_010092`: Groundnut seed rate (distractor).
  * `chunk_002692`: Duplicate KAU sheath blight rice recommendation.
* **Analysis**:
  * The retrieval pipeline successfully retrieved and delivered the exact primary source chunk (`chunk_002690`) containing 100% of the ground-truth facts.
  * In **Experiment R1-A**, the generator produced a complete answer and achieved **Score 2 (3/3 facts satisfied)**.
  * In **Experiment E1** (250 tokens) and **R1-C** (450 tokens), `openai/gpt-oss-20b`'s internal reasoning tokens expanded uncontrollably, consuming all 450 completion tokens before producing answer text:
    ```json
    "answer": "",
    "finish_reason": "length",
    "tokens": {"prompt": 612, "completion": 450, "total": 1062, "reasoning": 0}
    ```
* **Root Cause**: **Reasoning Token Starvation / Empty Completion Bug on 20B Model**, not a retrieval gap.

---

### Investigation 2: `Q11` — Mechanical Destruction of Congregating Insect Stages
* **Query**: `"Why is mechanical destruction of congregating insect stages recommended in sustainable pest management?"`
* **Expected Facts**:
  1. Physical destruction of sluggish and congregating pest stages
  2. Early detection enables localized, ecologically safe interventions
  3. Prevents widespread pest multiplication without synthetic chemical reliance
  4. Complemented by cultural practices like smoking organic wastes during flushing, flowering, and fruiting
* **Delivered Chunks in Benchmark**:
  * [chunk_011830](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data) (*The TNAU Horticulture PDF.pdf*):  
    > *"If the pest infestation is detected and located at the correct time, local and limited treatments, which are ecologically safe, can take care of the issues. Smoking: Smoking the plantation is found repelling many insect species harmful to the crop. Smoking the plantation with organic wastes during flushing, flowering and fruiting phases will keep TMB infestation low. Mechanical: Mechanical destruction of sluggish and congregating stages of the insect-pest are advisable."*
* **Generated Answer**:
  > *"Targets the most vulnerable, slow-moving stages (e.g., nymphs, pupae) that are easy to locate and remove. Reduces pest numbers before they reach damaging adult stages, lowering overall infestation pressure. Eliminates the need for chemical insecticides, preserving ecological balance... Works in conjunction with other low-impact tactics (smoking, ash, sticky traps) for integrated, sustainable control."*
* **Grader Rubric Evaluation**:
  * In `FACT_PATTERNS_V2["Q11"][0]`, the regex list is:  
    `["mechanical", "shaking", "destruction", "collect", "kill", "egg", "nymph"]`
  * The grader logic uses `all(re.search(pat, norm_answer) for pat in p_list)`.
  * The rubric strictly requires **all seven words** to be present in the output, including `"shaking"`.
  * However, neither the delivered chunk nor the source text in TNAU Horticulture contains the word `"shaking"`.
  * The semantic similarity fallback between the generated answer and Fact 0 was **0.711**, falling just below the deterministic threshold of **0.72**.
* **Root Cause**: **Deterministic Grader Pattern Over-Specification & Semantic Boundary**, not a retrieval gap.

---

### Investigation 3: `Q17` — Bacterial Wilt Resistant Varieties of Tomato (KAU)
* **Query**: `"Which bacterial wilt resistant varieties of tomato are recommended by Kerala Agricultural University?"`
* **Delivered Chunks in Benchmark**:
  * [chunk_003784](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data) (*KAU (Kerala Agricultural University).pdf*, Page 194):  
    > *"Varieties: Bacterial wilt resistant varieties: Sakthi, Mukthi, Anagha and Vellayani Vijai, Manulekshmi and Manuprabha. Rain shelter variety: Akshaya. Seed rate: 400 g ha-1. Raising seedlings: Tomato is a transplanted vegetable."*
* **Generated Answer**:
  > *"- Sakthi, Mukthi, Anagha, Vellayani Vijai, Manulekshmi, Manuprabha"*
* **Expected Ground-Truth Facts in Benchmark**:
  * Fact 0: `Bacterial wilt resistant varieties: Ujwala and Anugraha (as listed in source chunk)`
  * Fact 1: `High yielding varieties mentioned: Jwalasakhi, Jwalamukhi, Vellayani Athulya, Keerthi`
* **Discovery & Document Tracing**:
  * Inspection of `KAU (Kerala Agricultural University).pdf` at [chunk_003772](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data) reveals:  
    > *"Varieties: High yielding varieties: Jwalasakhi, Jwalamukhi, Jwala, Pant C-1, K-2, Vellayani Athulya, Keerthi... Bacterial wilt resistant varieties: Ujwala, Anugraha. Seed rate: 1.0 kg ha-1. Raising seedlings: Chilli is a transplanted crop."*
  * `Ujwala` and `Anugraha` are **CHILLI varieties**, not tomato varieties!
  * The benchmark ground truth was mistakenly annotated with chilli data from chunk 3772 instead of tomato data from chunk 3784.
  * The retrieval system correctly identified that the farmer asked for tomato and retrieved the tomato chunk (`chunk_003784`). The generator gave 100% accurate tomato varieties. The grader marked it 0/3 because the rubric demanded chilli varieties.
* **Root Cause**: **Benchmark Ground-Truth Annotation Error (Cross-Crop Contamination: Chilli vs Tomato)**, not a retrieval gap.

---

### Investigation 4: `Q28` — Coconut Coir Pith & Organic Manures
* **Query**: `"What organic manures and coir pith application practices are recommended for maintaining coconut palms?"`
* **Expected Facts**:
  1. Forest leaves, cattle manure, coir dust/pith, or coconut shreddings
  2. Dosage: 10 kg per pit
  3. Bio-inoculant: PGPR mix I (100–200 g)
  4. First 3 years in sandy/coastal soils
* **Source Chunks in Database**:
  * [chunk_011687](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data) (*The TNAU Horticulture PDF.pdf*): Exact 4/4 facts.
  * [chunk_003146](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/data) (*KAU (Kerala Agricultural University).pdf*): Exact 4/4 facts.
* **Pure Hybrid Retrieval vs Production Query Gate**:
  * When executing `retrieve_chunks` directly on the **raw user query**:
    * `chunk_011687` ranks **#1** (Dense: 0.7578, Sparse: 0.9188, Cross-Encoder: 5.7738)
    * `chunk_003146` ranks **#2** (Dense: 0.6949, Sparse: 0.8041, Cross-Encoder: 5.3853)
    * Both target chunks are retrieved at the very top of the context!
  * Why did the benchmark pipeline retrieve generic vegetable chunks (`chunk_000856`, `chunk_001572`, `chunk_001569`)?
  * **The Culprit: Step 0 Semantic Bridge**:
    * In [step6_query_gate.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/step6_query_gate.py) (lines 657–666), `apply_semantic_bridge(query)` is invoked before retrieval.
    * The bridge fuzzy-matches `"organic manures"` (97% match) to `"Farmyard Manure (Farm Yard Manure (FYM))"`.
    * The rewritten query becomes:
      > *"What organic manures (Farmyard Manure, Farm Yard Manure (FYM)) and coir pith application practices are recommended for maintaining coconut (Coconut, Cocos nucifera) palms?"*
    * This repetitive expansion triples the weight of `"Farmyard Manure"` and `"FYM"` in BM25.
    * BM25 and dense retrieval are flooded with generic FYM chunks for onion and radish from *Crop Protection.pdf*, completely displacing the real coconut chunks from the candidate pool!
* **Root Cause**: **Semantic Bridge Keyword Pollution & Query Drift**, actively degrading an otherwise perfect retrieval rank.

---

### Investigation 5: `Q13` — Strict Grader Spelling & Pattern Mismatch
* **Query**: `"What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?"`
* **Expected Fact 1**: `Chilli seed rate: 500 g for normal varieties; 200 g for hybrids`
* **Generated Answer in R1-C**:
  ```markdown
  - Tomato: Normal: 400–500 g/ha, Hybrid: 125–150 g/ha
  - Chili: Normal: 500 g/ha, Hybrid: 200 g/ha
  - Brinjal: Normal: 500 g/ha, Hybrid: 200 g/ha
  ```
* **Evaluation Trace**:
  1. Regex Pattern in `FACT_PATTERNS_V2["Q13"][1]`:  
     `["chilli", "1\\s*kg", "200-250\\s*g"]`
     * Issue A: Searches for `"1 kg"` (from KAU manual), while the ground-truth fact and TNAU manual state `"500 g"`.
     * Issue B: Requires double 'l' (`"chilli"`). The model output `"Chili"` (single 'l').
     * Result: Deterministic regex fails.
  2. Fallback Semantic Similarity:
     * When evaluating with `"Chili"`: Cosine similarity = **0.7114**.
     * Threshold = **0.7200**.
     * The fact fails by an infinitesimal delta of **0.0086**!
     * When evaluated with `"Chilli"`: Cosine similarity = **0.7274** (passes above 0.72).
* **Root Cause**: **Grader Regex Unit Conflict (1 kg vs 500 g) + Sub-Threshold Single-Letter Spelling Sensitivity**. The agronomic extraction was completely correct.

---

## 3. Categorized Taxonomy of Failures

```
                    ┌──────────────────────────────────────────────┐
                    │       BENCHMARK DEFICIT ROOT CAUSES         │
                    └──────────────────────┬───────────────────────┘
                                           │
         ┌──────────────────┬──────────────┴───────┬──────────────────┐
         │                  │                      │                  │
┌────────┴─────────┐ ┌──────┴───────────┐ ┌────────┴────────┐ ┌───────┴────────┐
│  Query Gate Step │ │   Model Runtime  │ │ Grader Rubrics  │ │  Dataset Truth │
│  Semantic Bridge │ │  Token Ceiling   │ │ Regex / Sim     │ │  Cross-Crop    │
├──────────────────┤ ├──────────────────┤ ├─────────────────┤ ├────────────────┤
│       Q28        │ │       Q04        │ │    Q11, Q13     │ │      Q17       │
│ Keyword flooding │ │ 450 tokens spent │ │ Impossible      │ │ Chilli truth   │
│ displaced #1/#2  │ │ on <think>,      │ │ regex (Q11);    │ │ assigned to    │
│ coconut chunks   │ │ empty completion │ │ sim 0.711 (Q13) │ │ tomato query   │
└──────────────────┘ └──────────────────┘ └─────────────────┘ └────────────────┘
```

---

## 4. Proposed Controlled Retrieval Experiment: Experiment SB1

### Experiment Name
**Experiment SB1: Semantic Bridge Precision & Query Drift Guardrails**

### Objective
Eliminate catastrophic query drift and keyword swamping on high-specificity agronomic queries while preserving scientific taxonomical bridging for genuine vernacular/dialect folk terms.

### Core Hypothesis
If the semantic bridge suppresses expansion when a query already contains explicit technical entity qualifiers (e.g. crop name `"coconut"` + practice `"coir pith"`), and deduplicates parenthesized synonym injection, retrieval will immediately recover `chunk_011687` and `chunk_003146` for `Q28` without regressing any vernacular query in the retrieval benchmark suite.

### Proposed Implementation Design (Evaluation-Only)
1. **Specificity Guardrail in `semantic_bridge.py`**:
   - Suppress fuzzy expansion of broad generic categories (such as `"organic manures" → "Farmyard Manure (Farm Yard Manure (FYM))"`) when the query already possesses specific crop and material identifiers.
2. **Synonym De-duplication**:
   - Ensure terms are appended at most once (e.g., `"organic manures (FYM)"` instead of triple repetition) to avoid artificially inflating BM25 term frequencies.
3. **Controlled Benchmark Protocol**:
   - Run an evaluation-only benchmark on [retrieval_benchmark_dataset.json](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/retrieval_benchmark_dataset.json) to measure **Recall@5**, **NDCG@5**, and **MRR** across all 30 queries before any generation execution.

---

## 5. Verification Checklist & Compliance

- [x] All 5 queries traced through source PDFs, chunks, embeddings, reranker, and grader.
- [x] Zero production code modified.
- [x] Zero index or embedding alterations.
- [x] No git commits or pushes.
- [x] Machine-readable audit saved to [retrieval_deficit_audit.json](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/retrieval_deficit_audit.json).
- [x] Markdown audit saved to [retrieval_deficit_audit.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/generation/retrieval_deficit_audit.md).
- [x] Proposed experiment (SB1) formulated and held pending user review.
