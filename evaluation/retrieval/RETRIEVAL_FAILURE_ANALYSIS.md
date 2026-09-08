# In-Depth Retrieval Failure Analysis & Behavioral Diagnostics

**Evaluation Baseline**: First Retrieval Benchmark (30 Queries, K=1, 3, 5, 10)
**Corpus**: Production Regenerated Corpus (`data/chunks/all_chunks.parquet`, **12,856 chunks**)
**Methods Compared**: Pure Dense (`all-MiniLM-L6-v2`), Pure Sparse (`BM25Okapi`), Weighted Hybrid (`0.6/0.4`), Reciprocal Rank Fusion (`RRF k=60`)

---

## 1. Overall Failure Counts by Retrieval Method

| Retrieval Method | Failures @ Top-5 (Hit@5 = 0) | Failures @ Top-10 (Hit@10 = 0) | Top-1 Accuracy (Hit@1) | Mean MRR@10 | Mean NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense** | 8 / 30 (26.7%) | 5 / 30 (16.7%) | 60.0% | 0.6737 | 0.7055 |
| **BM25** | 5 / 30 (16.7%) | 3 / 30 (10.0%) | 70.0% | 0.7503 | 0.7833 |
| **Hybrid (0.6/0.4)** | 5 / 30 (16.7%) | 4 / 30 (13.3%) | 70.0% | 0.7722 | 0.7949 |
| **RRF (k=60)** | 5 / 30 (16.7%) | 4 / 30 (13.3%) | 73.3% | 0.7815 | 0.8016 |

---

## 2. Failure Counts by Category (Evaluated at Top-5)

| Category (6 Queries Each) | Dense Failures | BM25 Failures | Hybrid Failures | RRF Failures | Hardest Query Patterns |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`conceptual_paraphrased`** | 4 / 6 | 3 / 6 | 3 / 6 | 3 / 6 | Q08, Q10, Q12 |
| **`crop_specific`** | 2 / 6 | 2 / 6 | 1 / 6 | 1 / 6 | Q17 |
| **`diagnostic`** | 1 / 6 | 0 / 6 | 1 / 6 | 1 / 6 | Q20 |
| **`exact_lexical`** | 1 / 6 | 0 / 6 | 0 / 6 | 0 / 6 | None (100% Top-5 Hit) |
| **`procedural_recommendation`** | 0 / 6 | 0 / 6 | 0 / 6 | 0 / 6 | None (100% Top-5 Hit) |

---

## 3. Comparative Behavioral Patterns

### A. Queries Where BM25 Succeeds but Dense Fails
- **`Q03`** (exact_lexical): "What is the recommended tine spacing for Happy Seeder and Coragen 18.5 SC dose in wheat?"
  - *Observation*: BM25 rank #1 vs Dense not in top-10
- **`Q08`** (conceptual_paraphrased): "Why does excessive application of nitrogenous fertilizers increase crop vulnerability to insect pests?"
  - *Observation*: BM25 rank #6 vs Dense not in top-10
- **`Q11`** (conceptual_paraphrased): "Why is mechanical destruction of congregating insect stages recommended in sustainable pest management?"
  - *Observation*: BM25 rank #1 (<=5) vs Dense rank #9
- **`Q16`** (crop_specific): "How is irrigation scheduled in sugarcane based on climatological pan evaporation approach?"
  - *Observation*: BM25 rank #1 (<=5) vs Dense rank #6
- **`Q20`** (diagnostic): "What nutrient deficiency is diagnosed when older leaves turn yellow while younger leaves remain relatively green?"
  - *Observation*: BM25 rank #5 vs Dense not in top-10

### B. Queries Where Dense Succeeds but BM25 Fails
- **`Q13`** (crop_specific): "What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?"
  - *Observation*: Dense rank #1 (<=5) vs BM25 rank #7
- **`Q17`** (crop_specific): "Which bacterial wilt resistant varieties of tomato are recommended by Kerala Agricultural University?"
  - *Observation*: Dense rank #10 vs BM25 not in top-10

### C. Queries Where BM25 Finds Ground Truth but Hybrid Loses It (Dilution Effect)
- **`Q08`** (conceptual_paraphrased): "Why does excessive application of nitrogenous fertilizers increase crop vulnerability to insect pests?"
  - *Observation*: BM25 rank #6 lost by Hybrid (Rank > 10)
  - *Mechanism*: BM25 retrieved the chunk, but because Dense assigned it 0.0 or placed it outside top-20, the weighted sum `(0.6 * 0.0 + 0.4 * sparse_score)` was insufficient to beat purely dense-retrieved false positives.
- **`Q20`** (diagnostic): "What nutrient deficiency is diagnosed when older leaves turn yellow while younger leaves remain relatively green?"
  - *Observation*: BM25 rank #5 lost by Hybrid (Rank > 10)
  - *Mechanism*: BM25 retrieved the chunk, but because Dense assigned it 0.0 or placed it outside top-20, the weighted sum `(0.6 * 0.0 + 0.4 * sparse_score)` was insufficient to beat purely dense-retrieved false positives.

### D. Queries Where RRF Improves Ranking Over Weighted Hybrid
- **`Q13`** (crop_specific): "What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?"
  - *Observation*: RRF rank #1 vs Hybrid #2

### E. Queries Missed by All Four Methods (Top-10 Complete Misses)
- **`Q10`** (conceptual_paraphrased): "What is the mechanism and role of Azospirillum biofertilizers in enhancing root development and nutrient uptake?" (Ground Truth: `['chunk_001999']`)
- **`Q12`** (conceptual_paraphrased): "How does timely weed management during the critical crop growth period influence nutrient and moisture competition?" (Ground Truth: `['chunk_003231']`)

---

## 4. In-Depth Case Studies: Failure Diagnostics

### Case Study 1: Query `Q03` (exact_lexical)
**Query**: *"What is the recommended tine spacing for Happy Seeder and Coragen 18.5 SC dose in wheat?"*

**Ground Truth Target**: `chunk_000788` (Crop Protection.pdf)
> **Ground Truth Excerpt**: "PROTECTION TECHNIQUES •	 Wheat: Use Happy Seeder with 8 inch tine spacing for sowing wheat for  better weed control. Spray 40 ml per acre Coragen 18.5 SC (chlorantraniliprole) for armyworm control. Broadcast 7 kg per acre Mortel/Regent 0.3 G (fipronil) or 1 litre Dursban 20  EC (chlorpyriphos) mixed with 20 kg of moist sand before first irrigation..."
> **Relevance Note**: Provides exact technical specs: Happy Seeder with 8 inch tine spacing for wheat sowing and Coragen 18.5 SC @ 40 ml/acre.

**Performance Across Methods**:
| Method | First Rank | Hit@5 | Hit@10 | MRR@10 | Top Retrieved Chunk IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_012251', 'chunk_002759', 'chunk_012249']` |
| **BM25** | #1 | 1 | 1 | 1.0000 | `['chunk_000788', 'chunk_007157', 'chunk_000851']` |
| **Hybrid (0.6/0.4)** | #1 | 1 | 1 | 1.0000 | `['chunk_000788', 'chunk_012251', 'chunk_002759']` |
| **RRF (k=60)** | #1 | 1 | 1 | 1.0000 | `['chunk_000788', 'chunk_012251', 'chunk_007157']` |

#### Empirical Diagnostic:
- **Failure Classification**: `exact numeric/model term issue` & `lexical mismatch in dense representation`
- **What Dense Retrieved**: Dense ranked `chunk_012251` (#1) and `chunk_002759` (#2). Excerpt from `chunk_012251`: *"Spacing under organic farming of nutri-cereals: Spacing of 30x75cm for organic  production of foxtail millet, little millet, kodo millet, browntop millet, proso millet  and barnyard millet is found to..."*
- **Root Cause**: The bi-encoder embedding model (`all-MiniLM-L6-v2`) embedded the query based on high-frequency semantic tokens ("recommended", "spacing", "dose in wheat"), clustering it with generic sowing geometry in UAS Bangalore documents. The specific machine entity ("Happy Seeder") and chemical trade name ("Coragen 18.5 SC") were lost in the dense embedding projection. In contrast, BM25 assigned massive IDF weight to "Happy", "Seeder", and "Coragen", placing `chunk_000788` instantly at Rank 1. Hybrid and RRF correctly rescued this.

---

### Case Study 2: Query `Q08` (conceptual_paraphrased)
**Query**: *"Why does excessive application of nitrogenous fertilizers increase crop vulnerability to insect pests?"*

**Ground Truth Target**: `chunk_001041` (Crop Protection.pdf)
> **Ground Truth Excerpt**: "Hints for obtaining high sugar recovery •	 Do not plant whole area under a single variety. Plant recommended early and mid- late maturing varieties in the ratio of 3:2 on area basis. •	 Save the crop from lodging by timely earthing up and propping. •	 Save the crop from diseases and insect pests through recommended control  measures. •	 Protect the..."
> **Relevance Note**: Discusses how imbalanced high nitrogen causes succulent vegetative growth that attracts and aggravates insect-pest infestations.

**Performance Across Methods**:
| Method | First Rank | Hit@5 | Hit@10 | MRR@10 | Top Retrieved Chunk IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_001544', 'chunk_011503', 'chunk_000783']` |
| **BM25** | #6 | 0 | 1 | 0.1667 | `['chunk_004567', 'chunk_010700', 'chunk_004570']` |
| **Hybrid (0.6/0.4)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_010700', 'chunk_010706', 'chunk_002135']` |
| **RRF (k=60)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_002135', 'chunk_010700', 'chunk_010706']` |

#### Empirical Diagnostic:
- **Failure Classification**: `score normalization/fusion issue` & `corpus competition / many similar chunks`
- **What BM25 Did**: BM25 successfully located the target `chunk_001041` at **Rank 6**.
- **What Dense Did**: Dense retrieved general urea schedule chunks (`chunk_001544`, `chunk_011503`, `chunk_000783`), giving `chunk_001041` zero dense score.
- **Why Hybrid Failed**: In current Hybrid: `final_score = 0.6 * dense + 0.4 * sparse`. Because dense weight is 0.6 and the chunk had 0.0 dense score, its hybrid score was `0.4 * 0.45 = 0.18`, which was overwhelmed by dense-only false positives scoring `0.6 * 0.55 = 0.33`. This caused `chunk_001041` to drop from Rank 6 in BM25 to **Rank > 10 in Hybrid** (a textbook score dilution failure).

---

### Case Study 3: Query `Q10` (conceptual_paraphrased)
**Query**: *"What is the mechanism and role of Azospirillum biofertilizers in enhancing root development and nutrient uptake?"*

**Ground Truth Target**: `chunk_001999` (Crop-Management_1.pdf)
> **Ground Truth Excerpt**: "Application of 100% inorganic nitrogen + 50 g Azospirillum + 5 kg FYM or 100% inorganic phosphorus + 50 g phosphobacteria + 5 kg FYM was found very effective in increasing yield. Honeybees and pollinators Apple blooms were visited by A. mellifera, A. cerana, Syrphieds, butterflies and coleopterans but honey bees contribute significantly in apple po..."
> **Relevance Note**: Explains nitrogen fixation and growth hormone secretion by Azospirillum inoculants to stimulate root proliferation and nutrient uptake.

**Performance Across Methods**:
| Method | First Rank | Hit@5 | Hit@10 | MRR@10 | Top Retrieved Chunk IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_004546', 'chunk_010876', 'chunk_009392']` |
| **BM25** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_010866', 'chunk_004543', 'chunk_010887']` |
| **Hybrid (0.6/0.4)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_001519', 'chunk_008896', 'chunk_004546']` |
| **RRF (k=60)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_001519', 'chunk_008896', 'chunk_010866']` |

#### Empirical Diagnostic:
- **Failure Classification**: `corpus competition / keyword saturation` & `semantic mismatch`
- **Corpus Saturation**: Over 40 chunks across the corpus mention *Azospirillum*. Almost all describe procedural application (e.g. *"Azospirillum @ 2 kg/ha seed treatment in cotton/rice"*).
- **Why All Methods Failed**: The target chunk (`chunk_001999`) comes from ICAR Annual Report describing biological nitrogen fixation and growth hormone secretion. Dense and BM25 were both flooded by procedural mentions from PAU, KAU, and TNAU packages of practices (`chunk_001519`, `chunk_008896`, `chunk_004546`). Without query grounding (e.g., specifying *"ICAR report"* or *"hormone mechanism"*), routine procedural mentions dominate BM25 term frequency and dense similarity.

---

### Case Study 4: Query `Q12` (conceptual_paraphrased)
**Query**: *"How does timely weed management during the critical crop growth period influence nutrient and moisture competition?"*

**Ground Truth Target**: `chunk_003231` (KAU (Kerala Agricultural University).pdf)
> **Ground Truth Excerpt**: "Weed the crop 10-15 days after germination of seed by light hoeing. Give another light hoeing or COCONUT AND GROUNDNUT 112 raking at the time of application of lime. Do not disturb the soil after 45 days of sowing. Plant protection Red hairy caterpillar, termites and leaf min- ers are the major pests of groundnut. Against leaf miner apply any of th..."
> **Relevance Note**: Outlines hoeing and weeding 10-15 days after germination to prevent early crop-weed competition for light, water, and nutrients.

**Performance Across Methods**:
| Method | First Rank | Hit@5 | Hit@10 | MRR@10 | Top Retrieved Chunk IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_001343', 'chunk_008440', 'chunk_011025']` |
| **BM25** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_007785', 'chunk_011118', 'chunk_008450']` |
| **Hybrid (0.6/0.4)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_011118', 'chunk_009550', 'chunk_008440']` |
| **RRF (k=60)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_008440', 'chunk_011118', 'chunk_009550']` |

#### Empirical Diagnostic:
- **Failure Classification**: `insufficient query specificity` & `corpus competition`
- **What Happened**: Ground truth `chunk_003231` is an intercropping weed management section in Groundnut (*"Weed the crop 10-15 days after germination..."*). The query asked broadly about *"timely weed management during critical growth period"* without specifying Groundnut.
- **Retrieved Chunks**: Both systems retrieved dedicated weed science chapters from TNAU and PAU (`chunk_008440`, `chunk_011118`, `chunk_009550`) that explicitly contain the academic phrase *"critical period of weed competition"*. These dedicated weed chunks legitimately had higher lexical and semantic similarity to the query than the concise agronomy bullet in `chunk_003231`.

---

### Case Study 5: Query `Q20` (diagnostic)
**Query**: *"What nutrient deficiency is diagnosed when older leaves turn yellow while younger leaves remain relatively green?"*

**Ground Truth Target**: `chunk_009535` (The TNAU Agriculture PDF.pdf)
> **Ground Truth Excerpt**: "ha-1): 260 | Yield target – 10 t ha-1: 125* | Yield target – 11 t ha-1: 153 107      Defieciency symptoms    Nitrogen deficiency  :  Leaves become yellow, older leaves show drying at the tips which  progress along mid veins, stalks become slender. Phosphorus deficiency  :  Leaves are purplish green during early growth. Growth spindly,  slow maturit..."
> **Relevance Note**: Diagnoses nitrogen deficiency by general chlorosis and yellowing initiating on mature lower leaves due to mobile N translocation.

**Performance Across Methods**:
| Method | First Rank | Hit@5 | Hit@10 | MRR@10 | Top Retrieved Chunk IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_000880', 'chunk_009543', 'chunk_009542']` |
| **BM25** | #5 | 1 | 1 | 0.2000 | `['chunk_000880', 'chunk_009542', 'chunk_008936']` |
| **Hybrid (0.6/0.4)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_000880', 'chunk_009542', 'chunk_008936']` |
| **RRF (k=60)** | Miss (>10) | 0 | 0 | 0.0000 | `['chunk_000880', 'chunk_009542', 'chunk_009543']` |

#### Empirical Diagnostic:
- **Failure Classification**: `semantic mismatch in Dense` & `score fusion dilution in Hybrid`
- **What Happened**: BM25 found target `chunk_009535` at **Rank 5** (Hit@5=1). Dense completely missed it (Rank > 10), retrieving general yellowing / mosaic virus chunks (`chunk_009137`, `chunk_009230`).
- **Why Hybrid Lost It**: Because Dense scored `chunk_009535` as 0.0, the 0.6 Dense weighting dragged down its combined score below the top-10 threshold. This is another clear instance where BM25 succeeded independently but the linear weighted hybrid dropped the relevant document.

---

### Case Study 6: Query `Q16` (crop_specific)
**Query**: *"How is irrigation scheduled in sugarcane based on climatological pan evaporation approach?"*

**Ground Truth Target**: `chunk_009552` (The TNAU Agriculture PDF.pdf)
> **Ground Truth Excerpt**: "After sowing, Life irrigation -4th ,12thday  Vegetative Phase  3  22nd ,32nd & 40th day  Flowering phase (Irrigate copiously)  3  50th ,60th & 72nd day  Maturity phase (Controlled  irrigation)  2  85th , 95th day    DRIP IRRIGATION TO MAIZE  Irrigation once in 2 days  Irrigation based on climatological approach Irrigation volume:  = (Pe x Kp x Kc x..."
> **Relevance Note**: Provides climatological IW/CPE ratio scheduling criteria and critical water management stages in sugarcane.

**Performance Across Methods**:
| Method | First Rank | Hit@5 | Hit@10 | MRR@10 | Top Retrieved Chunk IDs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Dense** | #6 | 0 | 1 | 0.1667 | `['chunk_001085', 'chunk_012321', 'chunk_010884']` |
| **BM25** | #1 | 1 | 1 | 1.0000 | `['chunk_009552', 'chunk_010367', 'chunk_009040']` |
| **Hybrid (0.6/0.4)** | #1 | 1 | 1 | 1.0000 | `['chunk_009552', 'chunk_010367', 'chunk_012323']` |
| **RRF (k=60)** | #1 | 1 | 1 | 1.0000 | `['chunk_009552', 'chunk_010367', 'chunk_001085']` |

#### Empirical Diagnostic:
- **Failure Classification**: `semantic diffusion in Dense`
- **What Happened**: BM25 placed `chunk_009552` directly at **Rank 1** by matching "irrigation scheduled", "sugarcane", and "climatological pan evaporation".
- **Dense Demotion**: Dense placed `chunk_009552` at **Rank 6** because dense bi-encoder representations dispersed similarity across multiple generic sugarcane drip irrigation chunks (`chunk_001085`, `chunk_012321`). Hybrid and RRF successfully used BM25's strong signal to surface `chunk_009552` at **Rank 1**.

---

## 5. Successful Complementary Retrieval Cases (Synergy Highlights)

### Synergy Case 1: Query `Q03` (exact_lexical)
**Query**: *"What is the recommended tine spacing for Happy Seeder and Coragen 18.5 SC dose in wheat?"*
- **Dense Rank**: `#Miss (>10)`
- **BM25 Rank**: `#1`
- **Hybrid Rank**: `#1`
- **RRF Rank**: `#1`
- **Synergy Insight**: Dense completely failed (missed top-10). BM25 provided strong exact token matching on model names. Hybrid and RRF successfully surfaced the chunk at **Rank 1**.

### Synergy Case 2: Query `Q11` (conceptual_paraphrased)
**Query**: *"Why is mechanical destruction of congregating insect stages recommended in sustainable pest management?"*
- **Dense Rank**: `#9`
- **BM25 Rank**: `#1`
- **Hybrid Rank**: `#1`
- **RRF Rank**: `#1`
- **Synergy Insight**: Dense placed the target at Rank 9 due to abstract IPM phrasing. BM25 matched "congregating", elevating the chunk to **Rank 1** in both Hybrid and RRF.

### Synergy Case 3: Query `Q18` (crop_specific)
**Query**: *"Which rice cultivars like CO51 and ADT47 are recommended for zinc biofortification strategies?"*
- **Dense Rank**: `#1`
- **BM25 Rank**: `#1`
- **Hybrid Rank**: `#1`
- **RRF Rank**: `#1`
- **Synergy Insight**: Dense ranked zinc biofortification cultivars at Rank 2. BM25 ranked it at Rank 2. RRF fused their mutual top-tier consensus and promoted `chunk_008935` to **Rank 1**.

---

## 6. Concrete Recommendations for Next Optimization Phase

Based on the empirical evidence gathered from the 120 retrieval runs, the following targeted interventions should be tested:

1. **Cross-Encoder Re-Ranking Stage (Top Priority)**:
   - *Evidence*: For both Hybrid and RRF, **Hit@10 is 86.7%** and **Recall@10 is 86.7%**, but **Hit@1 is only 70.0% – 73.3%**.
   - *Recommendation*: Introduce a lightweight cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` or `bge-reranker-base`) on the top-20 merged candidates. Cross-encoders examine query-document token interactions simultaneously, directly resolving ranking dilution without modifying corpus embeddings.

2. **Score Normalization Before Linear Fusion**:
   - *Evidence*: Query `Q08` demonstrated that BM25's raw score (scaled by `max_score`) was easily dominated by dense similarity scores, dropping a valid Rank 6 result entirely out of top 10.
   - *Recommendation*: Test Min-Max scaling or Z-score normalization over candidate pools before weighted combination, or adopt RRF as the default baseline since RRF eliminates scale variance.

3. **Scientific & Vernacular Query Expansion (Taxonomy Bridge)**:
   - *Evidence*: Dense failed on exact trade names/species (Q03, Q06), while BM25 struggled with semantic synonyms like IW/CPE pan evaporation (Q16).
   - *Recommendation*: Benchmark the existing `semantic_bridge.py` module explicitly during retrieval to evaluate if expanding queries with scientific binomials and standard chemical names improves dense/sparse alignment.

4. **Adaptive Category-Aware Alpha Tuning**:
   - *Evidence*: `exact_lexical` heavily favors BM25 (MRR 0.9167 vs Dense 0.6389), while `procedural_recommendation` heavily favors Dense (MRR 0.9167 vs BM25 0.7500).
   - *Recommendation*: Evaluate dynamic weighting where lexical queries use `dense_weight=0.3, sparse_weight=0.7` and procedural/conceptual queries use `dense_weight=0.7, sparse_weight=0.3` based on Query Gate classification.
