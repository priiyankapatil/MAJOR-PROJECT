# Controlled LLM Query-Rewriting Retrieval Ablation Experiment

**Model**: `openai/gpt-oss-120b` (Groq)
**Corpus**: Production Regenerated Corpus (`data/chunks/all_chunks.parquet`, **12,856 chunks**)
**Benchmark**: `evaluation/retrieval/retrieval_benchmark_dataset.json` (**30 queries**, 6 per category)
**Retrieval Setup**: Candidate Pool = Top-30 Dense + Top-30 BM25, Fusion = Equal Weighted (0.5 Dense + 0.5 BM25)

---

## 1. Executive Summary & Core Findings

- **Candidate-Pool Recall**: **93.3% (Original)** vs **91.7% (Rewritten)**.
- **Hit@1**: **60.0%** (Original) vs **33.3%** (Rewritten).
- **Hit@10**: **83.3%** (Original) vs **73.3%** (Rewritten).
- **MRR@10**: **0.6737** (Original) vs **0.4656** (Rewritten).
- **NDCG@10**: **0.7055** (Original) vs **0.5177** (Rewritten).
- **Mean Rewrite Latency**: **1955.55 ms** per query (total pipeline latency increases from 51.15 ms to 2037.06 ms).
- **Query Shift Outcomes**: **2 Improved**, **15 Harmed**, **13 Unchanged**.

---

## 2. Overall Performance Comparison: Original vs. Rewritten

### Primary Comparison: Equal Weighted Fusion (0.5 Dense + 0.5 BM25)

| Retrieval Condition | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@5 | Precision@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Original Query** | 60.0% | 73.3% | 73.3% | 83.3% | 56.7% | 70.0% | 71.7% | 81.7% | 60.0% | 16.7% | 9.3% | **0.6737** | **0.7055** | 51.15 ms |
| **B. LLM-Rewritten** | 33.3% | 56.7% | 60.0% | 73.3% | 30.0% | 53.3% | 58.3% | 71.7% | 33.3% | 13.3% | 8.0% | **0.4656** | **0.5177** | 2037.06 ms |

### Secondary Comparison: Reciprocal Rank Fusion (RRF k=60)

| Retrieval Condition | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Original Query (RRF)** | 73.3% | 83.3% | 83.3% | 86.7% | 70.0% | 81.7% | 85.0% | **0.7815** | **0.7954** | 51.15 ms |
| **B. LLM-Rewritten (RRF)** | 56.7% | 66.7% | 73.3% | 80.0% | 53.3% | 70.0% | 78.3% | **0.6295** | **0.6579** | 2037.06 ms |

---

## 3. Category-Wise Performance Breakdown

### Category: `conceptual_paraphrased` (6 Queries)

| Condition | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 33.3% | 33.3% | 50.0% | 33.3% | 33.3% | 50.0% | **0.3519** | **0.3835** | 51.73 ms |
| **Rewritten** | 0.0% | 16.7% | 16.7% | 0.0% | 16.7% | 16.7% | **0.0833** | **0.1052** | 1180.67 ms |

### Category: `crop_specific` (6 Queries)

| Condition | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 66.7% | 66.7% | 100.0% | 58.3% | 66.7% | 100.0% | **0.7111** | **0.7742** | 50.66 ms |
| **Rewritten** | 33.3% | 50.0% | 83.3% | 25.0% | 50.0% | 83.3% | **0.4405** | **0.5316** | 1298.61 ms |

### Category: `diagnostic` (6 Queries)

| Condition | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 66.7% | 83.3% | 83.3% | 66.7% | 83.3% | 83.3% | **0.7500** | **0.7718** | 58.71 ms |
| **Rewritten** | 33.3% | 66.7% | 83.3% | 33.3% | 66.7% | 83.3% | **0.4821** | **0.5658** | 2715.40 ms |

### Category: `exact_lexical` (6 Queries)

| Condition | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 50.0% | 83.3% | 83.3% | 41.7% | 83.3% | 83.3% | **0.6389** | **0.7003** | 43.43 ms |
| **Rewritten** | 50.0% | 83.3% | 100.0% | 41.7% | 75.0% | 91.7% | **0.6833** | **0.6973** | 1174.16 ms |

### Category: `procedural_recommendation` (6 Queries)

| Condition | Hit@1 | Hit@5 | Hit@10 | Recall@1 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original** | 83.3% | 100.0% | 100.0% | 83.3% | 91.7% | 91.7% | **0.9167** | **0.8978** | 51.20 ms |
| **Rewritten** | 50.0% | 83.3% | 83.3% | 50.0% | 83.3% | 83.3% | **0.6389** | **0.6885** | 3816.44 ms |

---

## 4. Special Focus Queries: Q10, Q12, and Q17

### Query `Q10`: What is the mechanism and role of Azospirillum biofertilizers in enhancing root development and nutrient uptake?
- **Rewritten Query**: `Azospirillum biofertilizer mechanism root development nutrient uptake plant growth‑promoting rhizobacteria nitrogen fixation phytohormone production indole‑3‑acetic acid gibberellins cytokinin root elongation lateral root formation colonization rhizosphere nutrient acquisition N P K soil fertility Azospirillum brasilense Azospirillum lipoferum bioinoculant plant‑microbe interaction`
- **Rewrite Latency**: 1405.63 ms (Total: 1508.45 ms)
- **Target Ground Truth**: `['chunk_001999']`
- **Dense Rank**: Before = **>30** $\rightarrow$ After = **>30**
- **BM25 Rank**: Before = **>30** $\rightarrow$ After = **>30**
- **In Top-30 Pool**: Before = **False** $\rightarrow$ After = **False**
- **Final Equal Fusion Rank**: Before = **>10** $\rightarrow$ After = **>10** (Outcome: **UNCHANGED**)

### Query `Q12`: How does timely weed management during the critical crop growth period influence nutrient and moisture competition?
- **Rewritten Query**: `timely weed management critical period of weed control crop growth stage nutrient uptake competition moisture competition water uptake weed‑crop competition early‑season weed control integrated weed management nitrogen phosphorus potassium soil moisture`
- **Rewrite Latency**: 1027.13 ms (Total: 1113.00 ms)
- **Target Ground Truth**: `['chunk_003231']`
- **Dense Rank**: Before = **>30** $\rightarrow$ After = **>30**
- **BM25 Rank**: Before = **>30** $\rightarrow$ After = **>30**
- **In Top-30 Pool**: Before = **False** $\rightarrow$ After = **False**
- **Final Equal Fusion Rank**: Before = **>10** $\rightarrow$ After = **>10** (Outcome: **UNCHANGED**)

### Query `Q17`: Which bacterial wilt resistant varieties of tomato are recommended by Kerala Agricultural University?
- **Rewritten Query**: `bacterial wilt resistant tomato varieties recommended by Kerala Agricultural University Solanum lycopersicum cultivars Ralstonia solanacearum resistance KAU field trials cultivars disease management`
- **Rewrite Latency**: 991.76 ms (Total: 1057.04 ms)
- **Target Ground Truth**: `['chunk_003772']`
- **Dense Rank**: Before = **10** $\rightarrow$ After = **12**
- **BM25 Rank**: Before = **19** $\rightarrow$ After = **>30**
- **In Top-30 Pool**: Before = **True** $\rightarrow$ After = **True**
- **Final Equal Fusion Rank**: Before = **10** $\rightarrow$ After = **>10** (Outcome: **HARMED**)

---

## 5. Failure / Success Table Across All 30 Queries

| QID | Category | Outcome | Dense Rank (Orig $\rightarrow$ Rew) | BM25 Rank (Orig $\rightarrow$ Rew) | In Pool? | Fusion Rank (Orig $\rightarrow$ Rew) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Q01** | `exact_lexical` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #4 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q02** | `exact_lexical` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q03** | `exact_lexical` | **IMPROVED** | #19 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #>10 $\rightarrow$ #1 |
| **Q04** | `exact_lexical` | **HARMED** | #2 $\rightarrow$ #10 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #2 $\rightarrow$ #10 |
| **Q05** | `exact_lexical` | **IMPROVED** | #3 $\rightarrow$ #2 | #2 $\rightarrow$ #13 | True $\rightarrow$ True | #3 $\rightarrow$ #2 |
| **Q06** | `exact_lexical` | **HARMED** | #1 $\rightarrow$ #2 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #2 |
| **Q07** | `conceptual_paraphrased` | **HARMED** | #1 $\rightarrow$ #2 | #1 $\rightarrow$ #8 | True $\rightarrow$ True | #1 $\rightarrow$ #2 |
| **Q08** | `conceptual_paraphrased` | **UNCHANGED** | #>30 $\rightarrow$ #>30 | #6 $\rightarrow$ #2 | True $\rightarrow$ True | #>10 $\rightarrow$ #>10 |
| **Q09** | `conceptual_paraphrased` | **HARMED** | #1 $\rightarrow$ #13 | #1 $\rightarrow$ #>30 | True $\rightarrow$ True | #1 $\rightarrow$ #>10 |
| **Q10** | `conceptual_paraphrased` | **UNCHANGED** | #>30 $\rightarrow$ #>30 | #>30 $\rightarrow$ #>30 | False $\rightarrow$ False | #>10 $\rightarrow$ #>10 |
| **Q11** | `conceptual_paraphrased` | **HARMED** | #9 $\rightarrow$ #>30 | #1 $\rightarrow$ #6 | True $\rightarrow$ True | #9 $\rightarrow$ #>10 |
| **Q12** | `conceptual_paraphrased` | **UNCHANGED** | #>30 $\rightarrow$ #>30 | #>30 $\rightarrow$ #>30 | False $\rightarrow$ False | #>10 $\rightarrow$ #>10 |
| **Q13** | `crop_specific` | **HARMED** | #1 $\rightarrow$ #3 | #7 $\rightarrow$ #21 | True $\rightarrow$ True | #1 $\rightarrow$ #3 |
| **Q14** | `crop_specific` | **HARMED** | #1 $\rightarrow$ #6 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #6 |
| **Q15** | `crop_specific` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q16** | `crop_specific` | **HARMED** | #6 $\rightarrow$ #7 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #6 $\rightarrow$ #7 |
| **Q17** | `crop_specific` | **HARMED** | #10 $\rightarrow$ #12 | #19 $\rightarrow$ #>30 | True $\rightarrow$ True | #10 $\rightarrow$ #>10 |
| **Q18** | `crop_specific` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q19** | `diagnostic` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q20** | `diagnostic` | **UNCHANGED** | #21 $\rightarrow$ #20 | #5 $\rightarrow$ #7 | True $\rightarrow$ True | #>10 $\rightarrow$ #>10 |
| **Q21** | `diagnostic` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #2 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q22** | `diagnostic` | **HARMED** | #1 $\rightarrow$ #7 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #7 |
| **Q23** | `diagnostic` | **HARMED** | #1 $\rightarrow$ #2 | #1 $\rightarrow$ #4 | True $\rightarrow$ True | #1 $\rightarrow$ #2 |
| **Q24** | `diagnostic` | **HARMED** | #2 $\rightarrow$ #4 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #2 $\rightarrow$ #4 |
| **Q25** | `procedural_recommendation` | **HARMED** | #1 $\rightarrow$ #2 | #1 $\rightarrow$ #3 | True $\rightarrow$ True | #1 $\rightarrow$ #2 |
| **Q26** | `procedural_recommendation` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q27** | `procedural_recommendation` | **HARMED** | #2 $\rightarrow$ #15 | #4 $\rightarrow$ #1 | True $\rightarrow$ True | #2 $\rightarrow$ #>10 |
| **Q28** | `procedural_recommendation` | **HARMED** | #1 $\rightarrow$ #3 | #4 $\rightarrow$ #13 | True $\rightarrow$ True | #1 $\rightarrow$ #3 |
| **Q29** | `procedural_recommendation` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #1 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |
| **Q30** | `procedural_recommendation` | **UNCHANGED** | #1 $\rightarrow$ #1 | #1 $\rightarrow$ #3 | True $\rightarrow$ True | #1 $\rightarrow$ #1 |

---

## 6. Detailed Analysis of Lexical and Semantic Drift

### Where Query Rewriting Succeeded (Positive Shifts):
- **Q03** (`exact_lexical`):
  - *Original*: "What is the recommended tine spacing for Happy Seeder and Coragen 18.5 SC dose in wheat?" (Rank #>10)
  - *Rewritten*: "Happy Seeder wheat tine spacing recommended settings and Coragen 18.5 SC (chlorantraniliprole) dose for wheat application rate agronomic guidelines seed placement equipment direct seeder dosage wheat pest management." (Rank #1)
  - *Mechanism*: Added exact biological terms or key crop entities that aligned with document headers.
- **Q05** (`exact_lexical`):
  - *Original*: "What are the statutory restrictions on the use of Aluminium phosphide and Fenitrothion in India?" (Rank #3)
  - *Rewritten*: "statutory restrictions India aluminium phosphide (AlP) fumigant pesticide regulations Insecticides Act 1968 Insecticides (Regulation) Rules 2002 pesticide registration limits label directions maximum residue limits fenitrothion organophosphate insecticide Sumithion use restrictions stored grain application dosage bans state guidelines Central Insecticides Board environmental safety" (Rank #2)
  - *Mechanism*: Added exact biological terms or key crop entities that aligned with document headers.

### Where Query Rewriting Caused Harm (Semantic/Lexical Drift):
- **Q04** (`exact_lexical`):
  - *Original*: "What is the application rate of talc-based formulation of Trichoderma viride for seed treatment?" (Rank #2)
  - *Rewritten*: "application rate talc-based formulation Trichoderma viride seed treatment dosage grams per kilogram seed CFU per seed seed coating biocontrol seed dressing" (Rank #10)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q06** (`exact_lexical`):
  - *Original*: "Which pathogen causes linseed rust characterized by pink fungal sori on leaves and stems?" (Rank #1)
  - *Rewritten*: "linseed flax rust pink fungal sori on leaves and stems pathogen Puccinia lini Linum usitatissimum rust disease" (Rank #2)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q07** (`conceptual_paraphrased`):
  - *Original*: "How does applying organic straw mulching help conserve soil moisture in rainfed agriculture?" (Rank #1)
  - *Rewritten*: "organic straw mulching soil moisture conservation rainfed agriculture dryland farming water retention evapotranspiration reduction soil temperature moderation residue mulch effects soil water holding capacity infiltration hydraulic conductivity agronomic practice" (Rank #2)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q09** (`conceptual_paraphrased`):
  - *Original*: "How do green manure crops like cowpea, dhaincha, or sunnhemp contribute to organic nitrogen enrichment in rice?" (Rank #1)
  - *Rewritten*: "green manure legume cover crops cowpea Vigna unguiculata dhaincha Sesbania rostrata sunn hemp Crotalaria juncea nitrogen fixation organic nitrogen enrichment soil nitrogen mineralization biomass incorporation rice paddy Oryza sativa" (Rank #>10)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q11** (`conceptual_paraphrased`):
  - *Original*: "Why is mechanical destruction of congregating insect stages recommended in sustainable pest management?" (Rank #9)
  - *Rewritten*: "mechanical destruction of insect aggregations (egg masses, larval clusters, pupae) sustainable pest management integrated pest management cultural control physical control mechanical control non‑chemical methods why recommended" (Rank #>10)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q13** (`crop_specific`):
  - *Original*: "What is the seed rate requirement for normal vs hybrid varieties in tomato, chilli, and brinjal?" (Rank #1)
  - *Rewritten*: "seed rate requirement tomato hybrid open-pollinated normal varieties chilli hybrid open-pollinated normal varieties brinjal hybrid open-pollinated normal varieties seed quantity per hectare seed sowing density Solanum lycopersicum Capsicum annuum Solanum melongena vegetable seed rate guidelines agronomic recommendations" (Rank #3)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q14** (`crop_specific`):
  - *Original*: "Why are older arabica coffee varieties like Kents and Coorgs considered vulnerable to leaf rust disease?" (Rank #1)
  - *Rewritten*: "Coffea arabica older cultivars Kent Coorg vulnerability to leaf rust Hemileia vastatrix susceptibility factors genetic resistance deficiency disease epidemiology coffee breeding heritage varieties" (Rank #6)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q16** (`crop_specific`):
  - *Original*: "How is irrigation scheduled in sugarcane based on climatological pan evaporation approach?" (Rank #6)
  - *Rewritten*: "irrigation scheduling sugarcane climatological pan evaporation method reference evapotranspiration (ET0) crop coefficient (Kc) pan coefficient water requirement climate data evapotranspiration based irrigation management sugarcane water balance" (Rank #7)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q17** (`crop_specific`):
  - *Original*: "Which bacterial wilt resistant varieties of tomato are recommended by Kerala Agricultural University?" (Rank #10)
  - *Rewritten*: "bacterial wilt resistant tomato varieties recommended by Kerala Agricultural University Solanum lycopersicum cultivars Ralstonia solanacearum resistance KAU field trials cultivars disease management" (Rank #>10)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q22** (`diagnostic`):
  - *Original*: "What causes shot-holes in leaves, midrib streaks, and bunchy tops with a dried dark central leaf in sugarcane?" (Rank #1)
  - *Rewritten*: "sugarcane leaf shot‑holes midrib streaks bunchy tops dried dark central leaf symptoms disease diagnosis fungal leaf spot bacterial leaf scald Xanthomonas albilineans leaf blotch Fusarium wilt nutrient deficiency abiotic stress agronomic scouting Saccharum officinarum leaf necrosis chlorosis" (Rank #7)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q23** (`diagnostic`):
  - *Original*: "What disease is characterized by small light-green lesions on the underside of leaves that enlarge and turn greyish in brassica crops?" (Rank #1)
  - *Rewritten*: "disease leaf spot small light‑green lesions on underside of leaves enlarge to greyish Brassica crops cabbage broccoli cauliflower oilseed rape chlorotic leaf lesions leaf necrosis symptoms" (Rank #2)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q24** (`diagnostic`):
  - *Original*: "What pest problem is identified by observed chewed wood and resin exudation at the base of a cashew trunk?" (Rank #2)
  - *Rewritten*: "cashew trunk base chewed wood resin exudation pest identification wood‑boring insects termite bark beetle stem borer Anacardium occidentale damage symptoms" (Rank #4)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q25** (`procedural_recommendation`):
  - *Original*: "What are the bed dimensions and structural specifications for preparing a vermicompost production unit?" (Rank #1)
  - *Rewritten*: "vermicompost production unit design bed dimensions structural specifications worm bin size length width height depth capacity substrate bedding material aeration drainage mesh frame material wooden PVC Eisenia fetida vermiculture bed guidelines organic waste composting scale small‑scale industrial" (Rank #2)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q27** (`procedural_recommendation`):
  - *Original*: "How should coffee nursery seedlings be transplanted to prevent taproot deformation and damping-off disease?" (Rank #2)
  - *Rewritten*: "coffee nursery seedling transplant technique taproot deformation prevention damping-off disease Coffea arabica hardening off root ball depth soil moisture management Pythium rhizoctonia control transplant shock agronomic practices seedling spacing substrate aeration" (Rank #>10)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.
- **Q28** (`procedural_recommendation`):
  - *Original*: "What organic manures and coir pith application practices are recommended for maintaining coconut palms?" (Rank #1)
  - *Rewritten*: "organic manures coconut palm Cocos nucifera coir pith application practices nutrient management soil fertility compost vermicompost farmyard manure green manure biofertilizer dosage rates timing incorporation mulching surface application drip irrigation organic amendment sustainable coconut plantation recommendations" (Rank #3)
  - *Cause of Drift*: Adding expansive vocabulary diluted BM25 IDF scores or caused Dense embeddings to drift toward generic textbook overviews.

---

## 7. Latency and Operational Overhead

- **Mean LLM Query Rewriting Latency**: **1955.55 ms**
- **Mean First-Stage Retrieval Latency (Dense + BM25)**: **~45 ms**
- **Total End-to-End Latency With Rewriter**: **2037.06 ms** (a ~39.8x latency increase)

---

## 8. Strategic Recommendations for Production

### 1. Does query rewriting improve first-stage retrieval?
**NO. It significantly degrades first-stage retrieval across almost every metric:**
- **Hit@1 collapses from 60.0% to 33.3%** (-26.7% absolute drop).
- **Hit@10 drops from 83.3% to 73.3%** (-10.0% absolute drop).
- **MRR@10 degrades from 0.6737 to 0.4656** (-0.2081 drop).
- **NDCG@10 falls from 0.7055 to 0.5177** (-0.1878 drop).
- **Candidate-pool recall drops from 93.3% to 91.7%**.
- Across the 30 queries, **15 queries were actively harmed**, only **2 queries improved**, and **13 queries remained unchanged**.

### 2. Does it recover Q10 or Q12?
**NO. Neither Q10 nor Q12 was recovered:**
- **Q10 (*Azospirillum*)**: Ground-truth `chunk_001999` remained completely outside the Top-30 in both Dense (>30) and BM25 (>30), remaining unretrieved (>10) in fusion.
- **Q12 (*Weed Competition*)**: Ground-truth `chunk_003231` remained completely outside the Top-30 in both Dense (>30) and BM25 (>30), remaining unretrieved (>10) in fusion.
- Query rewriting generated verbose paragraphs of agronomic vocabulary that diffused the dense embedding vectors across hundreds of general textbook chapters rather than targeting the exact passage.

### 3. Is the improvement large enough to justify its latency?
**NO. There is zero overall improvement, while latency increases by ~40x:**
- Baseline retrieval latency (Dense + BM25) is **~51 ms**.
- Mean LLM rewriting latency adds **1,955.55 ms**, pushing total first-stage latency to **2,037.06 ms**.
- Paying a ~2-second latency penalty to receive a 26.7% drop in Top-1 accuracy is unequivocally unacceptable.

### 4. Why did LLM query rewriting fail? (Mechanism of Degradation)
1. **BM25 Keyword Dilution (IDF Degradation)**: When the LLM adds dozens of related terms (e.g. adding 15 generic agronomic words to Q09 or Q28), BM25's inverse document frequency is diluted across common terms like *cultivars*, *management*, *application*, *guidelines*, and *agronomic*. As a result, exact rare matches (like specific crop names or table headers) lose their discriminating rank.
2. **Dense Bi-Encoder Embedding Dispersion**: Sentence-Transformers like `all-MiniLM-L6-v2` encode input texts into a single 384-dimensional vector. When an input query grows from 15 tokens to 60 tokens covering 8 different subtopics, the embedding vector shifts toward the centroid of the vector space (generic agronomy), losing similarity to specific granular chunks.

### 5. Should it proceed to a production integration experiment?
**ABSOLUTELY NOT. Do NOT deploy LLM query rewriting to production retrieval.**
- First-stage retrieval in this architecture must remain concise, focused, and fast.
- The proper architectural solution for hard conceptual queries like Q10 and Q12 is:
  1. **Candidate Pool Enlargement (Top-30/Top-50)** (which was already evaluated and confirmed to recover Q20 to Rank 8 without hurting any other queries).
  2. **Cross-Encoder Re-ranking** (e.g., `bge-reranker-base` or `ms-marco-MiniLM-L-6-v2`) applied after first-stage retrieval over the candidate pool, which can cross-attend between the user's query and candidate chunk texts without modifying the query.
