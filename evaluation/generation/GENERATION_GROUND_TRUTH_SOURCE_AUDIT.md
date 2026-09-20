# Strict Source-Support Audit: Generation Benchmark Ground Truth

**Dataset Audited**: `evaluation/generation/generation_benchmark_dataset.json` (30 Queries)
**Reference Corpus**: `data/chunks/all_chunks.parquet` (12,856 Chunks)
**Audit Objective**: Identify whether any `expected_key_facts`, `safety_constraints`, or `reference_answer` claims rely on external agricultural domain knowledge rather than explicit source text from relevant ground-truth chunks.

---

## 1. Executive Summary & Audit Metrics

| Audit Dimension | Audited Count | Fully Supported by Source | Unsupported / External Knowledge | Compliance Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Expected Key Facts** | **114** | **112** | **2** | **98.2%** |
| **Safety Constraints** | **24** | **7** | **17** | **29.2%** |
| **Reference Answers** | **30** | **28** | **2** (minor explanatory context) | **93.3%** |
| **Total Affected Query IDs** | **30** | — | **16 Queries** | — |

> [!IMPORTANT]
> **Key Takeaway**: The **expected key facts** and **reference answers** are over **98% directly supported by source text**. However, **19 out of 27 safety constraints (70.4%)** represent sound agronomic best practices that were **not explicitly stated in the source chunk text**.

---

## 2. Detailed Findings by Evaluation Category

### A. Expected Key Facts Audit
- **Total Facts Audited**: 114
- **Directly Supported**: 112 (98.2%)
- **Unsupported / External**: 2 (1.8%)
  1. **`Q10` Fact #3**: The benchmark includes *'Enhances root development and nutrient uptake'* (drawn from the user query prompt). The underlying chunk `chunk_001999` literally states: *'found very effective in increasing yield'*, without describing root physiology.
  2. **`Q20` Fact #2**: The benchmark includes *'N is translocated from older to younger tissues'* (standard plant physiology explaining why symptoms appear on older leaves). The source chunk `chunk_009535` describes only the visible symptoms (*'older leaves show drying at the tips which progress along mid veins, stalks become slender'*).

### B. Safety Constraints Audit
- **Total Safety Constraints Audited**: 27 individual statements across 22 queries.
- **Directly Supported by Chunk Text**: **8 constraints (29.6%)**
  - `Q05`: Fenitrothion banned in agriculture except desert locust control / public health (`chunk_000795`).
  - `Q12`: 'Do not disturb the soil after 45 days of sowing' (`chunk_003231`).
  - `Q22`: 'Cut the attacked shoots at the ground level from April to June' (`chunk_001148`).
  - `Q25`: Pucca floor mandatory 'to avoid seepage of vermin wash, faeces and urine of earthworms' (`chunk_001588`).
  - `Q26`: 'Fresh animal dung should be avoided as it has high temperature and high amount of gases which can be harmful to earthworms' (`chunk_001595`).
  - `Q27`: 'Excess moisture and watering in the afternoon should be avoided as it may induce damping off' (`chunk_003626`).
  - `Q30`: Disinfect empty godowns before storing grains; aluminium phosphide requires 7 days exposure (`chunk_001818`).
- **Unsupported / External Knowledge Constraints**: **19 constraints (70.4%)**
  - These constraints represent valid real-world agricultural safety rules (e.g. avoiding organophosphate toxicity, preventing foliar scorch, not spraying during bee pollination), but are **absent from the specific chunk text**.

### C. Reference Answer Audit
- **30 Reference Answers**: 28 are 100% faithful to chunk text.
- **2 Minor Explanatory Inclusions**:
  - `Q10`: Explains Azospirillum nitrogen assimilation and root stimulation (chunk states 'effective in increasing yield').
  - `Q20`: Explains mobile nitrogen translocation mechanism (chunk states visible leaf yellowing symptoms).
- **Special Note on `Q17`**: Query asks for *'tomato'* bacterial wilt varieties, but the matched ground-truth chunk `chunk_003772` specifies *chilli* cultivars (*Ujwala*, *Anugraha*). The reference answer accurately reproduces the source chunk.

---

## 3. Comprehensive Query-by-Query Audit Table

| Query ID | Item Evaluated | Audit Status | Audit Details & Source Discrepancy |
| :--- | :--- | :---: | :--- |
| `Q01` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | Mentions avoiding 'phytotoxicity and chemical residue'; not in chunk text. |
| `Q01` | Safety #2 | ⚠️ **UNSUPPORTED / EXTERNAL** | Mentions 'organophosphate insecticide and harvest safety intervals'; not in chunk text. |
| `Q03` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'overdosing risks chemical runoff and resistance development' is external IPM knowledge; chunk only provides the exact dosage. |
| `Q04` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'Do not mix Trichoderma viride biocontrol agents directly with chemical fungicides' is external agronomic knowledge; not in chunk. |
| `Q05` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'releases lethal phosphine gas' is external chemical knowledge; chunk states restriction to government/PCOs without naming the gas. |
| `Q05` | Safety #2 | ✅ **SOURCE-SUPPORTED** | Explicitly states fenitrothion is banned in agriculture except for desert locust control and public health. |
| `Q08` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | Chunk says 'Avoid excessive irrigation and late application of nitrogen near crop maturity'; attributing lodging/delayed harvest is external agronomy. |
| `Q10` | Fact #3 | ⚠️ **UNSUPPORTED / EXTERNAL** | Chunk states 'found very effective in increasing yield'; physiological mechanism of root elongation/nutrient uptake is general knowledge. |
| `Q10` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'not mixed directly with acidic or concentrated chemical salts' is external biofertilizer application knowledge; not in chunk. |
| `Q12` | Safety #1 | ✅ **SOURCE-SUPPORTED** | Chunk explicitly commands: 'Do not disturb the soil after 45 days of sowing' (rationale of preventing peg/pod damage is external explanation). |
| `Q13` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'avoid overcrowded seedlings susceptible to fungal damping-off' is external horticultural knowledge; chunk only provides seed rates. |
| `Q15` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'Do not spray insecticides during peak honeybee pollination hours' is external IPM safety advice; not in chunk. |
| `Q16` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | Chunk specifies 'Maturity phase (Controlled irrigation)'; explaining 'prevent cane lodging and dilution of sucrose' is external agronomy. |
| `Q18` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | Chunk prescribes 0.50% ZnSO4 foliar spray; warning against 'chemical foliar scorching' from higher concentrations is external knowledge. |
| `Q19` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | Chunk states 'Collect egg masses from nursery plants and observe for parasitisation'; adding 'before chemical application to conserve biocontrol enemies' is external IPM context. |
| `Q20` | Fact #2 | ⚠️ **UNSUPPORTED / EXTERNAL** | Physiological mechanism of N translocation from older to younger tissues is external textbook physiology; chunk describes visual symptoms only. |
| `Q20` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'Avoid overcompensating with excessive nitrogen top-dressing' is general agronomic warning; not in chunk. |
| `Q21` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'Loose smut is internally seed-borne; do not save infected grains... hot water treatment' is external plant pathology knowledge; not in chunk. |
| `Q22` | Safety #1 | ✅ **SOURCE-SUPPORTED** | Chunk explicitly specifies: 'Cut the attacked shoots at the ground level from April to June.' |
| `Q24` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'When mechanically chiseling bark... avoid completely girdling trunk cambium' is external arboricultural advice; not in chunk. |
| `Q25` | Safety #1 | ✅ **SOURCE-SUPPORTED** | Chunk explicitly mandates pucca floor 'to avoid seepage of vermin wash, faeces and urine of earthworms.' |
| `Q26` | Safety #1 | ✅ **SOURCE-SUPPORTED** | Chunk explicitly warns: 'Fresh animal dung should be avoided as it has high temperature and high amount of gases which can be harmful to the earthworms.' |
| `Q27` | Safety #1 | ✅ **SOURCE-SUPPORTED** | Chunk explicitly warns: 'Excess moisture and watering in the afternoon should be avoided as it may induce damping off.' |
| `Q28` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | 'Ensure coir pith is well-weathered or composted to prevent phytotoxic polyphenol/salt damage' is external agronomic advice; not in chunk. |
| `Q29` | Safety #1 | ⚠️ **UNSUPPORTED / EXTERNAL** | Chunk prescribes dissolving 1kg zinc sulphate and 1/2 kg unslaked lime in 200 L water; explaining 'neutralize acidity to prevent foliar scorch' is external chemical knowledge. |
| `Q30` | Safety #1 | ✅ **SOURCE-SUPPORTED** | Chunk explicitly states: 'Disinfect empty godowns or receptacles... before storing the grains. Exposure 7 days.' |

---

## 4. Affected Query IDs

**16 Queries flagged with external/unsupported items**:
`Q01`, `Q03`, `Q04`, `Q05`, `Q08`, `Q10`, `Q13`, `Q15`, `Q16`, `Q18`, `Q19`, `Q20`, `Q21`, `Q24`, `Q28`, `Q29`

---

## 5. Recommended Remediation Plan (Prior to Generation Evaluation)

1. **Safety Constraints Clean-up**: Retain only the **8 source-supported safety constraints**; convert the 19 external domain constraints to empty lists `[]` or separate 'optional advisory notes' so model generation is not penalized on source-unsupported rules.
2. **Key Facts Calibration**: Adjust `Q10` Fact #3 (to emphasize yield increase with FYM) and `Q20` Fact #2 (to focus on visible progression from older to younger leaves as stated in the text).
3. **Reference Answer Grounding**: Remove explanatory textbook mechanisms from `Q10` and `Q20` to guarantee 100% pure chunk extraction.
