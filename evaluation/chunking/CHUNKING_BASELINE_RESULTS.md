# Chunking Stage Baseline Evaluation Report

## 1. Objective

This report provides an isolated, reproducible baseline quality evaluation of the chunk corpus currently stored in `data/chunks/all_chunks.parquet`. 

The purpose is to quantitatively assess the output of the HDR (Hybrid Decomposition & Recombination) chunking stage (`step3_chunk.py`), establish baseline length distributions, detect structural and boundary anomalies, quantify duplicates and noise patterns, and identify candidates for manual inspection prior to any chunking pipeline adjustments.

> [!IMPORTANT]
> **Evaluation Scope Only**: No production pipeline files, chunk stores, embeddings, vector databases, or retrieval components were modified or regenerated during this analysis. All detections are heuristic indicators intended to guide manual verification.

---

## 2. Methodology & Configuration

The evaluation reuses the project's established chunking constants defined in `config.py`:
- **`MIN_CHUNK_SIZE`**: `50` characters (discard threshold for tiny fragments)
- **`MAX_CHUNK_SIZE`**: `500` characters (target maximum chunk character limit)
- **`NSP_THRESHOLD`**: `0.85` (BERT Next Sentence Prediction boundary threshold)
- **Storage**: Parquet engine loading `data/chunks/all_chunks.parquet`

### Heuristic Detection Categories
1. **Size Boundary Checks**:
   - Chunks below 100 characters (very small / near minimum threshold).
   - Chunks above `500` characters (exceeding configured target).
   - Chunks above 1,000 characters (extreme sentence boundary bypass).
2. **Boundary Completeness Checks**:
   - **Incomplete Endings**: Chunks ending with suspicious punctuation (commas, semicolons, hyphens, colons, opening brackets, ellipsis) or unpunctuated lowercase words.
   - **Continuation Starts**: Chunks beginning with lowercase characters, dangling closing brackets/punctuation, or coordinating conjunctions (`and`, `or`, `but`, `because`, etc.).
3. **Redundancy & Noise Checks**:
   - **Exact Duplicates**: Chunks sharing identical normalized text strings across documents or repetitive tables.
   - **Boilerplate Noise**: Recurring running headers, institutional signatures, or regulatory notices.
4. **Metadata & Tag Consistency**:
   - Substring collision detection in `extract_crop_tags()` (e.g. `"tea"` matched inside `"tea mosquito bug"`).
   - Mismatch between single-crop source titles and tagged entities.

---

## 3. Dataset Summary & Length Distribution Metrics

| Metric | Value |
| :--- | :--- |
| **Total Chunks in Corpus** | **14,766** |
| **Minimum Character Length** | **50** chars |
| **Maximum Character Length** | **14,916** chars |
| **Mean Character Length** | **401.67** chars |
| **Median Character Length** | **420.00** chars |
| **Standard Deviation** | **252.54** chars |

### Size Distribution Breakdown

| Chunk Size Bracket | Threshold | Count | Percentage of Corpus |
| :--- | :--- | :---: | :---: |
| **Very Small Chunks** | `< 100` characters | **418** | **2.83%** |
| **Target Size Compliance** | `<= 500` characters | **13,726** | **92.96%** |
| **Above Configured Max** | `> 500` characters | **1,040** | **7.04%** |
| **Extremely Large Chunks** | `> 1,000` characters | **129** | **0.87%** |

---

## 4. Heuristic Anomaly Detections

A total of **8,313** potential issues were flagged across the corpus and recorded in [`chunking_baseline_results.csv`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/chunking/chunking_baseline_results.csv).

| Category | Flagged Count | % of Corpus | Heuristic Description |
| :--- | :---: | :---: | :--- |
| **`POTENTIAL_EXACT_DUPLICATE`** | **3,596** | **24.35%** | Chunks sharing identical text (1,774 unique repeated text strings). |
| **`POTENTIAL_INCOMPLETE_ENDING`** | **1,289** | **8.73%** | Chunks ending in trailing punctuation (`,`, `;`, `-`, `:`) or unpunctuated lowercase words. |
| **`POTENTIAL_BOILERPLATE_NOISE`** | **1,117** | **7.56%** | Matches recurring document headers, advisory signatures, or statutory tables. |
| **`EXCEEDS_CONFIGURED_MAX_CHUNK_SIZE`** | **911** | **6.17%** | Chunks exceeding `500` chars (excluding extreme >1000 chars). |
| **`POTENTIAL_CONTINUATION_START`** | **819** | **5.55%** | Chunks beginning with lowercase characters, closing brackets, or conjunctions. |
| **`POTENTIAL_TINY_FRAGMENT`** | **418** | **2.83%** | Chunks under 100 characters, prone to lacking self-contained context. |
| **`POTENTIAL_OVERSIZED_CHUNK`** | **129** | **0.87%** | Massive chunks >1,000 characters caused by unpunctuated text bypassing spaCy. |
| **`POTENTIAL_METADATA_INCONSISTENCY`** | **34** | **0.23%** | Substring collisions (e.g. `"tea"` inside `"tea mosquito bug"`) or missing expected tags. |

---

## 5. Top Suspicious Chunks for Manual Inspection

Below are landmark failure cases and high-priority suspicious chunks identified for manual inspection:

| Chunk ID | Source Document | Length | Reason Flagged | Text Sample |
| :--- | :--- | :---: | :--- | :--- |
| `chunk_012110` | `Rabi-Agro-Advisory-2021-22_0.pdf` | 14,916 chars | Trace diagnostic landmark: Extreme oversized 14,916-char chunk with ICAR boilerplate | `Enterotoximia (Multivalent) ??????? | ??: ?????-? | ???????: 2ml S/C ICAR RABI AgRo-AdvIsoRy foR fARmeRs ICAR RABI AgRo-...` |
| `chunk_000317` | `ANGRAU Journal of Research.pdf` | 5,447 chars | Trace diagnostic landmark: Reversed/mirrored academic table (5,447 chars) | `xirB?(SST | atad delooP: atad delooP | T 3: T 3 | 21.0: 75.3 | 51.0: 04.3 | 71.0: 54.3 | 91.0: 53.3 | 61.0: 30.3 | 51.0:...` |
| `chunk_001683` | `Crop Protection.pdf` | 477 chars | Trace diagnostic landmark: Reversed seed certification glyph stream | `wedliM: 05.0 tums esooL: 01.0 | tums esooL: 01.0 | ynwoD wedliM: 50.0 skrameR: elbanoitcejbO )mumixam( sdeew % | tums es...` |
| `chunk_001743` | `Crop Protection.pdf` | 424 chars | Trace diagnostic landmark: Interleaved table columns and dropped specs | `e) Moong Spike tooth 7.5 hp 2.5 q/h Spike tooth thresher for wheat can Thresher thresher electric be used with following...` |
| `chunk_003222` | `KAU (Kerala Agricultural University).pdf` | 50 chars | Trace diagnostic landmark: Tiny 50-char orphan boundary fragment | `Transplanting is done regulate moisture retention.` |
| `chunk_007588` | `The TNAU Horticulture PDF.pdf` | 427 chars | Trace diagnostic landmark: False-positive 'tea' crop tag from cashew pest ('Tea Mosquito Bug') | `Mechanical Mechanical destruction of sluggish and congregating stages of the insect-pest are advisable. Tea Mosquito Bug...` |
| `chunk_001858` | `Crop Protection.pdf` | 298 chars | Trace diagnostic landmark: Non-agricultural administrative staff directory | `Dr. Amit Kaul, Agronomy 401 81464-00233 Dr. R K Gupta, Soil Science 506 81462-00940 Dr. Kulbir Singh, Vegetable Science ...` |
| `chunk_012614` | `list_of_pesticides_which_are_banned_refused_registration_and_restricted_in_use.pdf` | 51 chars | High repetition: identical text duplicated 7 times across the corpus | `Pesticides Banned for manufacture, import and use.:` |

---

## 6. Interpretation of Results

1. **Sentence Boundary Guard Limitations**:
   - While `MAX_CHUNK_SIZE = 500` is configured, **1,040 chunks (7.04%)** exceed this limit, with **129 chunks** exceeding 1,000 characters (maximum reaching **14,916 characters** in `chunk_012110`).
   - *Cause*: In `step3_chunk.py`, sentence splitting relies on spaCy. If extracted text lacks terminal punctuation (e.g. repeated advisory bullets or OCR noise), spaCy produces a single gigantic "sentence". The pipeline check `(current_len + len(sent)) > MAX_CHUNK_SIZE` only splits *between* sentences, never *within* an oversized sentence.
2. **High Corpus Redundancy**:
   - **3,596 chunks (24.35%)** are exact duplicates across 1,774 distinct text strings. Common sources include recurring statutory tables (e.g., banned pesticides repeated across state manuals) and uncleaned advisory headers.
3. **Boundary Incompleteness**:
   - **1,289 chunks (8.73%)** end with dangling punctuation, and **819 chunks (5.55%)** start with continuation lowercase words. This suggests cross-boundary semantic fragmentation during sentence segmentation.
4. **Metadata Tag False Positives**:
   - Simple lowercase substring matching in `extract_crop_tags()` triggers false-positive agricultural tags (e.g., tagging cashew recommendations as `tea` due to `"tea mosquito bug"`).

---

## 7. Limitations of this Baseline Evaluation

1. **Heuristic Nature of Detections**:
   - Flagged chunks represent potential defects identified by syntactic rules; a chunk ending with `:` or starting with lowercase may occasionally be a valid bullet item or table entry rather than an error.
   - Chunks are labeled as *potential* defects to be confirmed via human inspection.
2. **Page Untracked in Parquet**:
   - The production Parquet file (`all_chunks.parquet`) does not currently persist the originating PDF page number; page mapping requires external trace lookup against extracted JSONs.
3. **Baseline Snapshot Scope**:
   - This evaluation evaluates the existing production chunk store created prior to the Step 1 extraction fix. It serves as the baseline against which future chunking improvements can be measured.

---

## 8. Conclusion

The baseline chunking evaluation confirms that the chunk corpus is predominantly within viable operational bounds (median 420 chars, 92.96% <= 500), but exhibits specific structural vulnerabilities:
- Hard size upper-bound bypasses on unpunctuated text.
- Severe exact duplication (24.35% of chunks belong to duplicate clusters).
- Syntactic boundary fragmentation on ~8.7% of chunk endings.
- Naive substring metadata tagging false positives.

These quantitative findings establish the baseline for future chunking stage improvements.
