# Stage 1: Chunking Quality Evaluation

## 1. Overview & Motivation

In any Retrieval-Augmented Generation (RAG) system, **chunks are the atomic units of knowledge**. 
If a chunk is:
- **Truncated mid-sentence**, the LLM receives broken premises.
- **Missing essential context** (e.g. "Apply 2 ml per liter" without specifying the chemical or the crop), the LLM generates hallucinations or vague recommendations.
- **Corrupted by OCR or tabular extraction errors** (e.g., reversed text or mangled column lines), the retrieval embeddings will fail to match user queries.
- **Mixing unrelated topics** (e.g., combining tomato blight treatment with wheat sowing depths), retrieval similarity scores will be degraded.

Before evaluating retrieval algorithms (Dense vs. BM25 vs. Hybrid) or generating LLM answers, **we must evaluate whether the chunks produced by our pipeline are of high quality**.

---

## 2. Chunk Store Location & Schema

- **File Path**: `data/chunks/all_chunks.parquet`
- **Total Corpus Chunks**: 14,766 chunks across 32 agricultural PDF sources
- **Storage Engine**: Apache Parquet (Fast, columnar, memory-efficient)

### Available Schema Fields in Parquet

| Field Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `chunk_id` | `str` | Unique identifier formatted as `chunk_XXXXXX` | `chunk_000000` |
| `text` | `str` | Raw chunk text content | `"Nutrient management: 25:50:25:20 Kg/ha..."` |
| `char_count` | `int` | Character length of the chunk | `420` |
| `source_file` | `str` | Filename of the source PDF document | `Rabi-Agro-Advisory-2021-22_0.pdf` |
| `title` | `str` | Canonical title of the document | `Rabi Agro Advisory 2021-22` |
| `source_org` | `str` | Publishing agricultural university or institute | `ICAR / Directorate of Wheat Research` |
| `region` | `str` | Geographic region relevant to recommendations | `India` / `Andhra Pradesh, India` |
| `domain` | `str` | Domain category | `agriculture` / `agriculture_research` |
| `year` | `float`/`int` | Publication year (if known) | `2021` |
| `authority_score` | `float` | Source credibility weight (0.0 – 1.0) | `1.0` |
| `trust_weight` | `float` | Computed weight after bias penalty | `0.9` |
| `crop_tags` | `list` | Crops detected via agricultural entity recognition | `['wheat', 'mustard']` |

> [!NOTE]
> **Page Number Tracking**: Page numbers are not preserved in the current Parquet chunk schema because text extraction in `step3_chunk.py` concatenated page text prior to HDR (Hierarchical Discourse-based) sentence grouping. The evaluation sheet notes `page` as `N/A (untracked in parquet)`.

---

## 3. The 10 Chunk Quality Inspection Criteria

When reviewing chunks, evaluate against these 10 criteria:

1. **Context Completeness**: Does the chunk contain self-contained meaning, or does it start/end with dangling pronouns or fragments?
2. **Context Preservation**: Are critical referents (crop name, pest name, disease) clear from the text itself?
3. **Agricultural Measurements Preserved**: Are dosages (e.g., `2 ml/L`, `25:50:25 kg/ha`, `15-20 days`) intact without broken units or split numbers?
4. **Table Structure Preservation**: Are tables legible and coherent, or did column flattening turn them into gibberish?
5. **Clean Sentence Boundaries**: Did the chunk avoid cutting sentences mid-way?
6. **Topic Coherence**: Does the chunk focus on a single agricultural topic (e.g. weed control), or does it abruptly switch to an unrelated topic (e.g. tractor maintenance)?
7. **Question-Answering Utility**: Does this chunk provide enough evidence to answer a real farmer query?
8. **Appropriate Upper Size**: Is the chunk concise enough (avoiding massive multi-page dumps > 2,000 chars)?
9. **Appropriate Lower Size**: Is the chunk substantial enough (avoiding useless fragments < 100 chars like page numbers or headers)?
10. **Heading / Section Context**: Does the chunk retain section headers (e.g., `"Stem Borer:"`, `"Nutrient Management:"`)?

---

## 4. How to Run Chunk Inspection

The utility `evaluation/chunking/inspect_chunks.py` is provided to inspect the chunk corpus without modifying production files.

### Commands

1. **Display Corpus Statistics**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --stats
   ```

2. **List All Unique Source PDFs & Chunk Counts**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --list-sources
   ```

3. **List All Detected Crop Tags**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --list-crops
   ```

4. **Randomly Sample N Chunks**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --sample 5
   ```

5. **Sample Chunks from a Specific PDF**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --source "TNAU" --sample 5
   ```

6. **Sample Chunks for a Specific Crop**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --crop "wheat" --sample 5
   ```

7. **Inspect Length Outliers (Shortest vs. Longest)**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --shortest 5
   python evaluation/chunking/inspect_chunks.py --longest 5
   ```

8. **Export Sample to CSV for Human Evaluation**:
   ```bash
   python evaluation/chunking/inspect_chunks.py --sample 25 --export-csv evaluation/chunking/chunk_quality_results.csv
   ```

---

## 5. Human Evaluation Protocol (`chunk_quality_results.csv`)

Human evaluators inspect sampled chunks in `evaluation/chunking/chunk_quality_results.csv` and record scores:

| Column | Acceptable Values | Description |
| :--- | :--- | :--- |
| `chunk_id` | String | Fixed chunk identifier |
| `source` | String | Source PDF |
| `page` | String | Page number (`N/A` if untracked) |
| `crop` | String | Associated crops |
| `chunk_text` | String | Full chunk text |
| `chunk_length` | Integer | Character count |
| `context_complete` | `1` or `0` | `1` = complete thought; `0` = broken / truncated |
| `topic_coherent` | `1` or `0` | `1` = cohesive single topic; `0` = mixed unrelated topics |
| `contains_useful_information` | `1` or `0` | `1` = actionable agricultural info; `0` = junk/boilerplate |
| `table_information_preserved` | `1`, `0`, or `NA` | `1` = table intact; `0` = corrupted table; `NA` = not a table |
| `human_quality_score` | `1` to `5` | `1` = unusable, `3` = acceptable with flaws, `5` = perfect chunk |
| `notes` | Text | Specific notes on why it failed or succeeded (e.g. OCR reversed text, missing crop name) |

### Human Quality Scoring Guide
- **5 (Excellent)**: Contains full crop name, diagnosis/treatment/practice, clear dosages, clean sentence boundaries.
- **4 (Good)**: Completely intelligible and informative, minor whitespace or punctuation imperfections.
- **3 (Acceptable)**: Contains useful facts, but lacks some context (e.g., assumes user knows which crop is being discussed from previous sections).
- **2 (Poor)**: Severely truncated sentence, broken measurements, or mixed conflicting topics.
- **1 (Unusable / Junk)**: Table gibberish, reversed characters, or isolated fragments under 80 characters.
