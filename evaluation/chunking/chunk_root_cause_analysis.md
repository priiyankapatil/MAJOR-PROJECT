# Chunk Root-Cause Analysis Report

## 1. Overview & Objective

Following the chunk trace diagnostic and chunking baseline evaluation, this report establishes the definitive root cause for the landmark suspicious chunks identified in the corpus (`data/chunks/all_chunks.parquet`).

Each chunk was traced backward through:
1. **Raw PDF content** (`data/raw pdf/`)
2. **Step 1 Extraction** (`step1_extract.py`, extracted JSONs)
3. **Step 2 Cleaning** (`step2_clean.py`, noise filters)
4. **Step 3 Chunking** (`step3_chunk.py`, spaCy splitting, BERT NSP, hard boundaries, size guards)
5. **Metadata Tagging** (`extract_crop_tags()`, source metadata)

> [!IMPORTANT]
> **Evaluation Scope Only**: This is an analytical diagnostic. No production pipeline files (`step1_extract.py`, `step2_clean.py`, `step3_chunk.py`), chunk parquet stores, vector databases, or retrieval code were modified or regenerated.

---

## 2. Root-Cause Classification Table

| Chunk ID | Observed Problem | Root Cause Stage | Evidence | Production Fix Needed? |
| :--- | :--- | :---: | :--- | :---: |
| **`chunk_012110`** | Extreme oversized chunk (14,916 chars) with repetitive advisory headers and Tibetan livestock table text. | **MIXED** | Step 2 cleaning lacks a pattern for `ICAR RABI AgRo-AdvIsoRy foR fARmeRs`. Step 3 chunker relies on spaCy sentence splitting; unpunctuated bullet lists became a single 14,916-character "sentence" that bypassed the 500-char size guard due to the absence of an intra-sentence hard size cap. | **YES** (Step 2 filter + Step 3 hard size fallback) |
| **`chunk_000317`** | Mirrored/reversed table text (5,447 chars: `serunam diuqil cinagro...`, `'%AT)('`, `'atad delooP'`). | **STEP1_EXTRACTION** | Landscape-oriented academic table in ANGRAU p.69 read in reverse glyph byte stream order by `pdfplumber`. Text passed unpunctuated into Step 3, yielding an oversized, unreadable chunk. (Validated fixed upstream in Step 1 via PyMuPDF). | **NO** for Step 3 (Resolved upstream in Step 1) |
| **`chunk_001683`** | Reversed glyph stream in seed certification table (477 chars: `'II xidneppA sdeeS defiitreC'`). | **STEP1_EXTRACTION** | Appendix II table in Crop Protection p.146 had rotated fonts; `pdfplumber` extracted characters in reverse stream order. Step 3 grouped the corrupt tokens as-is. (Validated fixed upstream in Step 1 via PyMuPDF). | **NO** for Step 3 (Resolved upstream in Step 1) |
| **`chunk_001743`** | Horizontally interleaved columns and 80%+ loss of technical machinery specifications (424 chars). | **STEP1_EXTRACTION** | Merged-cell table on Crop Protection p.154 had `None` header cells in row 0, causing `table_to_readable_text()` to drop columns 1–4 while text mode horizontally collided adjacent columns. Step 3 merely grouped the distorted input. (Validated fixed upstream in Step 1 via PyMuPDF). | **NO** for Step 3 (Resolved upstream in Step 1) |
| **`chunk_003222`** | Undersized 50-character orphan boundary fragment (`"Transplanting is done regulate moisture retention."`). | **MIXED** | Primary upstream cause: Step 1 `pdfplumber` read across two columns horizontally on KAU p.177, splicing Column 1 line 6 into Column 2. Compounding Step 3 cause: `AGRI_SECTION_HEADERS` regex matched `r'^Transplanting'` inside the spliced line, triggering a split that produced a 50-char orphan barely meeting `MIN_CHUNK_SIZE = 50`. | **YES** (Step 1 column fix + Step 3 standalone header regex) |
| **`chunk_007588`** | High-quality cashew pest management content incorrectly tagged with crop `"tea"` (427 chars). | **METADATA** | Text and chunk boundaries are high quality. However, `extract_crop_tags()` in `step3_chunk.py` uses naive substring search (`c in text.lower()`); `"tea mosquito bug"` (*Helopeltis antonii*, a primary cashew pest) triggered a false-positive tag for crop `"tea"`. | **YES** in Step 3 (`extract_crop_tags` logic) |
| **`chunk_001858`** | Administrative directory listing university staff names, phone numbers, and office extensions (298 chars). | **STEP2_CLEANING** | Back-matter Appendix IX of Crop Protection p.171 contains staff contact directory. Step 1 extracted text cleanly; Step 2 cleaning lacks a noise filter for telephone directories and administrative back-matter; Step 3 grouped valid sentences into a chunk as designed. | **YES** in Step 2 (Back-matter filter) |
| **`chunk_000006`** | Oversized 1,906-character Table of Contents listing 5 unrelated crops (`rice`, `chilli`, `brinjal`, `tea`, `marigold`). | **MIXED** | Step 2 cleaning does not strip journal front-matter/TOCs. Step 3 has no intra-sentence size cap to split unpunctuated paper titles. Step 3 metadata tagging naively collected every crop mentioned across all paper titles in the table of contents into one chunk. | **YES** (Step 2 TOC filter + Step 3 hard cap) |

---

## 3. Detailed Root-Cause Analysis by Case

### 1. `chunk_012110` (14,916 chars) — `Rabi-Agro-Advisory-2021-22_0.pdf`
- **Observed Defect**: Massive single chunk (14,916 chars) comprising hundreds of unpunctuated bullet lines, repetitive ICAR advisory headers, and Tibetan livestock vaccination schedules.
- **Stage Attribution**: **`MIXED`**
- **Detailed Mechanics**:
  1. `step1_extract.py`: Extracted text from hundreds of advisory pages containing the running header `ICAR RABI AgRo-AdvIsoRy foR fARmeRs`.
  2. `step2_clean.py`: The `NOISE_PATTERNS` regex list explicitly includes university headers (`Tamil Nadu Agricultural University`, `Kerala Agricultural University`, etc.) but lacks an entry for ICAR advisory compendiums.
  3. `step3_chunk.py`: spaCy sentence segmentation splits on terminal sentence punctuation (`.`, `!`, `?`). The advisory bullets (`•`) had no periods. spaCy yielded a single 14,916-character sentence. In `hdr_chunk()`:
     ```python
     big = (current_len + len(sent)) > MAX_CHUNK_SIZE
     if hard or big:
         split = True
     ```
     Because the check splits *between* sentences, an individual sentence that already exceeds `MAX_CHUNK_SIZE` is pushed to `current_sents` as a solitary block and appended without intra-sentence splitting.
- **Remedy Location**: Step 2 (clean noise pattern) + Step 3 (enforce a hard character ceiling fallback if `len(sent) > MAX_CHUNK_SIZE`).

---

### 2. `chunk_000317` (5,447 chars) & `chunk_001683` (477 chars) — Reversed Glyph Streams
- **Observed Defect**: Words, table headers, and numerical values are rendered completely in reverse ASCII character order (e.g. `'serunam diuqil cinagro'` for `'organic liquid manures'`; `'II xidneppA sdeeS defiitreC'` for `'Certified Seeds Appendix II'`).
- **Stage Attribution**: **`STEP1_EXTRACTION`**
- **Detailed Mechanics**:
  - Both source pages (`ANGRAU p.69` and `Crop Protection p.146`) feature landscape-oriented tables printed on portrait PDF pages.
  - `pdfplumber` reads character stream objects in low-level PDF byte order without orientation font matrix normalization.
  - `step2_clean.py` and `step3_chunk.py` received reversed text. Neither stage is designed to reverse character streams.
- **Remedy Location**: **Step 1 Only**. Replacing `pdfplumber` text extraction with `PyMuPDF` in Step 1 has already been validated to completely eliminate this defect. Step 3 requires no modification for this issue.

---

### 3. `chunk_001743` (424 chars) — Merged-Cell Machinery Table Distortion
- **Observed Defect**: Horizontal line collision of machine specifications (`"e) Moong Spike tooth 7.5 hp 2.5 q/h Spike tooth thresher for wheat can Thresher thresher electric be used..."`) and total omission of horsepower/capacity specs from `tables_text`.
- **Stage Attribution**: **`STEP1_EXTRACTION`**
- **Detailed Mechanics**:
  - The table on Crop Protection p.154 has merged category cells in row 0 (`['Harvesting', None, None, None, None]`).
  - The extraction helper `table_to_readable_text()` mapped `headers = ['Harvesting', '', '', '', '']` and dropped every column where header was empty string `""`.
  - `pdfplumber.extract_text()` grouped characters across the same horizontal band, interweaving adjacent column text.
- **Remedy Location**: **Step 1 Only**. PyMuPDF block layout parsing groups machine items vertically, preserving complete technical specifications. Step 3 requires no modification for this issue.

---

### 4. `chunk_003222` (50 chars) — Orphan Boundary Split
- **Observed Defect**: 50-character orphan chunk (`"Transplanting is done regulate moisture retention."`) offering zero actionable agricultural context.
- **Stage Attribution**: **`MIXED`**
- **Detailed Mechanics**:
  1. `step1_extract.py`: Page 177 of KAU is a two-column layout. `pdfplumber` scanned horizontally across the central gutter, merging Column 1 line 6 (`"...to regulate moisture retention."`) with Column 2 (`"Transplanting is done..."`).
  2. `step3_chunk.py`: The `AGRI_SECTION_HEADERS` regex contains `r'Transplanting'`. It matched mid-sentence because the regex tests `r'^(...)'` on the spliced line. Step 3 split immediately before `"Transplanting"`.
  3. `step3_chunk.py`: The resulting fragment had exactly 50 characters, matching `MIN_CHUNK_SIZE = 50`. Thus, it bypassed the discard filter.
- **Remedy Location**: Step 1 (PyMuPDF column-aware extraction prevents line splicing) + Step 3 (refine header regex to require standalone lines or punctuation boundaries; consider raising `MIN_CHUNK_SIZE` to 60–80 chars).

---

### 5. `chunk_007588` (427 chars) — False-Positive Crop Tagging
- **Observed Defect**: High-quality cashew pest management recommendations tagged with `crop_tags: ['cashew', 'tea']`.
- **Stage Attribution**: **`METADATA`**
- **Detailed Mechanics**:
  - The chunk text describes the *Tea Mosquito Bug* (*Helopeltis antonii*), which is the major insect pest of cashew trees in India.
  - `step3_chunk.py` contains `extract_crop_tags(text)`:
    ```python
    CROPS = ["rice", "wheat", ..., "tea", "coffee", ...]
    tl = text.lower()
    return [c for c in CROPS if c in tl]
    ```
  - The substring `'tea'` was matched within `'tea mosquito bug'`, incorrectly tagging the chunk as relevant to tea cultivation (*Camellia sinensis*).
- **Remedy Location**: **Step 3 (`extract_crop_tags`)**. Needs word-boundary matching and exclusion of known entomological compound terms (e.g., `"tea mosquito bug"` -> `cashew`, not `tea`).

---

### 6. `chunk_001858` (298 chars) — Administrative Back-Matter Directory
- **Observed Defect**: Telephone directory of Punjab Agricultural University staff (`"Dr. Amit Kaul, Agronomy 401 81464-00233 Dr. R K Gupta..."`).
- **Stage Attribution**: **`STEP2_CLEANING`**
- **Detailed Mechanics**:
  - Step 1 extracted the text faithfully.
  - Step 2 cleaning has no filters for administrative back-matter appendices, faculty directories, or phone number patterns.
  - Step 3 chunker processed the text into valid grammatical sentences via spaCy and BERT NSP. The chunking algorithm performed as designed on non-agricultural input.
- **Remedy Location**: **Step 2 (Cleaning)**. Step 2 should detect and filter administrative directories (phone numbers, office extensions, faculty listings) or back-matter appendices.

---

### 7. `chunk_000006` (1,906 chars) — Front-Matter Table of Contents & Author Index
- **Observed Defect**: Oversized chunk (1,906 chars) containing journal article titles and author lists, falsely tagged with 5 crops (`rice`, `chilli`, `brinjal`, `tea`, `marigold`).
- **Stage Attribution**: **`MIXED`**
- **Detailed Mechanics**:
  - `step2_clean.py` removed excessive dots (`\.{4,}`), but allowed the raw TOC title/author text to pass through.
  - `step3_chunk.py` has no intra-sentence size cap. Because the TOC entries lack terminal periods, spaCy treated multiple paper titles as one gigantic sentence.
  - `extract_crop_tags()` naively scanned the multi-title block and accumulated tags for every crop mentioned across different research papers.
- **Remedy Location**: Step 2 (filter front-matter TOCs) + Step 3 (hard size cap + metadata disambiguation).

---

## 4. Root-Cause Summary by Pipeline Stage

| Pipeline Stage | Assigned Landmark Chunks | Primary Defect Pattern | Recommended Fix Location |
| :--- | :---: | :--- | :--- |
| **`STEP1_EXTRACTION`** | 3 chunks (`chunk_000317`, `chunk_001683`, `chunk_001743`) | Reversed glyph streams; merged-cell table column drops; horizontal column collision. | **Step 1 Only** (Adopt PyMuPDF primary text extractor; already validated). |
| **`STEP2_CLEANING`** | 1 chunk (`chunk_001858`) | Unfiltered administrative back-matter phone directory. | **Step 2 Only** (Add phone directory & back-matter filter patterns). |
| **`METADATA`** | 1 chunk (`chunk_007588`) | Substring false-positive crop tag on compound pest name (`"tea mosquito bug"`). | **Step 3 Metadata** (Refactor `extract_crop_tags()` with boundary/phrase exclusion). |
| **`MIXED`** | 3 chunks (`chunk_012110`, `chunk_003222`, `chunk_000006`) | Missing cleaning pattern + lack of intra-sentence hard size cap; extraction gutter collision + over-aggressive section boundary regex. | **Shared: Step 2 & Step 3** (Add cleaning rules + hard chunk size upper bound in Step 3). |
| **`STEP3_CHUNKING` (isolated)**| 0 chunks | None of the 8 landmark failures was caused *solely* by Step 3 chunking logic in isolation. | — |
| **`UNCERTAIN`** | 0 chunks | All 8 landmark chunks have clear, documented causal traces. | — |

---

## 5. What Problems Should Be Fixed in `step3_chunk.py`

1. **Intra-Sentence Hard Character Cap**:
   - When a single spaCy sentence exceeds `MAX_CHUNK_SIZE` (e.g. `chunk_012110` with 14,916 chars; `chunk_000006` with 1,906 chars), `step3_chunk.py` must split the sentence on punctuation fallbacks (semicolons, colons, newlines, commas) or hard character boundaries rather than letting it bypass `MAX_CHUNK_SIZE`.
2. **Crop Tag Disambiguation (`extract_crop_tags`)**:
   - Refactor substring matching to use regex word boundaries (`\btea\b`).
   - Add exclusion logic for compound agricultural terms (e.g. `"tea mosquito bug"` refers to a cashew/cocoa pest, not *Camellia sinensis*).
3. **Refine Agricultural Section Boundary Regex (`AGRI_SECTION_HEADERS`)**:
   - Ensure section breaks match standalone heading lines (e.g. `r'^\s*Transplanting\s*$'` or `r'^\s*Transplanting\s*:'`) rather than sentences that begin with common verbs or gerunds (`"Transplanting is done..."`).
4. **Tune `MIN_CHUNK_SIZE` Threshold**:
   - Slightly elevate `MIN_CHUNK_SIZE` from 50 to 60–80 characters to prevent uninformative fragments (`chunk_003222`) from persisting in the retrieval store.

---

## 6. What Problems Should NOT Be Fixed in `step3_chunk.py`

1. **Reversed Text Correction**:
   - Do **NOT** add character-reversal heuristics or string inversion logic to Step 3. Reversed text (`chunk_000317`, `chunk_001683`) is an extraction-layer defect caused by `pdfplumber`. It has already been resolved at the root in Step 1 using PyMuPDF.
2. **Table Flattening & Column Re-ordering**:
   - Do **NOT** attempt to re-stitch scrambled table columns (`chunk_001743`) in Step 3. Step 3 receives sequential text; multi-column layout interpretation belongs strictly in Step 1.
3. **Document Back-Matter & Staff Directory Filtering**:
   - Do **NOT** write custom chunk-level filters to discard staff phone numbers (`chunk_001858`). Administrative back-matter should be stripped during Step 2 cleaning or during document ingestion before tokenization and NSP modeling.
4. **Header / Boilerplate Stripping**:
   - Do **NOT** attempt to remove repeated institutional headers (`ICAR RABI AgRo-AdvIsoRy foR fARmeRs`, TNAU headers) in Step 3. Removing document-level noise is the explicit responsibility of Step 2 cleaning (`NOISE_PATTERNS`).

---

## 7. Recommended Fixes (Do NOT Implement Yet)

### Recommended Step 2 Cleaning Fixes:
```python
# In step2_clean.py:
NOISE_PATTERNS.extend([
    r'ICAR\s+RABI\s+AgRo-AdvIsoRy\s+foR\s+fARmeRs.*?\n',
    r'Appendix\s+[IVXLCDM]+\s*:?\s*Important\s+Telephone\s+Numbers.*',
    r'Dr\.\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*,\s+[A-Za-z\s&]+\s+\d{3,}\s+\d{5}-\d{5}',
])
```

### Recommended Step 3 Chunking Fixes:
```python
# 1. Hard intra-sentence size cap in hdr_chunk():
if len(sent) > MAX_CHUNK_SIZE:
    # Break oversized sentence by newline or punctuation fallback
    sub_parts = split_oversized_sentence(sent, MAX_CHUNK_SIZE)
    # process sub_parts...

# 2. Strict section header pattern:
AGRI_SECTION_HEADERS = re.compile(
    r'^\s*(Pests?|Diseases?|Varieties|Manuring|Fertilizers?|Irrigation|Harvesting|Transplanting)\s*[:\n]',
    re.IGNORECASE
)

# 3. Robust crop tagger with exclusion:
def extract_crop_tags(text):
    tl = text.lower()
    tags = []
    for crop in CROPS:
        if re.search(r'\b' + re.escape(crop) + r'\b', tl):
            if crop == 'tea' and 'tea mosquito' in tl:
                continue
            tags.append(crop)
    return tags
```

---

## 8. Evidentiary Confidence

All 8 landmark cases were traced directly to explicit source lines in `data/raw pdf/`, extracted JSONs, and `step3_chunk.py` logic. No case exhibits ambiguous or insufficient evidence (`UNCERTAIN: 0`).
