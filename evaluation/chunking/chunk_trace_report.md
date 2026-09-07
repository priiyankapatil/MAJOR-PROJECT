# Chunk Trace Diagnostic Report

## Methodology

Each suspicious chunk was traced backwards through three pipeline stages:

1. **Raw PDF text** — extracted directly via PyMuPDF/pdfplumber from the original PDF
2. **Extracted JSON** — the structured output saved by `step1_extract.py`, then cleaned by `step2_clean.py`
3. **Final chunk** — the text unit stored in `data/chunks/all_chunks.parquet` produced by `step3_chunk.py`

The goal is to identify **at which stage** the problem first appears, not to fix anything yet.

---

## Findings Summary

| chunk_id | Source | Page | Problem Type | First Problem Stage | Severity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `chunk_012110` | Rabi-Agro-Advisory-2021-22_0.pdf | 82–754 | TOO LARGE / NOISE | Cleaning | HIGH |
| `chunk_000317` | ANGRAU Journal of Research.pdf | 69 | CORRUPTED (reversed table) | Extraction | HIGH |
| `chunk_001683` | Crop Protection.pdf | 146 | CORRUPTED (reversed table) | Extraction | HIGH |
| `chunk_001858` | Crop Protection.pdf | 171 | NON-AGRICULTURAL NOISE | Cleaning | MEDIUM |
| `chunk_003222` | KAU (Kerala Agricultural University).pdf | 177 | INCOMPLETE | Extraction | HIGH |
| `chunk_007588` | TNAU Horticulture PDF.pdf | 105 | GOOD — FALSE METADATA | Metadata Tagging | LOW |
| `chunk_001743` | Crop Protection.pdf | 154 | CORRUPTED TABLE | Extraction | MEDIUM-HIGH |

---

## Detailed Analysis

---

### 1. `chunk_012110` — 14,916 chars — Rabi-Agro-Advisory-2021-22_0.pdf

**Problem Type:** TOO LARGE / NON-AGRICULTURAL NOISE

**Trace:**

- **Raw PDF (pages ~82 to ~754):** The PDF is a multi-crop, multi-state advisory compendium. It contains a running header `ICAR RABI AgRo-AdvIsoRy foR fARmeRs` repeated at the top of hundreds of pages. It also contains pages in Tibetan/Dzongkha script for livestock vaccination schedules.

- **Extracted JSON:** The header text `ICAR RABI AgRo-AdvIsoRy foR fARmeRs` was present on every extracted page and passed through unchanged.

- **Cleaning (step2_clean.py):** `step2_clean.py` has a `NOISE_PATTERNS` list that removes university-specific headers such as `Tamil Nadu Agricultural University` and `TNAU.*?Coimbatore`. The ICAR advisory header was **not included** in those patterns. It passed through uncleaned.

- **Chunking (step3_chunk.py):** The HDR chunker uses spaCy for sentence splitting. The repeated bullet points and Tibetan-script advisory content (`• • • • • • • •`) had **no terminal punctuation**. spaCy treated the entire 14,916-character block as a single sentence, never triggering a sentence boundary split. Since `MAX_CHUNK_SIZE` is checked against `current_len + len(sent)` but only after the full sentence is evaluated, a single sentence that is already 14,916 chars long bypasses the size guard.

**Root Cause:**  
Two compounding failures:
1. **Step 2 (Cleaning):** Missing noise pattern for `ICAR RABI AgRo-AdvIsoRy foR fARmeRs`
2. **Step 3 (Chunking):** No hard upper size cap applied when a single spaCy "sentence" exceeds `MAX_CHUNK_SIZE`

---

### 2. `chunk_000317` — 5,447 chars — ANGRAU Journal of Research.pdf

**Problem Type:** CORRUPTED (reversed/mirrored table text)

**Trace:**

- **Raw PDF (page 69):** Contains "Table 3: Per se performance of TSS (°Brix) and TA (%) of brinjal varieties as influenced by different organic liquid manures." The table is rendered in a landscape-oriented layout. The raw PDF text via PyMuPDF shows the table **correctly** on the page.

- **Extracted JSON (step1_extract.py, pdfplumber):** When pdfplumber extracted the table on this page, it rendered the columns in **reversed order** — column headers became `'%AT)('` (reversed `(AT%)`), `'atad delooP'` (reversed `Pooled data`), numeric values like `21.0`, `51.0` appear in reverse column sequence. This is a known pdfplumber behaviour on right-to-left or landscape-oriented table layouts in research PDFs.

- **Cleaning (step2_clean.py):** Passed through unchanged — no detection of reversed text.

- **Chunking (step3_chunk.py):** The reversed text contains no terminal punctuation, no hard boundaries, and no sentences. spaCy produced **one sentence** of 5,447 characters. The chunk was stored as-is.

**Root Cause:**  
**Step 1 (Extraction):** pdfplumber table extraction failure on a landscape-rotated multi-column academic table. The raw text stream read right-to-left, producing mirrored strings.

---

### 3. `chunk_001683` — 477 chars — Crop Protection.pdf

**Problem Type:** CORRUPTED (reversed/mirrored table text)

**Trace:**

- **Raw PDF (page 146):** Contains Appendix II "Field Standards for Foundation and Certified Seeds." The raw PyMuPDF extraction correctly shows: `"136 Crop Isolation distance (in metre) % Pollen shedders (maximum) % Off type plants..."` — this is a standard seed certification table with column headers.

- **Extracted JSON:** pdfplumber extracted the table characters in **reverse glyph order** from the PDF content stream. The result was: `"II xidneppA sdeeS defiitreC dna noitadnuoF rof sdradnatS dleiF"` (reversed: `"Field Standards for Foundation and Certified Seeds Appendix II"`). The column values `"wedliM ynwoD" = "Downy Mildew"`, `"tums esooL" = "Loose smut"` are all backwards.

- **Cleaning (step2_clean.py):** Passed through. No mechanism to detect reversed ASCII strings.

- **Chunking (step3_chunk.py):** The reversed text does not contain meaningful sentence boundaries, so HDR/NSP produced one chunk encompassing the reversed table content.

**Root Cause:**  
**Step 1 (Extraction):** PDF page 146 has a reversed glyph stream (a PDF encoding quirk). pdfplumber reads raw character positions and outputs them in encoding order rather than visual order. PyMuPDF correctly renders this page (as verified above), but the pipeline uses pdfplumber for table extraction, which does not perform glyph re-ordering.

---

### 4. `chunk_001858` — 298 chars — Crop Protection.pdf

**Problem Type:** NON-AGRICULTURAL NOISE (Staff contact directory)

**Trace:**

- **Raw PDF (page 171):** "Appendix IX: Important Telephone Numbers of Punjab Agricultural University, Ludhiana." This is back matter containing staff names, office extensions, and mobile numbers — entirely administrative.

- **Extracted JSON:** The text was extracted correctly. There is no agricultural content on this page.

- **Cleaning (step2_clean.py):** The cleaning step has no filter for back-matter appendices, staff directories, or phone number patterns. The text passed through.

- **Chunking (step3_chunk.py):** The HDR chunker treated each `"Dr. [Name], [Department]..."` line as a short, semantically separate sentence. These were grouped into a 298-character chunk by NSP scoring.

**Root Cause:**  
**Step 2 (Cleaning):** Missing noise filter for administrative back-matter content.  
The text is correctly extracted but should have been filtered before chunking. There are no agricultural knowledge signals on this page.

**Estimated Impact:**  
A query such as `"What does an agronomist recommend for soil management?"` might retrieve this chunk because it contains `"Agronomy"` as a staff department name.

---

### 5. `chunk_003222` — 50 chars — KAU (Kerala Agricultural University).pdf

**Problem Type:** INCOMPLETE

**Trace:**

- **Raw PDF (page 177):** A two-column page describing coffee nursery practices. Column 1 discusses coffee seed germination bed preparation and sowing. Column 2 discusses polybag nursery transplanting procedures.

- **Extracted JSON (step1_extract.py):** PyMuPDF read across both columns **horizontally** (a known two-column PDF issue). Instead of reading Column 1 top-to-bottom and then Column 2, it interleaved partial lines from both columns. The resulting text stitched fragments from left and right columns mid-sentence, creating internally incoherent runs.

- **Cleaning:** The interleaved text passed through unchanged.

- **Chunking (step3_chunk.py):** The `AGRI_SECTION_HEADERS` regex matched `"Transplanting"` as a hard boundary, causing a split immediately before `"Transplanting is done"`. The result was a 50-character orphan fragment: `"Transplanting is done\nregulate moisture retention."` which provides no actionable context.

**Root Cause:**  
**Step 1 (Extraction):** Multi-column PDF read in horizontal Z-order rather than column-by-column order, creating sentence fragments.  
**Step 3 (Chunking):** `AGRI_SECTION_HEADERS` pattern correctly split on `"Transplanting"` but exposed the extraction defect — the split produced a fragment below `MIN_CHUNK_SIZE` ... or one that barely passed (50 chars is the minimum threshold in config). This reveals `MIN_CHUNK_SIZE` may need to be raised.

---

### 6. `chunk_007588` — 427 chars — The TNAU Horticulture PDF.pdf

**Problem Type:** GOOD — FALSE METADATA LABEL (not a chunking problem)

**Trace:**

- **Raw PDF (page 105):** Cashew pest management section. The page correctly discusses the Tea Mosquito Bug (*Helopeltis antonii*) as a primary cashew pest. The treatment recommendations (neem oil, Pongamia oil) are specific to cashew.

- **Extracted JSON:** Correctly extracted with full agricultural context preserved.

- **Cleaning:** No issues. Text is clean.

- **Chunking:** The chunk boundary is appropriate. Content is topically coherent — the entire chunk is about cashew pest management.

- **crop_tags metadata:** The `extract_crop_tags()` function in `step3_chunk.py` uses a simple substring match against a list of crop names. The string `"Tea Mosquito Bug"` contains the substring `"tea"`, which matched the crop `"tea"` in the crop list. This incorrectly tagged the chunk as being about *tea (Camellia sinensis)* cultivation when it is actually about **cashew** pest management.

**Root Cause:**  
**Step 3 Metadata (extract_crop_tags):** Naive substring crop tagging without context validation. `"tea"` inside `"Tea Mosquito Bug"` is not a crop reference but an entomological common name.

**Estimated Impact:** Moderate. A crop-filtered retrieval query for `"tea pest management"` would incorrectly retrieve this cashew chunk. The chunk itself is valid; only its metadata is wrong.

---

### 7. `chunk_001743` — 424 chars — Crop Protection.pdf

**Problem Type:** CORRUPTED TABLE (badly flattened)

**Trace:**

- **Raw PDF (page 154):** A harvesting and threshing machinery table with 5 columns: Crop, Machine Name, Power Source, Capacity (q/h), and Modification Notes.

- **Extracted JSON:** pdfplumber's `table_to_readable_text()` function in `step1_extract.py` flattens tables using the first row as headers. However, when a table contains merged cells (e.g., the crop name "e) Moong" spans multiple rows), pdfplumber provides `None` for subsequent rows in the merged cell — and the `table_to_readable_text()` function skips any part where the header is empty or cell is empty. The result interleaves partial modification notes with power source specs mid-sentence: `"e) Moong Spike tooth 7.5 hp 2.5 q/h Spike tooth thresher for wheat can Thresher thresher electric be used..."`.

- **Cleaning:** No issue detected by cleaning.

- **Chunking:** HDR chunker treated the interleaved text as a single coherent sequence (NSP scores were ambiguous given the mixed technical context), producing one chunk with scrambled column data.

**Root Cause:**  
**Step 1 (Extraction):** Merged-cell table with `None` cells in pdfplumber extraction. The `table_to_readable_text()` function in `step1_extract.py` does not handle merged cells. Partial column flattening produced mixed machine specs.

---

## Categorised Problem Summary

### Extraction Problems (Step 1)
1. **Reversed glyph stream** — affects `chunk_000317`, `chunk_001683`  
   Cause: pdfplumber does not re-order glyphs from right-to-left or landscape-encoded content streams.

2. **Multi-column horizontal read** — affects `chunk_003222`  
   Cause: PyMuPDF reads in Z-order on two-column pages, stitching left and right column lines horizontally.

3. **Merged-cell table flattening** — affects `chunk_001743`  
   Cause: `table_to_readable_text()` does not handle `None` cells from merged rows in pdfplumber.

### Cleaning Problems (Step 2)
4. **Missing running header filter** — affects `chunk_012110`  
   Cause: `NOISE_PATTERNS` in `step2_clean.py` does not include `ICAR RABI AgRo-AdvIsoRy foR fARmeRs`.

5. **Missing back-matter appendix filter** — affects `chunk_001858`  
   Cause: No detection of staff contact directories or phone number blocks.

### Chunking Problems (Step 3)
6. **No hard upper size cap** — affects `chunk_012110`  
   Cause: A single sentence that is already 14,916 chars is stored as-is; the `MAX_CHUNK_SIZE` guard only applies when *adding* a sentence to an existing chunk, not to the sentence itself.

7. **Hard section boundary over-splitting** — affects `chunk_003222`  
   Cause: `AGRI_SECTION_HEADERS` regex split on `"Transplanting"` and exposed the upstream extraction defect.

### Metadata Problems (Step 3 extract_crop_tags)
8. **Substring crop tag false positive** — affects `chunk_007588`  
   Cause: `"tea"` matched inside `"Tea Mosquito Bug"`.

---

## Severity Distribution

| Severity | Count | Chunk IDs |
| :--- | :--- | :--- |
| HIGH | 3 | `chunk_012110`, `chunk_000317`, `chunk_003222` |
| MEDIUM-HIGH | 1 | `chunk_001683`, `chunk_001743` |
| MEDIUM | 1 | `chunk_001858` |
| LOW | 1 | `chunk_007588` |

---

## What Has NOT Been Changed

In accordance with the evaluation plan:

- `step1_extract.py` — **not modified**
- `step2_clean.py` — **not modified**
- `step3_chunk.py` — **not modified**
- `data/chunks/all_chunks.parquet` — **not modified**
- ChromaDB / BM25 / embeddings — **not modified**

---

## Next Steps (Proposed — Not Yet Executed)

> [!IMPORTANT]
> Do NOT proceed with these steps until explicitly instructed.

Possible future fixes to evaluate one at a time:

1. Add `ICAR RABI AgRo-AdvIsoRy foR fARmeRs` to `step2_clean.py` NOISE_PATTERNS
2. Add a hard sentence-level size cap in `step3_chunk.py` (force split if a single sentence > MAX_CHUNK_SIZE)
3. Add phone number / back-matter detection to `step2_clean.py`
4. Fix `extract_crop_tags()` to use word-boundary matching instead of substring matching
5. Investigate multi-column extraction alternatives for KAU PDF
6. Investigate whether PyMuPDF can be used for table extraction (instead of pdfplumber) for reversed-glyph pages
