# Broader PDF Extraction Benchmark (Evaluation Only)

## 1. Objective

Following the initial 4-page failure trace, this experiment conducts a broader, layout-varied benchmark across 8 distinct PDF pages to evaluate whether extraction behavior holds consistently across diverse document formats prior to considering modifications to `step1_extract.py`.

---

## 2. Pages and Layouts Selected

Eight pages were selected from across the existing PDF corpus representing 8 distinct visual layout archetypes (excluding the 4 previously evaluated failure pages):

| # | Source Document | Page | Layout Type |
| :-: | :--- | :-: | :--- |
| 1 | `Document-1-Guidelines-for-Soil-Health-Fertility-Scheme.pdf` | 2 | Normal narrative/text page |
| 2 | `The TNAU Agriculture PDF.pdf` | 20 | Normal agricultural table |
| 3 | `ANGRAU Journal of Research.pdf` | 68 | Landscape/rotated table |
| 4 | `KAU (Kerala Agricultural University).pdf` | 50 | Two-column page |
| 5 | `Crop Protection.pdf` | 153 | Complex/merged-cell table |
| 6 | `UAS Bangalore.pdf` | 15 | Headings/lists |
| 7 | `The TNAU Agriculture PDF.pdf` | 60 | Agricultural numerical/fertilizer info |
| 8 | `Banana.pdf` | 1 | Scanned/image-only page |

---

## 3. Methods Compared

- **`pdfplumber text extraction`**: `page.extract_text()` (geometric line clustering)
- **`PyMuPDF text extraction`**: `page.get_text("text")` (stream & block layout parsing)
- *Note*: `EasyOCR` was not evaluated because it is unavailable in the current environment (`ModuleNotFoundError`).

---

## 4. Quality Criteria

- **`GOOD`**: Readable text, correct reading order, no major missing content, no obvious corruption.
- **`PARTIAL`**: Mostly usable but has minor missing content, formatting problems, or incomplete structure.
- **`POOR`**: Major ordering problems, missing information, or serious structural problems, but text is still partly understandable.
- **`CORRUPTED`**: Reversed/mirrored characters, unreadable text, or severe corruption that makes the extracted content unusable.

---

## 5. Results

The full per-method extraction records are stored in [`broader_extraction_benchmark.csv`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/extraction/broader_extraction_benchmark.csv).

### Per-Page Comparison

| Page | Layout Type | `pdfplumber` Quality | `PyMuPDF` Quality | Winner | Rationale |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Doc-1 p.2** | Normal narrative/text | GOOD | GOOD | **Tie** | Both extract prose cleanly and accurately in logical order. |
| **TNAU Agri p.20** | Normal agricultural table | GOOD | GOOD | **Tie** | Both preserve standard row-column contents and grain metrics. |
| **ANGRAU p.68** | Landscape/rotated table | CORRUPTED | GOOD | **PyMuPDF** | `pdfplumber` outputs mirrored text (`diuqil cinagro`); `PyMuPDF` renders forward text. |
| **KAU p.50** | Two-column page | POOR | GOOD | **PyMuPDF** | `pdfplumber` scans horizontally across the gutter; `PyMuPDF` follows column blocks. |
| **Crop Prot. p.153** | Complex/merged-cell table | POOR | GOOD | **PyMuPDF** | `pdfplumber` collides column text; `PyMuPDF` keeps machine specs grouped. |
| **UASB p.15** | Headings/lists | GOOD | GOOD | **Tie** | Both preserve bulleted technological interventions and yield numbers. |
| **TNAU Agri p.60** | Numerical/fertilizer info | PARTIAL | GOOD | **PyMuPDF** | `pdfplumber` splits formula subscripts (`FP O`, `2 5`); `PyMuPDF` keeps formulas intact (`FP2O5`). |
| **Banana p.1** | Scanned/image-only | POOR | POOR | **Tie** | Both return 0 characters on pure raster images in the absence of OCR. |

### Summary Statistics

- **Total pages evaluated**: 8
- **Head-to-head outcome**:
  - `pdfplumber` better: **0**
  - `PyMuPDF` better: **4**
  - Ties: **4**
- **Quality breakdown**:
  - `pdfplumber`: GOOD = 3, PARTIAL = 1, POOR = 3, CORRUPTED = 1
  - `PyMuPDF`: GOOD = 7, PARTIAL = 0, POOR = 1 (scanned page), CORRUPTED = 0

---

## 6. Interpretation

1. **Equivalence on standard layouts**: On single-column prose, simple tables, and bulleted lists, both extractors achieve high fidelity (`GOOD`) with no significant performance gap.
2. **PyMuPDF advantages on complex structures**: `PyMuPDF` excels specifically where geometric or encoding complexities arise: rotated/landscape tables (correct character order), two-column flows (column-aware block parsing), merged-row tables, and formula subscripts.
3. **Common failure mode on raster imagery**: In the absence of an active OCR engine, both extractors fail equally on pure image-based scanned leaflets (`Banana.pdf`).

---

## 7. Limitations

- This benchmark uses a small, deliberately selected sample of PDF pages representing different layouts. It is intended as a diagnostic comparison, not as statistically representative evidence for the entire document corpus.
- EasyOCR was not evaluated because it is unavailable in the current environment.
- This is an evaluation-only benchmark; production extraction and downstream artifacts remain unaltered.
