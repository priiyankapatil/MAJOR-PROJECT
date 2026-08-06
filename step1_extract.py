# step1_extract.py
# =============================================
# PHASE 3: Extract text from all PDFs
# Supports both text-based and scanned PDFs (with OCR)
# =============================================

import fitz          # pymupdf
import pdfplumber
import os
import json
from tqdm import tqdm
from config import RAW_PDF_DIR, EXTRACTED_DIR, SOURCE_METADATA
from registry import is_already_processed

# OCR support for scanned PDFs
try:
    import easyocr
    from PIL import Image
    import io
    import numpy as np
    OCR_AVAILABLE = True
    # Initialize OCR reader for English (lazy-loaded on first use)
    OCR_READER = None
except ImportError:
    OCR_AVAILABLE = False
    OCR_READER = None

os.makedirs(EXTRACTED_DIR, exist_ok=True)

def extract_all_pdfs():
    pdf_files = [
        f for f in os.listdir(RAW_PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    print(f"\n📂 Found {len(pdf_files)} PDFs in folder")
    
    new_files  = []
    skip_files = []

    for f in pdf_files:
        # Check if extracted JSON already exists
        json_path = os.path.join(
            EXTRACTED_DIR, f.replace(".pdf", ".json")
        )
        if os.path.exists(json_path):
            skip_files.append(f)
        else:
            new_files.append(f)

    print(f"   ⏭️  Skipping {len(skip_files)} "
          f"already extracted files")
    print(f"   🆕 Processing {len(new_files)} new files")

    if not new_files:
        print("\n✅ Nothing new to extract!")
        return []

    all_extracted = []

    for pdf_file in tqdm(new_files, desc="Extracting new PDFs"):
        pdf_path  = os.path.join(RAW_PDF_DIR, pdf_file)
        extracted = extract_single_pdf(pdf_path)  # your existing function
        all_extracted.append(extracted)

        out_path = os.path.join(
            EXTRACTED_DIR,
            pdf_file.replace(".pdf", ".json")
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extracted {len(new_files)} new PDFs")
    return all_extracted

os.makedirs(EXTRACTED_DIR, exist_ok=True)


def table_to_readable_text(table):
    """
    Convert a table (list of lists) into
    readable sentence-style text.
    Example:
    [["Crop","Seed rate"],["Tomato","400g/ha"]]
    → "Crop: Tomato | Seed rate: 400g/ha"
    """
    if not table or len(table) < 2:
        return ""

    headers = [str(h).strip() if h else "" for h in table[0]]
    rows_text = []

    for row in table[1:]:
        parts = []
        for header, cell in zip(headers, row):
            cell = str(cell).strip() if cell else ""
            if header and cell:
                parts.append(f"{header}: {cell}")
        if parts:
            rows_text.append(" | ".join(parts))

    return "\n".join(rows_text)


def extract_text_from_image_ocr(page_pixmap):
    """
    Extract text from a page image using OCR (EasyOCR).
    Used for scanned PDFs.
    """
    global OCR_READER
    
    if not OCR_AVAILABLE:
        return ""
    
    try:
        # Initialize reader on first use (downloads models)
        if OCR_READER is None:
            print("   📥 Initializing OCR (first run, downloading models...)")
            OCR_READER = easyocr.Reader(['en'])
        
        # Convert pixmap to PIL Image then to numpy array
        img_data = page_pixmap.tobytes("ppm")
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        # Extract text using EasyOCR (expects numpy array)
        results = OCR_READER.readtext(img_array)
        text = "\n".join([text for (bbox, text, conf) in results])
        return text.strip()
    except Exception as e:
        print(f"   ⚠️  OCR failed: {e}")
        return ""


def extract_single_pdf(pdf_path):
    """
    Extract all text + tables from one PDF.
    Supports both text-based and scanned PDFs (with OCR).
    Returns a structured dictionary.
    """
    filename  = os.path.basename(pdf_path)
    result    = {
        "filename" : filename,
        "pages"    : [],
        "total_pages": 0,
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            result["total_pages"] = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                page_data = {
                    "page_number" : page_num + 1,
                    "text"        : "",
                    "tables_text" : [],
                }

                # ── Regular text ──
                raw_text = page.extract_text()
                if raw_text:
                    page_data["text"] = raw_text.strip()

                # ── Tables ──
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        readable = table_to_readable_text(table)
                        if readable:
                            page_data["tables_text"].append(readable)

                # ── If page has no text but has images, try OCR ──
                if not page_data["text"] and OCR_AVAILABLE:
                    # Use pymupdf to render page as image and OCR it
                    doc = fitz.open(pdf_path)
                    page_fitz = doc[page_num]
                    pix = page_fitz.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                    ocr_text = extract_text_from_image_ocr(pix)
                    if ocr_text:
                        page_data["text"] = ocr_text
                    doc.close()

                result["pages"].append(page_data)

    except Exception as e:
        print(f"  ⚠️  pdfplumber failed for {filename}: {e}")
        print(f"  🔄  Falling back to pymupdf...")

        # Fallback: use pymupdf if pdfplumber fails
        try:
            doc = fitz.open(pdf_path)
            result["total_pages"] = len(doc)
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                
                # If no text extracted and OCR available, try OCR
                if not text.strip() and OCR_AVAILABLE:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    text = extract_text_from_image_ocr(pix)
                
                result["pages"].append({
                    "page_number" : page_num + 1,
                    "text"        : text.strip(),
                    "tables_text" : [],
                })
            doc.close()
        except Exception as e2:
            print(f"  ❌  Both extractors failed for {filename}: {e2}")

    return result

    return result


def extract_all_pdfs():
    """
    Extract text from new PDFs only (skip already processed).
    Supports both text-based and scanned PDFs (with OCR).
    """
    pdf_files = [
        f for f in os.listdir(RAW_PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("❌ No PDFs found in data/raw pdf/")
        return []

    # Separate new vs already processed
    new_files  = []
    skip_files = []

    for f in pdf_files:
        json_path = os.path.join(
            EXTRACTED_DIR, f.replace(".pdf", ".json")
        )
        if os.path.exists(json_path):
            skip_files.append(f)
        else:
            new_files.append(f)

    print(f"\n📂 Found {len(pdf_files)} PDFs:")
    if skip_files:
        print(f"   ⏭️  Skipping {len(skip_files)} already extracted")
    print(f"   🆕 Processing {len(new_files)} new")

    if not new_files:
        print("\n✅ Nothing new to extract!")
        return []

    all_extracted = []

    for pdf_file in tqdm(new_files, desc="Extracting PDFs"):
        pdf_path = os.path.join(RAW_PDF_DIR, pdf_file)
        print(f"\n📄 Processing: {pdf_file}")

        extracted = extract_single_pdf(pdf_path)

        # Count extracted content
        total_chars = sum(
            len(p["text"]) for p in extracted["pages"]
        )
        total_tables = sum(
            len(p["tables_text"]) for p in extracted["pages"]
        )

        print(f"   ✅ Pages: {extracted['total_pages']} | "
              f"Characters: {total_chars:,} | "
              f"Tables: {total_tables}")

        all_extracted.append(extracted)

        # Save individual JSON
        out_name = pdf_file.replace(".pdf", ".json")
        out_path = os.path.join(EXTRACTED_DIR, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extraction complete! "
          f"Saved to: {EXTRACTED_DIR}")
    return all_extracted


if __name__ == "__main__":
    extract_all_pdfs()