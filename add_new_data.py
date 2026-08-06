# add_new_data.py
# =============================================
# ONE SCRIPT TO RUN WHEN YOU ADD NEW PDFs
# Just drop PDFs into data/raw pdf/ and run this
# =============================================

import os
from config import SOURCE_METADATA
from registry import get_registry_summary

def update_metadata_for_new_pdf(pdf_filename):
    """
    Helper to add metadata for a new PDF.
    Call this before running the pipeline.
    """
    if pdf_filename in SOURCE_METADATA:
        return  # Already configured

    print(f"\n⚠️  No metadata found for: {pdf_filename}")
    print("Adding default metadata. Please update config.py manually.")

    # Add default entry — user should update this
    SOURCE_METADATA[pdf_filename] = {
        "title"          : pdf_filename.replace(".pdf", ""),
        "source"         : "Agricultural University",
        "publisher"      : "India",
        "year"           : 2024,
        "authority_score": 0.9,   # High trust by default for university PDFs
        "bias_penalty"   : 0.0,
        "domain"         : "agriculture",
        "region"         : "India",
    }


def run_incremental_pipeline():
    """
    Full incremental pipeline:
    1. Check for new PDFs
    2. Extract text from new PDFs only
    3. Clean new extracted text only
    4. HDR chunk new PDFs only
    5. Store new chunks in ChromaDB only
    """
    print("\n" + "="*55)
    print("  🌾 AGRICULTURE RAG — INCREMENTAL UPDATE")
    print("="*55)

    from config import RAW_PDF_DIR, EXTRACTED_DIR
    from registry import is_already_processed

    # ── Find truly new PDFs ──
    all_pdfs = [
        f for f in os.listdir(RAW_PDF_DIR)
        if f.lower().endswith(".pdf")
    ]

    new_pdfs = [
        f for f in all_pdfs
        if not is_already_processed(f)
    ]

    if not new_pdfs:
        print("\n✅ No new PDFs found!")
        print("   Drop new PDFs into data/raw pdf/ and run again.")
        get_registry_summary()
        return

    print(f"\n🆕 Found {len(new_pdfs)} new PDFs:")
    for f in new_pdfs:
        print(f"   • {f}")

    # Ensure all new PDFs have metadata
    for pdf in new_pdfs:
        update_metadata_for_new_pdf(pdf)

    # ── Run each step ──
    print("\n" + "─"*55)
    print("STEP 1/4: Extracting text...")
    print("─"*55)
    from step1_extract import extract_all_pdfs
    extract_all_pdfs()

    print("\n" + "─"*55)
    print("STEP 2/4: Cleaning text...")
    print("─"*55)
    from step2_clean import clean_all_extracted
    clean_all_extracted()

    print("\n" + "─"*55)
    print("STEP 3/4: HDR Chunking...")
    print("─"*55)
    from step3_chunk import chunk_all_documents
    chunk_all_documents()

    print("\n" + "─"*55)
    print("STEP 4/4: Storing in ChromaDB...")
    print("─"*55)
    from step4_store import store_all_chunks
    store_all_chunks()

    print("\n" + "="*55)
    print("✅ INCREMENTAL UPDATE COMPLETE!")
    print("="*55)
    get_registry_summary()


if __name__ == "__main__":
    run_incremental_pipeline()