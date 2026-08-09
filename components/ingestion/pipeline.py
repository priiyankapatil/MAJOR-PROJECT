"""Ingestion facade for the current PDF-to-ChromaDB pipeline."""

from __future__ import annotations


def update_metadata_for_new_pdf(pdf_filename: str):
    from add_new_data import update_metadata_for_new_pdf as _update
    return _update(pdf_filename)


def run_incremental_pipeline():
    from add_new_data import run_incremental_pipeline as _run
    return _run()


def extract_all_pdfs():
    from step1_extract import extract_all_pdfs as _extract
    return _extract()


def clean_all_extracted():
    from step2_clean import clean_all_extracted as _clean
    return _clean()


def chunk_all_documents():
    from step3_chunk import chunk_all_documents as _chunk
    return _chunk()


def store_all_chunks():
    from step4_store import store_all_chunks as _store
    return _store()
