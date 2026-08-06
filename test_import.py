#!/usr/bin/env python
# Test if OCR is available in the step1_extract module

try:
    from step1_extract import OCR_AVAILABLE, OCR_READER
    print(f"OCR_AVAILABLE: {OCR_AVAILABLE}")
    print(f"OCR_READER: {OCR_READER}")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
