#!/usr/bin/env python
"""Pre-download and cache the sentence-transformers model."""

import sys
print("Pre-loading embedding model...", flush=True)

try:
    from sentence_transformers import SentenceTransformer
    print("Downloading model...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✓ Model loaded and cached", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
