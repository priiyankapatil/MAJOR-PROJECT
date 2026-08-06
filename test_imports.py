#!/usr/bin/env python
"""Test script to debug import issues."""

import sys

print("Starting test...", file=sys.stderr, flush=True)

try:
    print("Import 1: os", file=sys.stderr, flush=True)
    import os
    
    print("Import 2: time", file=sys.stderr, flush=True)
    import time
    
    print("Import 3: groq", file=sys.stderr, flush=True)
    from groq import Groq
    
    print("Import 4: chromadb", file=sys.stderr, flush=True)
    import chromadb
    
    print("Import 5: sentence_transformers", file=sys.stderr, flush=True)
    from sentence_transformers import SentenceTransformer
    
    print("Import 6: config", file=sys.stderr, flush=True)
    from config import EMBEDDING_MODEL, VECTOR_STORE
    
    print("All imports successful!", file=sys.stderr, flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
