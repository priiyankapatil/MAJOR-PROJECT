#!/usr/bin/env python
import sys
print("Test 1: Before import", flush=True)
sys.stdout.flush()

try:
    import step7_trust_scoring
    print("Test 2: After import", flush=True)
    sys.stdout.flush()
except Exception as e:
    print(f"Test ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
