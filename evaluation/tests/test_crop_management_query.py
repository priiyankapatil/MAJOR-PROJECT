import sys
import os

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("NO_INTERACTIVE_FEEDBACK", "1")

from step6_query_gate import QueryGate

qg = QueryGate()

# Test with crop-management-specific query
query = "What are the best crop management practices for increasing yield?"

print("\n" + "="*70)
print(f"Query: {query}")
print("="*70)

result = qg.process_query(query)

print(f"\nSources found:")
for source in set(m.get("source_file", "Unknown") for m in result.get("source_chunks", [])):
    print(f"  • {source}")

print(f"\nTop result:")
if result.get("answer"):
    print(result["answer"][:200])
