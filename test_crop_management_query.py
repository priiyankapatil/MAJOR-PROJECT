#!/usr/bin/env python3
"""Test query gate with crop management specific query"""

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
