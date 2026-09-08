"""
Validation Script: Retrieval Ground-Truth Dataset Integrity Check
Validates evaluation/retrieval/retrieval_benchmark_dataset.json against data/chunks/all_chunks.parquet.

Verification checks:
1. All required fields exist (query_id, query, category, relevant_chunk_ids, source_files, relevance_note)
2. Every query_id is unique
3. Category distribution is balanced (exactly 6 queries per category = 30 total)
4. No empty or missing relevant_chunk_ids
5. No duplicate chunk IDs within any query
6. Every referenced chunk_id exists in all_chunks.parquet
7. Total unique relevant chunk IDs accounting
"""

import os
import sys
import json
import pandas as pd

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_PATH = "evaluation/retrieval/retrieval_benchmark_dataset.json"
PARQUET_PATH = "data/chunks/all_chunks.parquet"

EXPECTED_CATEGORIES = {
    "exact_lexical": 6,
    "conceptual_paraphrased": 6,
    "crop_specific": 6,
    "diagnostic": 6,
    "procedural_recommendation": 6,
}

REQUIRED_FIELDS = [
    "query_id",
    "query",
    "category",
    "relevant_chunk_ids",
    "source_files",
    "relevance_note",
]

def validate():
    print("=" * 70)
    print("VALIDATING RETRIEVAL GROUND-TRUTH BENCHMARK DATASET")
    print("=" * 70)

    errors = []

    # 1. Load Parquet
    if not os.path.exists(PARQUET_PATH):
        print(f"ERROR: Missing corpus parquet: {PARQUET_PATH}")
        sys.exit(1)

    df_corpus = pd.read_parquet(PARQUET_PATH, columns=["chunk_id", "source_file"])
    corpus_chunk_ids = set(df_corpus["chunk_id"].unique())
    print(f"Corpus chunks loaded: {len(corpus_chunk_ids):,} from {PARQUET_PATH}")

    # 2. Load Benchmark Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Missing benchmark dataset: {DATASET_PATH}")
        sys.exit(1)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    print(f"Benchmark queries loaded: {total_queries} from {DATASET_PATH}\n")

    if total_queries != 30:
        errors.append(f"Expected exactly 30 queries, found {total_queries}")

    seen_qids = set()
    category_counts = {}
    all_referenced_chunks = set()

    for idx, item in enumerate(dataset):
        qid = item.get("query_id", f"INDEX_{idx}")

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in item or item[field] is None:
                errors.append(f"Query {qid}: Missing required field '{field}'")

        # Check unique query_id
        if qid in seen_qids:
            errors.append(f"Duplicate query_id detected: {qid}")
        seen_qids.add(qid)

        # Check query text
        query_text = item.get("query", "").strip()
        if not query_text:
            errors.append(f"Query {qid}: Query text is empty")

        # Check category
        cat = item.get("category", "")
        if cat not in EXPECTED_CATEGORIES:
            errors.append(f"Query {qid}: Invalid category '{cat}'")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        # Check relevant_chunk_ids
        cids = item.get("relevant_chunk_ids", [])
        if not cids or not isinstance(cids, list):
            errors.append(f"Query {qid}: 'relevant_chunk_ids' is empty or not a list")
            continue

        # Check for intra-query duplicate chunk IDs
        if len(cids) != len(set(cids)):
            errors.append(f"Query {qid}: Duplicate chunk IDs found in relevant_chunk_ids: {cids}")

        # Check that every chunk_id exists in corpus
        for cid in cids:
            if cid not in corpus_chunk_ids:
                errors.append(f"Query {qid}: Chunk ID '{cid}' does NOT exist in {PARQUET_PATH}")
            all_referenced_chunks.add(cid)

    # Check category balance
    print("Query Distribution by Category:")
    print("-" * 45)
    for cat, expected in EXPECTED_CATEGORIES.items():
        actual = category_counts.get(cat, 0)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {cat:<30}: {actual:>2} / {expected} [{status}]")
        if actual != expected:
            errors.append(f"Category '{cat}': expected {expected} queries, found {actual}")
    print("-" * 45)

    print(f"\nTotal Unique Relevant Chunks Referenced: {len(all_referenced_chunks)}")
    avg_chunks = sum(len(item.get("relevant_chunk_ids", [])) for item in dataset) / max(total_queries, 1)
    print(f"Average Relevant Chunks per Query:      {avg_chunks:.2f}")

    print("\n" + "=" * 70)
    if errors:
        print(f"VALIDATION FAILED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED: 100% INTEGRITY VERIFIED")
        print("=" * 70)

if __name__ == "__main__":
    validate()
