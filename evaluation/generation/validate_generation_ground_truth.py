"""
evaluation/generation/validate_generation_ground_truth.py
==========================================================
Comprehensive validation suite for the Ground-Truth Answer & Rubric Dataset.

Verifies:
1. Exactly 30 records.
2. Every query_id is unique (Q01 to Q30).
3. All query_ids match evaluation/retrieval/retrieval_benchmark_dataset.json.
4. All relevant_chunk_ids exist in data/chunks/all_chunks.parquet.
5. reference_answer is non-empty string for every record.
6. expected_key_facts is non-empty list of strings for every record.
7. grading_rubric contains explicit definitions for '0', '1', and '2'.
8. safety_constraints exists as a list for every record.
9. Balanced categories: exactly 6 queries per category across all 5 categories.
10. Statistics reporting: key facts per query, safety constraints distribution.
"""

import sys
import json
from pathlib import Path
import pyarrow.parquet as pq

# Set UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE = Path(__file__).resolve().parent.parent.parent


def validate_generation_ground_truth():
    gen_path = WORKSPACE / "evaluation" / "generation" / "generation_benchmark_dataset.json"
    ret_path = WORKSPACE / "evaluation" / "retrieval" / "retrieval_benchmark_dataset.json"
    parquet_path = WORKSPACE / "data" / "chunks" / "all_chunks.parquet"

    print("=" * 80)
    print(" 🌾 VALIDATING GENERATION GROUND-TRUTH ANSWER & RUBRIC DATASET")
    print("=" * 80)

    errors = []
    warnings = []

    # 1. Load files
    if not gen_path.exists():
        print(f"❌ Generation benchmark dataset not found at: {gen_path}")
        return False

    if not ret_path.exists():
        print(f"❌ Retrieval benchmark dataset not found at: {ret_path}")
        return False

    if not parquet_path.exists():
        print(f"❌ Parquet corpus not found at: {parquet_path}")
        return False

    with open(gen_path, "r", encoding="utf-8") as f:
        gen_data = json.load(f)

    with open(ret_path, "r", encoding="utf-8") as f:
        ret_data = json.load(f)

    table = pq.read_table(parquet_path)
    corpus_cids = set(table.column("chunk_id").to_pylist())
    print(f"✓ Loaded active corpus with {len(corpus_cids)} chunks from all_chunks.parquet")

    # 2. Record count check
    total_records = len(gen_data)
    print(f"\n1. RECORD COUNT CHECK:")
    print(f"   Total records: {total_records}")
    if total_records != 30:
        errors.append(f"Expected exactly 30 records, found {total_records}")
    else:
        print("   ✓ Exactly 30 records confirmed.")

    # 3. Unique query_ids and correspondence with retrieval benchmark
    print(f"\n2. QUERY ID & CORRESPONDENCE CHECK:")
    ret_dict = {q["query_id"]: q for q in ret_data}
    seen_qids = set()
    category_counts = {}

    for i, item in enumerate(gen_data):
        qid = item.get("query_id")
        if not qid:
            errors.append(f"Record #{i} is missing query_id")
            continue

        if qid in seen_qids:
            errors.append(f"Duplicate query_id found: {qid}")
        seen_qids.add(qid)

        if qid not in ret_dict:
            errors.append(f"query_id {qid} not found in retrieval benchmark dataset")
        else:
            ret_item = ret_dict[qid]
            # Verify query text matches
            if item.get("query") != ret_item.get("query"):
                errors.append(f"Query text mismatch for {qid}")
            # Verify category matches
            if item.get("category") != ret_item.get("category"):
                errors.append(f"Category mismatch for {qid}")
            # Verify relevant_chunk_ids match
            if set(item.get("relevant_chunk_ids", [])) != set(ret_item.get("relevant_chunk_ids", [])):
                errors.append(f"Relevant chunk IDs mismatch for {qid}")

        cat = item.get("category", "UNKNOWN")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if not errors:
        print(f"   ✓ All 30 query_ids unique and exactly match retrieval benchmark dataset.")

    # 4. Category Balance
    print(f"\n3. CATEGORY DISTRIBUTION CHECK:")
    expected_categories = {
        "exact_lexical": 6,
        "conceptual_paraphrased": 6,
        "crop_specific": 6,
        "diagnostic": 6,
        "procedural_recommendation": 6
    }
    for cat, exp_count in expected_categories.items():
        actual_count = category_counts.get(cat, 0)
        status = "✓" if actual_count == exp_count else "❌"
        print(f"   {status} {cat:<28}: {actual_count} / {exp_count}")
        if actual_count != exp_count:
            errors.append(f"Category {cat} has {actual_count} items, expected {exp_count}")

    # 5. Field Integrity & Content Checks
    print(f"\n4. FIELD INTEGRITY & SUBSTANCE CHECK:")
    total_key_facts = 0
    queries_with_safety = 0
    missing_cids = set()

    for item in gen_data:
        qid = item.get("query_id", "UNKNOWN")
        ref_ans = item.get("reference_answer", "")
        facts = item.get("expected_key_facts", [])
        rubric = item.get("grading_rubric", {})
        safety = item.get("safety_constraints", None)
        cids = item.get("relevant_chunk_ids", [])

        # Check reference answer
        if not isinstance(ref_ans, str) or len(ref_ans.strip()) < 20:
            errors.append(f"{qid}: reference_answer is missing or too short")

        # Check key facts
        if not isinstance(facts, list) or len(facts) == 0:
            errors.append(f"{qid}: expected_key_facts is missing or empty")
        else:
            total_key_facts += len(facts)
            for f_text in facts:
                if not isinstance(f_text, str) or len(f_text.strip()) == 0:
                    errors.append(f"{qid}: contains an empty key fact")

        # Check rubric
        if not isinstance(rubric, dict):
            errors.append(f"{qid}: grading_rubric must be a dictionary")
        else:
            for score_key in ["0", "1", "2"]:
                if score_key not in rubric or len(rubric[score_key].strip()) == 0:
                    errors.append(f"{qid}: grading_rubric missing definition for score '{score_key}'")

        # Check safety constraints
        if not isinstance(safety, list):
            errors.append(f"{qid}: safety_constraints must be a list")
        else:
            if len(safety) > 0:
                queries_with_safety += 1

        # Check chunks exist in parquet
        for cid in cids:
            if cid not in corpus_cids:
                missing_cids.add(cid)
                errors.append(f"{qid}: relevant chunk {cid} does not exist in all_chunks.parquet")

    if not errors:
        print("   ✓ All reference answers non-empty and well-formed.")
        print("   ✓ All expected_key_facts non-empty lists.")
        print("   ✓ All grading_rubrics contain complete 0/1/2 definitions.")
        print("   ✓ All safety_constraints properly typed.")
        print(f"   ✓ All {len(seen_qids)} queries map to verified chunks in the active 12,856 corpus.")

    # 6. Summary Statistics
    avg_facts = total_key_facts / total_records if total_records else 0
    print(f"\n5. BENCHMARK SUMMARY STATISTICS:")
    print(f"   • Total Ground-Truth Records  : {total_records}")
    print(f"   • Total Expected Key Facts    : {total_key_facts}")
    print(f"   • Average Key Facts per Query : {avg_facts:.2f}")
    print(f"   • Queries with Safety Rules   : {queries_with_safety} / {total_records} ({queries_with_safety/total_records*100:.1f}%)")
    print(f"   • Missing Chunks from Corpus  : {len(missing_cids)}")

    # 7. Verification Result
    print("\n" + "=" * 80)
    if errors:
        print(f"❌ VALIDATION FAILED WITH {len(errors)} ERRORS:")
        for err in errors[:10]:
            print(f"   • {err}")
        return False
    else:
        print("✅ VALIDATION RESULT: PASSED (100% compliant with ground-truth requirements)")
        print("=" * 80)
        return True


if __name__ == "__main__":
    success = validate_generation_ground_truth()
    sys.exit(0 if success else 1)
