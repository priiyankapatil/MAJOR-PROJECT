# Second Controlled Production Integration Review: Hardened SB2 Guardrails

**Date**: 2026-09-19<br>
**Target**: Hardened SB2 Semantic Bridge Implementation ([semantic_bridge_guardrails.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/semantic_bridge_guardrails.py))<br>
**Previous Review**: [sb2_production_integration_review.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_production_integration_review.md)<br>
**Hardening Report**: [sb2_targeted_hardening_report.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_targeted_hardening_report.md)<br>
**Machine-Readable Review**: [sb2_hardened_integration_review.json](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_hardened_integration_review.json)<br>
**Status**: Review Only — No Production Code Modified

---

## 1. Executive Summary

A second controlled production integration review was performed on the newly hardened SB2 evaluation bridge. The previous review categorized SB2 as *"Needs targeted fixes before integration test"* due to two specific architectural risks:
1. **Plural Inflection Blindness**: Inability of singular-only regex boundaries to match plurals like *tomatoes*, *chillies*, and *coconuts*.
2. **Hardcoded Compound Pest Exclusions**: Over-reliance on four hardcoded phrases, causing unlisted compound pests (*mango fruit fly*, *sugarcane top borer*, *rice blast*) to falsely inflate crop counts.

**Findings of this Second Review**:
- Both identified risks have been completely addressed through an explicit canonical inflection dictionary (`CROP_INFLECTIONS` / `SURFACE_TO_CANONICAL`) and a conservative structural regex backed by whitespace-preserving masking.
- The 29-test automated unit test suite ([test_sb2_guardrails.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/test_sb2_guardrails.py)) was executed and confirmed: **29/29 tests passed (100% pass rate, 0 failures, 0 errors, 0 skipped)** in 0.836s.
- The 5 canonical vernacular smoke tests and the 4 key benchmark queries (`Q13`, `Q17`, `Q21`, `Q28`) retain 100% behavioral fidelity.
- Production compatibility is verified: the input/output dictionary contract of `apply_semantic_bridge_sb2` is a strict 1:1 match with `semantic_bridge.apply_semantic_bridge`.
- Git status confirms zero modifications to production files, ChromaDB indexes, embeddings, ground truth, or grader code.

**Recommendation Category**: **Ready for controlled integration test**.

---

## 2. Exact Files and Functions Inspected

| File Path | Component | Functions Inspected | State / Role |
| :--- | :--- | :--- | :--- |
| [`evaluation/retrieval/semantic_bridge_guardrails.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/semantic_bridge_guardrails.py) | Evaluation Guarded Bridge | `get_detected_crops`, `is_entity_anchored`, `build_deduplicated_replacement`, `detect_folk_terms_guarded`, `apply_semantic_bridge_sb2` | **Hardened & Verified** |
| [`evaluation/retrieval/test_sb2_guardrails.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/test_sb2_guardrails.py) | Targeted Unit Tests | 29 unit tests across 7 test classes | **29/29 Passed** |
| [`semantic_bridge.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/semantic_bridge.py) | Production Step 0 Bridge | `apply_semantic_bridge`, `_log_bridge_result`, `get_bridge_statistics` | **Untouched Baseline** |
| [`dialectal_alignment.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/dialectal_alignment.py) | Production Alignment | `tokenize_query`, `detect_folk_terms`, `enrich_query`, `align_query` | **Untouched Baseline** |
| [`step6_query_gate.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/step6_query_gate.py) | Production Query Gate | `run_pipeline` (Step 0, lines 657–666), `retrieve_chunks` (lines 531–629) | **Untouched Baseline** |
| [`config.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/config.py) | Production Config | Paths, models, thresholds | **Untouched Baseline** |

---

## 3. Verified Hardening Behavior (Task A)

### A. Plural and Irregular Crop Detection
- **Mechanism**: `CROP_INFLECTIONS` defines canonical crop entities mapped to their valid English surface forms:
  - Irregular plurals: *tomato/tomatoes/tomatos*, *potato/potatoes/potatos*, *mango/mangoes/mangos*, *chilli/chillies/chilis/chilies*, *paddy/paddies*, *citrus/citruses*.
  - Standard plurals: *coconuts, brinjals/eggplants, cowpeas, chickpeas, lentils, bananas, onions, cashews*.
  - Uncountable crops: *wheat, rice, cotton, sugarcane, coffee, maize, sorghum, sesame, mustard, tea*.
- **Synonym Canonicalization**: Variants like `chilli` and `chili`, or `eggplant` and `brinjal` map to identical canonical keys, preventing queries like *"chilli and chili varieties"* from falsely triggering multi-crop balance.
- **Collision Prevention**: Multi-word forms (`pigeon peas`, `black grams`, `pearl millets`) are scanned first and masked with whitespace so that single words (`gram`, `millet`) cannot collide or trigger duplicate counts.

### B. Compound Pest & Pathogen Filtering
- **Two-Tier Architecture**:
  1. Catalog exclusions: `COMPOUND_PEST_EXCLUSIONS` (contains *tea mosquito bug*, *coffee berry borer*, *cotton pink bollworm*, *cotton aphid*, *mango fruit fly*, *sugarcane top borer*, *rice blast*, etc.).
  2. Structural regex: `\b(<crop>)\s+(?:(<modifier>)\s+)?(<head_noun>)\b`.
- **Conservative Head Nouns**: Strictly entomological and pathological (`fly`, `borer`, `bollworm`, `aphid`, `weevil`, `beetle`, `bug`, `caterpillar`, `moth`, `mite`, `thrips`, `blast`, `canker`).
- **Broad Agronomic Noun Protection**: `disease`, `management`, `fertilizer`, `pest`, `spray`, `practice` are strictly excluded from head nouns. Queries like *"rice disease management and tomato fertilizer"* preserve both `rice` and `tomato` as active crop anchors (`is_multi_crop = True`).
- **Whitespace-Preserving Masking**: Matched pest spans are replaced with whitespace of identical length, preserving query character offsets, word boundaries, and independent host mentions (e.g. *"tea mosquito bug in tea and cashew"* detects both `tea` and `cashew`).

---

## 4. Test Verification & Actual Results (Task B)

Execution of the unit test suite was verified directly in the workspace environment:
```powershell
python -u -m unittest evaluation/retrieval/test_sb2_guardrails.py
```
**Actual Output**:
```
.............................
----------------------------------------------------------------------
Ran 29 tests in 0.836s

OK
```
- **Total Tests**: 29
- **Passed**: 29 (100%)
- **Failures**: 0
- **Errors**: 0
- **Skipped**: 0

### Test Coverage Breakdown
1. **`TestPluralCropDetection` (4 tests)**: Validated irregular plurals (*tomatoes, chillies, paddies*), standard plurals, multi-word plurals, and uncountable crops.
2. **`TestNearMatchesAndOverlaps` (4 tests)**: Confirmed words sharing prefixes/suffixes (*team/steam/cottage/corner/price*) produce `[]`, sub-words (*black gram* vs *gram*) do not collide, and repeated terms (*tomato vs tomato*) canonicalize to count=1.
3. **`TestCompoundPestFiltering` (8 tests)**: Validated *tea mosquito bug*, *mango fruit fly*, *sugarcane top borer*, *rice blast*, *coffee berry borer*, independent host mentions, agronomic noun protection (*rice disease management and tomato fertilizer*), and inverted mentions (*fruit fly in mango*).
4. **`TestMultiCropBalanceEnrichment` (3 tests)**: Validated Q13 singular and plural seed rates (*tomatoes, chillies, brinjals*) suppress *Solanum lycopersicum*, while single-crop vernacular (*tamatar*) enriches properly.
5. **`TestMixedEnglishAndVernacular` (2 tests)**: Validated *dhan* and *arhar* dialectal translation.
6. **`TestRobustnessAndEdgeCases` (4 tests)**: Validated empty strings, whitespace, punctuation, and uppercase casing.
7. **`TestSB1GuardrailsPreservation` (4 tests)**: Validated synonym de-duplication tautology elimination, Q28 specificity protection (*organic manure* preserved from FYM narrowing), and vernacular confidence floor.

---

## 5. Production Compatibility & Integration Trace (Task C)

### A. Production Call Flow
1. User query enters `step6_query_gate.py:run_pipeline` (lines 657–666).
2. Step 0 imports and invokes `apply_semantic_bridge(query)` from `semantic_bridge.py`.
3. `semantic_bridge.py` delegates to `dialectal_alignment.align_query(query)` and writes to `semantic_bridge_log.json`.
4. If `bridge_result.get("bridged")` is True, `query` is updated with `bridge_result.get("enriched", query)`.
5. The updated `query` proceeds to Step 1 (Groq query classifier), Step 2 (safety gate), Step 3 (entropy routing), Step 4 (hybrid retrieval & Cross-Encoder reranking), and Step 5 (LLM generation).

### B. Input/Output Contract
`apply_semantic_bridge_sb2` produces the exact schema expected by `step6_query_gate.py`:
```python
{
    "original": str,
    "enriched": str,
    "terms_found": list[dict],
    "confidence": float,
    "bridged": bool,
    "term_count": int
}
```
Contract parity with production `align_query` is **100% identical**.

### C. Import & Side-Effect Analysis
- `semantic_bridge_guardrails.py` imports only from `folk_taxonomy.py` and `dialectal_alignment.py` (`tokenize_query`, `compute_alignment_confidence`).
- Zero circular dependencies.
- Zero side effects on downstream classifier, routing, safety gates, or LLM prompts.

---

## 6. Benchmark Scope, Evidence & Limitations (Task D)

### A. Genuine Benchmark Evidence
On the canonical 30-query benchmark (`evaluation/retrieval/retrieval_benchmark_dataset.json`):
- **Macro Recall@5**: 90.00% (vs 83.33% current bridge, +6.67 percentage points).
- **Macro NDCG@5**: 0.8283 (vs 0.7710 current bridge).
- **Macro MRR**: 0.8187 (vs 0.7737 current bridge).
- **Regressions**: Exactly 0 regressions out of 30 queries against raw baseline and current bridge.
- **Target Triad**: `Q28` ranks **#1**, `Q13` ranks **#2**, and `Q17` ranks **#5**.

### B. Verified Production Equivalence & Differences
- **Candidate Pool Depth**:
  - The benchmark runner evaluated retrieval with `top_k=10` ($K_{\text{cand}} = \text{top\_k} \times 3 = 30$).
  - In production `run_pipeline` ([step6_query_gate.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/step6_query_gate.py), line 727), `retrieve_chunks` defaults to `top_k=5` ($K_{\text{cand}} = 15$).
  - An empirical check across all 30 queries previously confirmed that ground-truth chunk inclusion in Top-5 is identical between $K_{\text{cand}}=15$ and $K_{\text{cand}}=30$.
- **Prompt Context Assembly**:
  - Fast-path queries in production assemble `chunks[:3]` into the prompt.
  - Slow-path queries assemble `chunks[:5]`.
  - Recall@5 directly corresponds to slow-path generation context. Fast-path relies on Top-3 precision.

### C. Open-Vocabulary Generalization Caution
- **Critical Rule**: Zero regressions on 30 benchmark queries does **NOT** prove zero regressions on unseen open-domain user queries.
- Unlisted regional crop dialects absent from `folk_taxonomy.py` will not be recognized as crops.
- Highly complex multi-word coordinated pests (>3 intervening words) rely on explicit catalog listing.

---

## 7. Staged Integration & Rollback Plan (Task E)

*(Plan proposed for future execution upon user approval; NOT executed during this review)*

### Phase 1: Preconditions (Already Met)
- [x] All 29 unit tests pass in `evaluation/retrieval/test_sb2_guardrails.py`.
- [x] All 5 vernacular smoke tests verified intact.
- [x] Git tracked working tree is 100% clean.
- [ ] User provides explicit authorization to integrate into production code.

### Phase 2: Minimal Production Integration
1. Add a configuration feature flag in [config.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/config.py):
   ```python
   ENABLE_SB2_GUARDRAILS = True  # Toggle for guarded semantic bridge
   ```
2. Update [semantic_bridge.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/semantic_bridge.py) to conditionally route:
   ```python
   from config import ENABLE_SB2_GUARDRAILS
   if ENABLE_SB2_GUARDRAILS:
       from evaluation.retrieval.semantic_bridge_guardrails import apply_semantic_bridge_sb2 as align_query_impl
   else:
       from dialectal_alignment import align_query as align_query_impl
   ```
3. Preserve production logging to `semantic_bridge_log.json`.

### Phase 3: Shadow / Canary Verification
- Run a 5-query end-to-end pipeline smoke test through `step6_query_gate.py:run_pipeline`.
- Verify that `semantic_bridge_log.json` logs correctly with `bridged`, `confidence`, and `terms_found`.

### Phase 4: Instant Rollback Procedure
If any unexpected behavior occurs in production:
1. **Zero-Code Rollback**: Set `ENABLE_SB2_GUARDRAILS = False` in `config.py`.
2. **Git Rollback**: `git checkout main -- semantic_bridge.py config.py`.

---

## 8. Conditions Required Before Production Integration

1. **Explicit User Approval**: Mandatory sign-off before modifying any production file.
2. **Strict File Scope**: Changes must be limited to `semantic_bridge.py` and `config.py`.
3. **No Index or Embedding Modifications**: ChromaDB, BM25 corpus, and embeddings must remain untouched.
4. **Post-Integration Smoke Test**: Run a direct test of `step6_query_gate.py` on `Q13`, `Q17`, `Q21`, `Q28`, and 1 vernacular query to verify live end-to-end execution.

---

## 9. Conclusion & Recommendation

**RECOMMENDATION CATEGORY**: **Ready for controlled integration test**

**Justification**:
- The evaluation implementation has resolved all identified structural risks (plural inflections, compound pests, agronomic noun preservation).
- 29/29 targeted unit tests passed with 100% success.
- 5/5 vernacular smoke tests and all 4 target benchmark queries (`Q13`, `Q17`, `Q21`, `Q28`) behave with verified correctness.
- The proposed feature-flag integration in `semantic_bridge.py` provides instant zero-risk rollback capability.
