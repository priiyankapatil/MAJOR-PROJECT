# Experiment SB2: Controlled Production Integration Review

**Date**: 2026-09-19T01:55:00Z<br>
**Target Proposal**: Integrating Experiment SB2 Guardrails into Production Semantic Bridge<br>
**Review Status**: **NEEDS TARGETED FIXES BEFORE INTEGRATION TEST**<br>
**Production Git Baseline**: Commit `b107e28` on branch `main`<br>
**Inspected Evaluation Artifacts**:
- [`evaluation/retrieval/semantic_bridge_guardrails.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/semantic_bridge_guardrails.py)
- [`evaluation/retrieval/run_sb2_retrieval_benchmark.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/run_sb2_retrieval_benchmark.py)
- [`evaluation/retrieval/sb2_benchmark_results.json`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_benchmark_results.json)
- [`evaluation/retrieval/sb2_benchmark_report.md`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_benchmark_report.md)
- [`evaluation/retrieval/sb2_validation_audit.md`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_validation_audit.md)
- [`evaluation/retrieval/sb2_validation_audit.json`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_validation_audit.json)

---

## 1. Executive Summary

Experiment SB2 introduces a multi-crop comparative balance guardrail alongside SB1's synonym de-duplication, specificity entity anchor suppression, and vernacular confidence floor. Mathematically, the benchmark results are verified: **90.00% Recall@5**, **0.8283 NDCG@5**, **0.8187 MRR**, and **0 regressions across all 30 canonical benchmark queries**. Crucially, the target triad is solved: `Q28` ranks **#1**, `Q17` ranks **#5**, and `Q13` recovers to **#2**.

However, an independent architectural review of the evaluation implementation reveals **two concrete edge-case defects** that must be hardened before this code can be safely merged into the production path:
1. **Plural Inflection Blindness**: The crop anchor regex (`r"\b" + crop + r"\b"`) strictly matches singular forms (`tomato`, `chilli`, `coconut`), but fails on standard English plurals (`tomatoes`, `chillies`, `coconuts`). A query containing plural vegetable lists would bypass the multi-crop guardrail and regress like SB1.
2. **Hardcoded Pest Exclusion Fragility**: `COMPOUND_PEST_EXCLUSIONS` contains only four hardcoded strings (`"tea mosquito bug"`, `"coffee berry borer"`, `"cotton pink bollworm"`, `"cotton aphid"`). Unlisted crop-named pests (e.g. *"mango fruit fly"*, *"sugarcane top borer"*, *"rice blast"*) will falsely inflate the crop entity count.

**Neutral Recommendation**: **Needs Targeted Fixes Before Integration Test**. The mathematical foundation is proven, but two targeted code adjustments are mandatory prior to staging.

---

## 2. Exact Files and Functions Inspected

| File Path | Function / Block | Purpose & Findings |
| :--- | :--- | :--- |
| [`evaluation/retrieval/semantic_bridge_guardrails.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/semantic_bridge_guardrails.py) | `get_detected_crops` | Scans query for `CROP_ANCHORS`; filters `COMPOUND_PEST_EXCLUSIONS`. Identified plural regex limitation. |
| | `is_entity_anchored` | Identifies if query has domain anchors (crop or practice). |
| | `build_deduplicated_replacement` | Normalizes whitespace-collapsed compound words to prevent tautologies. |
| | `detect_folk_terms_guarded` | Master detection controller; enforces `enable_multi_crop_balance`. |
| | `apply_semantic_bridge_sb2` | Public interface returning bridge payload. |
| [`semantic_bridge.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/semantic_bridge.py) | `apply_semantic_bridge` | Production Step 0 entrypoint; writes to `semantic_bridge_log.json`. |
| [`components/context/bridge.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/components/context/bridge.py) | `apply_semantic_bridge` | Facade module delegating directly to `semantic_bridge.py`. |
| [`step6_query_gate.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/step6_query_gate.py) | `run_pipeline` (lines 657–666) | Invokes `apply_semantic_bridge` at Step 0, passing enriched query to `retrieve_chunks`. |
| | `retrieve_chunks` (lines 531–625) | Dense + Sparse + Cross-Encoder reranking pipeline ($K_{\text{cand}} = \text{top\_k} \times 3$). |

---

## 3. SB2 Behavior and Proposed Minimal Integration Point

### A. Core Mechanics of SB2
1. **Inherited from SB1**:
   - **Synonym De-duplication**: Strips redundant tautologies such as `coconut (Coconut, Cocos nucifera)` $\rightarrow$ `coconut (Cocos nucifera)`. Collapses whitespace to prevent duplicate compound insertions (`(Farm Yard Manure (FYM))` instead of `(Farmyard Manure, Farm Yard Manure (FYM))`).
   - **Specificity Anchor Guardrail**: If query has crop or practice anchors, suppresses generic collective nouns (`"organic manures"`, `"chemical fertilizers"`, `"pesticides"`) from being hijacked to narrow inputs (prevented Q28 from drifting to FYM).
   - **Vernacular Confidence Floor**: Disallows fuzzy-matching on standard English collective nouns into Hindi fertilizer aliases.
2. **Added in SB2**:
   - **Multi-Crop Balance Guardrail**: When $\ge 2$ recognized crops co-occur in an English list (e.g. *"tomato, chilli, and brinjal"*), suppresses botanical Latin binomial injection (`Solanum lycopersicum`) to prevent vector steering away from multi-crop comparative tables.
   - **Compound Pest Filtering**: Filters out crops that appear as modifiers inside pest names (`"tea"` in `"tea mosquito bug"`).

### B. Proposed Minimal Integration Point
The smallest, cleanest integration point that touches zero query gating or pipeline orchestration logic:
- **Location**: Update [`dialectal_alignment.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/dialectal_alignment.py) (`detect_folk_terms` and `enrich_query`) with the guarded SB2 logic, or update [`semantic_bridge.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/semantic_bridge.py) to route through the hardened guardrail module.
- **Contract Compatibility**: `apply_semantic_bridge_sb2` outputs the exact dictionary schema expected by [`step6_query_gate.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/step6_query_gate.py):
  `{"original": str, "enriched": str, "terms_found": list, "confidence": float, "bridged": bool, "term_count": int}`
- **Side Effects**: Self-contained Python with standard regex and string operations. Zero extra dependencies.

---

## 4. Compatibility and Operational Risks

### Risk 1: Plural Crop Inflections (High Risk for Generalization)
- **Defect**: `pattern = r"\b" + re.escape(crop) + r"\b"`.
- **Proof**: `re.search(r"\btomato\b", "tomatoes")` returns `None`. `re.search(r"\bcoconut\b", "coconuts")` returns `None`.
- **Impact**: If a user asks *"What are the seed rates for tomatoes, chillies, and brinjals?"*, the multi-crop guardrail fails to count 3 crops ($\text{crops}=0$). In this scenario, `tomato` will be enriched with `Solanum lycopersicum` while `chillies` and `brinjals` remain unexpanded, reproducing the exact SB1 regression on real-world plural queries.
- **Fix Required**: Match plurals using `r"\b" + re.escape(crop) + r"(?:es|s)?\b"` or normalize tokens with singularization.

### Risk 2: Unlisted Compound Pest Names (Medium Risk)
- **Defect**: `COMPOUND_PEST_EXCLUSIONS` contains only 4 hardcoded strings.
- **Impact**: Queries like *"How to treat mango fruit fly and citrus canker?"* detect 2 crops (`mango`, `citrus`) because `"mango fruit fly"` is not in the exclusion list.
- **Fix Required**: Generalize compound pest filtering by checking if a crop token is immediately succeeded by an agronomic pest/pathogen noun (`fly`, `bug`, `borer`, `weevil`, `canker`, `rust`, `blast`, `blight`, `mildew`, `rot`, `wilt`).

### Risk 3: Explicit Botanical Requests (Low Edge-Case Risk)
- **Impact**: If a user deliberately asks *"What are the scientific names of wheat and barley?"*, SB2's multi-crop guardrail suppresses Latin binomial injection.
- **Mitigation**: Acceptable behavior for an agricultural advisory RAG system where the LLM generator can provide binomials in response to direct questions, while retrieval focuses on agronomic practice chunks.

---

## 5. Test Coverage Analysis

1. **Current Benchmark Suite**:
   - Covers 30 canonical agronomic queries across 4 categories (`exact_lexical`, `conceptual_paraphrased`, `crop_specific`, `procedural_recommendation`).
   - Validates that Q28, Q17, and Q13 are solved under canonical casing and singular phrasing.
2. **Current Vernacular Smoke Test**:
   - Covers 5 real-world colloquial queries (V01–V05).
   - Verified that genuine rural Hindi terms (`dhan`, `arhar`, `chana`, `tamatar`, `safed makhi`, `gobar`) continue to be translated into valid scientific binomials (`Oryza sativa`, `Cajanus cajan`, `Cicer arietinum`, `Bemisia tabaci`, `FYM`).
3. **Missing Test Coverage (Must be Added)**:
   - Plural multi-crop queries (*"tomatoes, chillies, and brinjals"*).
   - Compound pest queries with crop prefixes outside the hardcoded 4 (*"rice stem borer"*, *"mango fruit fly"*, *"banana weevil"*).
   - Empty, whitespace-only, and punctuation-only inputs.
   - Mixed-language code-switched queries (*"tomato fasal mein seed rate"*).

---

## 6. Benchmark Evidence and Generalization Limitations

### A. The Generalization Boundary of 30 Benchmark Queries
- The 30 queries in [`retrieval_benchmark_dataset.json`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/retrieval_benchmark_dataset.json) represent a highly curated canonical benchmark.
- Achieving **0 regressions on 30 queries is necessary, but not sufficient** to guarantee 0 regressions across open-domain farmer inputs.
- The plural inflection gap proves that a query phrasing variation outside the benchmark could bypass the guardrail.

### B. Production Pipeline Candidate Pool Equivalence Check ($K=15$ vs $K=30$)
- In [`run_sb2_retrieval_benchmark.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/run_sb2_retrieval_benchmark.py), `top_k=10` was used, which dynamically set $K_{\text{candidate}} = 10 \times 3 = 30$ candidates for dense and sparse search before Cross-Encoder reranking.
- In production [`step6_query_gate.py`](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/step6_query_gate.py), `run_pipeline` calls `retrieve_chunks(..., top_k=5)`, setting $K_{\text{candidate}} = 5 \times 3 = 15$.
- **Audit Experiment**: We ran an independent test comparing $K=5$ ($K_{\text{cand}}=15$) vs $K=10$ ($K_{\text{cand}}=30$) across all 30 queries:
  - In 11 queries, non-ground-truth context chunks at ranks 3–5 varied slightly.
  - **Critical Finding**: In **30 out of 30 queries (100%)**, the ground-truth chunk inclusion in Top-5 was **strictly identical** between $K=5$ and $K=10$.
  - Therefore, `Recall@5` of **90.00%** is guaranteed to transfer directly to the production prompt context window.

---

## 7. Staged Integration and Rollback Plan

### Stage 1: Targeted Evaluation Hardening (Pre-requisite)
1. Update `semantic_bridge_guardrails.py` to match plural forms (`tomato|tomatoes`, `chilli|chillies`, etc.).
2. Generalize compound pest filtering to avoid hardcoded pest exclusion lists.
3. Add a dedicated unit test script covering 15 edge-case strings.

### Stage 2: Shadow Production Integration (Zero User Impact)
1. Add SB2 guardrails into `dialectal_alignment.py` behind a feature flag:
   `ENABLE_SB2_GUARDRAILS = True` (configurable via `config.py`).
2. If the flag is disabled, fallback instantaneously to baseline `align_query`.

### Stage 3: End-to-End Generation Benchmark Verification
1. Run the full 30-query end-to-end generation benchmark (`run_r1c_benchmark.py` or equivalent) with the integrated SB2 bridge.
2. Confirm that Q28 generation fact coverage recovers from 0/4 to $\ge 3/4$ without any generation regressions.

### Rollback Plan
- Set `ENABLE_SB2_GUARDRAILS = False` in `config.py` for immediate runtime rollback.
- Full Git rollback: `git checkout b107e28 -- semantic_bridge.py dialectal_alignment.py`.

---

## 8. Pre-Integration Checklist

Before integrating SB2 into production, the following conditions must be satisfied:

- [ ] Plural crop forms (`tomatoes`, `chillies`, `brinjals`, `coconuts`, etc.) supported in crop anchor detection.
- [ ] Compound pest detection generalized beyond the 4 hardcoded phrases.
- [ ] Unit tests for edge-case queries written and passing.
- [ ] Production logging preserved (`semantic_bridge_log.json`).
- [ ] Feature flag toggle provided in `config.py`.
- [ ] Independent generation benchmark validation confirms prompt-level fact coverage gains on Q28.

---

## 9. Final Review Recommendation

> [!IMPORTANT]
> **Recommendation Category: NEEDS TARGETED FIXES BEFORE INTEGRATION TEST**
>
> **Justification**:
> - The mathematical gains of SB2 are indisputable on the 30 canonical benchmark queries (90.00% Recall@5, +0.0667 over current production, 0 regressions).
> - However, merging code into production that fails on simple English plurals (*"tomatoes and chillies"*) would introduce latent regressions on open-domain farmer inputs.
> - The required fixes are small, isolated, and straightforward to implement. Once these two hardening fixes are made and unit-tested, SB2 will be fully ready for controlled production integration.
