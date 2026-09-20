# SB2 Evaluation-Layer Targeted Hardening Report

**Date**: 2026-09-19<br>
**Review Target**: Experiment SB2 Semantic Bridge Guardrails (Evaluation Layer)<br>
**Status**: Completed & Validated (Review & Evaluation Only — Production Unmodified)<br>
**Artifacts Generated**:
- Test Suite: [test_sb2_guardrails.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/test_sb2_guardrails.py)
- JSON Report: [sb2_targeted_hardening_report.json](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_targeted_hardening_report.json)

---

## 1. Executive Summary

Following the formal SB2 Production Integration Review, targeted hardening has been implemented **strictly within the evaluation layer** (`evaluation/retrieval/semantic_bridge_guardrails.py`) without modifying any production code, vector indexes, embeddings, ground-truth benchmarks, or grader scripts.

The targeted hardening resolves the two structural edge-case defects identified during the review:
1. **Plural Inflection Blindness**: Added exhaustive canonical inflection mapping (`CROP_INFLECTIONS` & `SURFACE_TO_CANONICAL`) covering irregular plurals (*tomato/tomatoes*, *chilli/chillies*, *paddy/paddies*, *mango/mangoes*, *citrus/citruses*), standard plurals, and multi-word plurals (*pigeon peas*, *black grams*).
2. **Compound Pest/Pathogen Fragility**: Implemented an expanded Indian agricultural pest catalog and a conservative structural regex that masks crop modifiers inside pest/pathogen names (*mango fruit fly*, *sugarcane top borer*, *rice blast*, *coffee berry borer*) using non-destructive whitespace masking. Crucially, broad agronomic nouns (*disease*, *management*, *fertilizer*, *pest*, *practice*) are protected from false filtering, preserving queries like *"rice disease management and tomato fertilizer"*.
3. **Rigorous Unit Test Suite**: Created a 29-test automated test suite covering singular/plural recognition, near-matches, sub-word overlaps, compound pests, multi-crop balance, mixed vernacular queries, and edge cases. **29/29 tests (100%) passed.**

---

## 2. Files Changed and Created

### A. Modified (Evaluation Layer Only)
- [semantic_bridge_guardrails.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/semantic_bridge_guardrails.py)
  - Replaced naive word-boundary regex in `get_detected_crops` with whitespace-preserving compound pest masking and multi-word-first canonical crop detection.
  - Added `CROP_INFLECTIONS`, `SURFACE_TO_CANONICAL`, `ALL_CROP_SURFACE_FORMS`.
  - Added `STRUCTURAL_COMPOUND_PEST_REGEX` and `COORDINATED_PEST_REGEX`.
  - Updated `detect_folk_terms_guarded` to evaluate `ALL_CROP_SURFACE_FORMS` for symmetric suppression of both singular and plural English crop names in multi-crop lists.

### B. Created
- [test_sb2_guardrails.py](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/test_sb2_guardrails.py) (29 targeted unit tests)
- [sb2_targeted_hardening_report.json](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_targeted_hardening_report.json)
- [sb2_targeted_hardening_report.md](file:///c:/answers%201/major%20project%20final/MAJOR-PROJECT/evaluation/retrieval/sb2_targeted_hardening_report.md)

### C. Untouched Production Code
- `semantic_bridge.py` — **UNMODIFIED**
- `step6_query_gate.py` — **UNMODIFIED**
- `dialectal_alignment.py` — **UNMODIFIED**
- `folk_taxonomy.py` — **UNMODIFIED**
- `step5_vector_index.py` — **UNMODIFIED**
- `config.py` — **UNMODIFIED**

---

## 3. Exact Logic Changes

### Task 1: Plural Crop Detection Hardening
- **Root Cause Addressed**: Previously, `CROP_ANCHORS` contained only singular lemma strings (`tomato`, `chilli`, `coconut`). Queries containing plural nouns like *"tomatoes"* failed word-boundary matching `\btomato\b`, causing `get_detected_crops` to return `[]` and bypassing multi-crop balance.
- **Implementation**:
  ```python
  CROP_INFLECTIONS: dict[str, list[str]] = {
      "tomato": ["tomato", "tomatoes", "tomatos"],
      "potato": ["potato", "potatoes", "potatos"],
      "mango": ["mango", "mangoes", "mangos"],
      "chilli": ["chilli", "chillies", "chillis", "chili", "chilies", "chilis"],
      "paddy": ["paddy", "paddies"],
      "citrus": ["citrus", "citruses"],
      "coconut": ["coconut", "coconuts"],
      "brinjal": ["brinjal", "brinjals", "eggplant", "eggplants"],
      "cowpea": ["cowpea", "cowpeas"],
      "chickpea": ["chickpea", "chickpeas"],
      "pigeon pea": ["pigeon pea", "pigeon peas"],
      ...
  }
  ```
- **Collision Avoidance**: Multi-word crop surface forms (`"pigeon peas"`, `"black grams"`, `"pearl millets"`) are scanned first and masked with whitespace so that single words (`"gram"`, `"millet"`) cannot collide or trigger false sub-phrase entities.

### Task 2: Conservative Compound Pest & Pathogen Filtering
- **Root Cause Addressed**: Previously, `COMPOUND_PEST_EXCLUSIONS` was limited to 4 hardcoded strings (`"tea mosquito bug"`, `"coffee berry borer"`, `"cotton pink bollworm"`, `"cotton aphid"`). Any unlisted pest containing a crop name (e.g., *"mango fruit fly"*, *"sugarcane top borer"*, *"rice blast"*) falsely counted the crop modifier as an independent crop anchor.
- **Implementation**:
  1. Expanded `COMPOUND_PEST_EXCLUSIONS` with common Indian entomological and pathological names.
  2. Defined `STRUCTURAL_COMPOUND_PEST_REGEX` matching `<crop> + optional plant part + specific head noun`.
  3. **Strict Agronomic Noun Exclusion**: Specific head nouns are restricted to entomological/pathological entities (`fly`, `borer`, `bollworm`, `aphid`, `weevil`, `beetle`, `bug`, `caterpillar`, `moth`, `mite`, `thrips`, `blast`, `canker`). Broad agronomic nouns (`disease`, `management`, `fertilizer`, `pest`, `practice`, `spray`) are strictly excluded.
  4. **Equal-Length Whitespace Masking**: Instead of deleting or altering character indices, matched compound spans are replaced with whitespace of identical length. This guarantees:
     - Independent mentions of the crop elsewhere in the query are preserved (e.g. *"tea mosquito bug in tea and cashew"* correctly detects both `tea` and `cashew`).
     - Character offsets and surrounding word boundaries `\b` are preserved without distorting the query.

---

## 4. Test Results & Verification

### A. Targeted Unit Test Suite
Ran `python -u -m unittest evaluation/retrieval/test_sb2_guardrails.py`:
```
Ran 29 tests in 0.321s
OK
```

| Test Class | Test Name | Target Behavior | Result |
| :--- | :--- | :--- | :--- |
| `TestPluralCropDetection` | `test_irregular_plurals` | *tomatoes, potatoes, mangoes, chillies, paddies, citruses* | **PASS** |
| `TestPluralCropDetection` | `test_standard_plurals` | *coconuts, brinjals, chickpeas, cowpeas, bananas, onions* | **PASS** |
| `TestPluralCropDetection` | `test_multi_word_crop_plurals` | *pigeon peas, black grams, green grams, pearl millets* | **PASS** |
| `TestPluralCropDetection` | `test_uncountable_crops` | *wheat, paddy, sugarcane, cotton, coffee, sesame, tea* | **PASS** |
| `TestNearMatchesAndOverlaps` | `test_near_matches_do_not_trigger` | *team/steam/cottage/corner/price* -> `[]` | **PASS** |
| `TestNearMatchesAndOverlaps` | `test_overlapping_multiword_crops` | *black gram + green gram* -> `['black gram', 'green gram']` | **PASS** |
| `TestNearMatchesAndOverlaps` | `test_synonym_orthographic_variants` | *chilli + chili* -> canonical `['chilli']` (len=1) | **PASS** |
| `TestNearMatchesAndOverlaps` | `test_repeated_terms_do_not_inflate_count` | *tomato vs tomato* -> `['tomato']` (len=1) | **PASS** |
| `TestCompoundPestFiltering` | `test_tea_mosquito_bug_in_cashew` | Q21 canonical check -> `['cashew']` (`tea` masked) | **PASS** |
| `TestCompoundPestFiltering` | `test_mango_fruit_fly_in_cashew` | Unlisted compound pest -> `['cashew']` (`mango` masked) | **PASS** |
| `TestCompoundPestFiltering` | `test_sugarcane_top_borer_in_maize` | Unlisted compound pest -> `['maize']` (`sugarcane` masked) | **PASS** |
| `TestCompoundPestFiltering` | `test_rice_blast_in_finger_millet` | Pathogen compound check -> `['finger millet']` (`rice` masked) | **PASS** |
| `TestCompoundPestFiltering` | `test_coffee_berry_borer_in_cardamom` | Compound pest check -> `['cardamom']` (`coffee` masked) | **PASS** |
| `TestCompoundPestFiltering` | `test_compound_pest_plus_independent_host` | *tea mosquito bug in tea and cashew* -> `['tea', 'cashew']` | **PASS** |
| `TestCompoundPestFiltering` | `test_agronomic_noun_protection` | *rice disease management and tomato fertilizer* -> `['rice', 'tomato']` | **PASS** |
| `TestCompoundPestFiltering` | `test_inverted_crop_mention` | *fruit fly in mango and citrus* -> `['mango', 'citrus']` | **PASS** |
| `TestMultiCropBalanceEnrichment`| `test_q13_singular_seed_rates` | Singular multi-crop list -> Symmetrical preservation | **PASS** |
| `TestMultiCropBalanceEnrichment`| `test_q13_plural_seed_rates` | Plural multi-crop list -> Symmetrical preservation | **PASS** |
| `TestMultiCropBalanceEnrichment`| `test_single_crop_vernacular_enrichment`| Single crop vernacular (*tamatar*) -> Enriched with *Solanum lycopersicum* | **PASS** |
| `TestMixedEnglishAndVernacular` | `test_dhan_zinc_deficiency` | Mixed query (*dhan*) -> Enriched with *Oryza sativa* | **PASS** |
| `TestMixedEnglishAndVernacular` | `test_arhar_pod_borer` | Mixed query (*arhar*) -> Enriched with *Cajanus cajan* | **PASS** |
| `TestRobustnessAndEdgeCases` | `test_empty_string` | `""` -> `bridged: False`, `term_count: 0` | **PASS** |
| `TestRobustnessAndEdgeCases` | `test_whitespace_only` | `" \t \n "` -> `bridged: False`, `term_count: 0` | **PASS** |
| `TestRobustnessAndEdgeCases` | `test_punctuation_only` | `"... ,,, ??? !!!"` -> `bridged: False` | **PASS** |
| `TestRobustnessAndEdgeCases` | `test_uppercase_casing` | Uppercase query -> Case-insensitive detection | **PASS** |
| `TestSB1GuardrailsPreservation` | `test_synonym_deduplication_tautology` | Tautology check -> `None` | **PASS** |
| `TestSB1GuardrailsPreservation` | `test_synonym_clean_replacement` | Non-redundant parenthetical addition -> Verified | **PASS** |
| `TestSB1GuardrailsPreservation` | `test_specificity_guardrail_q28` | Q28 *organic manure* protected from FYM narrowing | **PASS** |
| `TestSB1GuardrailsPreservation` | `test_vernacular_confidence_floor` | *chemical fertilizer* not fuzzed to vernacular alias | **PASS** |

### B. Canonical Benchmark Target Queries Check
Verified key canonical queries on the hardened evaluation bridge:
- **Q13** (*tomato, chilli, brinjal seed rates*): `What are the recommended seed rate ranges (in g/ha or kg/ha) for tomato, chilli, and brinjal?` (Preserved without asymmetric tomato inflation).
- **Q17** (*KAU wilt-resistant tomato*): `Which tomato (Solanum lycopersicum) varieties from Kerala Agricultural University are resistant to bacterial wilt and what are their specific characteristics?` (Single crop accurately enriched).
- **Q21** (*Tea Mosquito Bug in cashew*): `What are the major damage symptoms and botanical oil control options for Tea Mosquito Bug in cashew?` (Cashew query protected from false multi-crop trigger).
- **Q28** (*Coconut organic manure and coir pith*): `What are the integrated organic manure and coir pith application practices for coconut (Cocos nucifera) basins to conserve moisture?` (Organic manure protected from FYM narrowing).

### C. Vernacular Smoke Tests Check
All 5 canonical vernacular queries verified intact:
1. `V01` (*dhan*): `dhan (Rice / Paddy, Oryza sativa) mein khaira rog ka lakshan aur upchar kya hai?`
2. `V02` (*arhar*): `arhar (Pigeon Pea, Cajanus cajan) ki fasal mein sundli ke niyantran ke liye kaun si dawai use karein?`
3. `V03` (*chana*): `chana (Chickpea, Cicer arietinum) mein ukhta rog lagne par kya karein?`
4. `V04` (*tamatar*): `tamatar (Tomato, Solanum lycopersicum) mein safed makhi (Whitefly, Bemisia tabaci) ka ilaj batayein`
5. `V05` (*ganna*): `ganna (Sugarcane, Saccharum officinarum) ki kheti mein gobar (Farm Yard Manure (FYM)) khad kitni matra mein daalna chahiye?`

---

## 5. Remaining Edge Cases & Boundaries

1. **Unlisted Vernacular Aliases**: Regional or dialectal crop names that do not exist in `folk_taxonomy.py` (e.g., obscure tribal dialects) are not detected as crops.
2. **Long Complex Coordinated Compound Pests**: Multi-pest coordination exceeding 3 intervening words (e.g., *"spotted stem, shoot, and root borer"*) relies on catalog listing or component head nouns.
3. **Multi-Crop Vernacular Lists**: Vernacular-only lists where all crops are non-English words (e.g. *"tamatar, mirchi aur baingan"*) are not balanced unless the vernacular terms are registered in crop anchors.

---

## 6. Confirmation of Production Code Isolation

- **Git Status**: Clean on all tracked files. No uncommitted modifications to production files.
- **Production Files**: `semantic_bridge.py`, `step6_query_gate.py`, `dialectal_alignment.py`, `folk_taxonomy.py`, `config.py` remain **100% untouched**.
- **Retrieval Indexes & Embeddings**: Unchanged.
- **Ground Truth & Graders**: Unchanged.

---

## 7. Recommendation Verdict

**CATEGORY**: **READY FOR CONTROLLED RE-REVIEW / INTEGRATION TEST**<br>
The two structural defects identified in the initial review have been resolved with mathematical precision in the evaluation layer. All 29 unit tests pass, and canonical query preservation is confirmed. Production integration can now be scheduled under a controlled integration plan when approved by the user.
