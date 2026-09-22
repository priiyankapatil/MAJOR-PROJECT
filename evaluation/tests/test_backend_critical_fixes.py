import sys
import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import config
from phenology_gate import get_phenological_stage, apply_phenological_gate
from regulatory_updater import get_regulatory_verification_status
from compliance_scanner import scan_answer_for_compliance, build_compliance_warning_box
from step6_query_gate import (
    QueryGate,
    query_gate,
    get_or_load_components,
    slow_path_answer,
    SLOW_PATH_PROMPTS
)


class TestBackendCriticalFixes(unittest.TestCase):
    """
    Test suite for the 9 backend critical issue resolutions:
    1. Rice diagnostic non-empty complete answer
    2. Phenological unknown stage on insufficient context & non-blocking
    3. Removal of unsafe temporal/phenological fallbacks
    4. Prohibition of invented diagnostic confidence percentages
    5. Transparent regulatory verification status reporting
    6. Skipping 5B/5C on empty/failed answer
    7. Skipping feedback adaptation on failed/insufficient answers
    8. Feedback prompt timing and non-blocking operation
    9. Recommendation output completeness
    """

    @classmethod
    def setUpClass(cls):
        os.environ["NO_INTERACTIVE_FEEDBACK"] = "1"
        cls.components = get_or_load_components()

    def test_01_phenological_stage_unknown_when_no_field_context(self):
        # Query with crop but without location or planting date
        res = get_phenological_stage("why my rice plant is getting yellow day by day", lat=None, lon=None)
        self.assertEqual(res["crop"], "rice")
        self.assertEqual(res["stage"], "unknown/insufficient_context")
        self.assertEqual(res["source"], "INSUFFICIENT_CONTEXT")

    def test_02_phenological_gate_permissive_on_unknown_stage(self):
        chunks = [
            {"text": "Harvesting rice in winter", "source_file": "doc1.pdf"},
            {"text": "Transplanting rice seedlings in nursery", "source_file": "doc2.pdf"}
        ]
        gate_res = apply_phenological_gate(chunks, "why my rice plant is getting yellow day by day", lat=None, lon=None)
        self.assertEqual(gate_res["stage"], "unknown/insufficient_context")
        self.assertEqual(gate_res["allowed_count"], 2)
        self.assertEqual(gate_res["blocked_count"], 0)
        self.assertEqual(len(gate_res["allowed_chunks"]), 2)

    def test_03_diagnostic_prompt_prohibits_invented_confidence(self):
        diag_prompt = SLOW_PATH_PROMPTS["DIAGNOSTIC"]
        self.assertIn("NEVER output invented probability percentages", diag_prompt)
        self.assertNotIn("[Most likely problem with confidence %]", diag_prompt)
        self.assertIn("CRITICAL FIELD OBSERVATIONS NEEDED", diag_prompt)

    def test_04_regulatory_status_explicitly_marks_offline(self):
        status = get_regulatory_verification_status()
        self.assertIn("mode", status)
        self.assertIn("disclaimer", status)
        self.assertFalse(status["cibrc_live_verified"])

        scan_res = scan_answer_for_compliance("Apply propiconazole 25 EC at 1ml per liter.")
        self.assertIn("regulatory_status", scan_res)
        box = build_compliance_warning_box(scan_res)
        self.assertIn("Regulatory Verification State", box)

    def test_05_slow_path_handles_empty_chunks_without_calling_llm(self):
        ans = slow_path_answer("Any query", chunks=[], query_type="DIAGNOSTIC")
        self.assertIn("answer", ans)
        self.assertTrue(ans["answer"].startswith("⚠️ Insufficient verified context"))
        self.assertEqual(ans["model_used"], "system/insufficient-context")

    def test_06_temporal_fallback_does_not_reintroduce_stale_chunks(self):
        # Stale chunks only
        stale_chunks = [
            {"chunk_id": "c1", "text": "Old 2010 practice", "source_file": "Crop Protection.pdf", "final_score": 0.3}
        ]
        with patch("step6_query_gate.retrieve_chunks", return_value=stale_chunks):
            res = query_gate(
                "Tana borer control in jowar",
                embedder=self.components[0],
                collection=self.components[1],
                bm25=self.components[2],
                corpus=self.components[3]
            )
            # If all chunks are stale, retrieval_insufficient is triggered and no stale chunks are used
            self.assertTrue(
                res["answer"].startswith("⚠️ Insufficient verified evidence") or
                len(res["sources"]) == 0 or
                "outdated" in res["answer"].lower() or
                "insufficient" in res["answer"].lower()
            )

    def test_07_rice_diagnostic_query_produces_complete_valid_answer(self):
        qg = QueryGate(*self.components)
        res = qg.process_query("why my rice plant is getting yellow day by day")
        answer = res.get("answer", "")

        self.assertTrue(len(answer) > 50, "Answer must not be empty or truncated")
        self.assertFalse(answer.endswith("|"), "Answer must not be cut off midway in table")
        # Verify no invented confidence percentage such as '70% confidence' or '30% confidence'
        self.assertIsNone(
            re.search(r"\b\d{1,2}\s*%\s*confidence\b", answer, re.IGNORECASE),
            "Diagnostic answer must not output invented probability percentages like '70% confidence'"
        )

    def test_08_adaptive_entropy_negative_feedback_and_threshold_stability(self):
        from adaptive_entropy import compute_path_accuracy, adapt_threshold

        # Verify accuracy calculation: negative feedback reduces accuracy
        fast_entries = [
            {"query": "q1", "entropy": 1.1, "was_accurate": True},
            {"query": "q2", "entropy": 1.2, "was_accurate": False},
            {"query": "q3", "entropy": 1.3, "was_accurate": False},
        ]
        acc = compute_path_accuracy(fast_entries)
        self.assertAlmostEqual(acc, 1.0 / 3.0, places=3)

        # Missing fast-path entries must evaluate to None and NOT trigger fast_low_accuracy
        empty_fast_log = {
            "current_threshold": 1.5,
            "total_feedbacks": 3,
            "fast_path_feedback": [],
            "slow_path_feedback": [
                {"query": "s1", "entropy": 1.8, "was_accurate": True},
                {"query": "s2", "entropy": 1.9, "was_accurate": True},
                {"query": "s3", "entropy": 2.0, "was_accurate": True},
            ],
            "threshold_history": []
        }
        adapted = adapt_threshold(empty_fast_log, min_path_observations=3)
        self.assertEqual(adapted["current_threshold"], 1.5)
        self.assertEqual(adapted["threshold_history"][-1]["reason"], "no_change")

        # When fast path has low accuracy with >= 3 observations, threshold decreases
        low_fast_log = {
            "current_threshold": 1.5,
            "total_feedbacks": 3,
            "fast_path_feedback": fast_entries,
            "slow_path_feedback": [],
            "threshold_history": []
        }
        adapted_low = adapt_threshold(low_fast_log, min_path_observations=3)
        self.assertLess(adapted_low["current_threshold"], 1.5)
        self.assertEqual(adapted_low["threshold_history"][-1]["reason"], "fast_low_accuracy")

    def test_09_provenance_candidate_match_not_entailed(self):
        from sentence_provenance import match_sentence_to_source, get_provenance_summary

        chunks = [
            {
                "source_file": "TNAU Guide.pdf",
                "text": "Iron deficiency causes interveinal chlorosis on young rice leaves.",
                "temporal_score": 0.85,
                "freshness_label": "FRESH"
            }
        ]

        # Sentence with word overlap evaluated without LLM entailment verification
        res = match_sentence_to_source("Iron deficiency causes chlorosis in rice leaves.", chunks, client=None)
        self.assertEqual(res["match_status"], "CANDIDATE_MATCH")
        self.assertFalse(res["entailment_verified"])
        self.assertIsNone(res["confidence"])
        self.assertIsNotNone(res["lexical_overlap"])
        self.assertNotEqual(res["match_status"], "ENTAILED")

        # Sentence with no overlap in context
        unsupported_res = match_sentence_to_source("Quantum mechanics principles apply to photosynthesis.", chunks, client=None)
        self.assertEqual(unsupported_res["match_status"], "UNSUPPORTED")
        self.assertFalse(unsupported_res["entailment_verified"])
        self.assertEqual(unsupported_res["confidence"], 0.0)

        # Provenance summary accounting
        summary = get_provenance_summary([res, unsupported_res])
        self.assertEqual(summary["total_sentences"], 2)
        self.assertEqual(summary["entailed_count"], 0)
        self.assertEqual(summary["candidate_match_count"], 1)
        self.assertEqual(summary["unsupported_count"], 1)

    def test_10_stale_chunks_disclosed_in_context_and_prompts(self):
        from step6_query_gate import SLOW_PATH_PROMPTS
        diag_prompt = SLOW_PATH_PROMPTS["DIAGNOSTIC"]
        self.assertIn("REGULATORY SAFETY MANDATE (OFFLINE_STATIC_KB", diag_prompt)
        self.assertIn("historical literature", diag_prompt.lower())

    def test_11_regulatory_safety_dimethoate_not_actionable_under_offline_kb(self):
        from step6_query_gate import reconcile_chemical_recommendations

        raw_answer = (
            "### Recommended Management\n"
            "Apply Dimethoate 0.05% as a foliar spray to control mealy-bug infestation."
        )
        reconciled = reconcile_chemical_recommendations(raw_answer, is_offline_reg=True)
        self.assertNotIn("Apply Dimethoate 0.05%", reconciled)
        self.assertIn("Chemical treatment blocked", reconciled)
        self.assertIn("Krishi Vigyan Kendra", reconciled)

    def test_12_reconciliation_blocks_tabular_chemical_prescriptions(self):
        from step6_query_gate import reconcile_chemical_recommendations

        raw_table_answer = (
            "| Pest | Product | Application Note |\n"
            "|---|---|---|\n"
            "| Rice mealy-bug | Dimethoate 0.05% spray | Apply 2L/ha when white fluff is observed |\n"
        )
        reconciled = reconcile_chemical_recommendations(raw_table_answer, is_offline_reg=True)
        self.assertNotIn("Apply 2L/ha", reconciled)
        self.assertIn("UNVERIFIED / BLOCKED", reconciled)
        self.assertIn("KVK", reconciled)

    def test_13_historical_dimethoate_dosage_sanitized_and_compliance_agrees(self):
        from step6_query_gate import reconcile_chemical_recommendations
        from compliance_scanner import scan_answer_for_compliance

        historical_answer = (
            "While historical literature (e.g., KAU) mentions Dimethoate 0.05% for rice mealy-bug, "
            "synthetic pesticides cannot be recommended as actionable under unverified regulatory status."
        )
        reconciled = reconcile_chemical_recommendations(historical_answer, is_offline_reg=True)
        self.assertNotIn("Dimethoate 0.05%", reconciled)
        self.assertNotIn("dimethoate", reconciled.lower())
        self.assertIn("withheld", reconciled.lower())

    def test_14_mixed_text_disclaimer_followed_by_actionable_prescription(self):
        from step6_query_gate import reconcile_chemical_recommendations
        from compliance_scanner import scan_answer_for_compliance

        mixed_text = (
            "⚠️ Live CIB&RC registration is unverified (OFFLINE_STATIC_KB) and historical literature is non-actionable. "
            "However, if infestation is severe, apply Dimethoate 0.05% @ 2 ml/L as a foliar spray."
        )

        # 1. Compliance scanner must NOT hide the actionable prescription on raw un-reconciled text
        raw_scan = scan_answer_for_compliance(mixed_text)
        self.assertIn("dimethoate", raw_scan["pesticides_found"])
        self.assertTrue(raw_scan["warning_needed"])
        self.assertLess(raw_scan["overall_compliance"], 0.80)

        # 2. Chemical reconciliation must block the actionable prescription despite the disclaimer
        reconciled = reconcile_chemical_recommendations(mixed_text, is_offline_reg=True)
        self.assertNotIn("Dimethoate 0.05%", reconciled)
        self.assertNotIn("@ 2 ml/L", reconciled)
        self.assertIn("Chemical treatment blocked", reconciled)

        # 3. Compliance scan on reconciled text must agree (0 detected pesticides)
        reconciled_scan = scan_answer_for_compliance(reconciled)
        self.assertEqual(reconciled_scan["pesticides_found"], [])
        self.assertEqual(reconciled_scan["overall_compliance"], 1.0)
        self.assertFalse(reconciled_scan["warning_needed"])

    def test_15_brand_and_alternate_name_coverage(self):
        from step6_query_gate import reconcile_chemical_recommendations
        from compliance_scanner import extract_pesticides_from_text

        # Alternate/brand names must be caught by both scanner and reconciliation
        cases = [
            ("Apply Rogor 30 EC @ 1.5 ml/L to foliage.", "dimethoate"),
            ("Spray Confidor at 0.5 ml/L for sucking pests.", "imidacloprid"),
            ("Drench soil with Dursban 20 EC @ 2 ml/L.", "chlorpyrifos"),
            ("Dust foliage with Sevin 50 WP @ 2 kg/ha.", "carbaryl"),
        ]

        for text, expected_key in cases:
            found = extract_pesticides_from_text(text)
            self.assertIn(expected_key, found, f"Scanner failed to detect brand name in: {text}")

            reconciled = reconcile_chemical_recommendations(text, is_offline_reg=True)
            self.assertIn("Chemical treatment blocked", reconciled, f"Reconciliation failed to block: {text}")
            self.assertNotIn("@", reconciled)

    def test_16_table_formatting_and_line_break_evasion(self):
        from step6_query_gate import reconcile_chemical_recommendations

        # Table row with brand name
        table_text = (
            "| Pest | Recommended Chemical | Application Rate |\n"
            "|---|---|---|\n"
            "| Rice mealy-bug | Rogor 30 EC | 2 ml/L water |\n"
        )
        reconciled_table = reconcile_chemical_recommendations(table_text, is_offline_reg=True)
        self.assertNotIn("Rogor", reconciled_table)
        self.assertNotIn("2 ml/L", reconciled_table)
        self.assertIn("UNVERIFIED / BLOCKED", reconciled_table)

        # Line break evasion (chemical on line 1, rate on line 2)
        broken_text = (
            "Recommended product:\n"
            "Rogor 30 EC\n"
            "Apply 2 ml/L in morning\n"
        )
        reconciled_broken = reconcile_chemical_recommendations(broken_text, is_offline_reg=True)
        self.assertNotIn("Rogor", reconciled_broken)
        self.assertNotIn("Apply 2 ml/L", reconciled_broken)
        self.assertIn("Chemical treatment blocked", reconciled_broken)

    def test_17_compliance_scanner_does_not_hide_actionable_prescriptions_with_disclaimers(self):
        from compliance_scanner import extract_pesticides_from_text

        # Mixed text with disclaimer + prescription must extract pesticide
        mixed_1 = "Live CIB&RC is unverified. Apply Rogor 30 EC at 1.5 ml/L."
        self.assertIn("dimethoate", extract_pesticides_from_text(mixed_1))

        mixed_2 = "Historical reference notes: Spray Confidor at 0.5 ml/L."
        self.assertIn("imidacloprid", extract_pesticides_from_text(mixed_2))

        # Purely blocked statement without actionable prescription
        pure_blocked = (
            "• Chemical treatment blocked (OFFLINE_STATIC_KB): Live CIB&RC registration is unverified. "
            "Synthetic chemical prescriptions are blocked without live regulatory verification. "
            "Consult local Krishi Vigyan Kendra (KVK) for current label-approved options."
        )
        self.assertEqual(extract_pesticides_from_text(pure_blocked), [])

    def test_18_unverified_biological_rates_suppressed_with_kvk_referral(self):
        from step6_query_gate import reconcile_chemical_recommendations

        input_text = (
            "Encourage natural enemies: release *Trichogramma* spp. (≈1 lakh per ha) and "
            "*Beauveria bassiana* (≈10 g L⁻¹) as a biological spray – note that the exact rates are "
            "from historical references; verify current recommended rates with your KVK."
        )
        reconciled = reconcile_chemical_recommendations(input_text, is_offline_reg=True)

        # Numerical rates must be suppressed
        self.assertNotIn("1 lakh per ha", reconciled)
        self.assertNotIn("10 g L⁻¹", reconciled)
        self.assertNotIn("10 g/l", reconciled.lower())

        # Organisms must be preserved
        self.assertIn("Trichogramma", reconciled)
        self.assertIn("Beauveria bassiana", reconciled)

        # Safe qualitative guidance and KVK referral must be present
        self.assertIn("KVK", reconciled)
        self.assertIn("consult local kvk", reconciled.lower())

    def test_19_biological_rates_non_parentheses_and_newline_formatting(self):
        from step6_query_gate import reconcile_chemical_recommendations

        # Outside parentheses format
        outside_text = (
            "Release Trichogramma spp. at 1 lakh per ha and spray Beauveria bassiana @ 10 g/l for control."
        )
        reconciled_outside = reconcile_chemical_recommendations(outside_text, is_offline_reg=True)
        self.assertNotIn("1 lakh per ha", reconciled_outside)
        self.assertNotIn("10 g/l", reconciled_outside)
        self.assertIn("Trichogramma", reconciled_outside)
        self.assertIn("Beauveria bassiana", reconciled_outside)
        self.assertIn("KVK", reconciled_outside)

        # Line break formatting
        newline_text = (
            "• Encourage natural enemies: release Trichogramma spp.\n"
            "  (≈1 lakh per ha) as a biological spray"
        )
        reconciled_newline = reconcile_chemical_recommendations(newline_text, is_offline_reg=True)
        self.assertNotIn("1 lakh per ha", reconciled_newline)
        self.assertIn("Trichogramma", reconciled_newline)
        self.assertIn("KVK", reconciled_newline)

    def test_20_biological_rates_table_formatting(self):
        from step6_query_gate import reconcile_chemical_recommendations

        table_text = (
            "| Pest | Recommended Biocontrol | Field Action |\n"
            "|---|---|---|\n"
            "| Brown plant hopper | Release Trichogramma spp. (≈1 lakh per ha) and Beauveria bassiana @ 10 g/l | Monitor field |\n"
        )
        reconciled_table = reconcile_chemical_recommendations(table_text, is_offline_reg=True)
        self.assertNotIn("1 lakh per ha", reconciled_table)
        self.assertNotIn("10 g/l", reconciled_table)
        self.assertIn("Trichogramma", reconciled_table)
        self.assertIn("Beauveria bassiana", reconciled_table)
        self.assertIn("KVK", reconciled_table)
        self.assertTrue(reconciled_table.strip().startswith("|"))
        self.assertTrue(reconciled_table.strip().endswith("|"))

    def test_21_stale_fertilizer_rate_suppression(self):
        from step6_query_gate import reconcile_chemical_recommendations

        stale_fert_text = (
            "Apply 120 kg N, 60 kg P2O5, and 40 kg K2O per hectare immediately for crop growth."
        )
        reconciled = reconcile_chemical_recommendations(stale_fert_text, is_offline_reg=True)
        self.assertNotIn("120 kg N", reconciled)
        self.assertNotIn("60 kg P2O5", reconciled)
        self.assertNotIn("40 kg K2O", reconciled)
        self.assertIn("soil testing", reconciled.lower())
        self.assertIn("Package of Practices", reconciled)
        self.assertIn("KVK", reconciled)

    def test_22_verified_current_rates_preserved(self):
        from step6_query_gate import reconcile_chemical_recommendations

        verified_text = "Apply neem oil @ 3 ml/l or spray Beauveria bassiana @ 5 g/l as recommended."
        fresh_chunks = [
            {
                "text": "TNAU 2022 Guide: Spray neem oil @ 3 ml/l or apply Beauveria bassiana @ 5 g/l for caterpillar control.",
                "freshness_label": "FRESH",
                "temporal_score": 0.95,
                "source_file": "The TNAU Agriculture PDF.pdf",
            }
        ]
        reconciled = reconcile_chemical_recommendations(verified_text, is_offline_reg=True, chunks=fresh_chunks)
        # Verified rate in fresh chunk MUST be preserved
        self.assertIn("5 g/l", reconciled)
        self.assertIn("Beauveria bassiana", reconciled)

    def test_23_reconciliation_before_compliance_precedence(self):
        from step6_query_gate import reconcile_chemical_recommendations
        from compliance_scanner import scan_answer_for_compliance

        # An unverified prescription with a disclaimer must be reconciled first
        raw_answer = (
            "⚠️ Notice: Historical reference only. Spray Dimethoate 0.05% @ 1.5 ml/L for mealy-bug.\n"
            "Release Trichogramma spp. (≈1 lakh per ha)."
        )
        reconciled = reconcile_chemical_recommendations(raw_answer, is_offline_reg=True)
        compliance_report = scan_answer_for_compliance(reconciled)

        # Reconciliation must have stripped the chemical prescription
        self.assertEqual(len(compliance_report["pesticides_found"]), 0)
        self.assertNotIn("0.05%", reconciled)
        self.assertNotIn("1 lakh per ha", reconciled)
        self.assertIn("Trichogramma", reconciled)

    def test_24_pest_parasitoid_mismatch_trichogramma_bph_and_mealybug(self):
        from step6_query_gate import reconcile_chemical_recommendations

        # Text line mismatch
        bph_text = (
            "• Brown plant hopper (Nilaparvata lugens): Release Trichogramma japonicum for biological control."
        )
        reconciled_bph = reconcile_chemical_recommendations(bph_text, is_offline_reg=True)
        # Trichogramma must NOT be recommended as an effective agent for BPH
        self.assertNotIn("Release Trichogramma japonicum for biological control", reconciled_bph)
        self.assertIn("egg parasitoid of lepidopteran borers and unsuited for sucking pests", reconciled_bph)
        self.assertIn("predatory mirid bugs", reconciled_bph)
        self.assertIn("KVK", reconciled_bph)

        # Table row mismatch for mealybug
        mealy_table = (
            "| Rice mealy bug | Release Trichogramma spp. for biological control | Follow up |\n"
        )
        reconciled_table = reconcile_chemical_recommendations(mealy_table, is_offline_reg=True)
        self.assertNotIn("Release Trichogramma spp. for biological control", reconciled_table)
        self.assertIn("not recommended for sucking pests", reconciled_table)

    def test_25_unsupported_numerical_water_depth_suppression(self):
        from step6_query_gate import reconcile_chemical_recommendations

        unsupported_water = (
            "Maintain 5–7 cm of standing water during tillering and maintain 10–12 cm water depth during heading."
        )
        reconciled = reconcile_chemical_recommendations(unsupported_water, is_offline_reg=True)
        self.assertNotIn("5–7 cm", reconciled)
        self.assertNotIn("10–12 cm", reconciled)
        self.assertIn("Package of Practices", reconciled)
        self.assertIn("KVK", reconciled)
        self.assertIn("drainage", reconciled.lower())

        # Verified water depth from fresh chunks must be preserved
        fresh_chunk = [
            {
                "text": "CRRI 2023 Guidelines: Maintain 2-3 cm water depth during transplanting to facilitate root establishment.",
                "freshness_label": "FRESH",
                "temporal_score": 0.95,
                "source_file": "CRRI Guidelines.pdf"
            }
        ]
        verified_water = "Maintain 2-3 cm water depth during transplanting."
        reconciled_verified = reconcile_chemical_recommendations(verified_water, is_offline_reg=True, chunks=fresh_chunk)
        self.assertIn("2-3 cm", reconciled_verified)

    def test_26_destructive_burning_advice_suppression(self):
        from step6_query_gate import reconcile_chemical_recommendations

        burn_text = "Remove and burn infested leaves and tillers to prevent disease spread."
        reconciled = reconcile_chemical_recommendations(burn_text, is_offline_reg=True)
        self.assertNotIn("burn infested leaves", reconciled)
        self.assertIn("field sanitation", reconciled.lower())
        self.assertIn("avoid burning", reconciled.lower())

        stubble_burn = "Open burning of crop residues and stubbles is recommended for field clearance."
        reconciled_stubble = reconcile_chemical_recommendations(stubble_burn, is_offline_reg=True)
        self.assertNotIn("open burning", reconciled_stubble.lower())
        self.assertIn("avoid burning", reconciled_stubble.lower())

    def test_27_provenance_invariant_lexical_overlap_never_entails(self):
        from sentence_provenance import match_sentence_to_source, is_entailment_verified

        scored_chunk = [{
            "text": "Rice crops require balanced nitrogen, phosphorus, and potassium application based on regional guidelines.",
            "source_file": "TNAU Agriculture.pdf",
            "temporal_score": 0.99,
            "freshness_label": "FRESH"
        }]
        sentence = "Rice crops require balanced nitrogen, phosphorus, and potassium application."

        # When evaluated by lexical overlap (no NLI client)
        res = match_sentence_to_source(sentence, scored_chunk, client=None)

        # Even with high lexical overlap and fresh chunk, it must be CANDIDATE_MATCH, never ENTAILED
        self.assertEqual(res["match_status"], "CANDIDATE_MATCH")
        self.assertFalse(res["entailment_verified"])
        self.assertFalse(is_entailment_verified(res))
        self.assertIsNotNone(res["lexical_overlap"])

    def test_28_provenance_validation_flag_false_when_zero_entailed(self):
        from sentence_provenance import get_provenance_summary

        # Provenance map with candidate matches and unsupported claims, but 0 entailed
        prov_map = [
            {"sentence": "S1", "match_status": "CANDIDATE_MATCH", "entailment_verified": False, "temporal_score": 0.5, "combined_score": 0.5, "matched_source": "doc1.pdf", "lexical_overlap": 0.4},
            {"sentence": "S2", "match_status": "UNSUPPORTED", "entailment_verified": False, "temporal_score": 0.0, "combined_score": 0.0, "matched_source": "UNSUPPORTED / UNVERIFIED"}
        ]
        summary = get_provenance_summary(prov_map)
        self.assertEqual(summary["entailed_count"], 0)
        self.assertEqual(summary["candidate_match_count"], 1)
        self.assertEqual(summary["unsupported_count"], 1)

        # Query gate validation flag contract
        factually_validated = False if summary.get("entailed_count", 0) == 0 else True
        self.assertFalse(factually_validated)

    def test_29_unverified_rate_cooccurrence_and_stale_rejection(self):
        from step6_query_gate import _is_rate_verified_in_context

        # 1. Fresh chunk with item and rate far apart (>100 chars) must NOT be verified
        distant_chunk = [{
            "text": "Water management in rice requires constant monitoring. " + ("general agronomic background text " * 20) + " Apply 10 kg/ha nitrogen fertilizer.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "TestDoc.pdf"
        }]
        self.assertFalse(_is_rate_verified_in_context("water", "10 kg/ha", chunks=distant_chunk))

        stale_chunk = [{
            "text": "Spray Beauveria bassiana @ 10 g/l for control.",
            "freshness_label": "STALE",
            "temporal_score": 0.45,
            "source_file": "OldKAU.pdf"
        }]
        self.assertFalse(_is_rate_verified_in_context("Beauveria bassiana", "10 g/l", chunks=stale_chunk))

    def test_30_unverified_agronomic_claims_qualified_with_kvk(self):
        from step6_query_gate import reconcile_chemical_recommendations

        unverified_claims = (
            "• Practice alternate wetting and drying (AWD) as advised in the state Package of Practices.\n"
            "• Apply iron chelate foliar spray for chlorosis.\n"
            "• Remove and destroy heavily infested tillers to manage mealy-bug.\n"
            "• Rotate rice with non-host crops (e.g., pulses, millets) to control nematodes."
        )
        reconciled = reconcile_chemical_recommendations(unverified_claims, is_offline_reg=True)

        self.assertIn("AWD is not verified in retrieved context", reconciled)
        self.assertIn("specific iron chelate formulations are not documented in retrieved context", reconciled)
        self.assertIn("tiller removal is unverified in retrieved context", reconciled)
        self.assertIn("crop rotation sequences are unverified in retrieved context", reconciled)
        self.assertIn("KVK", reconciled)

    def test_31_strict_rate_verification_formulation_bound_rate_crop_pest(self):
        from step6_query_gate import _is_rate_verified_in_context

        chunk_stem_borer = [{
            "text": "In rice crops, manage yellow stem borer by spraying chlorpyrifos 20 EC at 2 ml/l when ETL is reached.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RiceProtection.pdf"
        }]
        chunk_cooccur = [{
            "text": "Wait 10 days after transplanting rice. Then spray chlorpyrifos 20 EC at 2 ml/l for stem borer.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RiceProtection.pdf"
        }]
        chunk_cotton = [{
            "text": "In cotton crops, bollworm can be controlled by chlorpyrifos 20 EC at 2 ml/l.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "CottonPackage.pdf"
        }]

        # 1. Valid exact match (treatment, crop, pest, formulation, rate, units)
        self.assertTrue(
            _is_rate_verified_in_context("chlorpyrifos", "20 EC at 2 ml/l", chunks=chunk_stem_borer, target_crop="rice", target_pest="stem borer")
        )

        # 2. Co-occurrence false-positive rejection (10 from days must NOT match 10 ml/l)
        self.assertFalse(
            _is_rate_verified_in_context("chlorpyrifos", "10 ml/l", chunks=chunk_cooccur, target_crop="rice", target_pest="stem borer")
        )

        # 3. Formulation mismatch rejection (50 EC vs 20 EC)
        self.assertFalse(
            _is_rate_verified_in_context("chlorpyrifos", "50 EC at 2 ml/l", chunks=chunk_stem_borer, target_crop="rice", target_pest="stem borer")
        )

        # 4. Unit mismatch rejection (2 kg/ha vs 2 ml/l)
        self.assertFalse(
            _is_rate_verified_in_context("chlorpyrifos", "2 kg/ha", chunks=chunk_stem_borer, target_crop="rice", target_pest="stem borer")
        )

        # 5. Cross-pest mismatch rejection (stem borer treatment claimed for brown plant hopper)
        self.assertFalse(
            _is_rate_verified_in_context("chlorpyrifos", "2 ml/l", chunks=chunk_stem_borer, target_crop="rice", target_pest="brown plant hopper")
        )

        # 6. Cross-crop mismatch rejection (cotton treatment claimed for rice)
        self.assertFalse(
            _is_rate_verified_in_context("chlorpyrifos", "2 ml/l", chunks=chunk_cotton, target_crop="rice", target_pest="stem borer")
        )

        # 7. Treatment/rate association rejection (nearby rate belongs to different chemical/fertilizer)
        chunk_association_error = [{
            "text": "In rice fields, spray neem oil. Apply urea at 50 kg/ha for vegetative growth.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RicePackage.pdf"
        }]
        self.assertFalse(
            _is_rate_verified_in_context("neem oil", "50 kg/ha", chunks=chunk_association_error, target_crop="rice")
        )

        # 8. Cross-row table association rejection
        chunk_table_multiroow = [{
            "text": "| Stem Borer | Chlorpyrifos 20 EC | 2 ml/l |\n| Mealybug | Dimethoate | 0.05% |",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RiceTable.pdf"
        }]
        self.assertFalse(
            _is_rate_verified_in_context("Chlorpyrifos", "0.05%", chunks=chunk_table_multiroow, target_crop="rice")
        )
        self.assertTrue(
            _is_rate_verified_in_context("Chlorpyrifos", "20 EC at 2 ml/l", chunks=chunk_table_multiroow, target_crop="rice")
        )

    def test_32_agronomic_scrubber_table_preservation_and_verified_context(self):
        from step6_query_gate import _scrub_unverified_agronomic_claims

        # 1. Table format preservation: pipe structure and column alignment must not break
        table_input = (
            "| Problem Area | Suspected Cause | Management Practice |\n"
            "|---|---|---|\n"
            "| Water Regime | Waterlogging | Practice Alternate Wetting and Drying (AWD) |\n"
            "| Nutrient | Chlorosis | Apply iron chelates (Fe-EDDHA) at 500 g/ha |"
        )
        scrubbed_table = _scrub_unverified_agronomic_claims(table_input, chunks=[])
        lines = scrubbed_table.strip().splitlines()
        self.assertEqual(len(lines), 4)
        for line in lines:
            self.assertTrue(line.startswith("|") and line.endswith("|"), f"Malformed table row: {line}")
        self.assertIn("AWD is not verified in retrieved context", scrubbed_table)
        self.assertIn("specific iron chelate formulations are not documented in retrieved context", scrubbed_table)
        self.assertNotIn("500 g/ha", scrubbed_table, "Unverified iron chelate rate must be eliminated")

        # 2. Preserves verified context when practice is explicitly present in chunks
        verified_chunks = [{
            "text": "Alternate wetting and drying (AWD) can reduce water use in rice by 25%.",
            "source_file": "IRRI Water.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.92
        }]
        text_with_awd = "Farmers can practice Alternate Wetting and Drying (AWD) for water management."
        preserved = _scrub_unverified_agronomic_claims(text_with_awd, chunks=verified_chunks)
        self.assertIn("Alternate Wetting and Drying", preserved)
        self.assertNotIn("AWD is not verified in retrieved context", preserved)

    def test_33_non_rate_quantities_and_reversed_wording_association(self):
        from step6_query_gate import _is_rate_verified_in_context

        # Context with dosage, waiting period, and temperature
        chunk = [{
            "text": "In rice fields, apply Chlorpyrifos 20 EC at 2 ml/l for stem borer. Waiting period is 15 days. Apply when temperature is below 30°C.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RicePackage.pdf"
        }]

        # 1. Valid rate passes
        self.assertTrue(
            _is_rate_verified_in_context("Chlorpyrifos", "20 EC at 2 ml/l", chunks=chunk, target_crop="rice", target_pest="stem borer")
        )

        # 2. Waiting period must NOT be accepted as rate
        self.assertFalse(
            _is_rate_verified_in_context("Chlorpyrifos", "15 days", chunks=chunk, target_crop="rice")
        )
        self.assertFalse(
            _is_rate_verified_in_context("Chlorpyrifos", "15 days waiting period", chunks=chunk, target_crop="rice")
        )
        # Unrelated number (15) must not be bound to ml/l
        self.assertFalse(
            _is_rate_verified_in_context("Chlorpyrifos", "15 ml/l", chunks=chunk, target_crop="rice")
        )

        # 3. Temperature must NOT be accepted as rate
        self.assertFalse(
            _is_rate_verified_in_context("Chlorpyrifos", "30°C", chunks=chunk, target_crop="rice")
        )
        self.assertFalse(
            _is_rate_verified_in_context("Chlorpyrifos", "30", chunks=chunk, target_crop="rice")
        )

        # 4. Reversed wording (rate before treatment) in same sentence must be accepted
        reversed_chunk = [{
            "text": "For rice vegetative growth, apply 50 kg/ha urea. At 2 ml/l, spray Chlorpyrifos 20 EC for stem borer.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RiceReversed.pdf"
        }]
        self.assertTrue(
            _is_rate_verified_in_context("urea", "50 kg/ha", chunks=reversed_chunk, target_crop="rice")
        )
        self.assertTrue(
            _is_rate_verified_in_context("Chlorpyrifos", "2 ml/l", chunks=reversed_chunk, target_crop="rice", target_pest="stem borer")
        )

        # 5. Reversed cross-treatment rejection: rate in previous sentence must NOT be associated with next item
        reversed_cross_chunk = [{
            "text": "Apply 50 kg/ha of urea. Spray neem oil for insect deterrence.",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "source_file": "RiceRotation.pdf"
        }]
        self.assertFalse(
            _is_rate_verified_in_context("neem oil", "50 kg/ha", chunks=reversed_cross_chunk, target_crop="rice")
        )

    def test_34_scrubber_table_column_preservation_and_tiller_variants(self):
        from step6_query_gate import _scrub_bio_rates, _scrub_unverified_agronomic_claims

        # 1. Biocontrol table column preservation
        table_bio = (
            "| Organism | Application Rate | Target |\n"
            "|---|---|---|\n"
            "| Beauveria bassiana | 10 g/l | caterpillars |"
        )
        scrubbed = _scrub_bio_rates(table_bio, chunks=[])
        lines = scrubbed.strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[2].startswith("|") and lines[2].endswith("|"))
        self.assertNotIn("10 g/l", scrubbed)
        self.assertIn("consult local KVK for approved application rate", scrubbed)
        # Verify 3 columns preserved: Organism | Rate | Target
        cols = [c.strip() for c in lines[2].split("|")[1:-1]]
        self.assertEqual(len(cols), 3)
        self.assertEqual(cols[0], "Beauveria bassiana")
        self.assertIn("KVK", cols[1])
        self.assertEqual(cols[2], "caterpillars")

        # 2. Tiller removal variants
        tiller_text = "Farmers must remove infested tillers or destroy infested tillers to control mealybugs."
        scrubbed_tiller = _scrub_unverified_agronomic_claims(tiller_text, chunks=[])
        self.assertNotIn("remove infested tillers", scrubbed_tiller.lower())
        self.assertNotIn("destroy infested tillers", scrubbed_tiller.lower())
        self.assertIn("field sanitation", scrubbed_tiller)
        self.assertIn("KVK", scrubbed_tiller)

    def test_35_regulatory_verification_state_disclosures_offline_unknown_live(self):
        from compliance_scanner import scan_answer_for_compliance, build_compliance_warning_box

        # Scan text with neem oil (organic status)
        text = "Neem oil application is recommended for insect deterrence."
        scan = scan_answer_for_compliance(text)

        # 1. Test OFFLINE_STATIC_KB (Default)
        scan["regulatory_status"] = {
            "mode": "OFFLINE_STATIC_KB",
            "cibrc_live_verified": False,
            "disclaimer": "Static reference KB active; live gazette checks skipped."
        }
        box_offline = build_compliance_warning_box(scan)
        self.assertNotIn("SAFE ✅", box_offline, "Under OFFLINE_STATIC_KB, compliance box must NOT state SAFE ✅")
        self.assertIn("Not Listed as Banned in Static KB", box_offline)
        self.assertNotIn("India Banned     : No\n", box_offline)
        self.assertNotIn("EU Banned        : No\n", box_offline)
        self.assertNotIn("Organic Compliant : Yes\n", box_offline)
        self.assertIn("PROVISIONAL", box_offline)
        self.assertIn("Unverified", box_offline)
        self.assertIn("OFFLINE_STATIC_KB", box_offline)

        # 2. Test UNKNOWN state
        scan["regulatory_status"] = {
            "mode": "UNKNOWN",
            "cibrc_live_verified": False,
            "disclaimer": "Regulatory sources could not be checked."
        }
        box_unknown = build_compliance_warning_box(scan)
        self.assertIn("UNKNOWN", box_unknown)
        self.assertNotIn("SAFE ✅", box_unknown)
        self.assertNotIn("India Banned     : No\n", box_unknown)

        # 3. Test LIVE_VERIFIED state
        scan["regulatory_status"] = {
            "mode": "LIVE_VERIFIED",
            "cibrc_live_verified": True,
            "eu_sante_live_verified": True,
            "disclaimer": "Status verified against live CIB&RC and EU SANTE gazettes."
        }
        box_live = build_compliance_warning_box(scan)
        self.assertIn("SAFE ✅", box_live)
        self.assertIn("India Banned     : No\n", box_live)
        self.assertIn("EU Banned        : No\n", box_live)
        self.assertIn("Organic Compliant : Yes\n", box_live)
        self.assertIn("LIVE_VERIFIED", box_live)

    def test_36_provenance_gating_unverified_notice_attached_when_zero_entailed(self):
        from step6_query_gate import query_gate

        # Mock retrieval with chunks and mock LLM response
        sample_chunks = [{
            "chunk_id": "c1",
            "text": "Yellowing in rice can occur from multiple abiotic and biotic factors.",
            "source_file": "The TNAU Agriculture PDF.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90,
            "final_score": 0.90
        }]
        mock_raw_answer = (
            "**🔍 POTENTIAL CAUSES (SOURCE-GROUNDED)**\n"
            "Nitrogen deficiency causes yellow streaks. Brown plant hopper causes hopper burn."
        )
        mock_prov_map = [
            {
                "sentence": "Nitrogen deficiency causes yellow streaks.",
                "match_status": "CANDIDATE_MATCH",
                "entailment_verified": False,
                "confidence": 0.5,
                "lexical_overlap": 0.3,
                "source": "The TNAU Agriculture PDF.pdf",
                "temporal_score": 0.90,
                "freshness_label": "FRESH"
            },
            {
                "sentence": "Brown plant hopper causes hopper burn.",
                "match_status": "UNSUPPORTED",
                "entailment_verified": False,
                "confidence": 0.0,
                "lexical_overlap": 0.1,
                "source": "UNSUPPORTED / UNVERIFIED",
                "temporal_score": 0.0,
                "freshness_label": "UNKNOWN"
            }
        ]

        with patch("step6_query_gate.retrieve_chunks", return_value=sample_chunks), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": mock_raw_answer, "model_used": "mock-model", "sources": []}), \
             patch("sentence_provenance.build_provenance_map", return_value=mock_prov_map):
            res = query_gate(
                "why my rice plant is getting yellow day by day",
                embedder=self.components[0],
                collection=self.components[1],
                bm25=self.components[2],
                corpus=self.components[3]
            )
            ans = res.get("answer", "")
            self.assertIn("UNVERIFIED ANSWER NOTICE — NOT FACTUALLY PROVEN", ans)
            self.assertNotIn("SOURCE-GROUNDED", ans)
            self.assertIn("POSSIBLE CAUSES (UNVERIFIED HYPOTHESES / LITERATURE PATTERNS)", ans)
            self.assertFalse(res.get("is_factually_proven", True))
            self.assertFalse(res.get("factually_validated", True))

    def test_37_fast_path_and_step5b_exception_provenance_flags_fail_closed(self):
        from step6_query_gate import query_gate

        # 1. Fast-path query (provenance skipped) must NOT claim factually proven / validated
        res_fast = query_gate(
            "What does NPK stand for?",
            embedder=self.components[0],
            collection=self.components[1],
            bm25=self.components[2],
            corpus=self.components[3]
        )
        self.assertEqual(res_fast["path"], "fast")
        self.assertIsNone(res_fast["provenance_summary"])
        self.assertFalse(res_fast.get("is_factually_proven", True), "Fast path must NOT claim is_factually_proven")
        self.assertFalse(res_fast.get("factually_validated", True), "Fast path must NOT claim factually_validated")

        # 2. Slow path when Step 5B encounters an exception must fail closed
        sample_chunks = [{
            "chunk_id": "c1",
            "text": "Yellowing in rice can occur from multiple factors.",
            "source_file": "TNAU.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90,
            "final_score": 0.90
        }]
        with patch("step6_query_gate.retrieve_chunks", return_value=sample_chunks), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": "Rice yellowing advice.", "model_used": "mock-model", "sources": []}), \
             patch("sentence_provenance.build_provenance_map", side_effect=RuntimeError("Groq API timeout")):
            res_err = query_gate(
                "why my rice plant is getting yellow day by day",
                embedder=self.components[0],
                collection=self.components[1],
                bm25=self.components[2],
                corpus=self.components[3]
            )
            self.assertFalse(res_err.get("is_factually_proven", True))
            self.assertFalse(res_err.get("factually_validated", True))

    def test_38_regulatory_status_dependency_presence_does_not_forge_live_verified(self):
        import regulatory_updater
        from regulatory_updater import get_regulatory_verification_status

        orig_state = dict(regulatory_updater._LIVE_VERIFICATION_STATE)
        try:
            # 1. Dependencies simulated available but NO live scrapes executed
            regulatory_updater._LIVE_VERIFICATION_STATE["cibrc_live_verified"] = False
            regulatory_updater._LIVE_VERIFICATION_STATE["eu_sante_live_verified"] = False
            status_offline = get_regulatory_verification_status()
            self.assertEqual(status_offline["mode"], "OFFLINE_STATIC_KB")
            self.assertFalse(status_offline["cibrc_live_verified"])
            self.assertFalse(status_offline["eu_sante_live_verified"])
            self.assertIn("OFFLINE_STATIC_KB", status_offline["mode"])
            self.assertNotIn("LIVE_VERIFIED", status_offline["mode"])

            # 2. Partial verification: CIB&RC verified, EU SANTE not verified
            regulatory_updater._LIVE_VERIFICATION_STATE["cibrc_live_verified"] = True
            regulatory_updater._LIVE_VERIFICATION_STATE["eu_sante_live_verified"] = False
            status_partial = get_regulatory_verification_status()
            self.assertEqual(status_partial["mode"], "PARTIAL_LIVE_VERIFIED")
            self.assertTrue(status_partial["cibrc_live_verified"])
            self.assertFalse(status_partial["eu_sante_live_verified"])
            self.assertIn("PARTIAL", status_partial["mode"])

            # 3. Full verification: both sources verified
            regulatory_updater._LIVE_VERIFICATION_STATE["cibrc_live_verified"] = True
            regulatory_updater._LIVE_VERIFICATION_STATE["eu_sante_live_verified"] = True
            status_live = get_regulatory_verification_status()
            self.assertEqual(status_live["mode"], "LIVE_VERIFIED")
            self.assertTrue(status_live["cibrc_live_verified"])
            self.assertTrue(status_live["eu_sante_live_verified"])
        finally:
            regulatory_updater._LIVE_VERIFICATION_STATE = orig_state

    def test_39_water_depth_scrubber_preserves_spacing_and_trap_height(self):
        from step6_query_gate import _scrub_water_depths

        # 1. Non-water distance, spacing, and height measurements must be preserved exactly
        spacing_text = (
            "Transplant seedlings at 15–20 cm spacing between rows and 10 cm within rows. "
            "Install pheromone traps 30 cm above crop canopy. Dig a trench 15 cm wide. "
            "Sow seeds at 2–3 cm planting depth."
        )
        scrubbed_spacing = _scrub_water_depths(spacing_text, chunks=[])
        self.assertIn("15–20 cm spacing", scrubbed_spacing)
        self.assertIn("10 cm within rows", scrubbed_spacing)
        self.assertIn("30 cm above crop canopy", scrubbed_spacing)
        self.assertIn("15 cm wide", scrubbed_spacing)
        self.assertIn("2–3 cm planting depth", scrubbed_spacing)
        self.assertNotIn("drainage", scrubbed_spacing.lower())

        # 2. Non-depth water phrases (pipe, trap, tube) must be preserved
        pipe_trap_text = "Install a 5 cm water pipe for drip inflow. Place a 5 cm water trap near the border."
        scrubbed_pipe_trap = _scrub_water_depths(pipe_trap_text, chunks=[])
        self.assertIn("5 cm water pipe", scrubbed_pipe_trap)
        self.assertIn("5 cm water trap", scrubbed_pipe_trap)

        # 3. Actual unverified water depth claims (including inverted 'depth of water') must still be scrubbed
        water_text = (
            "Maintain standing water of 5–7 cm in the field. "
            "Keep water depth at 10 cm during tillering. "
            "Maintain 5 cm depth of water throughout vegetative stage."
        )
        scrubbed_water = _scrub_water_depths(water_text, chunks=[])
        self.assertNotIn("5–7 cm", scrubbed_water)
        self.assertNotIn("10 cm", scrubbed_water)
        self.assertNotIn("5 cm depth of water", scrubbed_water)
        self.assertIn("maintain proper field drainage", scrubbed_water)

        # 4. Table-separated water depth label and quantity
        table_text = (
            "| Water Management | Parameter |\n"
            "|---|---|\n"
            "| Water Depth | 5–7 cm |\n"
            "| Standing Water | 5 cm |"
        )
        scrubbed_table = _scrub_water_depths(table_text, chunks=[])
        self.assertNotIn("5–7 cm", scrubbed_table)
        self.assertNotIn("5 cm", scrubbed_table)
        self.assertIn("| Water Depth | recommended levels", scrubbed_table)
        self.assertIn("| Standing Water | recommended levels", scrubbed_table)

        # 5. Verified water depth in authoritative context must be preserved
        chunk_with_water = [{
            "text": "For wetland rice, maintain water depth of 2–3 cm during vegetative stage.",
            "source_file": "RiceIrrigation.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.95
        }]
        verified_water = "Maintain water depth of 2–3 cm during tillering."
        preserved = _scrub_water_depths(verified_water, chunks=chunk_with_water)
        self.assertIn("2–3 cm", preserved)

    def test_40_unrelated_chunk_keywords_do_not_bypass_tiller_scrubber(self):
        from step6_query_gate import _scrub_unverified_agronomic_claims

        advice = "Farmers should remove infested tillers to control rice mealybug."

        # Case 1: Chunk has 'remove' and 'tiller' in completely unrelated sentences
        unrelated_chunk = [{
            "text": "Remove weeds from field bunds before sowing. Healthy tillers develop during active tillering phase.",
            "source_file": "WeedControl.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90
        }]
        scrubbed = _scrub_unverified_agronomic_claims(advice, chunks=unrelated_chunk)
        self.assertNotIn("remove infested tillers", scrubbed.lower())
        self.assertIn("field sanitation", scrubbed)
        self.assertIn("tiller removal is unverified", scrubbed)

        # Case 2: Chunk has unrelated mentions in table rows
        table_chunk = [{
            "text": "| Weed Management | Remove weeds at 20 DAS |\n| Growth Stages | Tillers emerge rapidly |",
            "source_file": "RiceAgronomy.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90
        }]
        scrubbed_tab = _scrub_unverified_agronomic_claims(advice, chunks=table_chunk)
        self.assertNotIn("remove infested tillers", scrubbed_tab.lower())
        self.assertIn("field sanitation", scrubbed_tab)

        # Case 3: Line-wrapped or newline-separated tiller removal advice
        wrapped_advice = "Farmers should remove and destroy\ninfested tillers in affected patches."
        scrubbed_wrapped = _scrub_unverified_agronomic_claims(wrapped_advice, chunks=[])
        self.assertNotIn("infested tillers", scrubbed_wrapped)
        self.assertIn("field sanitation", scrubbed_wrapped)

        # Case 4: Action verbs (uproot, destroy, pull, cut, clip)
        for verb in ["uproot", "destroy", "pull out", "cut", "clip"]:
            action_advice = f"Farmers should {verb} infested tillers immediately."
            scrubbed_act = _scrub_unverified_agronomic_claims(action_advice, chunks=[])
            self.assertNotIn("infested tillers", scrubbed_act, f"Verb '{verb}' must be scrubbed when unverified")
            self.assertIn("field sanitation", scrubbed_act)

        # Case 5: Reverse syntax ('infested tillers must be destroyed')
        rev_advice = "All heavily infested tillers must be destroyed."
        scrubbed_rev = _scrub_unverified_agronomic_claims(rev_advice, chunks=[])
        self.assertNotIn("must be destroyed", scrubbed_rev)
        self.assertIn("field sanitation", scrubbed_rev)

        # Case 6: Genuine same-sentence directive to remove tillers
        genuine_chunk = [{
            "text": "In case of severe mealybug attack, remove infested tillers and destroy them to prevent spread.",
            "source_file": "MealybugGuide.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90
        }]
        preserved = _scrub_unverified_agronomic_claims(advice, chunks=genuine_chunk)
        self.assertEqual(preserved, advice, "Genuine same-sentence tiller removal documentation must preserve advice")

    def test_41_chemical_table_column_counts_preserved_across_3_4_5_columns(self):
        from step6_query_gate import reconcile_chemical_recommendations

        # 3-column table
        table_3col = (
            "| Target Pest | Treatment | Dosage |\n"
            "|---|---|---|\n"
            "| Rice Stem Borer | Chlorpyrifos 20 EC | 2 ml/l foliar spray |"
        )
        rec_3 = reconcile_chemical_recommendations(table_3col, is_offline_reg=True, chunks=[])
        row_3 = [c.strip() for c in rec_3.strip().splitlines()[2].split("|")[1:-1]]
        self.assertEqual(len(row_3), 3, "3-column table row must preserve 3 columns")
        self.assertEqual(row_3[0], "Rice Stem Borer")
        self.assertIn("⛔ UNVERIFIED / BLOCKED", row_3[1])

        # 4-column table
        table_4col = (
            "| Target Pest | Chemical | Formulation | Dosage |\n"
            "|---|---|---|---|\n"
            "| Rice Mealybug | Dimethoate | 30 EC | 1.5 ml/l spray |"
        )
        rec_4 = reconcile_chemical_recommendations(table_4col, is_offline_reg=True, chunks=[])
        row_4 = [c.strip() for c in rec_4.strip().splitlines()[2].split("|")[1:-1]]
        self.assertEqual(len(row_4), 4, "4-column table row must preserve 4 columns")
        self.assertEqual(row_4[0], "Rice Mealybug")
        self.assertIn("⛔ UNVERIFIED / BLOCKED", row_4[1])

        # 5-column table
        table_5col = (
            "| Target Pest | Chemical | Formulation | Dosage | Pre-Harvest Interval |\n"
            "|---|---|---|---|---|\n"
            "| Brown Plant Hopper | Imidacloprid | 17.8 SL | 0.5 ml/l foliar | 15 days |"
        )
        rec_5 = reconcile_chemical_recommendations(table_5col, is_offline_reg=True, chunks=[])
        row_5 = [c.strip() for c in rec_5.strip().splitlines()[2].split("|")[1:-1]]
        self.assertEqual(len(row_5), 5, "5-column table row must preserve 5 columns")
        self.assertEqual(row_5[0], "Brown Plant Hopper")
        self.assertIn("⛔ UNVERIFIED / BLOCKED", row_5[1])

    def test_42_provenance_outcome_flags_consistency(self):
        from step6_query_gate import query_gate

        sample_chunks = [{
            "chunk_id": "c1",
            "text": "Nitrogen deficiency causes yellow streaks in rice leaves.",
            "source_file": "TNAU.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90,
            "final_score": 0.90
        }]
        mock_answer = "Nitrogen deficiency causes yellow streaks in rice leaves."
        entailed_map = [{
            "sentence": "Nitrogen deficiency causes yellow streaks in rice leaves.",
            "match_status": "ENTAILED",
            "entailment_verified": True,
            "confidence": 0.9,
            "source": "TNAU.pdf",
            "temporal_score": 0.90,
            "freshness_label": "FRESH"
        }]

        with patch("step6_query_gate.retrieve_chunks", return_value=sample_chunks), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": mock_answer, "model_used": "mock-model", "sources": ["TNAU.pdf"]}), \
             patch("sentence_provenance.build_provenance_map", return_value=entailed_map):
            res = query_gate(
                "why my rice plant is getting yellow day by day",
                embedder=self.components[0],
                collection=self.components[1],
                bm25=self.components[2],
                corpus=self.components[3]
            )
            self.assertTrue(res.get("is_factually_proven"), "When sentences are entailed, is_factually_proven must be True")
            self.assertTrue(res.get("factually_validated"), "When sentences are entailed, factually_validated must be True")
            self.assertIn("PROVENANCE VERIFICATION NOTICE", res.get("answer", ""))

    def test_43_real_response_path_scrubs_table_water_depth_and_wrapped_tillers(self):
        from step6_query_gate import query_gate

        sample_chunks = [{
            "chunk_id": "c1",
            "text": "Rice crops require balanced irrigation and field sanitation.",
            "source_file": "GeneralRice.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.90,
            "final_score": 0.90
        }]
        # Unverified LLM output containing both table water depth and newline-wrapped tiller advice
        mock_answer = (
            "### Recommended Water and Agronomic Practices\n\n"
            "| Practice | Target Depth |\n"
            "|---|---|\n"
            "| Water Depth | 5–7 cm |\n\n"
            "For pest control, farmers should remove and destroy\n"
            "infested tillers in damaged patches. Also install 5 cm water pipe for drainage.\n"
            "Transplant seedlings at 15–20 cm spacing."
        )

        with patch("step6_query_gate.retrieve_chunks", return_value=sample_chunks), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": mock_answer, "model_used": "mock-model", "sources": ["GeneralRice.pdf"]}), \
             patch("step6_query_gate.fast_path_answer", return_value={"answer": mock_answer, "model_used": "mock-model", "sources": ["GeneralRice.pdf"]}):
            res = query_gate(
                "what water depth and practices for rice",
                embedder=self.components[0],
                collection=self.components[1],
                bm25=self.components[2],
                corpus=self.components[3]
            )
            final_ans = res.get("answer", "")
            # Table-separated water depth must be scrubbed
            self.assertNotIn("5–7 cm", final_ans)
            self.assertIn("recommended levels", final_ans)
            # Wrapped tiller destruction must be scrubbed
            self.assertNotIn("infested tillers", final_ans)
            self.assertIn("field sanitation", final_ans)
            # Non-depth water phrase and spacing must be preserved
            self.assertIn("5 cm water pipe", final_ans)
            self.assertIn("15–20 cm spacing", final_ans)


if __name__ == "__main__":
    unittest.main()
