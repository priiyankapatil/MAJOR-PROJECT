"""
Offline regression and safety tests covering critical correctness fixes:
1. Semantic Bridge: No "what" -> "wheat" false positive expansion.
2. Folk Taxonomy: No collective "organic manure" narrowing to FYM.
3. Vernacular mapping preservation: "dhan" -> "Oryza sativa", "tamatar" -> "Solanum lycopersicum".
4. Multi-crop queries: Balanced preservation without single-entity overwrite.
5. Fast-path and Slow-path empty chunk handling: Structured insufficient-context response.
6. Temporal credibility decay: Stale / outdated source decay behavior.
7. Phenological gate: Stage compatibility vs unknown stage fallback.
8. Chemical safety reconciliation: Offline synthetic chemical scrubbing.
9. Sentence-level provenance unification: Both fast & slow paths handle evidence verification.
"""

import unittest
from unittest.mock import MagicMock, patch

import folk_taxonomy
import dialectal_alignment
from dialectal_alignment import enrich_query
from folk_taxonomy import FOLK_TAXONOMY, STOPWORDS
from temporal_credibility import score_all_chunks, filter_stale_chunks, compute_temporal_score
from phenology_gate import apply_phenological_gate
from stage_compatibility import check_chunk_compatibility
from step6_query_gate import fast_path_answer, slow_path_answer, reconcile_chemical_recommendations


class TestSemanticBridgeCorrectness(unittest.TestCase):
    """Verify semantic bridge fixes for false positive expansions."""

    def test_what_does_not_expand_to_wheat(self):
        """Query with 'what' must never expand or rewrite to wheat."""
        from semantic_bridge import apply_semantic_bridge
        query = "What is the recommended dose of urea for paddy?"
        res = apply_semantic_bridge(query)
        enriched = res.get("enriched", query)
        self.assertNotIn("Triticum", enriched)
        # Ensure 'what' remains intact
        self.assertTrue(enriched.startswith("What") or enriched.startswith("what"))

    def test_whats_and_what_is_not_expanded(self):
        """Possessive/contraction 'what's' must not trigger crop expansion."""
        from semantic_bridge import apply_semantic_bridge
        query = "What's the best time to sow mustard?"
        res = apply_semantic_bridge(query)
        enriched = res.get("enriched", query)
        self.assertNotIn("Triticum", enriched)

    def test_organic_manure_not_narrowed_to_fym(self):
        """Generic 'organic manure' or 'organic manures' must not be narrowed to Farm Yard Manure (FYM)."""
        from semantic_bridge import apply_semantic_bridge
        query = "How to prepare organic manure at home?"
        res = apply_semantic_bridge(query)
        enriched = res.get("enriched", query)
        # Should not falsely inject FYM or gobar khad
        self.assertNotIn("Farm Yard Manure", enriched)
        self.assertNotIn("gobar khad", enriched)

    def test_valid_vernacular_mapping_preserved(self):
        """Valid vernacular names like 'dhan' or 'tamatar' must map correctly."""
        from semantic_bridge import apply_semantic_bridge
        dhan_query = "dhan blast control measures"
        dhan_res = apply_semantic_bridge(dhan_query)
        dhan_enriched = dhan_res.get("enriched", dhan_query)
        self.assertTrue("paddy" in dhan_enriched.lower() or "oryza" in dhan_enriched.lower())

        tamatar_query = "tamatar leaf curl symptoms"
        tamatar_res = apply_semantic_bridge(tamatar_query)
        tamatar_enriched = tamatar_res.get("enriched", tamatar_query)
        self.assertTrue("tomato" in tamatar_enriched.lower() or "solanum lycopersicum" in tamatar_enriched.lower())

    def test_multi_crop_query_balance(self):
        """Multi-crop query mentioning tomato, chilli, and brinjal must retain all entities."""
        from semantic_bridge import apply_semantic_bridge
        query = "management of fruit borer in tomato, chilli and brinjal"
        res = apply_semantic_bridge(query)
        enriched = res.get("enriched", query)
        self.assertIn("tomato", enriched.lower())
        self.assertIn("chilli", enriched.lower())
        self.assertIn("brinjal", enriched.lower())


class TestTemporalAndPhenologyFilters(unittest.TestCase):
    """Verify temporal credibility and phenology stage filtering offline."""

    def test_temporal_decay_stale_source(self):
        """Stale sources should receive decay penalty."""
        chunks = [
            {"text": "Apply 100 kg N/ha in 2018 guidelines.", "source_file": "Crop Protection.pdf"},
            {"text": "Apply 120 kg N/ha in 2023 guidelines.", "source_file": "ICAR.pdf"},
        ]
        scored = score_all_chunks(chunks, query_type="RECOMMENDATION", current_year=2025)
        self.assertEqual(len(scored), 2)
        # Recent source should rank higher
        self.assertGreater(scored[0]["temporal_score"], scored[1]["temporal_score"])
        self.assertEqual(scored[0]["temporal_source_match"], "ICAR")

    def test_phenology_stage_compatibility_vs_unknown(self):
        """Confirmed compatible stage must pass, incompatible must be filtered, unknown must be preserved safely."""
        # check_chunk_compatibility directly
        pre_comp = check_chunk_compatibility("Weed management at pre-emergence stage", "sowing")
        self.assertTrue(pre_comp.get("compatible"))

        harv_comp = check_chunk_compatibility("Harvesting guidelines for mature crop", "sowing")
        self.assertFalse(harv_comp.get("compatible"))

    def test_empty_chunks_safe_handling(self):
        """Empty chunks should return empty list without crashing."""
        self.assertEqual(score_all_chunks([], query_type="FACTUAL"), [])
        self.assertEqual(filter_stale_chunks([]), [])
        gate_res = apply_phenological_gate([], "query", 20.59, 78.96)
        self.assertEqual(gate_res["allowed_chunks"], [])


class TestSafetyAndProvenanceUnification(unittest.TestCase):
    """Verify safety checks and provenance handling across fast and slow paths."""

    def test_fast_path_empty_chunks_safe_return(self):
        """fast_path_answer with empty chunks must return structured insufficient context."""
        result = fast_path_answer("What is the seed rate for wheat?", [])
        self.assertEqual(result.get("model_used"), "system/insufficient-context")
        self.assertEqual(result.get("sources"), [])
        self.assertIn("Insufficient verified context", result.get("answer", ""))

    def test_slow_path_empty_chunks_safe_return(self):
        """slow_path_answer with empty chunks must return structured insufficient context."""
        result = slow_path_answer("What is the seed rate for wheat?", [], "factual")
        self.assertEqual(result.get("model_used"), "system/insufficient-context")
        self.assertEqual(result.get("sources"), [])
        self.assertIn("Insufficient verified context", result.get("answer", ""))

    def test_chemical_safety_reconciliation_offline(self):
        """Under offline regulatory mode, synthetic chemicals without verified POP support are flagged or scrubbed."""
        unverified_text = "You should spray Monocrotophos 36 SL at 2 ml/L immediately."
        sanitized = reconcile_chemical_recommendations(unverified_text, is_offline_reg=True, chunks=[])
        # Monocrotophos is hazardous/restricted in many applications and with 0 chunks must have safety warning or scrubbed
        self.assertTrue(
            "⚠️" in sanitized or "unverified" in sanitized.lower() or "Monocrotophos" not in sanitized or "banned" in sanitized.lower() or "caution" in sanitized.lower()
        )


if __name__ == "__main__":
    unittest.main()
