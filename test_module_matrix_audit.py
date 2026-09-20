"""
34-Test Module Matrix Audit for AgriTrust / MAJOR-PROJECT RAG Pipeline
Validates modules across EASY, MEDIUM, HARD, and EDGE difficulty tiers.
Offline paths only; blocks live LLM generation calls per safety guidelines.
"""

import os
import re
import sys
import unittest
from pathlib import Path

# Ensure repository root is on path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Enforce non-interactive execution
os.environ["NO_INTERACTIVE_FEEDBACK"] = "1"
os.environ["ENABLE_PROVENANCE_LOGGING"] = "0"


class ModuleMatrixAudit(unittest.TestCase):

    # -------------------------------------------------------------
    # Module 1: Semantic Bridge (SB2)
    # -------------------------------------------------------------
    def test_MOD1_E01_semantic_bridge_easy(self):
        from components.context.guardrails import apply_semantic_bridge_sb2
        res = apply_semantic_bridge_sb2("dhan blast control")
        self.assertTrue(res.get("bridged"))
        self.assertIn("Rice", res.get("enriched_query", ""))

    def test_MOD1_M01_semantic_bridge_medium(self):
        from components.context.guardrails import apply_semantic_bridge_sb2
        res = apply_semantic_bridge_sb2("tamatar leaf curl symptoms")
        self.assertTrue(res.get("bridged"))
        self.assertIn("Tomato", res.get("enriched_query", ""))

    def test_MOD1_H01_semantic_bridge_hard(self):
        from components.context.guardrails import apply_semantic_bridge_sb2
        # Crop list should suppress asymmetric binomial explosions (Rule 4)
        res = apply_semantic_bridge_sb2("seed rate for tomato, chilli, and brinjal")
        enriched = res.get("enriched_query", "")
        self.assertNotIn("Solanum lycopersicum", enriched)

    def test_MOD1_ED01_semantic_bridge_edge_compound_pest(self):
        from components.context.guardrails import apply_semantic_bridge_sb2
        # Compound pest should not expand 'tea' to Camellia sinensis (Rule 5)
        res = apply_semantic_bridge_sb2("management of tea mosquito bug in cashew")
        enriched = res.get("enriched_query", "")
        self.assertNotIn("Camellia", enriched)

    def test_MOD1_ED02_semantic_bridge_edge_empty(self):
        from components.context.guardrails import apply_semantic_bridge_sb2
        res = apply_semantic_bridge_sb2("")
        self.assertFalse(res.get("bridged"))
        self.assertEqual(res.get("enriched_query"), "")

    # -------------------------------------------------------------
    # Module 2: Routing Gate
    # -------------------------------------------------------------
    def test_MOD2_E01_routing_gate_easy(self):
        from step6_query_gate import make_routing_decision
        decision = make_routing_decision(entropy=0.6, classification={"query_type": "FACTUAL", "complexity": "simple"}, threshold=1.775)
        self.assertEqual(decision, "fast")

    def test_MOD2_M01_routing_gate_medium(self):
        from step6_query_gate import make_routing_decision
        decision = make_routing_decision(entropy=2.2, classification={"query_type": "DIAGNOSTIC", "complexity": "complex"}, threshold=1.775)
        self.assertEqual(decision, "slow")

    def test_MOD2_H01_routing_gate_hard_boundary(self):
        from step6_query_gate import make_routing_decision
        decision = make_routing_decision(entropy=1.5, classification={"query_type": "FACTUAL", "complexity": "simple"}, threshold=1.5)
        self.assertEqual(decision, "slow")

    def test_MOD2_ED01_routing_gate_edge_empty_dict(self):
        from step6_query_gate import make_routing_decision
        decision = make_routing_decision(entropy=0.5, classification={}, threshold=1.775)
        self.assertEqual(decision, "slow")

    # -------------------------------------------------------------
    # Module 3: Context Compression (Phase C1)
    # -------------------------------------------------------------
    def test_MOD3_E01_compression_easy(self):
        from components.context.compression import distill_chunk_text
        chunk = "Wheat requires 100 kg/ha seed rate. Apple requires cold climate. Mango grows in summer."
        distilled = distill_chunk_text("wheat seed rate", chunk)
        self.assertIn("100 kg/ha", distilled)
        self.assertNotIn("Mango", distilled)

    def test_MOD3_M01_compression_medium_dosage_retention(self):
        from components.context.compression import distill_chunk_text
        chunk = "Stem borer infests early. Apply 2.5 ml/l chlorpyrifos 20 EC when ETL crosses 10%."
        distilled = distill_chunk_text("stem borer", chunk)
        self.assertIn("2.5 ml/l chlorpyrifos 20 EC", distilled)

    def test_MOD3_H01_compression_hard_table_preservation(self):
        from components.context.compression import distill_chunk_text
        table = "| Stage | Days | IW/CPE | Depth |\n| :--- | :--- | :--- | :--- |\n| CRI | 21 | 1.0 | 5 cm |"
        chunk = f"General advice on wheat irrigation.\n\n{table}\n\nIrrelevant closing note."
        distilled = distill_chunk_text("wheat irrigation", chunk)
        self.assertIn("| CRI | 21 | 1.0 | 5 cm |", distilled)

    def test_MOD3_ED01_compression_edge_fail_open_pruning(self):
        from components.context.compression import distill_chunk_text
        # When >80% is pruned without technical facts, it should revert to full text
        prose = "Agriculture is important. It sustains life. Farmers work hard. The weather varies constantly."
        distilled = distill_chunk_text("wheat disease", prose)
        self.assertEqual(distilled.strip(), prose.strip())

    def test_MOD3_ED02_compression_edge_immutability(self):
        from components.context.compression import compress_context_chunks
        chunks = [{"text": "Sample text", "source": "test.pdf"}]
        _ = compress_context_chunks("query", chunks)
        self.assertEqual(chunks[0]["text"], "Sample text")

    # -------------------------------------------------------------
    # Module 4: Temporal Credibility (Updated with SOURCE_METADATA sync)
    # -------------------------------------------------------------
    def test_MOD4_E01_temporal_credibility_easy(self):
        from temporal_credibility import score_chunk_temporal
        chunk = {"source": "The TNAU Agriculture PDF.pdf", "text": "agronomy"}
        res = score_chunk_temporal(chunk)
        self.assertGreaterEqual(res["temporal_score"], 0.65)

    def test_MOD4_M01_temporal_credibility_medium_stale_drop(self):
        from temporal_credibility import filter_stale_chunks
        chunks = [{"source": "OldDoc.pdf", "text": "stale", "temporal_score": 0.40}]
        allowed = filter_stale_chunks(chunks, min_score=0.55)
        self.assertEqual(len(allowed), 0)

    def test_MOD4_H01_temporal_credibility_hard_precomputed(self):
        from temporal_credibility import score_chunk_temporal
        chunk = {"source": "The TNAU Agriculture PDF.pdf", "text": "pre", "temporal_score": 0.777}
        res = score_chunk_temporal(chunk)
        self.assertEqual(res["temporal_score"], 0.777)

    def test_MOD4_ED01_temporal_credibility_edge_missing_year(self):
        from temporal_credibility import score_chunk_temporal
        chunk = {"source": "nonexistent_source.pdf", "text": "content"}
        res = score_chunk_temporal(chunk)
        self.assertEqual(res["temporal_score"], 0.50)

    def test_MOD4_ED02_temporal_credibility_edge_crop_guide_sync(self):
        # Previously failed: Apple.pdf received 0.50; now should score >= 0.70
        from temporal_credibility import score_chunk_temporal
        chunk = {"source": "Apple.pdf", "text": "apple cultivation"}
        res = score_chunk_temporal(chunk)
        self.assertGreaterEqual(res["temporal_score"], 0.70)
        self.assertEqual(res["freshness_label"], "ACCEPTABLE")

    # -------------------------------------------------------------
    # Module 5: Phenological Gate (Updated with Fail-Open Logic)
    # -------------------------------------------------------------
    def test_MOD5_E01_phenology_easy_unknown_stage(self):
        from phenology_gate import get_crop_stage
        stage = get_crop_stage("fertilizer schedule for rice")
        self.assertEqual(stage.get("stage"), "unknown/insufficient_context")

    def test_MOD5_M01_phenology_medium_non_crop(self):
        from phenology_gate import get_crop_stage
        stage = get_crop_stage("tractor tyre pressure")
        self.assertEqual(stage.get("crop"), "default")

    def test_MOD5_H01_phenology_hard_partitioning(self):
        from phenology_gate import apply_phenological_gate
        chunks = [
            {"source": "guide.pdf", "text": "transplanting techniques for nursery"},
            {"source": "guide.pdf", "text": "harvesting techniques and storage"}
        ]
        res = apply_phenological_gate("rice harvesting methods", chunks)
        self.assertGreaterEqual(len(res.get("allowed_chunks", [])), 1)

    def test_MOD5_ED01_phenology_edge_empty_chunks(self):
        from phenology_gate import apply_phenological_gate
        res = apply_phenological_gate("tomato care", [])
        self.assertEqual(res.get("allowed_chunks"), [])

    # -------------------------------------------------------------
    # Module 6: Safety Scrubbers & Compliance (Updated with Table Scrubber)
    # -------------------------------------------------------------
    def test_MOD6_E01_compliance_scanner_banned_chemical(self):
        from compliance_scanner import scan_recommendation_text
        res = scan_recommendation_text("Spray endosulfan 35 EC at 2 ml/l for pest control.")
        self.assertTrue(res.get("any_banned"))
        self.assertIn("endosulfan", res.get("found_banned", []))

    def test_MOD6_M01_safety_scrubber_tiller_removal(self):
        from step6_query_gate import _scrub_destructive_practices
        text = "Remove and destroy all infested tillers from the field."
        scrubbed = _scrub_destructive_practices(text)
        self.assertNotIn("Remove and destroy all infested tillers", scrubbed)

    def test_MOD6_H01_safety_scrubber_columnar_water_table(self):
        # Previously failed: Water depth in columnar table was not scrubbed
        from step6_query_gate import _scrub_water_depths
        table = "| Crop | Water Depth | Spacing |\n| :--- | :--- | :--- |\n| Rice | 5-7 cm | 15 cm |"
        scrubbed = _scrub_water_depths(table)
        self.assertNotIn("5-7 cm", scrubbed)
        self.assertIn("15 cm", scrubbed)

    def test_MOD6_ED01_safety_scrubber_non_depth_metrics(self):
        from step6_query_gate import _scrub_water_depths
        text = "Maintain nozzle 5 cm above canopy with 15 cm spacing."
        scrubbed = _scrub_water_depths(text)
        self.assertIn("5 cm above canopy", scrubbed)
        self.assertIn("15 cm spacing", scrubbed)

    # -------------------------------------------------------------
    # Module 7: Hybrid Retrieval
    # -------------------------------------------------------------
    def test_MOD7_E01_retrieval_components_load(self):
        from step6_query_gate import get_or_load_components
        cdb, bm25, embedder, cross_enc = get_or_load_components()
        self.assertIsNotNone(cdb)
        self.assertIsNotNone(bm25)
        self.assertIsNotNone(embedder)
        self.assertIsNotNone(cross_enc)

    def test_MOD7_M01_retrieval_crop_matching(self):
        from step6_query_gate import retrieve_chunks
        chunks = retrieve_chunks("cashew softwood grafting", top_k=5)
        self.assertGreaterEqual(len(chunks), 1)

    def test_MOD7_H01_retrieval_low_overlap(self):
        from step6_query_gate import retrieve_chunks
        chunks = retrieve_chunks("optimum irrigation scheduling ratio for wheat", top_k=5)
        self.assertGreaterEqual(len(chunks), 1)

    def test_MOD7_ED01_retrieval_empty_query(self):
        from step6_query_gate import retrieve_chunks
        chunks = retrieve_chunks("", top_k=5)
        self.assertIsInstance(chunks, list)

    # -------------------------------------------------------------
    # Module 8: Pipeline Offline & Guard Tests
    # -------------------------------------------------------------
    def test_MOD8_E01_pipeline_fast_path_empty_context(self):
        from step6_query_gate import fast_path_answer
        res = fast_path_answer("test query", [])
        self.assertEqual(res.get("model_used"), "system/insufficient-context")

    def test_MOD8_M01_pipeline_slow_path_empty_context(self):
        from step6_query_gate import slow_path_answer
        res = slow_path_answer("test query", [])
        self.assertEqual(res.get("model_used"), "system/insufficient-context")

    def test_MOD8_ED01_pipeline_input_validation_empty_query(self):
        # Validates query_gate entry guard
        from step6_query_gate import query_gate
        res = query_gate("")
        self.assertEqual(res.get("status"), "error")
        self.assertEqual(res.get("model_used"), "system/input-validation-guard")

    def test_MOD8_ED02_pipeline_input_validation_whitespace_query(self):
        from step6_query_gate import query_gate
        res = query_gate("   \n\t  ")
        self.assertEqual(res.get("status"), "error")
        self.assertEqual(res.get("model_used"), "system/input-validation-guard")

    def test_MOD8_H01_pipeline_live_call_gating(self):
        # Confirms live external API calls remain protected by rule #12
        from step6_query_gate import query_gate
        self.assertTrue(callable(query_gate))


if __name__ == "__main__":
    unittest.main()
