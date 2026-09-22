"""
Unit and regression tests for Phase C1: Selective Context Compression.
Verifies distillation rules, technical token preservation, table preservation,
safeguards (empty and >80% pruning), immutability of chunk objects,
and feature flag behavior.
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import config
from components.context.compression import (
    distill_chunk_text,
    compress_context_chunks,
    is_sentence_relevant,
    extract_query_keywords
)
import step6_query_gate


class TestContextCompression(unittest.TestCase):

    def test_01_query_relevant_sentences_retained(self):
        query = "how to control brown plant hopper in rice"
        text = (
            "Brown plant hopper causes hopper burn in susceptible rice fields. "
            "Farmers should monitor the base of tillers regularly. "
            "Apple cultivation in Himachal Pradesh requires adequate winter chilling hours."
        )
        distilled, kept, dropped = distill_chunk_text(query, text)
        self.assertIn("Brown plant hopper", distilled)
        self.assertIn("hopper burn", distilled)
        self.assertIn("Apple cultivation", " ".join(dropped))
        self.assertNotIn("Apple cultivation", distilled)

    def test_02_irrelevant_sentences_pruned(self):
        query = "nitrogen fertilizer schedule for sugarcane"
        text = (
            "Apply nitrogen in three split doses at planting, 45 days, and 90 days. "
            "Weeds can also be removed manually along border roads to keep surroundings tidy. "
            "Sugarcane requires adequate nitrogen for vegetative stalk growth."
        )
        distilled, kept, dropped = distill_chunk_text(query, text)
        self.assertIn("Apply nitrogen in three split doses", distilled)
        self.assertIn("Sugarcane requires adequate nitrogen", distilled)
        self.assertIn("Weeds can also be removed manually along border roads", " ".join(dropped))
        self.assertNotIn("Weeds can also be removed manually along border roads", distilled)

    def test_03_technical_examples_preserved(self):
        """
        Verify preservation of technical numbers, units, dosages, chemical names,
        formulas, and ratios even if the query does not mention them directly.
        """
        query = "general management of pests and irrigation"
        text = (
            "Foliar spray of Chlorpyrifos 20 EC at 2.5 ml/l provides effective control against borers. "
            "Basal application of 50 kg/ha nitrogen is recommended before transplanting. "
            "Irrigate the field when IW/CPE = 0.8 is reached. "
            "Weather in northern plains is generally pleasant during early spring."
        )
        distilled, kept, dropped = distill_chunk_text(query, text)
        # All 4 technical lines must be preserved
        self.assertIn("2.5 ml/l", distilled)
        self.assertIn("Chlorpyrifos 20 EC", distilled)
        self.assertIn("50 kg/ha", distilled)
        self.assertIn("IW/CPE = 0.8", distilled)
        # Narrative fluff pruned
        self.assertIn("pleasant during early spring", " ".join(dropped))
        self.assertNotIn("pleasant during early spring", distilled)

    def test_04_markdown_tables_remain_verbatim(self):
        query = "rice mealybug management"
        text = (
            "Infested fields require careful monitoring.\n\n"
            "| Pest | Recommended Practice | Dosage |\n"
            "|---|---|---|\n"
            "| Rice Mealybug | Field sanitation and removal | As needed |\n"
            "| Stem Borer | Pheromone traps | 5 traps/ha |\n\n"
            "Tourists often visit the agricultural research station during harvest."
        )
        distilled, kept, dropped = distill_chunk_text(query, text)
        # Table must remain verbatim
        self.assertIn("| Pest | Recommended Practice | Dosage |", distilled)
        self.assertIn("| Rice Mealybug | Field sanitation and removal | As needed |", distilled)
        self.assertIn("| Stem Borer | Pheromone traps | 5 traps/ha |", distilled)
        # Non-table irrelevant sentence pruned
        self.assertNotIn("Tourists often visit", distilled)

    def test_05_empty_result_and_over_80_pruning_safeguard(self):
        query = "wheat blast control"
        # Case A: Entirely irrelevant text that would drop all sentences
        text_all_irrelevant = (
            "The historical museum was established in 1952. "
            "Local architecture reflects colonial construction styles."
        )
        distilled_a, kept_a, dropped_a = distill_chunk_text(query, text_all_irrelevant)
        # Safeguard must retain original text rather than returning empty
        self.assertEqual(distilled_a, text_all_irrelevant)

        # Case B: Pruning would remove >80% of characters
        text_over_80 = (
            "Wheat blast causes spike bleaching. "  # ~36 chars
            "A very long historical preamble discussing the origins of regional trade routes across ancient territories "
            "which spanned hundreds of miles and influenced culinary practices for several centuries without any connection to agronomy."  # ~230 chars
        )
        distilled_b, kept_b, dropped_b = distill_chunk_text(query, text_over_80)
        # >80% dropped -> safeguard retains original
        self.assertEqual(distilled_b, text_over_80)

    def test_06_original_chunk_metadata_and_objects_not_mutated(self):
        query = "fertilizer for rice"
        original_chunk = {
            "chunk_id": "chunk_0042",
            "source_file": "TNAU_Rice.pdf",
            "text": (
                "Apply 50 kg/ha urea at tillering. "
                "The administrative office is closed on second Saturdays."
            ),
            "temporal_score": 0.92,
            "freshness_label": "FRESH"
        }
        original_text_copy = original_chunk["text"]

        compressed_chunks = compress_context_chunks(query, [original_chunk])

        # 1. Returned chunk is a new object
        self.assertIsNot(compressed_chunks[0], original_chunk)
        # 2. Original chunk text is completely unmutated
        self.assertEqual(original_chunk["text"], original_text_copy)
        # 3. Compressed chunk has updated distilled text
        self.assertIn("50 kg/ha urea", compressed_chunks[0]["text"])
        self.assertNotIn("administrative office is closed", compressed_chunks[0]["text"])
        # 4. Metadata and IDs are preserved exactly
        self.assertEqual(compressed_chunks[0]["chunk_id"], "chunk_0042")
        self.assertEqual(compressed_chunks[0]["source_file"], "TNAU_Rice.pdf")
        self.assertEqual(compressed_chunks[0]["temporal_score"], 0.92)
        self.assertEqual(compressed_chunks[0]["freshness_label"], "FRESH")

    def test_07_flag_off_behavior_preserves_prompt_context_exactly(self):
        """When ENABLE_CONTEXT_COMPRESSION is False, slow_path_answer must not compress context."""
        query = "rice pest control"
        sample_chunk = {
            "chunk_id": "c1",
            "source_file": "TNAU.pdf",
            "text": "Apply field sanitation. The station canteen serves tea at 4 PM.",
            "freshness_label": "FRESH"
        }

        captured_contexts = []

        def mock_chat_create(*args, **kwargs):
            messages = kwargs.get("messages", [])
            for m in messages:
                if "Farmer's Question:" in m.get("content", ""):
                    captured_contexts.append(m["content"])
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Sanitation is recommended."
            return mock_resp

        # Ensure flag is OFF
        with patch.object(config, "ENABLE_CONTEXT_COMPRESSION", False), \
             patch("step6_query_gate.client.chat.completions.create", side_effect=mock_chat_create):
            res = step6_query_gate.slow_path_answer(query, [sample_chunk], "RECOMMENDATION")
            self.assertIn("The station canteen serves tea at 4 PM.", captured_contexts[0])

    def test_08_compression_exceptions_fail_open_to_original_context(self):
        """When compression encounters an unexpected error, slow_path_answer must fail open to original text."""
        query = "rice pest control"
        sample_chunk = {
            "chunk_id": "c1",
            "source_file": "TNAU.pdf",
            "text": "Apply field sanitation. Keep bunds clean.",
            "freshness_label": "FRESH"
        }

        captured_contexts = []

        def mock_chat_create(*args, **kwargs):
            messages = kwargs.get("messages", [])
            for m in messages:
                if "Farmer's Question:" in m.get("content", ""):
                    captured_contexts.append(m["content"])
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Sanitation is recommended."
            return mock_resp

        # Simulate exception inside distill_chunk_text
        with patch.object(config, "ENABLE_CONTEXT_COMPRESSION", True), \
             patch("components.context.compression.distill_chunk_text", side_effect=RuntimeError("Regex segmentation error")), \
             patch("step6_query_gate.client.chat.completions.create", side_effect=mock_chat_create):
            res = step6_query_gate.slow_path_answer(query, [sample_chunk], "RECOMMENDATION")
            # Must fail open to original text without crashing
            self.assertIn("Keep bunds clean.", captured_contexts[0])


if __name__ == "__main__":
    unittest.main()
