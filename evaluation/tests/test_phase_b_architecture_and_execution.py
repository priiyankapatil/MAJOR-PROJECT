"""
Regression and verification test suite for Phase B:
Pipeline Architecture and Execution Issues (B5 - B8).
"""

import sys
import os
import io
import re
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

WORKSPACE = Path(__file__).resolve().parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import step6_query_gate
import regulatory_updater


class TestPhaseBArchitectureAndExecution(unittest.TestCase):
    """
    Test suite for B5-B8:
    - B5: Singular definition and correctness of slow_path_answer (no duplicate stubs)
    - B7: Core pipeline query_gate execution is strictly non-interactive by default
    - B8: Encoding-safe console printing preventing UnicodeEncodeError on Windows codepages
    """

    def test_b5_singular_slow_path_answer_definition(self):
        """B5: Confirm exactly one definition of slow_path_answer exists in step6_query_gate.py."""
        source_path = WORKSPACE / "step6_query_gate.py"
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()

        defs = re.findall(r"^[ \t]*def slow_path_answer\(", content, flags=re.MULTILINE)
        self.assertEqual(len(defs), 1, f"Expected exactly 1 definition of slow_path_answer, found {len(defs)}")

        # Verify slow_path_answer function is callable and behaves safely on empty chunks
        res = step6_query_gate.slow_path_answer("test query", chunks=[], query_type="RECOMMENDATION")
        self.assertEqual(res["model_used"], "system/insufficient-context")
        self.assertEqual(res["path"], "slow")
        self.assertIn("Insufficient verified context", res["answer"])

    def test_b7_core_pipeline_is_non_interactive_by_default(self):
        """
        B7: Normal query_gate pipeline execution must NEVER prompt or wait for terminal input,
        even when running in a terminal where sys.stdin.isatty() is True.
        """
        # Save and restore environment flags
        orig_no_feedback = os.environ.pop("NO_INTERACTIVE_FEEDBACK", None)
        orig_enable_feedback = os.environ.pop("ENABLE_INTERACTIVE_FEEDBACK", None)

        sample_chunks = [{
            "chunk_id": "c1",
            "text": "Rice requires balanced fertilizers.",
            "source_file": "TNAU.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "final_score": 0.95
        }]
        mock_answer_data = {
            "answer": "Rice requires balanced fertilizers.",
            "model_used": "mock-llm",
            "sources": ["TNAU.pdf"],
            "path": "slow"
        }

        try:
            # Simulate an interactive terminal session where isatty() == True
            with patch.object(sys.stdin, "isatty", return_value=True), \
                 patch("builtins.input", side_effect=AssertionError("CRITICAL BUG: input() was called during normal non-interactive pipeline execution!")) as mock_input, \
                 patch("step6_query_gate.retrieve_chunks", return_value=sample_chunks), \
                 patch("step6_query_gate.slow_path_answer", return_value=mock_answer_data), \
                 patch("sentence_provenance.build_provenance_map", return_value=[]):

                # Run query_gate without explicit interactive feedback
                result = step6_query_gate.query_gate(
                    "how to fertilize rice",
                    embedder=MagicMock(),
                    collection=MagicMock(),
                    bm25=MagicMock(),
                    corpus=MagicMock()
                )

                # Assert that input() was never called
                mock_input.assert_not_called()
                self.assertIsNotNone(result)
                self.assertIn("answer", result)
        finally:
            if orig_no_feedback is not None:
                os.environ["NO_INTERACTIVE_FEEDBACK"] = orig_no_feedback
            if orig_enable_feedback is not None:
                os.environ["ENABLE_INTERACTIVE_FEEDBACK"] = orig_enable_feedback

    def test_b7_interactive_feedback_when_explicitly_enabled(self):
        """B7: When explicitly enabled at CLI/caller level, feedback prompt is presented and handled."""
        orig_no_feedback = os.environ.pop("NO_INTERACTIVE_FEEDBACK", None)

        sample_chunks = [{
            "chunk_id": "c1",
            "text": "Rice requires balanced fertilizers.",
            "source_file": "TNAU.pdf",
            "freshness_label": "FRESH",
            "temporal_score": 0.95,
            "final_score": 0.95
        }]
        mock_answer_data = {
            "answer": "Rice requires balanced fertilizers.",
            "model_used": "mock-llm",
            "sources": ["TNAU.pdf"],
            "path": "slow"
        }

        try:
            with patch.object(sys.stdin, "isatty", return_value=True), \
                 patch("builtins.input", return_value="y") as mock_input, \
                 patch("step6_query_gate.retrieve_chunks", return_value=sample_chunks), \
                 patch("step6_query_gate.slow_path_answer", return_value=mock_answer_data), \
                 patch("sentence_provenance.build_provenance_map", return_value=[]), \
                 patch("step6_query_gate.record_feedback") as mock_record:

                result = step6_query_gate.query_gate(
                    "how to fertilize rice",
                    embedder=MagicMock(),
                    collection=MagicMock(),
                    bm25=MagicMock(),
                    corpus=MagicMock(),
                    enable_interactive_feedback=True
                )

                mock_input.assert_called_once()
                mock_record.assert_called_once()
                self.assertTrue(mock_record.call_args[1]["was_accurate"])
        finally:
            if orig_no_feedback is not None:
                os.environ["NO_INTERACTIVE_FEEDBACK"] = orig_no_feedback

    def test_b8_safe_print_handles_strict_ascii_stream_without_raising(self):
        """B8: _safe_print handles streams with restricted encodings (e.g. ASCII, cp1252) safely."""
        byte_stream = io.BytesIO()
        strict_ascii_writer = io.TextIOWrapper(byte_stream, encoding="ascii", errors="strict")

        # Test printing unicode and emojis through strict ASCII stream
        emoji_text = "🌾 Agricultural Advisory: ⚠️ Warning — Approved ✅ Chemical ⛔ Banned 🛡️ Trust"

        # Direct write raises UnicodeEncodeError
        with self.assertRaises(UnicodeEncodeError):
            strict_ascii_writer.write(emoji_text)

        # _safe_print catches and handles UnicodeEncodeError cleanly
        step6_query_gate._safe_print(emoji_text, file=strict_ascii_writer, flush=True)

        output_bytes = byte_stream.getvalue()
        self.assertGreater(len(output_bytes), 0)
        decoded_output = output_bytes.decode("ascii", errors="replace")
        self.assertIn("Agricultural Advisory", decoded_output)
        self.assertIn("Warning", decoded_output)

    def test_b8_regulatory_updater_uses_safe_print(self):
        """B8: regulatory_updater module print function safely handles restricted encodings."""
        byte_stream = io.BytesIO()
        strict_ascii_writer = io.TextIOWrapper(byte_stream, encoding="ascii", errors="strict")

        log_text = "🔄 REGULATORY UPDATE: ✅ CIB&RC scrape successful | ⚠️ Warning notice"
        regulatory_updater._safe_print(log_text, file=strict_ascii_writer, flush=True)

        output_str = byte_stream.getvalue().decode("ascii", errors="replace")
        self.assertIn("REGULATORY UPDATE", output_str)
        self.assertIn("CIB&RC scrape successful", output_str)


if __name__ == "__main__":
    unittest.main()
