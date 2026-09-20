"""
evaluation/generation/test_conformal_trust_integration.py
=========================================================
Focused test suite for Conformal Trust Scorer integration:
1. Supported answer and evidence
2. Unsupported or hallucinated claims
3. Empty answer or empty evidence
4. Scorer/API failure and timeout
5. Fast path remains unchanged
6. Feature flag OFF preserves legacy behavior
7. Feature flag ON returns trust metadata without breaking existing schema
8. Telemetry appends safely without overwriting existing logs
"""

import os
import sys
import json
import tempfile
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import config
import step6_query_gate as s6
import conformal_trust_scorer as cts


class TestConformalTrustIntegration(unittest.TestCase):

    def setUp(self):
        self.orig_flag = getattr(config, "ENABLE_CONFORMAL_TRUST", False)
        config.ENABLE_CONFORMAL_TRUST = False
        os.environ["NO_INTERACTIVE_FEEDBACK"] = "1"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.isolated_log_path = os.path.join(self.temp_dir.name, "isolated_trust_log.json")
        os.environ["CONFORMAL_TRUST_LOG_PATH"] = self.isolated_log_path

    def tearDown(self):
        config.ENABLE_CONFORMAL_TRUST = self.orig_flag
        os.environ.pop("CONFORMAL_TRUST_LOG_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # ── Test 1: Supported Answer and Evidence ──
    def test_01_supported_answer_and_evidence(self):
        query = "What is vermicompost and what is its C:N ratio?"
        answer = "Vermicompost is organic fertilizer produced by earthworms. Its carbon to nitrogen ratio is 15:1."
        chunks = [
            {"text": "Vermicompost is organic fertilizer produced from farm waste using earthworms. The carbon to nitrogen ratio is 15:1."},
            {"text": "Earthworms decompose organic matter over 80 to 90 days to produce high-grade vermicompost."}
        ]

        # Mock LLM decomposition to deterministic claims
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='["Vermicompost is organic fertilizer produced by earthworms.", "Its carbon to nitrogen ratio is 15:1."]'))]
        mock_client.chat.completions.create.return_value = mock_resp

        res = cts.run_conformal_trust_scorer(
            query=query,
            answer=answer,
            chunks=chunks,
            client=mock_client
        )

        self.assertIn("aggregate", res)
        agg = res["aggregate"]
        self.assertGreaterEqual(agg["overall_trust_score"], 0.65)
        self.assertEqual(agg["supported_claims"], 2)
        self.assertEqual(agg["total_claims"], 2)
        self.assertIn("EVIDENCE ALIGNMENT", agg["verdict"])
        print(f"  ✓ Test 1 Passed: Supported answer scores evidence alignment ({agg['overall_trust_score']:.4f})")

    # ── Test 2: Unsupported or Hallucinated Claims ──
    def test_02_unsupported_or_hallucinated_claims(self):
        query = "What pesticide for tomato blight?"
        answer = "Apply quantum nano-pesticide X7 at 500ml per acre to cure all fungal diseases instantly."
        chunks = [
            {"text": "Mancozeb 2.5g per liter controls late blight in tomato effectively."},
            {"text": "Copper oxychloride is recommended for tomato fungal diseases at 3g per liter."}
        ]

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='["Apply quantum nano-pesticide X7 at 500ml per acre to cure all fungal diseases instantly."]'))]
        mock_client.chat.completions.create.return_value = mock_resp

        res = cts.run_conformal_trust_scorer(
            query=query,
            answer=answer,
            chunks=chunks,
            client=mock_client
        )

        self.assertIn("aggregate", res)
        agg = res["aggregate"]
        self.assertLess(agg["overall_trust_score"], 0.50)
        self.assertIn("UNSUPPORTED", agg["verdict"])
        print("  ✓ Test 2 Passed: Hallucinated claim correctly detected as unsupported")

    # ── Test 3: Empty Answer or Empty Evidence ──
    def test_03_empty_answer_and_evidence(self):
        # Empty answer
        res_empty_ans = cts.run_conformal_trust_scorer(
            query="test query",
            answer="",
            chunks=[{"text": "Some valid chunk."}]
        )
        self.assertEqual(res_empty_ans["aggregate"]["overall_trust_score"], 0.0)
        self.assertEqual(res_empty_ans["claims"], [])
        self.assertEqual(res_empty_ans["aggregate"]["supported_claims"], 0)

        # Empty evidence chunks
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content='["Claim 1 valid text."]'))]
        mock_client.chat.completions.create.return_value = mock_resp

        res_empty_chunks = cts.run_conformal_trust_scorer(
            query="test query",
            answer="Claim 1 valid text.",
            chunks=[],
            client=mock_client
        )
        self.assertEqual(res_empty_chunks["aggregate"]["overall_trust_score"], 0.0)
        self.assertEqual(res_empty_chunks["aggregate"]["supported_claims"], 0)
        self.assertEqual(res_empty_chunks["aggregate"]["support_ratio"], 0.0)
        print("  ✓ Test 3 Passed: Empty inputs handled gracefully without crash")

    # ── Test 4: Scorer/API Failure and Timeout ──
    def test_04_scorer_failure_preserves_answer(self):
        config.ENABLE_CONFORMAL_TRUST = True

        with patch("step6_query_gate.classify_query", return_value={"query_type": "DIAGNOSTIC", "confidence": 0.9, "key_topics": ["wheat"], "complexity": "complex", "reasoning": "test reasoning"}), \
             patch("step6_query_gate.measure_query_entropy", return_value={"entropy": 2.0, "token_logprobs": [-1.0], "first_tokens": ["test"]}), \
             patch("step6_query_gate.retrieve_chunks", return_value=[{"chunk_id": "c1", "text": "wheat disease text", "source_file": "ICAR.pdf", "final_score": 0.9}]), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": "Ensure proper field drainage and apply organic compost.", "model_used": "test-model", "sources": ["ICAR.pdf"], "path": "slow"}), \
             patch("conformal_trust_scorer.run_conformal_trust_scorer", side_effect=TimeoutError("Groq API request timed out")):

            mock_embedder = MagicMock()
            mock_coll = MagicMock()
            mock_bm25 = MagicMock()
            mock_corpus = MagicMock()

            out = s6.query_gate("wheat rust treatment", mock_embedder, mock_coll, mock_bm25, mock_corpus)

            # Answer must be intact
            self.assertIn("Ensure proper field drainage", out["answer"])
            # Trust score must be explicitly None (never fabricated)
            self.assertIsNone(out["trust_score"])
            self.assertIsNone(out["trust_verdict"])
            self.assertIsNone(out["trust_data"])
            print("  ✓ Test 4 Passed: Scorer timeout preserves answer and returns null trust metadata")

    # ── Test 5: Fast Path Remains Unchanged ──
    def test_05_fast_path_remains_unchanged(self):
        config.ENABLE_CONFORMAL_TRUST = True  # Even if flag is ON

        with patch("step6_query_gate.classify_query", return_value={"query_type": "FACTUAL", "confidence": 0.95, "key_topics": ["NPK"], "complexity": "simple", "reasoning": "test reasoning"}), \
             patch("step6_query_gate.measure_query_entropy", return_value={"entropy": 0.5, "token_logprobs": [-0.1], "first_tokens": ["NPK"]}), \
             patch("step6_query_gate.retrieve_chunks", return_value=[{"chunk_id": "c1", "text": "NPK is nitrogen phosphorus potassium", "source_file": "TNAU.pdf", "final_score": 0.9}]), \
             patch("step6_query_gate.fast_path_answer", return_value={"answer": "NPK stands for Nitrogen, Phosphorus, and Potassium.", "model_used": "fast-70b", "sources": ["TNAU.pdf"], "path": "fast"}), \
             patch("conformal_trust_scorer.run_conformal_trust_scorer") as mock_scorer:

            mock_embedder = MagicMock()
            mock_coll = MagicMock()
            mock_bm25 = MagicMock()
            mock_corpus = MagicMock()

            out = s6.query_gate("What does NPK stand for?", mock_embedder, mock_coll, mock_bm25, mock_corpus)

            # Scorer must NOT be called on fast path
            mock_scorer.assert_not_called()
            self.assertEqual(out["path"], "fast")
            self.assertIsNone(out["trust_score"])
            self.assertEqual(out["answer"], "NPK stands for Nitrogen, Phosphorus, and Potassium.")
            print("  ✓ Test 5 Passed: Fast path bypasses trust scoring completely")

    # ── Test 6: Feature Flag OFF Preserves Legacy Behavior ──
    def test_06_flag_off_preserves_legacy_behavior(self):
        config.ENABLE_CONFORMAL_TRUST = False

        with patch("step6_query_gate.classify_query", return_value={"query_type": "DIAGNOSTIC", "confidence": 0.9, "key_topics": ["tomato"], "complexity": "complex", "reasoning": "test reasoning"}), \
             patch("step6_query_gate.measure_query_entropy", return_value={"entropy": 2.0, "token_logprobs": [-1.0], "first_tokens": ["test"]}), \
             patch("step6_query_gate.retrieve_chunks", return_value=[{"chunk_id": "c1", "text": "tomato blight text", "source_file": "TNAU.pdf", "final_score": 0.9}]), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": "Spray copper oxychloride.", "model_used": "test-model", "sources": ["TNAU.pdf"], "path": "slow"}), \
             patch("conformal_trust_scorer.run_conformal_trust_scorer") as mock_scorer:

            mock_embedder = MagicMock()
            mock_coll = MagicMock()
            mock_bm25 = MagicMock()
            mock_corpus = MagicMock()

            out = s6.query_gate("tomato blight", mock_embedder, mock_coll, mock_bm25, mock_corpus)

            mock_scorer.assert_not_called()
            self.assertIsNone(out["trust_score"])
            self.assertIsNone(out["trust_verdict"])
            self.assertIsNone(out["trust_data"])
            self.assertIn("Spray copper oxychloride.", out["answer"])
            print("  ✓ Test 6 Passed: Flag OFF guarantees zero call to trust scorer")

    # ── Test 7: Feature Flag ON Returns Trust Metadata Without Breaking Schema ──
    def test_07_flag_on_returns_valid_schema(self):
        config.ENABLE_CONFORMAL_TRUST = True

        mock_trust_res = {
            "query": "tomato blight",
            "claims": ["Spray copper oxychloride."],
            "claim_results": [{"claim": "Spray copper oxychloride.", "trust_score": 0.88, "supported": True}],
            "aggregate": {
                "overall_trust_score": 0.88,
                "verdict": "HIGH EVIDENCE ALIGNMENT ✅",
                "support_ratio": 1.0,
                "total_claims": 1,
                "supported_claims": 1
            },
            "q_hat": 0.12,
            "alpha": 0.10
        }

        with patch("step6_query_gate.classify_query", return_value={"query_type": "DIAGNOSTIC", "confidence": 0.9, "key_topics": ["tomato"], "complexity": "complex", "reasoning": "test reasoning"}), \
             patch("step6_query_gate.measure_query_entropy", return_value={"entropy": 2.0, "token_logprobs": [-1.0], "first_tokens": ["test"]}), \
             patch("step6_query_gate.retrieve_chunks", return_value=[{"chunk_id": "c1", "text": "tomato blight text", "source_file": "TNAU.pdf", "final_score": 0.9}]), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": "Spray copper oxychloride.", "model_used": "test-model", "sources": ["TNAU.pdf"], "path": "slow"}), \
             patch("conformal_trust_scorer.run_conformal_trust_scorer", return_value=mock_trust_res):

            mock_embedder = MagicMock()
            mock_coll = MagicMock()
            mock_bm25 = MagicMock()
            mock_corpus = MagicMock()

            out = s6.query_gate("tomato blight", mock_embedder, mock_coll, mock_bm25, mock_corpus)

            # All 11 existing keys must exist
            existing_keys = [
                "query", "query_type", "path", "entropy", "model_used",
                "answer", "sources", "chunks_used", "routing_reason",
                "weather_enrichment_applied", "weather_data"
            ]
            for k in existing_keys:
                self.assertIn(k, out, f"Existing key {k} missing from output")

            # 3 new trust keys
            self.assertEqual(out["trust_score"], 0.88)
            self.assertEqual(out["trust_verdict"], "HIGH EVIDENCE ALIGNMENT ✅")
            self.assertEqual(out["trust_data"], mock_trust_res)
            print("  ✓ Test 7 Passed: Flag ON returns complete schema without breaking existing keys")

    # ── Test 8: Telemetry Isolated and Appends Safely ──
    def test_08_telemetry_isolated_and_appends_safely(self):
        log_path = self.isolated_log_path
        initial_len = 0
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    initial_len = len(data) if isinstance(data, list) else 0
            except Exception:
                initial_len = 0

        mock_agg = {
            "overall_trust_score": 0.85,
            "verdict": "HIGH EVIDENCE ALIGNMENT ✅",
            "support_ratio": 1.0,
            "total_claims": 1,
            "statistical_guarantee": "Semantic evidence alignment: 1/1 claims supported"
        }
        cts.save_trust_log(
            query="test telemetry query",
            answer="test telemetry answer",
            aggregate=mock_agg,
            claim_results=[],
            log_path=log_path
        )

        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            updated_data = json.load(f)

        self.assertEqual(len(updated_data), initial_len + 1)
        last_entry = updated_data[-1]
        self.assertEqual(last_entry["query"], "test telemetry query")
        self.assertEqual(last_entry["overall_trust"], 0.85)
        print(f"  ✓ Test 8 Passed: Telemetry log isolated to temp directory and cleanly appended (count: {initial_len} → {len(updated_data)})")

    # ── Test 9: Telemetry IOError Handled Gracefully ──
    def test_09_telemetry_ioerror_handled_gracefully(self):
        # 1. Direct save_trust_log handles IOError without raising
        with patch("builtins.open", side_effect=IOError("Simulated disk write failure")):
            try:
                cts.save_trust_log(
                    query="test error query",
                    answer="test error answer",
                    aggregate={"overall_trust_score": 0.5, "verdict": "LOW", "support_ratio": 0.5, "total_claims": 2, "statistical_guarantee": ""},
                    claim_results=[],
                    log_path=self.isolated_log_path
                )
            except Exception as e:
                self.fail(f"save_trust_log raised an unhandled exception: {e}")

        # 2. query_gate delivers answer intact even if save_trust_log fails
        config.ENABLE_CONFORMAL_TRUST = True
        with patch("step6_query_gate.classify_query", return_value={"query_type": "DIAGNOSTIC", "confidence": 0.9, "key_topics": ["wheat"], "complexity": "complex", "reasoning": "r"}), \
             patch("step6_query_gate.measure_query_entropy", return_value={"entropy": 2.0, "token_logprobs": [-1.0], "first_tokens": ["t"]}), \
             patch("step6_query_gate.retrieve_chunks", return_value=[{"chunk_id": "c1", "text": "wheat disease text", "source_file": "ICAR.pdf", "final_score": 0.9}]), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": "Ensure proper field drainage and apply organic compost.", "model_used": "test-model", "sources": ["ICAR.pdf"], "path": "slow"}), \
             patch("conformal_trust_scorer.save_trust_log", side_effect=IOError("Simulated disk error")):

            mock_embedder = MagicMock()
            mock_embedder.encode.return_value = np.zeros((1, 384))
            mock_coll = MagicMock()
            mock_bm25 = MagicMock()
            mock_corpus = MagicMock()

            out = s6.query_gate("wheat rust treatment", mock_embedder, mock_coll, mock_bm25, mock_corpus)
            self.assertIn("Ensure proper field drainage", out["answer"])
            print("  ✓ Test 9 Passed: Telemetry IOError handled gracefully without interrupting answer delivery")

    # ── Test 10: Claim Decomposition Fallback Chain ──
    def test_10_claim_decomposition_fallback_chain(self):
        # Case A: Truncated / malformed JSON recovers via quoted strings
        client_malformed = MagicMock()
        client_malformed.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Malathion 50 EC at 100ml per 10 liters.", "Spray the godown walls and floor thoroughly.'))]
        )
        claims_a = cts.decompose_answer_to_claims("dummy answer", client_malformed)
        self.assertGreaterEqual(len(claims_a), 1)
        self.assertIn("Malathion 50 EC", claims_a[0])

        # Case B: Unstructured bullet list without JSON
        client_bullets = MagicMock()
        client_bullets.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='- Apply copper oxychloride at 3g per liter.\n* Ensure complete coverage of the foliage.\n1. Repeat after 14 days if symptoms persist.'))]
        )
        claims_b = cts.decompose_answer_to_claims("dummy answer", client_bullets)
        self.assertEqual(len(claims_b), 3)
        self.assertIn("Apply copper oxychloride", claims_b[0])

        # Case C: Multi-sentence prose without JSON or bullets
        client_prose = MagicMock()
        client_prose.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='Vermicompost is organic fertilizer produced by earthworms. It enriches the soil microbiology. The ideal carbon to nitrogen ratio is 15:1.'))]
        )
        claims_c = cts.decompose_answer_to_claims("dummy answer", client_prose)
        self.assertGreaterEqual(len(claims_c), 2)
        for c in claims_c:
            self.assertGreater(len(c), 12)
            self.assertFalse(c.startswith('-') or c.startswith('*') or c.startswith('|'))
        print("  ✓ Test 10 Passed: Claim decomposition fallback chain handles malformed JSON, bullets, and prose")

    # ── Test 11: Embedding Dimension Mismatch Preserves Answer ──
    def test_11_embedding_dimension_mismatch_preserves_answer(self):
        # Mock embedder returning dimension 384 for claims, but dimension 768 for chunks
        mock_mismatch_embedder = MagicMock()
        def mock_encode(texts, **kwargs):
            if any("Claim" in str(t) or "Vermicompost" in str(t) for t in texts):
                return np.zeros((len(texts), 384))
            else:
                return np.zeros((len(texts), 768))
        mock_mismatch_embedder.encode.side_effect = mock_encode

        res = cts.compute_nonconformity_scores(
            claims=["Vermicompost is organic fertilizer.", "Vermicompost improves soil structure."],
            chunks=[{"text": "Chunk 1 content here."}, {"text": "Chunk 2 content here."}],
            embedder=mock_mismatch_embedder
        )
        self.assertEqual(len(res), 0)

        # Full query_gate execution under dimension mismatch
        config.ENABLE_CONFORMAL_TRUST = True
        with patch("step6_query_gate.classify_query", return_value={"query_type": "DIAGNOSTIC", "confidence": 0.9, "key_topics": ["wheat"], "complexity": "complex", "reasoning": "r"}), \
             patch("step6_query_gate.measure_query_entropy", return_value={"entropy": 2.0, "token_logprobs": [-1.0], "first_tokens": ["t"]}), \
             patch("step6_query_gate.retrieve_chunks", return_value=[{"chunk_id": "c1", "text": "wheat disease text", "source_file": "ICAR.pdf", "final_score": 0.9}]), \
             patch("step6_query_gate.slow_path_answer", return_value={"answer": "Ensure proper field drainage and apply organic compost.", "model_used": "test-model", "sources": ["ICAR.pdf"], "path": "slow"}), \
             patch("conformal_trust_scorer.decompose_answer_to_claims", return_value=["Claim 1 with mismatch.", "Claim 2 with mismatch."]):

            mock_coll = MagicMock()
            mock_bm25 = MagicMock()
            mock_corpus = MagicMock()

            out = s6.query_gate("wheat rust treatment", mock_mismatch_embedder, mock_coll, mock_bm25, mock_corpus)
            self.assertIn("Ensure proper field drainage", out["answer"])
            self.assertEqual(out["trust_verdict"], "UNVERIFIED ⚠️")
            print("  ✓ Test 11 Passed: Embedding dimension mismatch handled safely without losing generated answer")


if __name__ == "__main__":
    unittest.main()
