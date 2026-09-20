import sys
import os
import unittest
from pathlib import Path

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
from step6_query_gate import (
    QueryGate,
    query_gate,
    get_or_load_components,
    classify_query,
    measure_query_entropy,
)


class TestBackendQueryPipeline(unittest.TestCase):
    """
    Focused unit tests for backend fault resolutions:
    1. QueryGate class import and instantiation
    2. Lazy component loading and caching
    3. Return dictionary schema including source_chunks
    4. Guardrail preservation (ENABLE_CONFORMAL_TRUST = False)
    5. Safe non-interactive feedback execution
    """

    @classmethod
    def setUpClass(cls):
        os.environ["NO_INTERACTIVE_FEEDBACK"] = "1"
        cls.components = get_or_load_components()

    def test_01_querygate_class_available(self):
        qg = QueryGate(
            embedder=self.components[0],
            collection=self.components[1],
            bm25=self.components[2],
            corpus=self.components[3]
        )
        self.assertIsNotNone(qg)
        self.assertTrue(hasattr(qg, "process_query"))

    def test_02_querygate_lazy_initialization(self):
        # Initializing without arguments should successfully retrieve cached components
        qg = QueryGate()
        self.assertIsNotNone(qg.embedder)
        self.assertIsNotNone(qg.collection)
        self.assertIsNotNone(qg.bm25)
        self.assertIsNotNone(qg.corpus)

    def test_03_query_classification(self):
        result = classify_query("What is vermicompost?")
        self.assertIn("query_type", result)
        self.assertIn(result["query_type"], ["FACTUAL", "DIAGNOSTIC", "RECOMMENDATION", "PROCEDURAL"])

    def test_04_query_entropy_measurement(self):
        entropy_data = measure_query_entropy("What is NPK?")
        self.assertIn("entropy", entropy_data)
        self.assertIsInstance(entropy_data["entropy"], (int, float))

    def test_05_return_schema_contains_source_chunks(self):
        # Test fast path query
        qg = QueryGate(*self.components)
        result = qg.process_query("What does NPK stand for?")

        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("source_chunks", result)
        self.assertIn("path", result)
        self.assertIn("query_type", result)
        self.assertIsInstance(result["source_chunks"], list)
        self.assertTrue(len(result["answer"]) > 0)

    def test_06_conformal_trust_remains_guarded(self):
        self.assertFalse(
            getattr(config, "ENABLE_CONFORMAL_TRUST", False),
            "ENABLE_CONFORMAL_TRUST must remain False"
        )


if __name__ == "__main__":
    unittest.main()
