import unittest
import re
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = WORKSPACE / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from step6_query_gate import (
    reconcile_chemical_recommendations,
    _scrub_unverified_agronomic_claims,
    _get_banned_pesticide_directive,
)


class TestAnswersQualityFixes(unittest.TestCase):
    def test_no_recursive_iron_chelate_nesting(self):
        """Repeated passes of _scrub_unverified_agronomic_claims must NOT nest recursively."""
        text = (
            "• Apply iron chelate (foliar) if chlorosis observed.\n"
            "• If leaves show iron deficiency, spray iron chelate (≈ 0.1 % solution)."
        )
        pass1 = _scrub_unverified_agronomic_claims(text, chunks=[])
        pass2 = _scrub_unverified_agronomic_claims(pass1, chunks=[])
        pass3 = _scrub_unverified_agronomic_claims(pass2, chunks=[])

        # Must not contain nested 'consult local KVK' or repeated 'formulations are not documented'
        self.assertNotIn("specific consult local KVK", pass3)
        self.assertNotIn("formulations are not documented in retrieved context) formulations", pass3)
        # Should be completely identical between pass1 and pass3 (idempotent)
        self.assertEqual(pass1, pass3)

    def test_banned_monocrotophos_unequivocal_prohibition(self):
        """Query asking 'Can I spray monocrotophos for aphids?' must produce strict prohibition, not 'Yes, [chemical withheld]'."""
        query = "Can I spray monocrotophos for aphids?"
        raw_affirmative_answer = (
            "Yes, monocrotophos can be used to control aphids. "
            "Apply 1.5 ml per litre of water (or 750 ml per 500 l of water) as a foliar spray."
        )
        sanitized = reconcile_chemical_recommendations(raw_affirmative_answer, is_offline_reg=True, chunks=[], query=query)

        self.assertNotIn("Yes", sanitized)
        self.assertNotIn("[chemical withheld] can be used", sanitized)
        self.assertIn("No. Monocrotophos is strictly prohibited and banned", sanitized)
        self.assertIn("CIB&RC", sanitized)

    def test_banned_directive_injection(self):
        """_get_banned_pesticide_directive detects banned chemicals in user queries."""
        directive = _get_banned_pesticide_directive("Can I spray monocrotophos for aphids?")
        self.assertIn("CRITICAL STATUTORY SAFETY MANDATE", directive)
        self.assertIn("monocrotophos", directive.lower())
        self.assertIn("CIB&RC", directive)

    def test_table_chemical_withheld_clean_tag(self):
        """Tables with synthetic chemicals show clean UNVERIFIED / BLOCKED tag without broken brackets."""
        table_text = (
            "| Step | Action | Practical Details |\n"
            "|---|---|---|\n"
            "| 3. Seed rate | Apply chlorpyrifos 2 ml/kg seed | Consult local KVK |\n"
        )
        sanitized = reconcile_chemical_recommendations(table_text, is_offline_reg=True, chunks=[])
        self.assertIn("UNVERIFIED / BLOCKED", sanitized)
        self.assertNotIn("Apply chlorpyrifos", sanitized)


if __name__ == "__main__":
    unittest.main()
