"""
evaluation/retrieval/test_sb2_guardrails.py

Unit tests for hardened evaluation-layer SB2 Semantic Bridge Guardrails:
1. Plural and irregular crop inflection detection (tomato/tomatoes, chilli/chillies, paddy/paddies).
2. Crop-name near-matches, substrings, and overlapping terms.
3. Compound pest and pathogen filtering (catalog and structural regex).
4. Agronomic noun protection (preventing false pest filtering on 'rice disease management and tomato fertilizer').
5. Multi-crop balance guardrail (symmetrical Latin binomial suppression on singular and plural crop lists).
6. Mixed English and vernacular dialectal translation preservation.
7. Robustness on empty strings, punctuation, casing, and repeated terms.
8. Preservation of SB1 guardrails (de-duplication, specificity anchoring, vernacular confidence floor).
"""

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from components.context.guardrails import (
    get_detected_crops,
    is_entity_anchored,
    build_deduplicated_replacement,
    detect_folk_terms_guarded,
    enrich_query_guarded,
    apply_semantic_bridge_sb1,
    apply_semantic_bridge_sb2,
    CROP_INFLECTIONS,
    SURFACE_TO_CANONICAL,
    ALL_CROP_SURFACE_FORMS
)


class TestPluralCropDetection(unittest.TestCase):
    """Verifies singular, irregular plural, and spelling-variant crop recognition."""

    def test_irregular_plurals(self):
        cases = [
            ("tomatoes in polyhouse", ["tomato"]),
            ("potatoes storage temperature", ["potato"]),
            ("mangoes export grade", ["mango"]),
            ("chillies nursery management", ["chilli"]),
            ("chilies pest control", ["chilli"]),
            ("water management in paddies", ["paddy"]),
            ("citruses rootstock", ["citrus"])
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                detected = get_detected_crops(query)
                self.assertEqual(detected, expected, f"Failed on query: '{query}'")

    def test_standard_plurals(self):
        cases = [
            ("organic coconuts yield", ["coconut"]),
            ("hybrid brinjals spacing", ["brinjal"]),
            ("high yielding eggplants", ["brinjal"]),
            ("cowpeas seed rate", ["cowpea"]),
            ("desi chickpeas cultivation", ["chickpea"]),
            ("export quality lentils", ["lentil"]),
            ("tissue culture bananas", ["banana"]),
            ("red onions storage", ["onion"]),
            ("grafted cashews spacing", ["cashew"])
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                detected = get_detected_crops(query)
                self.assertEqual(detected, expected, f"Failed on query: '{query}'")

    def test_multi_word_crop_plurals(self):
        cases = [
            ("spacing for pigeon peas", ["pigeon pea"]),
            ("intercropping black grams with maize", ["black gram", "maize"]),
            ("green grams fertilizer dosage", ["green gram"]),
            ("improved pearl millets varieties", ["pearl millet"]),
            ("finger millets nutrition facts", ["finger millet"])
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                detected = get_detected_crops(query)
                self.assertEqual(detected, expected, f"Failed on query: '{query}'")

    def test_uncountable_crops(self):
        cases = [
            ("wheat rust management", ["wheat"]),  # Wheat is the crop being managed
            ("wheat sowing date", ["wheat"]),
            ("paddy transplantation", ["paddy"]),
            ("sugarcane ratoon management", ["sugarcane"]),
            ("cotton defoliation", ["cotton"]),
            ("coffee shade trees", ["coffee"]),
            ("sesame seed rate", ["sesame"]),
            ("mustard oil percentage", ["mustard"]),
            ("tea pruning cycle", ["tea"])
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                detected = get_detected_crops(query)
                self.assertEqual(detected, expected, f"Failed on query: '{query}'")


class TestNearMatchesAndOverlaps(unittest.TestCase):
    """Verifies that near-matches and sub-word strings do not trigger false positives."""

    def test_near_matches_do_not_trigger(self):
        # Queries containing words that share prefixes/suffixes with crop names
        query = "team work with steam tractor near cottage price corner"
        detected = get_detected_crops(query)
        self.assertEqual(detected, [], f"Expected empty crop list for near-matches, got: {detected}")

    def test_overlapping_multiword_crops(self):
        # "black gram" should not falsely trigger "gram" (chickpea) as a second crop
        query = "black gram and green gram intercropping"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["black gram", "green gram"])
        self.assertEqual(len(detected), 2)

    def test_synonym_orthographic_variants(self):
        # "chilli" and "chili" are orthographic variants of the same crop
        query = "chilli and chili varieties comparison"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["chilli"])
        self.assertEqual(len(detected), 1, "Orthographic synonyms of the same crop must canonicalize to 1 entity")

    def test_repeated_terms_do_not_inflate_count(self):
        # Multiple mentions of the same crop must not trigger multi-crop balance
        query = "tomato nursery vs tomato open field transplanting"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["tomato"])
        self.assertEqual(len(detected), 1)


class TestCompoundPestFiltering(unittest.TestCase):
    """Verifies that crop names inside compound pest/pathogen names are not counted as independent crop anchors."""

    def test_tea_mosquito_bug_in_cashew(self):
        # Q21 canonical benchmark query: Cashew is the crop, Tea is inside the pest name
        query = "What are the major damage symptoms and botanical oil control options for Tea Mosquito Bug in cashew?"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["cashew"])
        self.assertEqual(len(detected), 1, "Tea Mosquito Bug must not count 'tea' as a crop anchor")

    def test_mango_fruit_fly_in_cashew(self):
        query = "How to control mango fruit fly in cashew orchard?"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["cashew"])
        self.assertEqual(len(detected), 1, "Mango fruit fly must not count 'mango' as a crop anchor")

    def test_sugarcane_top_borer_in_maize(self):
        query = "sugarcane top borer damage in maize fields"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["maize"])
        self.assertEqual(len(detected), 1, "Sugarcane top borer must not count 'sugarcane' as a crop anchor")

    def test_rice_blast_in_finger_millet(self):
        query = "rice blast symptoms on finger millet"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["finger millet"])
        self.assertEqual(len(detected), 1, "Rice blast must not count 'rice' as a crop anchor")

    def test_coffee_berry_borer_in_cardamom(self):
        query = "How to control coffee berry borer in cardamom plantations?"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["cardamom"])
        self.assertEqual(len(detected), 1)

    def test_compound_pest_plus_independent_host_mention(self):
        # When a crop is in a compound pest AND mentioned independently as a host
        query = "tea mosquito bug damage in tea and cashew"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["tea", "cashew"])
        self.assertEqual(len(detected), 2, "Independent mention of 'tea' must be detected alongside 'cashew'")

    def test_agronomic_noun_protection(self):
        # Critical protection: Broad agronomic nouns MUST NOT suppress genuine crops
        query = "rice disease management and tomato fertilizer"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["rice", "tomato"])
        self.assertEqual(len(detected), 2, "Broad agronomic nouns must not trigger pest filtering")

    def test_inverted_crop_mention(self):
        # Prepositional structure: pest before crop
        query = "fruit fly in mango and citrus"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["mango", "citrus"])
        self.assertEqual(len(detected), 2)


class TestMultiCropBalanceEnrichment(unittest.TestCase):
    """Verifies that SB2 suppresses asymmetric botanical injection on multi-crop lists."""

    def test_q13_singular_seed_rates(self):
        query = "What are the recommended seed rate ranges (in g/ha or kg/ha) for tomato, chilli, and brinjal?"
        res = apply_semantic_bridge_sb2(query)
        # Symmetrical protection: tomato should NOT be injected with Solanum lycopersicum
        self.assertNotIn("Solanum lycopersicum", res["enriched"])
        self.assertEqual(res["enriched"], query)

    def test_q13_plural_seed_rates(self):
        query = "What are the recommended seed rate ranges (in g/ha or kg/ha) for tomatoes, chillies, and brinjals?"
        res = apply_semantic_bridge_sb2(query)
        # Hardened plural protection: tomatoes should NOT be injected with Solanum lycopersicum
        self.assertNotIn("Solanum lycopersicum", res["enriched"])
        self.assertEqual(res["enriched"], query)

    def test_single_crop_vernacular_enrichment(self):
        query = "tamatar mein safed makhi ka ilaj"
        res = apply_semantic_bridge_sb2(query)
        self.assertTrue(res["bridged"])
        self.assertIn("Solanum lycopersicum", res["enriched"])
        self.assertIn("Bemisia tabaci", res["enriched"])


class TestMixedEnglishAndVernacular(unittest.TestCase):
    """Verifies preservation of dialectal translation for mixed queries."""

    def test_dhan_zinc_deficiency(self):
        query = "dhan mein khaira rog ka lakshan aur upchar kya hai?"
        res = apply_semantic_bridge_sb2(query)
        self.assertTrue(res["bridged"])
        self.assertIn("Oryza sativa", res["enriched"])

    def test_arhar_pod_borer(self):
        query = "arhar ki fasal mein sundli ke niyantran ke liye dawai"
        res = apply_semantic_bridge_sb2(query)
        self.assertTrue(res["bridged"])
        self.assertIn("Cajanus cajan", res["enriched"])


class TestRobustnessAndEdgeCases(unittest.TestCase):
    """Verifies handling of empty queries, punctuation, casing, and whitespace."""

    def test_empty_string(self):
        res = apply_semantic_bridge_sb2("")
        self.assertFalse(res["bridged"])
        self.assertEqual(res["enriched"], "")
        self.assertEqual(res["term_count"], 0)

    def test_whitespace_only(self):
        res = apply_semantic_bridge_sb2("   \t  \n  ")
        self.assertFalse(res["bridged"])
        self.assertEqual(res["term_count"], 0)

    def test_punctuation_only(self):
        res = apply_semantic_bridge_sb2("... ,,, ;;; ??? !!!")
        self.assertFalse(res["bridged"])

    def test_uppercase_casing(self):
        query = "WHAT ARE THE SEED RATES FOR TOMATOES, CHILLIES, AND BRINJALS?"
        detected = get_detected_crops(query)
        self.assertEqual(detected, ["tomato", "chilli", "brinjal"])
        res = apply_semantic_bridge_sb2(query)
        self.assertNotIn("Solanum lycopersicum", res["enriched"])


class TestSB1GuardrailsPreservation(unittest.TestCase):
    """Verifies that core SB1 guardrails remain fully intact."""

    def test_synonym_deduplication_tautology(self):
        # Tautological replacement returning None
        rep = build_deduplicated_replacement("Green Manure", "Green Manure", "Green Manure")
        self.assertIsNone(rep)

    def test_synonym_clean_replacement(self):
        # Clean parenthetical addition
        rep = build_deduplicated_replacement("coconut", "Coconut", "Cocos nucifera")
        self.assertEqual(rep, "coconut (Cocos nucifera)")

    def test_specificity_guardrail_q28(self):
        # Q28 query: "organic manure" should NOT be narrowed to FYM when anchored by coconut + coir pith
        query = "What are the integrated organic manure and coir pith application practices for coconut basins to conserve moisture?"
        res = apply_semantic_bridge_sb2(query)
        self.assertNotIn("Farmyard Manure", res["enriched"])
        self.assertNotIn("FYM", res["enriched"])

    def test_vernacular_confidence_floor(self):
        # English collective term "chemical fertilizer" should not be fuzzed into a Hindi alias
        query = "chemical fertilizer recommendations for wheat"
        res = apply_semantic_bridge_sb2(query)
        self.assertNotIn("Gobar", res["enriched"])


class TestWordBoundaryReplacement(unittest.TestCase):
    """Verifies that word boundaries prevent substring corruption of unrelated host words."""

    def test_host_words_remain_intact(self):
        cases = [
            ("corn yield in corner plot", "corner", "corn (Maize / Corn, Zea mays)", "corner plot"),
            ("urea application through agricultural bureau", "bureau", "urea (Urea (CO(NH2)2) — 46% N)", "agricultural bureau"),
            ("chana crop in archana agricultural farm", "archana", "chana (Chickpea, Cicer arietinum)", "archana agricultural farm"),
            ("dhan production under vidhan sabha scheme", "vidhan", "dhan (Rice / Paddy, Oryza sativa)", "vidhan sabha scheme"),
        ]
        for query, host_word, expected_enrichment, expected_phrase in cases:
            with self.subTest(query=query):
                res = apply_semantic_bridge_sb2(query)
                self.assertTrue(res["bridged"])
                self.assertIn(expected_enrichment, res["enriched"])
                self.assertIn(expected_phrase, res["enriched"])
                self.assertIn(host_word, res["enriched"])

    def test_exact_standalone_matches(self):
        cases = [
            ("corn", "corn (Maize / Corn, Zea mays)"),
            ("urea", "urea (Urea (CO(NH2)2) — 46% N)"),
            ("chana", "chana (Chickpea, Cicer arietinum)"),
            ("dhan", "dhan (Rice / Paddy, Oryza sativa)"),
        ]
        for query, expected_enriched in cases:
            with self.subTest(query=query):
                res = apply_semantic_bridge_sb2(query)
                self.assertTrue(res["bridged"])
                self.assertEqual(res["enriched"], expected_enriched)

    def test_case_insensitive_matching(self):
        cases = [
            ("CORN yield in CORNER plot", "corn (Maize / Corn, Zea mays) yield in CORNER plot"),
            ("UREA in BUREAU", "urea (Urea (CO(NH2)2) — 46% N) in BUREAU"),
            ("CHANA in ARCHANA", "chana (Chickpea, Cicer arietinum) in ARCHANA"),
            ("DHAN in VIDHAN", "dhan (Rice / Paddy, Oryza sativa) in VIDHAN"),
        ]
        for query, expected_enriched in cases:
            with self.subTest(query=query):
                res = apply_semantic_bridge_sb2(query)
                self.assertTrue(res["bridged"])
                self.assertEqual(res["enriched"], expected_enriched)

    def test_multiple_occurrences(self):
        query = "corn and corn seeds in corner plot with corn stalks"
        res = apply_semantic_bridge_sb2(query)
        self.assertTrue(res["bridged"])
        self.assertIn("corner plot", res["enriched"])
        count_corn_enriched = res["enriched"].count("corn (Maize / Corn, Zea mays)")
        self.assertEqual(count_corn_enriched, 3)

    def test_multi_word_terms(self):
        # "safed makhi" (Whitefly)
        query = "safed makhi control in corner plot"
        res = apply_semantic_bridge_sb2(query)
        self.assertTrue(res["bridged"])
        self.assertIn("safed makhi (Whitefly, Bemisia tabaci)", res["enriched"])
        self.assertIn("corner plot", res["enriched"])

        # Word boundary prevents partial multi-word substring match
        query_unrelated = "asafed makhia pest"
        res_unrelated = apply_semantic_bridge_sb2(query_unrelated)
        self.assertNotIn("Bemisia tabaci", res_unrelated["enriched"])


class TestMissingCropAliasesRegression(unittest.TestCase):
    """Regression tests for BUG 1: Missing English crop aliases in CROP_INFLECTIONS."""

    def test_multicrop_detection_missing_aliases(self):
        cases = [
            ("mung bean and tomato seed rate", ["mung bean", "tomato"]),
            ("canola and mustard yield", ["canola", "mustard"]),
            ("black lentil and red lentil spacing", ["black lentil", "red lentil"]),
            ("red gram and chickpea varieties", ["red gram", "chickpea"]),
            ("spiked millet and finger millet comparison", ["spiked millet", "finger millet"]),
            ("canola and tomato spacing", ["canola", "tomato"]),
        ]
        for query, expected_crops in cases:
            with self.subTest(query=query):
                detected = get_detected_crops(query)
                self.assertEqual(detected, expected_crops, f"Failed detecting crops in: '{query}'")

    def test_comparative_balance_missing_aliases(self):
        cases = [
            "mung bean and tomato seed rate",
            "canola and mustard yield",
            "black lentil and red lentil spacing",
            "red gram and chickpea varieties",
            "spiked millet and finger millet comparison",
            "canola and tomato spacing",
        ]
        for query in cases:
            with self.subTest(query=query):
                res = apply_semantic_bridge_sb2(query)
                # Symmetrical balance: English crops must NOT be injected with Latin binomials
                self.assertEqual(res["enriched"], query, f"Multi-crop query was unexpectedly altered: '{res['enriched']}'")
                self.assertNotIn("Solanum lycopersicum", res["enriched"])
                self.assertNotIn("Brassica", res["enriched"])
                self.assertNotIn("Vigna", res["enriched"])
                self.assertNotIn("Lens culinaris", res["enriched"])

    def test_single_crop_enrichment_missing_aliases(self):
        cases = [
            ("mung bean cultivation", "Vigna radiata"),
            ("canola cultivation", "Brassica napus"),
            ("black lentil cultivation", "Vigna mungo"),
            ("red lentil cultivation", "Lens culinaris"),
        ]
        for query, expected_sci in cases:
            with self.subTest(query=query):
                res = apply_semantic_bridge_sb2(query)
                self.assertTrue(res["bridged"])
                self.assertIn(expected_sci, res["enriched"])


class TestNestedReplacementRegression(unittest.TestCase):
    """Regression tests for BUG 2: Nested/duplicate replacement in enrich_query_guarded."""

    def test_black_gram_no_nested_chickpea_corruption(self):
        query = "black gram cultivation and fertilizer"
        res = apply_semantic_bridge_sb2(query)
        self.assertTrue(res["bridged"])
        self.assertIn("black gram (Vigna mungo)", res["enriched"])
        self.assertNotIn("Chickpea", res["enriched"])
        self.assertNotIn("Cicer arietinum", res["enriched"])

    def test_overlapping_and_standalone_tokens(self):
        query = "black gram and gram cultivation"
        terms = [
            {"original_term": "gram", "scientific": "Cicer arietinum", "english": "Chickpea"},
            {"original_term": "black gram", "scientific": "Vigna mungo", "english": "Black Gram"},
        ]
        enriched = enrich_query_guarded(query, terms)
        self.assertIn("black gram (Vigna mungo)", enriched)
        self.assertIn("gram (Chickpea, Cicer arietinum)", enriched)

    def test_generated_replacement_not_reprocessed(self):
        # Even if a replacement string introduces a word that matches another term,
        # it must never be reprocessed
        query = "special crop"
        terms = [
            {"original_term": "special crop", "scientific": "Custom sci", "english": "urea"},
            {"original_term": "urea", "scientific": "CO(NH2)2", "english": "Urea"},
        ]
        enriched = enrich_query_guarded(query, terms)
        self.assertIn("special crop (urea, Custom sci)", enriched)
        self.assertNotIn("CO(NH2)2", enriched)


if __name__ == "__main__":
    unittest.main(verbosity=2)
