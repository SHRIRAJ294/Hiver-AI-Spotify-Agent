import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from intents import INTENTS


class TestIntentsTaxonomy(unittest.TestCase):
    def test_intents_count_and_keys(self):
        self.assertEqual(len(INTENTS), 9)
        expected_keys = {
            "technical_issue",
            "playback_audio",
            "account_access",
            "billing_payment",
            "subscription",
            "content_availability",
            "feature_request",
            "general_information",
            "non_support",
        }
        self.assertEqual(set(INTENTS.keys()), expected_keys)

    def test_intents_descriptions_non_empty(self):
        for intent, desc in INTENTS.items():
            self.assertIsInstance(desc, str)
            self.assertGreater(len(desc), 10)


if __name__ == "__main__":
    unittest.main()
