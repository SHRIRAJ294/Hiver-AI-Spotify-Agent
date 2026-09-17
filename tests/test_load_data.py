import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from load_data import clean_text, load_raw_twcs


class TestLoadData(unittest.TestCase):
    def test_clean_text(self):
        raw_text = "@SpotifyCares my app is crashing! http://t.co/xyz123 help please   "
        cleaned = clean_text(raw_text)
        self.assertNotIn("@SpotifyCares", cleaned)
        self.assertNotIn("http://t.co/xyz123", cleaned)
        self.assertEqual(cleaned, "my app is crashing! help please")

    def test_clean_text_empty_and_non_string(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_load_raw_twcs_missing_file(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_raw_twcs("data/missing_file.csv")
        self.assertIn("thoughtvector/customer-support-on-twitter", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
