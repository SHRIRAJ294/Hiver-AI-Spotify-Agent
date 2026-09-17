import sys
import os
import unittest

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from retriever import retrieve_similar_cases


class TestRetriever(unittest.TestCase):
    def test_retriever_ranking_and_bounds(self):
        query = "My Spotify desktop app keeps crashing when playing songs"
        results, best_sim = retrieve_similar_cases(query, top_k=5)

        self.assertEqual(len(results), 5)
        self.assertIn("customer_message", results.columns)
        self.assertIn("support_response", results.columns)
        self.assertIn("similarity", results.columns)

        # Verify best similarity score bounds
        self.assertGreaterEqual(best_sim, -1.0)
        self.assertLessEqual(best_sim, 1.0)

        # Verify results are sorted descending by similarity
        sims = results["similarity"].tolist()
        self.assertEqual(sims, sorted(sims, reverse=True))


if __name__ == "__main__":
    unittest.main()
