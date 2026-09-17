import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from decision import decide_action


class TestDecision(unittest.TestCase):
    def test_decide_action_auto_handle(self):
        evidence = {"sufficient": True, "reason": "Historical cases directly address the crash issue."}
        decision = decide_action(evidence)
        self.assertEqual(decision["decision"], "auto_handle")
        self.assertEqual(decision["reason"], evidence["reason"])

    def test_decide_action_escalate(self):
        evidence = {"sufficient": False, "reason": "Insufficient historical evidence for user query."}
        decision = decide_action(evidence)
        self.assertEqual(decision["decision"], "escalate")
        self.assertEqual(decision["reason"], evidence["reason"])


if __name__ == "__main__":
    unittest.main()
