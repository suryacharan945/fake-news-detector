import unittest
import os
import sys
import ast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class TestFakeNewsPipeline(unittest.TestCase):

    def test_python_source_syntax_validity(self):
        for script in ["app.py", "inference.py", "train.py"]:
            path = os.path.join(PROJECT_ROOT, script)
            self.assertTrue(os.path.exists(path), f"Missing script: {script}")
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            # Verify AST parses without SyntaxError
            parsed = ast.parse(source, filename=script)
            self.assertIsNotNone(parsed)

    def test_app_configuration_constants(self):
        app_path = os.path.join(PROJECT_ROOT, "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn('REPO_ID = "suryacharan945/fake-news-detector"', content)
        self.assertIn('MAX_LEN = 256', content)
        self.assertIn('LABELS  = ["FAKE", "REAL"]', content)

    def test_softmax_probability_math(self):
        import math
        logits = [2.0, -1.0]
        max_l = max(logits)
        exp_vals = [math.exp(x - max_l) for x in logits]
        sum_exp = sum(exp_vals)
        probs = [x / sum_exp for x in exp_vals]
        
        self.assertEqual(len(probs), 2)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertGreater(probs[0], probs[1])

if __name__ == "__main__":
    unittest.main()
