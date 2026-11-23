import unittest
from unittest.mock import Mock
from run_experiments import CONDITIONS, find_source_tokens, make_prompt, relative_preference

class Tests(unittest.TestCase):
    def test_conditions(self):
        self.assertEqual(CONDITIONS, ("A", "B", "AB", "BA"))

    def test_positions(self):
        offsets = [(0, 2), (2, 5), (5, 9), (9, 12)]
        self.assertEqual(find_source_tokens(offsets, (4, 10)), [1, 2, 3])

    def test_order_and_spans(self):
        tk = Mock()
        tk.apply_chat_template.side_effect = lambda msgs, **kwargs: msgs[1]["content"]
        prompt, spans = make_prompt(tk, "Q?", "first", "second", "BA")
        self.assertLess(spans["B"][0], spans["A"][0])
        self.assertEqual(prompt[slice(*spans["A"])], "first")
        self.assertEqual(prompt[slice(*spans["B"])], "second")

    def test_preference_symmetry(self):
        a = {"avg_token_logprob": -1.0}
        b = {"avg_token_logprob": -1.0}
        self.assertAlmostEqual(relative_preference(a, b)["normalized_pair_score_A"], .5)
        self.assertGreater(relative_preference(a, {"avg_token_logprob": -2})["normalized_pair_score_A"], .5)

if __name__ == "__main__":
    unittest.main()
