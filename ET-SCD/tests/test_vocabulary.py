import unittest
from etscd.vocabulary import build_object_map


class Tokenizer:
    all_special_ids = [0]
    words = {"cat": 1, " cat": 2, "dog": 3, " dog": 4, "bird": 0}

    def encode(self, text, add_special_tokens=False):
        return [self.words[text]] if text in self.words else [7, 8]

    def decode(self, ids, skip_special_tokens=False):
        return {0: "bird", 1: "cat", 2: " cat", 3: "dog", 4: " dog"}.get(ids[0], "?")


class VocabularyTests(unittest.TestCase):
    def test_single_token_aliases_and_skipped_multiword_are_explicit(self):
        tokens, skipped = build_object_map(Tokenizer(), {"cat": ["cat"], "dog": ["dog"],
                                                        "sports ball": ["sports ball"], "bird": ["bird"]})
        self.assertEqual(tokens, {1: "cat", 2: "cat", 3: "dog", 4: "dog"})
        self.assertEqual(skipped, ["bird", "sports ball"])

    def test_empty_result_fails(self):
        with self.assertRaises(ValueError):
            build_object_map(Tokenizer(), {"sports ball": ["sports ball"]})

    def test_concept_collision_fails(self):
        with self.assertRaises(ValueError):
            build_object_map(Tokenizer(), {"cat": ["cat"], "another concept": ["cat"]})


if __name__ == "__main__":
    unittest.main()
