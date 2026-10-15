"""Runner, generation-prefix and benchmark input protocol tests."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from etscd.adapters import BinaryTokenPair
from etscd.config import ETConfig
from etscd.data import CHAIR_PROMPT
from etscd.runner import resolve_image, run_sample, validate_questions


class FakeBackend:
    """Record exact input contexts and return a three-step synthetic script."""

    eos_ids = {2}
    visual_input_size = (12, 10)

    def __init__(self):
        self.calls = []
        self.syncs = 0
        self.decoded = []

    def next_logits(self, image, question, prefix):
        branch = image.info["branch"]
        self.calls.append((branch, question, tuple(prefix)))
        if len(prefix) == 0:
            # At an activated first step ET-SCD must Defer, while ordinary
            # greedy and VCD select token0 under the fixed token-ID tie rule.
            return np.array([1.0, 0.9, -10.0]) if branch == "focused" else np.array([1.0, 1.0, -10.0])
        if len(prefix) == 1:
            return np.array([-4.0, 0.0, -10.0]) if branch == "weakened" else np.array([4.0, 0.0, -10.0])
        return np.array([-5.0, -5.0, 10.0])

    def decode(self, token_ids):
        self.decoded.append(list(token_ids))
        return "decoded:" + ",".join(str(i) for i in token_ids)

    def synchronize(self):
        self.syncs += 1

    def memory_bytes(self):
        return None


def branch_image(name):
    image = Image.new("RGB", (12, 10), "gray")
    image.info["branch"] = name
    return image


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.original = branch_image("original")
        self.weakened = branch_image("weakened")
        self.query_targets = []

    def views(self, image, targets, config):
        self.assertIs(image, self.original)
        self.query_targets.append(tuple(targets))
        yield "focused", branch_image("focused")
        yield "unchanged", branch_image("unchanged")

    def test_methods_have_lazy_auditable_branch_counts_across_steps(self):
        expected = {
            "regular": ([0, 0, 2], {"original": 3, "weakened": 0, "requery": 0}),
            "vcd": ([0, 0, 2], {"original": 3, "weakened": 3, "requery": 0}),
            "etscd": ([1, 0, 2], {"original": 3, "weakened": 3, "requery": 2}),
        }
        for method, (tokens, counts) in expected.items():
            with self.subTest(method=method):
                backend = FakeBackend()
                self.query_targets.clear()
                with patch("etscd.runner.corner_views", self.views):
                    result = run_sample(
                        backend, self.original, self.weakened, CHAIR_PROMPT,
                        benchmark="chair", object_map={0: "dog", 1: "cat"},
                        max_new_tokens=64, method=method,
                    )
                self.assertEqual(result["token_ids"], tokens)
                self.assertEqual(result["efficiency"]["forward_calls"], counts)
                self.assertEqual(result["efficiency"]["total_forward_calls"], sum(counts.values()))
                self.assertEqual(len(self.query_targets), 1 if method == "etscd" else 0)
                self.assertEqual(backend.syncs, 2)
                self.assertEqual(backend.decoded, [tokens])
                self.assertIsNone(result["efficiency"]["peak_allocated_cuda_bytes"])
                self.assertIsNone(result["efficiency"]["flops"])

    def test_every_view_uses_same_question_and_exact_committed_prefix(self):
        backend = FakeBackend()
        with patch("etscd.runner.corner_views", self.views):
            result = run_sample(
                backend, self.original, self.weakened, CHAIR_PROMPT,
                benchmark="chair", object_map={0: "dog", 1: "cat"},
            )
        self.assertEqual(result["token_ids"], [1, 0, 2])
        self.assertEqual(backend.calls, [
            ("original", CHAIR_PROMPT, ()),
            ("weakened", CHAIR_PROMPT, ()),
            ("focused", CHAIR_PROMPT, ()),
            ("unchanged", CHAIR_PROMPT, ()),
            ("original", CHAIR_PROMPT, (1,)),
            ("weakened", CHAIR_PROMPT, (1,)),
            ("original", CHAIR_PROMPT, (1, 0)),
            ("weakened", CHAIR_PROMPT, (1, 0)),
        ])
        self.assertEqual(result["steps"][0]["route"][0]["action"], "Defer")
        self.assertEqual(result["steps"][1]["reason"], "accept")

    def test_eos_stops_generation_before_limit(self):
        backend = FakeBackend()
        result = run_sample(
            backend, self.original, self.weakened, CHAIR_PROMPT,
            benchmark="chair", method="regular", max_new_tokens=64,
        )
        self.assertEqual(result["token_ids"], [0, 0, 2])
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(result["efficiency"]["generated_token_count_including_eos"], 3)

    def test_binary_query_retains_full_attribute_proposition(self):
        backend = FakeBackend()
        question = "Is the color of the dog black? Please answer yes or no."
        pair = BinaryTokenPair(0, 1, "Yes", "No")
        with patch("etscd.runner.corner_views", self.views):
            result = run_sample(
                backend, self.original, self.weakened, question,
                benchmark="mme", binary_pair=pair,
            )
        self.assertEqual(result["text"], "No")
        self.assertEqual(result["token_ids"], [1])
        self.assertEqual(self.query_targets, [(question,)])
        self.assertTrue(all(q == question and prefix == () for _, q, prefix in backend.calls))

    def test_binary_pair_is_required_before_any_forward(self):
        backend = FakeBackend()
        with self.assertRaises(ValueError):
            run_sample(backend, self.original, self.weakened, "Is a dog present?",
                       benchmark="pope", binary_pair=None)
        self.assertEqual(backend.calls, [])


class InferenceInputTests(unittest.TestCase):
    @staticmethod
    def row(**extra):
        return {"id": "1", "image": "image.jpg", "question": "Is a dog present?", **extra}

    def test_input_requires_the_three_field_schema(self):
        for field in ("label", "answer", "ground_truth", "category", "metadata"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_questions([self.row(**{field: "yes"})], "pope")

    def test_questions_require_unique_ids_and_string_fields(self):
        with self.assertRaises(ValueError):
            validate_questions([self.row(), self.row()], "pope")
        for field, value in (("id", 1), ("image", ""), ("question", None)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_questions([self.row(**{field: value})], "pope")

    def test_chair_requires_exact_prompt_even_for_diagnostic_subset(self):
        for question in ("Describe this photo in detail.", CHAIR_PROMPT + " ", ""):
            with self.subTest(question=question), self.assertRaises(ValueError):
                validate_questions([self.row(question=question)], "chair", allow_subset=True)
        validate_questions([self.row(question=CHAIR_PROMPT)], "chair", allow_subset=True)

    def test_chair_full_protocol_requires_500_distinct_images(self):
        rows = [{"id": str(i), "image": f"{i}.jpg", "question": CHAIR_PROMPT} for i in range(500)]
        validate_questions(rows, "chair")
        with self.assertRaises(ValueError):
            validate_questions(rows[:-1], "chair")
        rows[-1]["image"] = rows[0]["image"]
        with self.assertRaises(ValueError):
            validate_questions(rows, "chair")

    def test_image_paths_must_stay_in_declared_root(self):
        root = Path(__file__).resolve().parent.parent
        self.assertEqual(resolve_image(root, "configs/etscd_tmm.json"), root / "configs" / "etscd_tmm.json")
        with self.assertRaises(ValueError):
            resolve_image(root, "../outside.jpg")

    def test_shipped_default_configuration_exactly_matches_dataclass(self):
        path = Path(__file__).resolve().parent.parent / "configs" / "etscd_tmm.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(values, ETConfig().to_dict())
        self.assertEqual(ETConfig.from_dict(values), ETConfig())


if __name__ == "__main__":
    unittest.main()
