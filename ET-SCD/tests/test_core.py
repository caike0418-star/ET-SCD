"""Numerical and control-flow tests for ET-SCD using synthetic logits."""

import math
import unittest
from dataclasses import replace

import numpy as np

from etscd.config import ETConfig
from etscd.core import (
    Evidence,
    boundary_set,
    candidate_set,
    choose_action,
    decode_step,
    evidence,
    normalized_entropy,
    risks,
    softmax,
)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config = ETConfig()

    def test_softmax_is_stable_and_shift_invariant(self):
        expected = np.array([1.0, math.e]) / (1.0 + math.e)
        np.testing.assert_allclose(softmax([1000.0, 1001.0]), expected)
        np.testing.assert_allclose(softmax([-1000.0, -999.0]), expected)

    def test_entropy_is_normalized_over_original_vocabulary(self):
        self.assertAlmostEqual(normalized_entropy(np.zeros(100)), 1.0)
        self.assertLess(normalized_entropy([100.0, 0.0, 0.0]), 1e-20)

    def test_candidate_cap_can_truncate_probability_target(self):
        # Uniform entropy gives m=.95 and k=50. Only .25 mass is retained.
        selected = candidate_set(np.zeros(200), self.config)
        self.assertEqual(selected, list(range(50)))
        self.assertLess(len(selected) / 200, 0.95)

    def test_candidate_minimum_cap_is_not_minimum_set_size(self):
        self.assertEqual(candidate_set([100.0, 0.0, 0.0], self.config), [0])

    def test_relative_support_intervals_match_equations(self):
        result = evidence(np.log([0.8, 0.2]), 0.4, self.config)
        np.testing.assert_allclose(result.support, [0.8, 0.2])
        # W=.05*.4+.8*.8*.2=.148 for both candidates.
        np.testing.assert_allclose(result.width, [0.148, 0.148])
        np.testing.assert_allclose(result.lower, [0.652, 0.052])
        np.testing.assert_allclose(result.upper, [0.948, 0.348])

    def test_endpoint_risks_and_deterministic_action_ties(self):
        values = risks(0.3, 0.7, self.config)
        self.assertAlmostEqual(values["Accept"], 0.7)
        self.assertAlmostEqual(values["Reject"], 0.7)
        self.assertAlmostEqual(values["Defer"], 0.35)
        self.assertEqual(choose_action(0.3, 0.7, self.config), "Defer")
        # Accept > Reject > Defer at equal minimum risk.
        self.assertEqual(choose_action(0.65, 0.8, self.config), "Accept")
        self.assertEqual(choose_action(0.1, 0.35, self.config), "Reject")

    def test_asymmetric_defer_cost_uses_worst_endpoint(self):
        increasing = replace(self.config, lambda_d0=0.1, lambda_d1=0.6)
        decreasing = replace(self.config, lambda_d0=0.6, lambda_d1=0.1)
        self.assertAlmostEqual(risks(0.2, 0.7, increasing)["Defer"], 0.45)
        self.assertAlmostEqual(risks(0.2, 0.7, decreasing)["Defer"], 0.50)

    def test_intervals_are_clipped_to_unit_range(self):
        result = evidence([20.0, 0.0], 1.0, self.config)
        self.assertEqual(result.upper[0], 1.0)
        self.assertEqual(result.lower[1], 0.0)

    def test_boundary_keeps_leader_masks_rejections_and_caps_size(self):
        candidates = list(reversed(range(15)))
        values = evidence(np.zeros(15), 0.0, self.config)
        result = boundary_set(candidates, 14, {0, 1}, values, self.config)
        self.assertEqual(result, [14] + list(range(2, 13)))

    def test_boundary_prioritizes_overlap_before_higher_support(self):
        # Candidate1 only touches leader0 at an endpoint; candidate2 has
        # higher nominal support but cannot displace that overlap competitor.
        values = Evidence(
            support=np.array([0.35, 0.15, 0.50]),
            width=np.array([0.10, 0.10, 0.01]),
            lower=np.array([0.25, 0.05, 0.49]),
            upper=np.array([0.45, 0.25, 0.51]),
        )
        config = replace(self.config, k_boundary=2)
        self.assertEqual(boundary_set([0, 1, 2], 0, set(), values, config), [0, 1])


class SelectiveDecodingTests(unittest.TestCase):
    def setUp(self):
        self.config = ETConfig()

    @staticmethod
    def forbidden_query(_targets):
        raise AssertionError("Visual acquisition must remain lazy on this route")

    def test_inactive_caption_uses_full_vocabulary_base_score(self):
        # The original image favors token0, but base CD favors token2.
        clean = np.array([4.0, 3.0, 0.0])
        weak = np.array([4.0, 3.0, -20.0])
        result = decode_step(
            clean, weak, config=self.config, object_ids=frozenset(),
            query_views=self.forbidden_query,
        )
        self.assertEqual(result.token_id, 2)
        np.testing.assert_allclose(result.scores, [4.0, 3.0, 20.0])

    def test_accept_commits_without_query(self):
        result = decode_step(
            [2.0, 1.0], [-4.0, 1.0], config=self.config,
            binary_ids=[0, 1], question_target="dog",
            query_views=self.forbidden_query,
        )
        self.assertEqual(result.token_id, 0)

    def test_one_rejection_falls_back_to_nonrejected_candidate(self):
        # Base leader0 has almost zero support, so it is rejected.
        result = decode_step(
            [20.0, 0.0], [20.0, -5.0], config=self.config,
            binary_ids=[0, 1], question_target="dog",
            query_views=self.forbidden_query,
        )
        self.assertEqual(result.token_id, 1)
        self.assertEqual(result.trace["rejected"], [0])
        self.assertEqual(result.trace["reason"], "reject_budget_exhausted")

    def test_all_rejected_fallback_restores_original_candidates(self):
        config = replace(
            self.config, k_reject=2, lambda_fa=10.0,
            lambda_d0=2.0, lambda_d1=2.0,
        )
        result = decode_step(
            [2.0, 1.0], [2.0, 1.0], config=config,
            binary_ids=[0, 1], question_target="dog",
            query_views=self.forbidden_query,
        )
        self.assertEqual(result.token_id, 0)
        self.assertEqual(result.trace["rejected"], [0, 1])
        self.assertEqual(len(result.trace["route"]), 2)

    def test_defer_without_target_falls_back_without_query(self):
        result = decode_step(
            [2.0, 1.0], [2.0, 1.0], config=self.config,
            binary_ids=[0, 1], query_views=self.forbidden_query,
        )
        self.assertEqual(result.token_id, 0)

    def test_gain_gate_preserves_original_evidence(self):
        calls = []

        def query(targets):
            calls.append(targets)
            yield "no-improvement", np.array([2.0, 1.0])

        result = decode_step(
            [2.0, 1.0], [2.0, 1.0], config=self.config,
            binary_ids=[0, 1], question_target="dog", query_views=query,
        )
        self.assertEqual(calls, [("dog",)])
        self.assertEqual(result.token_id, 0)
        self.assertFalse(result.trace["updated"])
        np.testing.assert_allclose(result.scores, [2.0, 1.0])

    def test_js_constraint_and_adaptive_fusion_can_change_decision(self):
        clean = np.array([1.0, 1.0, 0.0])
        weak = np.array([1.0, 1.0, -2.0])
        calls = []

        def query(targets):
            calls.append(targets)
            # Large gain, but JS exceeds .12 and the view must be excluded.
            yield "inadmissible-shift", np.array([-3.0, 1.0, -99.0])
            # Original-reference difference on B is [0,.1].
            yield "small-valid-gain", np.array([1.0, 0.9, -99.0])

        result = decode_step(
            clean, weak, config=self.config, binary_ids=[0, 1],
            question_target="the dog's color is black", query_views=query,
        )
        self.assertEqual(calls, [("the dog's color is black",)])
        self.assertEqual(result.token_id, 1)
        self.assertFalse(result.trace["views"][0]["admissible"])
        self.assertTrue(result.trace["views"][1]["admissible"])
        self.assertEqual(result.trace["selected_view"], "small-valid-gain")

        support = softmax([0.0, 0.1])
        base_width = 0.05 * normalized_entropy(clean) + 0.8 * 0.25
        view_width = 0.05 * normalized_entropy(clean) + 0.8 * support[0] * support[1]
        # Both interval pairs overlap, so their overlap-ratio difference is0.
        gain = support[1] - support[0] + 0.2 * (base_width - view_width)
        rho = min(0.3, gain + 0.5 * base_width)
        self.assertGreater(rho, 0.0)
        self.assertLess(rho, 0.3)
        np.testing.assert_allclose(result.scores, [1.0, 1.0 + rho * 0.1, 2.0])
        # Non-boundary token2 retains its original CD evidence.

    def test_no_shift_admissible_view_falls_back(self):
        def query(_targets):
            yield "inadmissible", np.array([-3.0, 1.0])

        result = decode_step(
            [1.0, 1.0], [1.0, 1.0], config=self.config,
            binary_ids=[0, 1], question_target="dog", query_views=query,
        )
        self.assertEqual(result.token_id, 0)
        self.assertEqual(result.trace["reason"], "no_admissible_view")
        self.assertFalse(result.trace["updated"])

    def test_view_budget_stops_lazy_generator_at_four(self):
        visited = []

        def query(_targets):
            for index in range(10):
                visited.append(index)
                yield f"view-{index}", np.array([2.0, 1.0])

        result = decode_step(
            [2.0, 1.0], [2.0, 1.0], config=self.config,
            binary_ids=[0, 1], question_target="dog", query_views=query,
        )
        self.assertEqual(result.token_id, 0)
        self.assertEqual(visited, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
