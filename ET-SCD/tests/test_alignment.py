"""Fixed-profile and result-label checks."""

from dataclasses import replace
import unittest

from etscd.alignment import label_method, validate_alignment
from etscd.config import ETConfig


class AlignmentTests(unittest.TestCase):
    def test_fixed_profile_accepts_every_benchmark_and_method(self):
        for benchmark in ("pope", "mme", "chair"):
            for method in ("etscd", "regular", "vcd"):
                with self.subTest(benchmark=benchmark, method=method):
                    report = validate_alignment(ETConfig(), benchmark=benchmark, method=method)
                    self.assertTrue(report["policy_matches_table_vi"])
                    self.assertTrue(report["generation_limit_matches_paper"])
                    self.assertEqual(report["changed_parameter_names"], [])
                    self.assertFalse(report["diagnostic_subset"])
                    self.assertEqual(label_method(method, report), method)

    def test_fixed_profile_rejects_multiple_parameter_changes_by_name(self):
        changed = replace(ETConfig(), gamma_o=0.10, rho_max=0.20)
        with self.assertRaises(ValueError) as caught:
            validate_alignment(changed, benchmark="chair")
        self.assertIn("gamma_o=0.1", str(caught.exception))
        self.assertIn("rho_max=0.2", str(caught.exception))
        self.assertIn("profile='custom'", str(caught.exception))

    def test_fixed_profile_rejects_alternate_generation_limit(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens=32"):
            validate_alignment(ETConfig(), benchmark="chair", max_new_tokens=32)

    def test_custom_profile_reports_changes_and_labels_all_methods(self):
        config = replace(ETConfig(), max_views=2, k_targets=1)
        for method in ("etscd", "regular", "vcd"):
            with self.subTest(method=method):
                report = validate_alignment(config, benchmark="pope", profile="custom",
                                            method=method, max_new_tokens=32, allow_subset=True)
                self.assertEqual(report["profile"], "custom")
                self.assertFalse(report["policy_matches_table_vi"])
                self.assertFalse(report["generation_limit_matches_paper"])
                self.assertEqual(report["changed_parameter_names"],
                                 ["k_targets", "max_views", "max_new_tokens"])
                self.assertTrue(report["diagnostic_subset"])
                self.assertEqual(label_method(method, report), f"{method}-custom")

    def test_custom_label_is_explicit_even_with_default_values(self):
        report = validate_alignment(ETConfig(), benchmark="mme", profile="custom")
        self.assertTrue(report["policy_matches_table_vi"])
        self.assertEqual(report["changed_parameter_names"], [])
        self.assertEqual(label_method("etscd", report), "etscd-custom")

    def test_et_scd_subset_has_distinct_label(self):
        report = validate_alignment(ETConfig(), benchmark="pope", allow_subset=True)
        self.assertEqual(label_method("etscd", report), "etscd-diagnostic")
        for method in ("regular", "vcd"):
            report = validate_alignment(ETConfig(), benchmark="pope", method=method, allow_subset=True)
            self.assertEqual(label_method(method, report), method)

    def test_invalid_profile_task_method_or_generation_limit_are_rejected(self):
        for kwargs in ({"benchmark": "unknown"}, {"benchmark": "pope", "method": "unknown"},
                       {"benchmark": "pope", "profile": "unknown"},
                       {"benchmark": "pope", "max_new_tokens": 0},
                       {"benchmark": "pope", "max_new_tokens": 2.5},
                       {"benchmark": "pope", "max_new_tokens": True},
                       {"benchmark": "pope", "allow_subset": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                validate_alignment(ETConfig(), **kwargs)

    def test_method_label_must_match_validated_report(self):
        report = validate_alignment(ETConfig(), benchmark="pope")
        with self.assertRaisesRegex(ValueError, "differs"):
            label_method("regular", report)
        with self.assertRaises(ValueError):
            label_method("unknown", report)
        with self.assertRaises(ValueError):
            label_method("etscd", {"method": "etscd"})


if __name__ == "__main__":
    unittest.main()
