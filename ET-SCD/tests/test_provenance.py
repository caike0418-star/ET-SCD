import unittest
from unittest.mock import patch

from etscd.evaluation import aggregate_runs, POPE_SEEDS, sample_ids_sha256, score_files


class ProvenanceTests(unittest.TestCase):
    def fixture(self):
        return [{"benchmark": "pope", "dataset": "aokvqa", "setting": "random",
                 "model": "llava", "method": "etscd", "seed": seed,
                 "sample_ids_sha256": "a" * 64, "config_sha256": "b" * 64,
                 "input_sha256": "c" * 64, "labels_sha256": "d" * 64,
                 "source_sha256": {"core.py": "e" * 64}, "status": "complete",
                 "image_sha256": {"a": "1" * 64, "b": "2" * 64},
                 "weak_image_sha256": {"a": "3" * 64, "b": "4" * 64},
                 "diagnostic_subset": False, "metrics": {"accuracy": 80., "f1": 80.}}
                for seed in POPE_SEEDS]

    def test_diagnostic_and_failed_runs_rejected(self):
        for field, value in (("status", "failed"), ("diagnostic_subset", True)):
            rows = self.fixture()
            rows[-1][field] = value
            with self.assertRaises(ValueError):
                aggregate_runs(rows)

    def test_question_source_and_label_consistency_required(self):
        for field, value in (("input_sha256", "f" * 64), ("labels_sha256", "f" * 64),
                             ("source_sha256", {"core.py": "f" * 64})):
            rows = self.fixture()
            rows[-1][field] = value
            with self.assertRaises(ValueError):
                aggregate_runs(rows)

    def test_partial_provenance_rejected(self):
        for field in ("source_sha256", "image_sha256", "weak_image_sha256"):
            rows = self.fixture()
            del rows[-1][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                aggregate_runs(rows)

    def test_changed_original_or_weak_images_rejected(self):
        for field in ("image_sha256", "weak_image_sha256"):
            rows = self.fixture()
            rows[-1][field]["a"] = "f" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                aggregate_runs(rows)

    def test_matching_image_maps_are_independent_of_key_order(self):
        rows = self.fixture()
        for field in ("image_sha256", "weak_image_sha256"):
            rows[-1][field] = dict(reversed(list(rows[-1][field].items())))
        result = aggregate_runs(rows)["groups"][0]
        self.assertEqual(result["metrics"]["accuracy"]["mean"], 80.0)
        self.assertEqual(result["metrics"]["accuracy"]["std"], 0.0)

    def test_inline_weak_images_allow_all_none_but_not_mixed_maps(self):
        rows = self.fixture()
        for row in rows:
            row["weak_image_sha256"] = None
        self.assertEqual(aggregate_runs(rows)["groups"][0]["num_seeds"], 5)
        rows[-1]["weak_image_sha256"] = {"a": "3" * 64, "b": "4" * 64}
        with self.assertRaisesRegex(ValueError, "weak_image_sha256"):
            aggregate_runs(rows)

    def test_score_preserves_original_and_optional_weak_image_hashes(self):
        predictions = [{"id": "a", "text": "yes"}, {"id": "b", "text": "no"}]
        labels = [{"id": "a", "label": "yes"}, {"id": "b", "label": "no"}]
        for weak_hashes in ({"a": "3" * 64, "b": "4" * 64}, None):
            manifest = {key: value for key, value in self.fixture()[0].items() if key != "metrics"}
            manifest["sample_ids_sha256"] = sample_ids_sha256(["a", "b"])
            manifest["weak_image_sha256"] = weak_hashes
            with (self.subTest(weak_hashes=weak_hashes),
                  patch("etscd.evaluation.read_jsonl", side_effect=[predictions, labels]),
                  patch("etscd.evaluation.Path.read_bytes", return_value=b"input")):
                report = score_files("pope", "predictions.jsonl", "labels.jsonl", manifest)
            self.assertEqual(report["image_sha256"], manifest["image_sha256"])
            self.assertEqual(report["weak_image_sha256"], weak_hashes)
