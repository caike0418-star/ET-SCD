"""Tests for benchmark scoring and matched-seed aggregation."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from etscd.evaluation import (
    MME_CATEGORIES,
    POPE_SEEDS,
    aggregate_runs,
    evaluate_records,
    main,
    parse_answer,
    sample_ids_sha256,
    score_files,
)


class EvaluationTests(unittest.TestCase):
    def test_answer_parser_rejects_substrings_explanations_and_both_answers(self):
        for text in ("yesterday", "nobody", "yes or no", "Yes, there is a cat.", "", None):
            self.assertEqual(parse_answer(text), "unknown")
        self.assertEqual(parse_answer(" Yes. "), "yes")
        self.assertEqual(parse_answer("NO!"), "no")

    def test_pope_unknown_is_wrong_even_for_negative_truth(self):
        labels = [{"id": "a", "label": "yes"}, {"id": "b", "label": "no"},
                  {"id": "c", "label": "yes"}, {"id": "d", "label": "no"}]
        predictions = [{"id": "d", "text": "ambiguous"}, {"id": "c", "text": "no"},
                       {"id": "b", "text": "yes"}, {"id": "a", "text": "yes"}]
        result = evaluate_records("pope", predictions, labels)
        self.assertEqual(result["accuracy"], 25.0)
        self.assertEqual(result["precision"], 50.0)
        self.assertEqual(result["recall"], 50.0)
        self.assertEqual(result["f1"], 50.0)
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["tn"], 0)

    def test_exact_unique_ids_required(self):
        labels = [{"id": "a", "label": "yes"}, {"id": "b", "label": "no"}]
        cases = [
            [{"id": "a", "text": "yes"}],
            [{"id": "a", "text": "yes"}, {"id": "c", "text": "no"}],
            [{"id": "a", "text": "yes"}, {"id": "a", "text": "no"}],
        ]
        for predictions in cases:
            with self.assertRaises(ValueError):
                evaluate_records("pope", predictions, labels)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_records("pope", [{"id": "a", "text": "yes"}], labels[:1] * 2)

    def test_labels_are_separate_and_canonical(self):
        predictions = [{"id": "a", "text": "no", "label": "no"}]
        result = evaluate_records("pope", predictions, [{"id": "a", "label": "yes"}])
        self.assertEqual(result["accuracy"], 0.0)
        with self.assertRaises(ValueError):
            evaluate_records("pope", predictions, [{"id": "a", "label": "Yes"}])

    @staticmethod
    def mme_fixture():
        labels, predictions = [], []
        for category in MME_CATEGORIES:
            for question in (0, 1):
                identifier = f"{category}:{question}"
                labels.append({"id": identifier, "image_id": "same_image", "category": category,
                               "label": "yes" if question == 0 else "no"})
                answer = "yes" if question == 0 else "no"
                if category == "count" and question == 1:
                    answer = "unknown"
                if category == "position":
                    answer = "no" if question == 0 else "yes"
                predictions.append({"id": identifier, "text": answer})
        return predictions, labels

    def test_mme_acc_plus_counts_complete_image_pairs(self):
        predictions, labels = self.mme_fixture()
        result = evaluate_records("mme", predictions, labels)
        self.assertEqual(result["categories"]["existence"]["score"], 200.0)
        self.assertEqual(result["categories"]["count"]["accuracy"], 50.0)
        self.assertEqual(result["categories"]["count"]["accuracy_plus"], 0.0)
        self.assertEqual(result["categories"]["position"]["score"], 0.0)
        self.assertEqual(result["hall_total"], 450.0)
        self.assertEqual(result["unknown"], 1)

    def test_mme_rejects_missing_category_and_incomplete_pair(self):
        predictions, labels = self.mme_fixture()
        with self.assertRaisesRegex(ValueError, "all four categories"):
            evaluate_records("mme", predictions[:-2], labels[:-2])
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            evaluate_records("mme", predictions[:-1], labels[:-1])
        labels[-1] = dict(labels[-1], image_id="different_image")
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            evaluate_records("mme", predictions, labels)

    @staticmethod
    def runs(setting="random", values=(80, 81, 82, 83, 84)):
        return [{"benchmark": "pope", "dataset": "aokvqa", "setting": setting,
                 "model": "llava", "method": "etscd", "seed": seed,
                 "sample_ids_sha256": "a" * 64, "config_sha256": "b" * 64,
                 "metrics": {"accuracy": value, "f1": value}}
                for seed, value in zip(POPE_SEEDS, values)]

    def test_seed_sd_is_not_dispersion_across_settings(self):
        runs = self.runs(values=(80, 80, 80, 80, 80))
        runs += self.runs("popular", (10, 10, 10, 10, 10))
        runs += self.runs("adversarial", (50, 50, 50, 50, 50))
        result = aggregate_runs(runs)
        self.assertEqual(len(result["groups"]), 3)
        for group in result["groups"]:
            self.assertEqual(group["metrics"]["accuracy"]["std"], 0.0)
            self.assertEqual(group["std_ddof"], 1)

    def test_sample_sd_denominator_is_explicit(self):
        result = aggregate_runs(self.runs())["groups"][0]
        self.assertEqual(result["metrics"]["accuracy"]["mean"], 82.0)
        self.assertAlmostEqual(result["metrics"]["accuracy"]["std"], math.sqrt(2.5))

    def test_seed_aggregation_rejects_duplicate_missing_and_changed_provenance(self):
        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            aggregate_runs(self.runs()[:-1])
        with self.assertRaisesRegex(ValueError, "duplicate seeds"):
            aggregate_runs(self.runs() + self.runs()[:1])
        for field in ("sample_ids_sha256", "config_sha256"):
            runs = self.runs()
            runs[-1][field] = "c" * 64
            with self.assertRaises(ValueError):
                aggregate_runs(runs)
        runs = self.runs()
        runs[-1]["metrics"]["accuracy"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            aggregate_runs(runs)

    def test_sample_hash_is_order_independent_and_unique(self):
        self.assertEqual(sample_ids_sha256(["b", "a"]), sample_ids_sha256(["a", "b"]))
        with self.assertRaises(ValueError):
            sample_ids_sha256([1, "1"])

    def test_score_files_manifest_and_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            predictions = directory / "predictions.jsonl"
            labels = directory / "labels.jsonl"
            output = directory / "metrics.json"
            predictions.write_text('{"id":"a","text":"yes"}\n', encoding="utf-8")
            labels.write_text('{"id":"a","label":"yes"}\n', encoding="utf-8")
            manifest = {"benchmark": "pope", "dataset": "aokvqa", "setting": "random",
                        "model": "llava", "method": "etscd", "seed": 42,
                        "image_sha256": {"a": "d" * 64}, "weak_image_sha256": {"a": "e" * 64},
                        "config_sha256": "b" * 64, "sample_ids_sha256": sample_ids_sha256(["a"])}
            report = score_files("pope", predictions, labels, manifest)
            self.assertEqual(report["metrics"]["accuracy"], 100.0)
            self.assertEqual(report["image_sha256"], manifest["image_sha256"])
            self.assertEqual(report["weak_image_sha256"], manifest["weak_image_sha256"])
            Path(str(predictions) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            main(["score", "--benchmark", "pope", "--predictions", str(predictions),
                  "--labels", str(labels), "--output", str(output)])
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 42)
            self.assertEqual(saved["image_sha256"], manifest["image_sha256"])
            self.assertEqual(saved["weak_image_sha256"], manifest["weak_image_sha256"])
            manifest["sample_ids_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "does not match"):
                score_files("pope", predictions, labels, manifest)


if __name__ == "__main__":
    unittest.main()
