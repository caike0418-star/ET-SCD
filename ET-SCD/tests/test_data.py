"""Check task preparation strips supervision before inference."""

from dataclasses import replace
import unittest

import numpy as np
from PIL import Image

from etscd.config import ETConfig
from etscd.data import CHAIR_PROMPT, prepare_records
from etscd.views import corner_views, resize_regional_view


class DataPreparationTests(unittest.TestCase):
    def test_legacy_pope_fields_are_normalized_and_labels_split_out(self):
        source = [{"question_id": 7, "image": "0007.jpg", "text": "Is a dog present?",
                   "label": " YES ", "category": "random", "metadata": {"index": 7}}]
        questions, labels = prepare_records(source, "pope")
        self.assertEqual(questions, [{"id": "7", "image": "0007.jpg", "question": "Is a dog present?"}])
        self.assertEqual(labels, [{"id": "7", "label": "yes"}])
        self.assertIn("metadata", source[0])

    def test_flipping_labels_does_not_change_inference_questions(self):
        row = {"id": "a", "image": "a.jpg", "question": "Is a dog present?"}
        first, _ = prepare_records([dict(row, label="yes")], "pope")
        second, _ = prepare_records([dict(row, label="no")], "pope")
        self.assertEqual(first, second)

    def test_mme_retains_hall_categories_and_separate_grouping_metadata(self):
        source = [{"id": "a", "image": "a.jpg", "question": "Are there two dogs?",
                   "label": "no", "category": "count", "image_id": "group-a"},
                  {"id": "b", "image": "b.jpg", "question": "Read the text.",
                   "label": "yes", "category": "ocr"}]
        questions, labels = prepare_records(source, "mme")
        self.assertEqual(questions, [{"id": "a", "image": "a.jpg", "question": "Are there two dogs?"}])
        self.assertEqual(labels, [{"id": "a", "label": "no", "category": "count", "image_id": "group-a"}])

    def test_invalid_or_duplicate_binary_labels_are_rejected(self):
        row = {"id": "a", "image": "a.jpg", "question": "Is a dog present?", "label": "yes"}
        with self.assertRaises(ValueError):
            prepare_records([row, row], "pope")
        for label in (None, "unknown", "1", True):
            with self.subTest(label=label), self.assertRaises(ValueError):
                prepare_records([dict(row, label=label)], "pope")

    def test_chair_preparation_replaces_legacy_prompt_and_emits_no_labels(self):
        source = [{"question_id": i, "image": f"{i}.jpg", "text": "Describe this photo in detail."}
                  for i in range(500)]
        questions, labels = prepare_records(source, "chair")
        self.assertEqual(len(questions), 500)
        self.assertEqual({row["question"] for row in questions}, {CHAIR_PROMPT})
        self.assertEqual(labels, [])
        source[-1]["image"] = source[0]["image"]
        with self.assertRaises(ValueError):
            prepare_records(source, "chair")


class FixedViewTests(unittest.TestCase):
    def test_regional_crop_resizes_to_native_width_and_height(self):
        crop = Image.new("RGB", (14, 7), "red")
        crop.info["branch"] = "focus"
        result = resize_regional_view(crop, (12, 10))
        self.assertEqual(result.size, (12, 10))
        self.assertEqual(result.info["branch"], "focus")
        self.assertEqual(crop.size, (14, 7))

    def test_native_size_requires_positive_integer_dimensions(self):
        crop = Image.new("RGB", (14, 7))
        for size in (None, (0, 10), (12, -1), (True, 10), (12., 10), (12,)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                resize_regional_view(crop, size)

    def test_default_corner_budget_exposes_first_target_only(self):
        image = Image.new("RGB", (20, 10), "red")
        views = list(corner_views(image, ("dog", "cat"), ETConfig()))
        self.assertEqual([name for name, _ in views], [
            "dog:top_left", "dog:top_right", "dog:bottom_left", "dog:bottom_right",
        ])
        self.assertEqual([view.size for _, view in views], [(14, 7)] * 4)
        self.assertEqual(image.size, (20, 10))
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_two_targets_have_identical_crop_geometry_without_localizer(self):
        pixels = np.arange(20 * 10 * 3, dtype=np.uint8).reshape(10, 20, 3)
        image = Image.fromarray(pixels)
        config = replace(ETConfig(), max_views=8)
        views = list(corner_views(image, ("dog", "cat"), config))
        self.assertEqual(len(views), 8)
        for index in range(4):
            np.testing.assert_array_equal(np.asarray(views[index][1]), np.asarray(views[index + 4][1]))

    def test_no_valid_target_means_no_regional_views(self):
        image = Image.new("RGB", (20, 10))
        self.assertEqual(list(corner_views(image, (), ETConfig())), [])
        self.assertEqual(list(corner_views(image, (" ",), ETConfig())), [])


if __name__ == "__main__":
    unittest.main()
