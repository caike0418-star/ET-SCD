"""Frequency filtering, gradient overlays, and image preparation checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from etscd.weak_images import edge_overlay, frequency_lowpass, prepare_weak_images


class ImageTransformTests(unittest.TestCase):
    def test_frequency_filter_suppresses_high_more_than_low_frequencies(self):
        size = 129
        x = np.arange(size)
        amplitudes = []
        for cycles in (2, 32):
            wave = np.cos(2 * np.pi * cycles * x / size)
            signal = np.round(128 + 60 * wave).astype(np.uint8)
            pixels = np.repeat(np.tile(signal, (size, 1))[:, :, None], 3, axis=2)
            filtered = np.asarray(frequency_lowpass(Image.fromarray(pixels)))[:, :, 0].mean(axis=0)
            amplitudes.append(abs(np.dot(filtered - filtered.mean(), wave) * 2 / size))
        self.assertGreater(amplitudes[0], 50)
        self.assertLess(amplitudes[1], amplitudes[0] * 0.1)

    def test_transforms_return_rgb_at_original_size(self):
        image = Image.new("RGBA", (19, 17), (80, 160, 240, 64))
        for transform in (frequency_lowpass, edge_overlay):
            with self.subTest(transform=transform.__name__):
                result = transform(image)
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.size, image.size)

    def test_constant_color_has_no_edges_and_blends_toward_black(self):
        image = Image.new("RGB", (9, 7), (200, 100, 50))
        self.assertTrue(np.all(np.asarray(edge_overlay(image, 0.5)) == (100, 50, 25)))
        self.assertTrue(np.all(np.asarray(edge_overlay(image, 1.0)) == 0))
        self.assertTrue(np.array_equal(np.asarray(edge_overlay(image, 0)), np.asarray(image)))

    def test_transform_parameter_boundaries(self):
        image = Image.new("RGB", (9, 7), "gray")
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(cutoff=value), self.assertRaises(ValueError):
                frequency_lowpass(image, cutoff_ratio=value)
            with self.subTest(order=value), self.assertRaises(ValueError):
                frequency_lowpass(image, order=value)
        for value in (-0.01, 1.01, float("nan"), float("inf"), True):
            with self.subTest(strength=value), self.assertRaises(ValueError):
                edge_overlay(image, value)
        with self.assertRaises(ValueError):
            edge_overlay(Image.new("RGB", (1, 7)))


class PrepareImagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "nested").mkdir()
        Image.new("RGB", (12, 10), (50, 100, 150)).save(self.images / "nested/a.jpg")
        Image.new("RGB", (16, 14), (70, 90, 110)).save(self.images / "b.jpg")
        self.questions = self.root / "questions.jsonl"
        self.rows = [
            {"id": "2", "image": "b.jpg", "question": "Is there a dog?"},
            {"id": "1", "image": "nested/a.jpg", "question": "Is there a cat?"},
            {"id": "3", "image": "b.jpg", "question": "Is there a bird?"},
        ]
        self.write_questions()

    def write_questions(self):
        self.questions.write_text("".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8")

    def prepare(self, output="weak", **kwargs):
        return prepare_weak_images(self.questions, self.images, self.root / output,
                                   transform="gaussian-blur", **kwargs)

    def test_images_are_unique_sorted_encoded_and_hashed(self):
        manifest = self.prepare()
        self.assertEqual(manifest["images"], ["b.jpg", "nested/a.jpg"])
        self.assertEqual(manifest["question_count"], 3)
        self.assertEqual(manifest["image_count"], 2)
        self.assertEqual(manifest["parameters"], {"radius": 3.0})
        self.assertEqual(manifest["jpeg_quality"], 95)
        self.assertEqual(manifest["status"], "complete")
        for name in manifest["images"]:
            path = self.root / "weak" / name
            self.assertEqual(manifest["weak_image_sha256"][name], hashlib.sha256(path.read_bytes()).hexdigest())
            with Image.open(path) as result, Image.open(self.images / name) as source:
                self.assertEqual(result.format, "JPEG")
                self.assertEqual(result.size, source.size)
        self.assertEqual(manifest["questions_sha256"], hashlib.sha256(self.questions.read_bytes()).hexdigest())
        self.assertEqual(json.loads((self.root / "weak/manifest.json").read_text(encoding="utf-8")), manifest)

    def test_existing_output_is_not_reused(self):
        self.prepare()
        original = (self.root / "weak/manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.root / "weak/manifest.json").read_bytes(), original)

    def test_labels_duplicate_ids_and_escaping_paths_are_rejected_before_writing(self):
        invalid = [
            [{**self.rows[0], "label": "yes"}],
            [self.rows[0], self.rows[0]],
            [{**self.rows[0], "image": "../b.jpg"}],
            [{**self.rows[0], "image": "/tmp/image.jpg"}],
            [{**self.rows[0], "image": "C:/images/a.jpg"}],
        ]
        for rows in invalid:
            self.rows = rows
            self.write_questions()
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse((self.root / "weak").exists())

    def test_invalid_save_and_transform_parameters_are_rejected_before_writing(self):
        for kwargs in ({"jpeg_quality": 0}, {"jpeg_quality": 101},
                       {"jpeg_quality": True}, {"blur_radius": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.prepare(**kwargs)
            self.assertFalse((self.root / "weak").exists())


if __name__ == "__main__":
    unittest.main()
