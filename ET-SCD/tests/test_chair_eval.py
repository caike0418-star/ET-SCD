import unittest
from etscd.chair_eval import validate_caption_records
from etscd.data import CHAIR_PROMPT


class ChairProtocolTests(unittest.TestCase):
    def fixture(self):
        questions = [{"id": str(i), "image": f"COCO_val2014_{i:012d}.jpg",
                      "question": CHAIR_PROMPT} for i in range(500)]
        predictions = [{"id": str(i), "text": "A dog."} for i in range(500)]
        return predictions, questions

    def test_fixed500_exact_ids_and_numeric_coco_ids(self):
        predictions, questions = self.fixture()
        result = validate_caption_records(predictions[::-1], questions)
        self.assertEqual(len(result), 500)
        self.assertEqual(result[0], {"image_id": 0, "caption": "A dog."})

    def test_missing_and_duplicate_predictions_fail(self):
        predictions, questions = self.fixture()
        with self.assertRaises(ValueError):
            validate_caption_records(predictions[:-1], questions)
        with self.assertRaises(ValueError):
            validate_caption_records(predictions + predictions[:1], questions)

    def test_prompt_and_distinct_images_are_required(self):
        predictions, questions = self.fixture()
        questions[-1]["image"] = questions[0]["image"]
        with self.assertRaises(ValueError):
            validate_caption_records(predictions, questions)
        predictions, questions = self.fixture()
        questions[0]["question"] = "Describe this photo in detail."
        with self.assertRaises(ValueError):
            validate_caption_records(predictions, questions)
