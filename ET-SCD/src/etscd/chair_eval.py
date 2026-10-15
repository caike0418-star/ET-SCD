"""Strict fixed500 CHAIR/BLEU scoring, with optional metrics imported lazily."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from .data import CHAIR_PROMPT, read_jsonl
from .evaluation import sample_ids_sha256


def validate_caption_records(predictions, questions):
    expected = {}
    for question in questions:
        if question.get("question") != CHAIR_PROMPT:
            raise ValueError("CHAIR requires the exact TMM caption prompt")
        key = str(question["id"])
        if key in expected:
            raise ValueError("Duplicate question ID")
        expected[key] = question
    if len(expected) != 500 or len({row["image"] for row in questions}) != 500:
        raise ValueError("CHAIR requires exactly 500 distinct images")
    actual = {}
    for row in predictions:
        key = str(row["id"])
        if key in actual:
            raise ValueError("Duplicate prediction ID")
        if not isinstance(row.get("text"), str):
            raise ValueError("Caption must be a string")
        actual[key] = row["text"]
    if set(actual) != set(expected):
        raise ValueError("Prediction IDs must exactly match the fixed500 questions")
    captions = [{"image_id": int(key), "caption": actual[key]} for key in expected]
    if len({row["image_id"] for row in captions}) != 500:
        raise ValueError("IDs must identify 500 distinct numeric COCO image IDs")
    return captions


def score(predictions_path, questions_path, annotation_dir):
    predictions = read_jsonl(predictions_path)
    questions = read_jsonl(questions_path)
    captions = validate_caption_records(predictions, questions)
    manifest_path = Path(str(predictions_path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    prediction_hash = hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest()
    if manifest:
        if manifest.get("status") != "complete" or manifest.get("diagnostic_subset"):
            raise ValueError("CHAIR scoring requires a complete non-diagnostic run")
        if manifest.get("benchmark") != "chair" or manifest.get("predictions_sha256") != prediction_hash:
            raise ValueError("CHAIR manifest benchmark or prediction checksum mismatch")
        if manifest.get("sample_ids_sha256") != sample_ids_sha256(row["id"] for row in questions):
            raise ValueError("CHAIR manifest sample IDs mismatch")
    try:
        from .metrics.chair import load_or_build
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-eval.txt and Java before CHAIR/BLEU scoring") from exc
    annotation_dir = Path(annotation_dir)
    caption_path = annotation_dir / "captions_val2014.json"
    annotations = json.loads(caption_path.read_text(encoding="utf-8"))
    references = {}
    for annotation in annotations["annotations"]:
        references.setdefault(int(annotation["image_id"]), []).append({"caption": annotation["caption"]})
    missing = [row["image_id"] for row in captions if not references.get(row["image_id"])]
    if missing:
        raise ValueError(f"Missing COCO caption references for IDs {missing[:5]}")
    evaluator = load_or_build(str(annotation_dir), cache="")
    # No pickle cache is loaded. Temporary captions contain predictions only.
    with tempfile.TemporaryDirectory(prefix="etscd_chair_") as directory:
        minimal = Path(directory) / "captions.jsonl"
        minimal.write_text("".join(json.dumps(row) + "\n" for row in captions), encoding="utf-8")
        chair = evaluator.compute(str(minimal))
    gts = {row["image_id"]: references[row["image_id"]] for row in captions}
    res = {row["image_id"]: [{"caption": row["caption"]}] for row in captions}
    tokenizer = PTBTokenizer()
    gts, res = tokenizer.tokenize(gts), tokenizer.tokenize(res)
    bleu, _ = Bleu(4).compute_score(gts, res)
    overall = chair["overall_metrics"]
    if overall["NumCaptions"] != 500:
        raise ValueError("CHAIR evaluator did not score all500 captions")
    metrics = {key: 100 * overall[key] for key in ("CHAIRs", "CHAIRi", "Recall", "F1")}
    metrics["BLEU-4"] = 100 * float(bleu[3])
    report = {"benchmark": "chair", "n": 500, "metrics": metrics, "units": "percent",
              "sample_ids_sha256": sample_ids_sha256(row["id"] for row in questions),
              "predictions_sha256": prediction_hash, "evaluator_version": overall["EvaluatorVersion"],
              "annotation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(annotation_dir.glob("*.json"))},
              "chair_details": chair}
    if manifest:
        for key in ("dataset", "setting", "model", "method", "seed", "config_sha256", "source_sha256"):
            report[key] = manifest.get(key)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "questions", "annotation-dir", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.resolve() in {Path(args.predictions).resolve(), Path(args.questions).resolve()} or output.exists():
        p.error("Choose a fresh output path different from all inputs")
    report = score(args.predictions, args.questions, args.annotation_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
