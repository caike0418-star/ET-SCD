"""POPE/MME scoring and matched-seed aggregation.

Prediction and label files are joined by their unique sample IDs. Standard
deviation is computed across matched seeds within each benchmark setting,
using sample SD (ddof=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POPE_SEEDS = (42, 55, 123, 2024, 3407)
MME_CATEGORIES = ("existence", "count", "position", "color")
GROUP_FIELDS = ("benchmark", "dataset", "setting", "model", "method")
SD_CONVENTION = "sample standard deviation across matched seeds (ddof=1)"
_ANSWER = re.compile(r"\s*(yes|no)\s*[.!?]?\s*", re.IGNORECASE)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def parse_answer(text: Any) -> str:
    """Parse a standalone Yes/No answer; other text is an unknown answer.

    A single terminal '.', '!', or '?' is allowed. Substrings, explanations,
    both-answer expressions, and missing answers are deliberately unknown.
    Unknown predictions receive no credit, including on negative questions.
    """
    match = _ANSWER.fullmatch(text) if isinstance(text, str) else None
    return match.group(1).lower() if match else "unknown"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
    return records


def _canonical_id(value: Any, field: str = "id") -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a nonempty string or integer")
    result = str(value)
    if not result.strip():
        raise ValueError(f"{field} must be nonempty")
    return result


def _index(records: Sequence[Mapping[str, Any]], name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError(f"{name}: expected object records")
        if "id" not in row:
            raise ValueError(f"{name}: record is missing id")
        identifier = _canonical_id(row["id"])
        if identifier in indexed:
            raise ValueError(f"{name}: duplicate id {identifier!r}")
        indexed[identifier] = row
    if not indexed:
        raise ValueError(f"{name}: no records")
    return indexed


def sample_ids_sha256(ids: Iterable[str | int]) -> str:
    """Hash sorted canonical IDs as compact, UTF-8 JSON (ensure_ascii=False)."""
    canonical = [_canonical_id(identifier) for identifier in ids]
    if len(canonical) != len(set(canonical)):
        raise ValueError("duplicate sample IDs cannot be hashed as a run")
    payload = json.dumps(sorted(canonical), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _align(predictions: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]):
    prediction_index = _index(predictions, "predictions")
    label_index = _index(labels, "labels")
    missing = sorted(set(label_index) - set(prediction_index))
    extra = sorted(set(prediction_index) - set(label_index))
    if missing or extra:
        raise ValueError(f"prediction/label IDs differ: missing={missing[:5]}, extra={extra[:5]}")
    for identifier, row in label_index.items():
        if row.get("label") not in {"yes", "no"}:
            raise ValueError(f"labels: id {identifier!r} must have canonical label 'yes' or 'no'")
    return prediction_index, label_index


def _pope_metrics(predictions, labels) -> dict[str, int | float]:
    tp = tn = fp = fn = unknown = correct = predicted_yes = 0
    positives = sum(row["label"] == "yes" for row in labels.values())
    for identifier, row in labels.items():
        truth = row["label"]
        prediction = parse_answer(predictions[identifier].get("text"))
        unknown += prediction == "unknown"
        correct += prediction == truth
        predicted_yes += prediction == "yes"
        if truth == "yes":
            tp += prediction == "yes"
            fn += prediction != "yes"
        else:
            tn += prediction == "no"
            fp += prediction == "yes"
    count = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": count,
        "accuracy": 100.0 * correct / count,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "yes_rate": 100.0 * predicted_yes / count,
        "unknown": unknown,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _mme_metrics(predictions, labels) -> dict[str, Any]:
    pairs: dict[tuple[str, str], list[tuple[str, bool]]] = defaultdict(list)
    category_correct = defaultdict(int)
    category_count = defaultdict(int)
    unknown = 0
    for identifier, row in labels.items():
        category = row.get("category")
        if category not in MME_CATEGORIES:
            raise ValueError(f"MME id {identifier!r}: category must be one of {MME_CATEGORIES}")
        image_value = row.get("image_id", row.get("image"))
        image_id = _canonical_id(image_value, "MME image_id/image")
        prediction = parse_answer(predictions[identifier].get("text"))
        correct = prediction == row["label"]
        unknown += prediction == "unknown"
        pairs[(category, image_id)].append((identifier, correct))
        category_correct[category] += correct
        category_count[category] += 1
    if set(category_count) != set(MME_CATEGORIES):
        missing = sorted(set(MME_CATEGORIES) - set(category_count))
        raise ValueError(f"MME-Hall requires all four categories; missing={missing}")
    category_pairs = defaultdict(list)
    for (category, image_id), questions in pairs.items():
        if len(questions) != 2:
            raise ValueError(f"MME pair ({category!r}, {image_id!r}) must have exactly 2 distinct questions; got {len(questions)}")
        category_pairs[category].append(all(correct for _, correct in questions))
    details: dict[str, Any] = {}
    for category in MME_CATEGORIES:
        pair_results = category_pairs[category]
        accuracy = 100.0 * category_correct[category] / category_count[category]
        accuracy_plus = 100.0 * sum(pair_results) / len(pair_results)
        details[category] = {
            "n": category_count[category],
            "pairs": len(pair_results),
            "accuracy": accuracy,
            "accuracy_plus": accuracy_plus,
            "score": accuracy + accuracy_plus,
        }
    return {
        "n": len(labels),
        "unknown": unknown,
        "categories": details,
        "hall_total": sum(details[category]["score"] for category in MME_CATEGORIES),
    }


def evaluate_records(
    benchmark: str,
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate predictions and labels joined by exact sample IDs."""
    prediction_index, label_index = _align(predictions, labels)
    if benchmark == "pope":
        return _pope_metrics(prediction_index, label_index)
    if benchmark == "mme":
        return _mme_metrics(prediction_index, label_index)
    raise ValueError("benchmark must be 'pope' or 'mme'; CHAIR requires its official evaluator")


def _check_hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def score_files(
    benchmark: str,
    predictions_path: str | Path,
    labels_path: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    predictions = read_jsonl(predictions_path)
    labels = read_jsonl(labels_path)
    metrics = evaluate_records(benchmark, predictions, labels)
    sample_hash = sample_ids_sha256(row["id"] for row in labels)
    report: dict[str, Any] = {
        "benchmark": benchmark,
        "sample_ids_sha256": sample_hash,
        "metric_units": "percent; MME category score=Acc+Acc_plus in [0,200]",
        "answer_parser": "standalone yes/no with optional terminal punctuation; unknown answers receive no credit",
        "metrics": metrics,
        "labels_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(Path(predictions_path).read_bytes()).hexdigest(),
    }
    if manifest is not None:
        if "status" in manifest and manifest["status"] != "complete":
            raise ValueError("Only a complete inference manifest may be scored")
        if ("predictions_sha256" in manifest
                and manifest["predictions_sha256"] != report["predictions_sha256"]):
            raise ValueError("Prediction checksum does not match the completed manifest")
        if manifest.get("benchmark", benchmark) != benchmark:
            raise ValueError("manifest benchmark does not match requested benchmark")
        if "sample_ids_sha256" in manifest and manifest["sample_ids_sha256"] != sample_hash:
            raise ValueError("manifest sample_ids_sha256 does not match evaluated IDs")
        for name in (*GROUP_FIELDS[1:], "seed", "config_sha256"):
            if name not in manifest:
                raise ValueError(f"manifest is missing {name}")
            report[name] = manifest[name]
        for name in ("status", "diagnostic_subset", "input_sha256", "source_sha256",
                     "image_sha256", "weak_image_sha256"):
            if name in manifest:
                report[name] = manifest[name]
        _validate_run(report)
    return report


def _flatten_numeric(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(item, Mapping):
            result.update(_flatten_numeric(item, key))
        elif isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"metric {key!r} must be finite and numeric")
        else:
            result[key] = float(item)
    if not result:
        raise ValueError("metrics must be a nonempty numeric mapping")
    return result


def _validate_run(report: Mapping[str, Any]) -> None:
    for field in GROUP_FIELDS:
        if field not in report or not isinstance(report[field], str) or not report[field].strip():
            raise ValueError(f"run is missing nonempty string provenance field {field}")
    if isinstance(report.get("seed"), bool) or not isinstance(report.get("seed"), int):
        raise ValueError("run seed must be an integer")
    _check_hash(report.get("sample_ids_sha256"), "sample_ids_sha256")
    _check_hash(report.get("config_sha256"), "config_sha256")
    if not isinstance(report.get("metrics"), Mapping):
        raise ValueError("run metrics must be an object")
    _flatten_numeric(report["metrics"])


def aggregate_runs(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate matched seeds independently within each benchmark setting."""
    if not reports:
        raise ValueError("no metric runs provided")
    groups = defaultdict(list)
    for report in reports:
        _validate_run(report)
        if report.get("diagnostic_subset"):
            raise ValueError("Diagnostic subset runs cannot be aggregated as benchmark results")
        if "status" in report and report["status"] != "complete":
            raise ValueError("Only complete inference runs may be aggregated")
        key = tuple(report[field] for field in GROUP_FIELDS)
        groups[key].append(report)
    output_groups = []
    for key, runs in sorted(groups.items()):
        seeds = [run["seed"] for run in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate seeds within group {key}")
        if key[0] == "pope" and set(seeds) != set(POPE_SEEDS):
            raise ValueError(f"POPE group {key} must contain exactly seeds {list(POPE_SEEDS)}; got {sorted(seeds)}")
        if len(seeds) < 2:
            raise ValueError(f"group {key} needs at least two seed runs to report sample SD")
        sample_hashes = {run["sample_ids_sha256"] for run in runs}
        config_hashes = {run["config_sha256"] for run in runs}
        if len(sample_hashes) != 1:
            raise ValueError(f"sample IDs differ across seeds in group {key}")
        if len(config_hashes) != 1:
            raise ValueError(f"configuration differs across seeds in group {key}")
        for field in ("input_sha256", "source_sha256", "labels_sha256",
                      "image_sha256", "weak_image_sha256"):
            present = [field in run for run in runs]
            if any(present) and (not all(present) or len({json.dumps(run[field], sort_keys=True) for run in runs}) != 1):
                raise ValueError(f"{field} differs or is missing across seeds in group {key}")
        ordered_runs = sorted(runs, key=lambda run: run["seed"])
        flattened = [_flatten_numeric(run["metrics"]) for run in ordered_runs]
        metric_names = set(flattened[0])
        if any(set(item) != metric_names for item in flattened[1:]):
            raise ValueError(f"metric keys differ across seeds in group {key}")
        summaries = {
            name: {
                "mean": statistics.mean(item[name] for item in flattened),
                "std": statistics.stdev(item[name] for item in flattened),
            }
            for name in sorted(metric_names)
        }
        output_groups.append({
            **dict(zip(GROUP_FIELDS, key)),
            "seeds": [run["seed"] for run in ordered_runs],
            "num_seeds": len(ordered_runs),
            "sample_ids_sha256": next(iter(sample_hashes)),
            "config_sha256": next(iter(config_hashes)),
            "std_ddof": 1,
            "std_convention": SD_CONVENTION,
            "metrics": summaries,
        })
    return {"std_convention": SD_CONVENTION, "std_ddof": 1, "groups": output_groups}


def aggregate_files(paths: Sequence[str | Path]) -> dict[str, Any]:
    reports = []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError(f"{path}: expected a metric report object")
        reports.append(report)
    return aggregate_runs(reports)


def _write_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="score predictions against separate labels")
    score.add_argument("--benchmark", choices=("pope", "mme"), required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--labels", required=True)
    score.add_argument("--manifest", help="optional frozen inference manifest; auto-detects PREDICTIONS.manifest.json")
    score.add_argument("--output", required=True)
    aggregate = subparsers.add_parser("aggregate", help="aggregate matched-seed metric reports by setting")
    aggregate.add_argument("--inputs", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output_path = Path(args.output).resolve()
    source_paths = ([args.predictions, args.labels, args.manifest] if args.command == "score" else args.inputs)
    if output_path in {Path(v).resolve() for v in source_paths if v}:
        parser.error("Scoring output must not overwrite an input file")
    try:
        if args.command == "score":
            manifest_path = Path(args.manifest) if args.manifest else Path(str(args.predictions) + ".manifest.json")
            manifest = None
            if args.manifest or manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if not isinstance(manifest, dict):
                    raise ValueError("manifest must be a JSON object")
            report = score_files(args.benchmark, args.predictions, args.labels, manifest)
        else:
            report = aggregate_files(args.inputs)
        _write_report(args.output, report)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
