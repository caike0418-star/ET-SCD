"""Prepare inference questions and separate benchmark labels.

Normalize question_id/text fields to id/question, set the CHAIR caption prompt,
and select the four MME-Hall categories.
"""
import argparse
import json
from pathlib import Path

CHAIR_PROMPT = "Please describe this image in detail."
HALL_CATEGORIES = {"existence", "count", "position", "color"}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def prepare_records(records, benchmark):
    questions, labels, seen = [], [], set()
    for row in records:
        if benchmark == "mme" and row.get("category") not in HALL_CATEGORIES:
            continue
        identity = row.get("id", row.get("question_id"))
        if identity is None or str(identity) in seen:
            raise ValueError("Missing or duplicate sample ID")
        seen.add(str(identity))
        question = CHAIR_PROMPT if benchmark == "chair" else row.get("question", row.get("text"))
        if not question or not row.get("image"):
            raise ValueError(f"Sample {identity} requires image and question")
        sample = {"id": str(identity), "image": row["image"], "question": question}
        questions.append(sample)
        if benchmark != "chair":
            label = row.get("label")
            if str(label).strip().lower() not in {"yes", "no"}:
                raise ValueError(f"Sample {identity} requires an explicit yes/no label")
            entry = {"id": str(identity), "label": str(label).strip().lower()}
            if benchmark == "mme":
                entry.update(category=row["category"], image_id=str(row.get("image_id", row["image"])))
            labels.append(entry)
    if not questions:
        raise ValueError("No matching samples")
    if benchmark == "chair" and (len(questions) != 500 or len({q['image'] for q in questions}) != 500):
        raise ValueError("TMM CHAIR protocol requires exactly 500 distinct images")
    return questions, labels


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["pope", "mme", "chair"], required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--labels")
    args = p.parse_args()
    if args.benchmark != "chair" and not args.labels:
        p.error("Binary benchmarks require a separate --labels output")
    paths = [Path(v).resolve() for v in (args.input, args.questions, args.labels) if v]
    if len(set(paths)) != len(paths):
        p.error("Input, questions, and labels paths must be different")
    if any(Path(v).exists() for v in (args.questions, args.labels) if v):
        p.error("Output already exists; choose fresh paths")
    questions, labels = prepare_records(read_jsonl(args.input), args.benchmark)
    write_jsonl(args.questions, questions)
    if args.labels:
        write_jsonl(args.labels, labels)
    print(json.dumps({"benchmark": args.benchmark, "samples": len(questions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

