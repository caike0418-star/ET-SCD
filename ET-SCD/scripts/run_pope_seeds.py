"""Run and score five matched POPE seeds, or print the command sequence.

Labels are passed only to the scoring subprocess, never to inference.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

SEEDS = (42, 55, 123, 2024, 3407)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "labels", "model-path", "image-folder", "output-dir"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--family", choices=["llava", "qwen", "instructblip"], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--setting", choices=["random", "popular", "adversarial"], required=True)
    p.add_argument("--method", choices=["etscd", "regular", "vcd"], default="etscd")
    p.add_argument("--device", default="cuda")
    p.add_argument("--llm-model")
    p.add_argument("--config")
    p.add_argument("--profile", choices=["tmm", "custom"], default="tmm")
    p.add_argument("--weakening", choices=["gaussian-blur", "provided"], required=True)
    p.add_argument("--blur-radius", type=float)
    p.add_argument("--weak-image-folder")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    commands, reports = [], []
    for seed in SEEDS:
        prediction = Path(args.output_dir) / f"seed{seed}.jsonl"
        metrics = Path(args.output_dir) / f"seed{seed}.metrics.json"
        reports.append(str(metrics))
        infer = [sys.executable, "-m", "etscd.runner", "--benchmark", "pope",
                 "--family", args.family, "--model-path", args.model_path,
                 "--questions", args.questions, "--image-folder", args.image_folder,
                 "--output", str(prediction), "--dataset", args.dataset,
                 "--setting", args.setting, "--seed", str(seed), "--method", args.method,
                 "--device", args.device, "--profile", args.profile, "--weakening", args.weakening]
        for option, value in (("--config", args.config), ("--blur-radius", args.blur_radius),
                              ("--weak-image-folder", args.weak_image_folder), ("--llm-model", args.llm_model)):
            if value is not None:
                infer += [option, str(value)]
        score = [sys.executable, "-m", "etscd.evaluation", "score", "--benchmark", "pope",
                 "--predictions", str(prediction), "--labels", args.labels, "--output", str(metrics)]
        commands += [infer, score]
    commands.append([sys.executable, "-m", "etscd.evaluation", "aggregate", "--inputs", *reports,
                     "--output", str(Path(args.output_dir) / "five_seed_summary.json")])
    if args.dry_run:
        print(json.dumps({"seeds": SEEDS, "commands": commands}, indent=2))
        return
    for command in commands:
        print(json.dumps(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
