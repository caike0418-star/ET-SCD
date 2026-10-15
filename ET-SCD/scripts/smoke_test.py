"""Load one backbone and run short POPE, MME-Hall and captioning generations."""
import argparse
import json
from pathlib import Path
from PIL import Image
from etscd.adapters import load_backend, resolve_binary_token_ids
from etscd.data import CHAIR_PROMPT
from etscd.runner import run_sample
from etscd.views import weaken_gaussian
from etscd.vocabulary import build_object_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=["llava", "qwen", "instructblip"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--llm-model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", default="Is there a dog in the image? Please answer yes or no.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lexicon", default=str(Path(__file__).resolve().parents[1] / "data/coco_object_concepts.json"))
    parser.add_argument("--output", default="outputs/smoke.json")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--blur-radius", type=float, default=5.0)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.blur_radius <= 0:
        parser.error("Generation length and blur radius must be positive")
    if args.llm_model and args.family != "instructblip":
        parser.error("--llm-model is used with instructblip")
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a fresh output path")
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
    weakened = weaken_gaussian(image, args.blur_radius)
    options = {"llm_model": args.llm_model} if args.llm_model else {}
    backend = load_backend(args.family, args.model_path, args.device, **options)
    try:
        pair = resolve_binary_token_ids(backend.tokenizer)
        concepts = json.loads(Path(args.lexicon).read_text(encoding="utf-8"))
        object_map, _ = build_object_map(backend.tokenizer, concepts)
        result = {"model": backend.metadata(), "image": str(Path(args.image).resolve()), "tasks": {}}
        for benchmark in ("pope", "mme", "chair"):
            question = CHAIR_PROMPT if benchmark == "chair" else args.question
            prediction = run_sample(backend, image, weakened, question, benchmark=benchmark,
                                    binary_pair=pair, object_map=object_map,
                                    max_new_tokens=args.max_new_tokens)
            result["tasks"][benchmark] = prediction
            print(f"{benchmark}: {prediction['text']}", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved {output}")
    finally:
        close = getattr(backend, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
