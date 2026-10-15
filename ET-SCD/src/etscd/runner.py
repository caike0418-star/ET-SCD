"""ET-SCD inference for POPE, MME-Hall and CHAIR.

Compute semantic contrastive evidence, route candidate decisions and selectively
evaluate regional views using the original question and generated prefix.
"""
import argparse
from collections import Counter
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .adapters import load_backend, resolve_binary_token_ids
from .alignment import label_method, validate_alignment
from . import __version__
from .config import ETConfig
from .core import decode_step
from .data import CHAIR_PROMPT, read_jsonl
from .views import corner_views, resize_regional_view, weaken_gaussian


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_questions(rows, benchmark, *, allow_subset=False):
    allowed = {"id", "image", "question"}
    seen = set()
    for row in rows:
        extra = set(row) - allowed
        if extra or not allowed <= set(row):
            raise ValueError(f"Inference input must contain ONLY id,image,question; unexpected={sorted(extra)}")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in allowed):
            raise ValueError("id,image,question must be nonempty strings")
        if row["id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["id"])
        if benchmark == "chair" and row["question"] != CHAIR_PROMPT:
            raise ValueError("CHAIR must use the exact TMM caption prompt")
    if not rows:
        raise ValueError("Empty inference input")
    if benchmark == "chair" and not allow_subset and (len(rows) != 500 or len({r['image'] for r in rows}) != 500):
        raise ValueError("CHAIR requires the fixed 500 distinct images; --allow-subset marks a diagnostic run")


def resolve_image(folder, relative):
    folder = Path(folder).resolve()
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder):
        raise ValueError("Image path escapes the declared image folder")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_sample(backend, image, weakened, question, *, config=ETConfig(), benchmark,
               binary_pair=None, object_map=None, max_new_tokens=64, method="etscd"):
    """Single sample with lazy re-query, exact prefix IDs and real forward counts."""
    if benchmark not in {"pope", "mme", "chair"}:
        raise ValueError("Unsupported benchmark")
    if method not in {"etscd", "regular", "vcd"}:
        raise ValueError("Unsupported decoding method")
    if benchmark != "chair" and binary_pair is None:
        raise ValueError("Binary benchmark requires resolved answer tokens")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prefix = []
    traces = []
    counts = Counter(original=0, weakened=0, requery=0)
    backend.synchronize()
    started = time.perf_counter()
    for _ in range(1 if benchmark != "chair" else max_new_tokens):
        ids = tuple(prefix)
        le = backend.next_logits(image, question, ids)
        counts["original"] += 1
        if method == "regular":
            la = le.copy()
        else:
            la = backend.next_logits(weakened, question, ids)
            counts["weakened"] += 1

        def query(targets):
            for view_id, view in corner_views(image, targets, config):
                view = resize_regional_view(view, backend.visual_input_size)
                logits = backend.next_logits(view, question, ids)
                counts["requery"] += 1
                yield view_id, logits

        pair = tuple(binary_pair) if benchmark != "chair" else None
        if method != "etscd":
            scores = le if method == "regular" else le + config.alpha * (le - la)
            candidates = pair if pair is not None else range(len(le))
            token = min(candidates, key=lambda i: (-scores[i], i))
            trace = {"token_id": int(token), "reason": method, "activated": False,
                     "route": [], "view_forwards": 0, "requery_rounds": 0, "updated": False}
        else:
            step = decode_step(le, la, config=config, binary_ids=pair,
                               object_ids=frozenset(object_map or {}), target_for_token=object_map,
                               question_target=question if pair else None, query_views=query)
            token, trace = step.token_id, step.trace
        prefix.append(int(token))
        traces.append(trace)
        if int(token) in backend.eos_ids:
            break
    backend.synchronize()
    elapsed = time.perf_counter() - started
    if benchmark != "chair":
        text = "Yes" if prefix[-1] == binary_pair.yes_id else "No"
    else:
        text = backend.decode(prefix)
    return {"text": text, "token_ids": prefix, "steps": traces,
            "efficiency": {"forward_calls": dict(counts), "total_forward_calls": sum(counts.values()),
                           "generation_seconds": elapsed, "generated_token_count_including_eos": len(prefix),
                           "peak_allocated_cuda_bytes": backend.memory_bytes(),
                           "flops": None, "cache_mode": "full_context_recompute"}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, choices=["pope", "mme", "chair"])
    p.add_argument("--family", required=True, choices=["llava", "qwen", "instructblip"])
    p.add_argument("--model-path", required=True)
    p.add_argument("--llm-model", help="Local Vicuna language-model directory for InstructBLIP")
    p.add_argument("--device", default="cuda")
    p.add_argument("--questions", required=True)
    p.add_argument("--image-folder", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", help="Optional JSON override; default is TMM supplement Table VI")
    p.add_argument("--profile", choices=["tmm", "custom"], default="tmm")
    p.add_argument("--method", choices=["etscd", "regular", "vcd"], default="etscd")
    p.add_argument("--weakening", choices=["gaussian-blur", "provided"], required=True)
    p.add_argument("--blur-radius", type=float)
    p.add_argument("--weak-image-folder")
    p.add_argument("--object-token-map", help="JSON token ID -> canonical object concept, CHAIR only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", required=True)
    p.add_argument("--setting", required=True)
    p.add_argument("--allow-subset", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=64)
    args = p.parse_args()
    if args.weakening == "gaussian-blur" and (args.blur_radius is None or not np.isfinite(args.blur_radius) or args.blur_radius <= 0):
        p.error("gaussian-blur requires a positive --blur-radius")
    if args.weakening == "provided" and not args.weak_image_folder:
        p.error("provided weakening requires --weak-image-folder")
    if args.benchmark == "chair" and args.method == "etscd" and not args.object_token_map:
        p.error("CHAIR ET-SCD requires an explicit tokenizer-specific --object-token-map")
    if args.llm_model and args.family != "instructblip":
        p.error("--llm-model is used with --family instructblip")
    output = Path(args.output)
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        p.error("Output or manifest already exists; choose a fresh output path")
    rows = read_jsonl(args.questions)
    validate_questions(rows, args.benchmark, allow_subset=args.allow_subset)
    config = (ETConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
              if args.config else ETConfig())
    try:
        profile = validate_alignment(config, benchmark=args.benchmark, profile=args.profile,
                                     allow_subset=args.allow_subset, max_new_tokens=args.max_new_tokens,
                                     method=args.method)
    except ValueError as exc:
        p.error(str(exc))
    method_label = label_method(args.method, profile)
    object_map = None
    if args.object_token_map:
        raw = json.loads(Path(args.object_token_map).read_text(encoding="utf-8"))
        raw = raw.get("tokens", raw)
        object_map = {int(k): str(v) for k, v in raw.items()}
        if not object_map or any(i < 0 or not c.strip() for i, c in object_map.items()):
            p.error("Object map must contain nonnegative token IDs and nonempty concepts")
    # Validate every image before model loading or producing partial output.
    image_paths = [resolve_image(args.image_folder, row["image"]) for row in rows]
    weak_paths = [resolve_image(args.weak_image_folder, row["image"]) for row in rows] if args.weakening == "provided" else None
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    backend_options = {"llm_model": args.llm_model} if args.llm_model else {}
    backend = load_backend(args.family, args.model_path, args.device, **backend_options)
    pair = resolve_binary_token_ids(backend.tokenizer) if args.benchmark != "chair" else None
    settings = {"policy": config.to_dict(), "profile": profile, "method": method_label, "weakening": args.weakening,
                "blur_radius": args.blur_radius, "object_map_sha256": file_hash(args.object_token_map) if args.object_token_map else None,
                "max_new_tokens": args.max_new_tokens, "binary_tokens": pair.metadata() if pair else None,
                "backend": backend.metadata(), "tie_order": "ascending_token_id_then_view_order",
                "view_entropy": "original_full_vocabulary_entropy", "native_stop": "backend_eos_ids",
                "eos_token_ids": sorted(backend.eos_ids)}
    from .evaluation import sample_ids_sha256
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    source = Path(__file__).parent
    source_hashes = {path.relative_to(source).as_posix(): file_hash(path)
                     for path in sorted(source.rglob("*.py"))}
    manifest = {"status": "running", "implementation": "ET-SCD", "implementation_version": __version__,
                "benchmark": args.benchmark, "dataset": args.dataset, "setting": args.setting,
                "model": args.family, "method": method_label, "seed": args.seed,
                "diagnostic_subset": args.allow_subset, "sample_count": len(rows),
                "sample_ids_sha256": sample_ids_sha256([row["id"] for row in rows]),
                "input_sha256": file_hash(args.questions), "config_sha256": canonical_hash(settings),
                "settings": settings, "source_sha256": source_hashes, "git_revision": revision,
                "command": sys.argv, "python": platform.python_version(),
                "random_seeds": {"python": args.seed, "numpy": args.seed, "torch": args.seed},
                "image_sha256": {row["id"]: file_hash(path) for row, path in zip(rows, image_paths)},
                "weak_image_sha256": {row["id"]: file_hash(path) for row, path in zip(rows, weak_paths)} if weak_paths else None}
    output.parent.mkdir(parents=True, exist_ok=True)
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_manifest()
    totals = Counter()
    try:
        with output.open("x", encoding="utf-8") as handle:
            for n, (row, image_path) in enumerate(zip(rows, image_paths)):
                with Image.open(image_path) as raw:
                    image = raw.convert("RGB")
                if weak_paths:
                    with Image.open(weak_paths[n]) as raw:
                        weakened = raw.convert("RGB")
                else:
                    weakened = weaken_gaussian(image, args.blur_radius)
                result = run_sample(backend, image, weakened, row["question"], config=config,
                                    benchmark=args.benchmark, binary_pair=pair, object_map=object_map,
                                    max_new_tokens=args.max_new_tokens, method=args.method)
                result.update(id=row["id"], image=row["image"], question=row["question"])
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                totals.update(result["efficiency"]["forward_calls"])
                print(f"{n + 1}/{len(rows)} {row['id']}", flush=True)
        manifest.update(status="complete", forward_calls=dict(totals), predictions_sha256=file_hash(output))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save_manifest()
        close = getattr(backend, "close", None)
        if close:
            close()


if __name__ == "__main__":
    main()
